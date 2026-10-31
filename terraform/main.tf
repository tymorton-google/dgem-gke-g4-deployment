# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# 1. Enable Required Google Cloud APIs & Define G4 Hardware / Pattern Locals
# ==============================================================================
locals {
  required_apis = [
    "compute.googleapis.com",
    "container.googleapis.com",
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "secretmanager.googleapis.com",
  ]

  # Official GKE G4 accelerator count mapping per machine type
  # Verified against https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-create-g2-g4:
  # Fractional vGPU shapes (g4-standard-12, g4-standard-24) and single-GPU g4-standard-48 all require count = 1.
  g4_gpu_count_by_shape = {
    "g4-standard-12"  = 1 # 1/4 vGPU (24 GB GDDR7), 12 vCPUs, 48 GB RAM
    "g4-standard-24"  = 1 # 1/2 vGPU (48 GB GDDR7), 24 vCPUs, 96 GB RAM
    "g4-standard-48"  = 1 # 1x full GPU (96 GB GDDR7), 48 vCPUs, 180 GB RAM
    "g4-standard-96"  = 2 # 2x full GPUs (192 GB GDDR7), 96 vCPUs, 360 GB RAM
    "g4-standard-192" = 4 # 4x full GPUs (384 GB GDDR7), 192 vCPUs, 720 GB RAM
    "g4-standard-384" = 8 # 8x full GPUs (768 GB GDDR7), 384 vCPUs, 1,440 GB RAM
  }

  fractional_g4_shapes = [
    "g4-standard-6",
    "g4-standard-12",
    "g4-standard-24",
  ]

  # Effective pool enablement and baseline sizing derived from deployment_pattern
  enable_spot_pool_effective = (
    var.deployment_pattern == "ondemand_only" || var.deployment_pattern == "dws_flex_batch"
    ? false
    : var.enable_spot_pool
  )

  enable_dws_flex_pool_effective = (
    var.deployment_pattern == "ondemand_only"
    ? false
    : var.enable_dws_flex_pool
  )

  enable_ondemand_pool_effective = (
    var.deployment_pattern == "dws_flex_batch"
    ? false
    : var.enable_ondemand_pool
  )

  # Pattern 4 (hybrid_baseline_plus_burst) and Pattern 3 (ondemand_only) enforce >=1 always-on On-Demand node
  ondemand_initial_nodes_effective = (
    var.deployment_pattern == "hybrid_baseline_plus_burst" || var.deployment_pattern == "ondemand_only"
    ? max(1, var.ondemand_min_nodes)
    : 0
  )

  ondemand_min_nodes_effective = (
    var.deployment_pattern == "hybrid_baseline_plus_burst" || var.deployment_pattern == "ondemand_only"
    ? max(1, var.ondemand_min_nodes)
    : var.ondemand_min_nodes
  )

  ondemand_max_nodes_effective = max(local.ondemand_min_nodes_effective, var.ondemand_max_nodes)

  baseline_gpu_count = lookup(local.g4_gpu_count_by_shape, var.baseline_g4_machine_type, var.g4_gpu_count)
  burst_gpu_count    = lookup(local.g4_gpu_count_by_shape, var.burst_g4_machine_type, var.g4_gpu_count)
}

resource "google_project_service" "apis" {
  for_each           = toset(local.required_apis)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

data "google_project" "current" {
  project_id = var.project_id
  depends_on = [google_project_service.apis]
}

# ==============================================================================
# 2. Custom VPC Network, Private Subnet & Cloud NAT
#    (Complies with compute.vmExternalIpAccess & compute.skipDefaultNetworkCreation)
# ==============================================================================
resource "google_compute_network" "dgem_vpc" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false
  project                 = var.project_id
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "dgem_subnet" {
  name                     = "${var.cluster_name}-subnet"
  ip_cidr_range            = "10.20.0.0/20"
  region                   = var.region
  network                  = google_compute_network.dgem_vpc.id
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "pods-range"
    ip_cidr_range = "10.24.0.0/14"
  }

  secondary_ip_range {
    range_name    = "services-range"
    ip_cidr_range = "10.28.0.0/20"
  }
}

resource "google_compute_router" "dgem_router" {
  name    = "${var.cluster_name}-router"
  region  = var.region
  network = google_compute_network.dgem_vpc.id
}

resource "google_compute_router_nat" "dgem_nat" {
  name                               = "${var.cluster_name}-nat"
  router                             = google_compute_router.dgem_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# ==============================================================================
# 3. Artifact Registry & GCS Buckets (Weights + HNS Pod Snapshot Bucket)
# ==============================================================================
resource "google_artifact_registry_repository" "dgem_repo" {
  location      = var.region
  repository_id = "dgem-images"
  description   = "Container images for dgem GPU engine (vLLM + structured_server) and dgem gateway"
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# Model Weights, FlashInfer JIT Cache, and Domain Calibration Receipts Bucket
resource "google_storage_bucket" "dgem_weights" {
  name                        = "${var.project_id}-dgem-weights"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.apis]
}

# GKE Pod Snapshots Bucket (Requires Hierarchical Namespace + Uniform Access + Soft Delete = 0)
resource "google_storage_bucket" "dgem_pod_snapshots" {
  provider                    = google-beta
  name                        = "${var.project_id}-dgem-pod-snapshots"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true

  hierarchical_namespace {
    enabled = true
  }

  soft_delete_policy {
    retention_duration_seconds = 0
  }

  lifecycle_rule {
    condition {
      age = 14
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.apis]
}

# ==============================================================================
# 4. Least-Privilege Service Accounts & Workload Identity IAM
# ==============================================================================

# 4a. GKE Node Pool Service Account
resource "google_service_account" "gke_node_sa" {
  account_id   = "dgem-gke-node-sa"
  display_name = "dgem GKE G4 Node Pool Service Account"
}

resource "google_project_iam_member" "node_sa_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/artifactregistry.reader",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_node_sa.email}"
}

resource "google_storage_bucket_iam_member" "node_sa_weights_reader" {
  bucket = google_storage_bucket.dgem_weights.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.gke_node_sa.email}"
}

# Required for GKE Pod Snapshots (node SA, workload SA, and GKE service agent)
# Verified against https://docs.cloud.google.com/kubernetes-engine/docs/how-to/pod-snapshots-prepare
resource "google_storage_bucket_iam_member" "node_sa_snapshot_admin" {
  for_each = toset([
    "roles/storage.bucketViewer",
    "roles/storage.objectUser",
    "roles/storage.insightsCollectorService",
  ])
  bucket = google_storage_bucket.dgem_pod_snapshots.name
  role   = each.value
  member = "serviceAccount:${google_service_account.gke_node_sa.email}"
}

resource "google_storage_bucket_iam_member" "gke_service_agent_snapshot_user" {
  bucket = google_storage_bucket.dgem_pod_snapshots.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:service-${data.google_project.current.number}@container-engine-robot.iam.gserviceaccount.com"
}

# 4b. dgem GPU Engine Workload Identity Service Account
resource "google_service_account" "dgem_engine_gsa" {
  account_id   = "dgem-engine-gsa"
  display_name = "dgem GPU Engine Workload Service Account"
}

resource "google_storage_bucket_iam_member" "engine_weights_reader" {
  bucket = google_storage_bucket.dgem_weights.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.dgem_engine_gsa.email}"
}

resource "google_storage_bucket_iam_member" "engine_snapshot_user" {
  for_each = toset([
    "roles/storage.bucketViewer",
    "roles/storage.objectUser",
  ])
  bucket = google_storage_bucket.dgem_pod_snapshots.name
  role   = each.value
  member = "serviceAccount:${google_service_account.dgem_engine_gsa.email}"
}

resource "google_service_account_iam_member" "engine_wi_binding" {
  service_account_id = google_service_account.dgem_engine_gsa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.k8s_namespace}/dgem-engine-ksa]"
  depends_on         = [google_container_cluster.dgem_cluster]
}

# 4c. dgem Gateway Workload Identity Service Account (Calls Vertex AI Gemini 3.5 Flash-Lite)
resource "google_service_account" "dgem_gateway_gsa" {
  account_id   = "dgem-gateway-gsa"
  display_name = "dgem Gateway & Stage 2 Gemini Cascade Service Account"
}

resource "google_project_iam_member" "gateway_vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.dgem_gateway_gsa.email}"
}

resource "google_storage_bucket_iam_member" "gateway_weights_reader" {
  bucket = google_storage_bucket.dgem_weights.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.dgem_gateway_gsa.email}"
}

resource "google_service_account_iam_member" "gateway_wi_binding" {
  service_account_id = google_service_account.dgem_gateway_gsa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.k8s_namespace}/dgem-gateway-ksa]"
  depends_on         = [google_container_cluster.dgem_cluster]
}

# ==============================================================================
# 5. GKE Standard Cluster (G4 Ready, 1.35.8-gke.1518000+, Image Streaming, WI)
# ==============================================================================
resource "google_container_cluster" "dgem_cluster" {
  provider       = google-beta
  name           = var.cluster_name
  location       = var.region
  node_locations = var.zones

  network    = google_compute_network.dgem_vpc.id
  subnetwork = google_compute_subnetwork.dgem_subnet.id

  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false

  min_master_version = var.gke_min_master_version

  release_channel {
    channel = var.gke_release_channel
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods-range"
    services_secondary_range_name = "services-range"
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
    gcs_fuse_csi_driver_config {
      enabled = true
    }
    http_load_balancing {
      disabled = false
    }
  }

  enable_shielded_nodes = true

  depends_on = [
    google_project_service.apis,
    google_compute_router_nat.dgem_nat,
  ]
}

# Lightweight CPU System Node Pool (Hosts dgem-gateway, Kueue controller, & kube-system)
resource "google_container_node_pool" "system_cpu_pool" {
  provider   = google-beta
  name       = "system-cpu-pool"
  location   = var.region
  cluster    = google_container_cluster.dgem_cluster.name
  node_count = 1

  autoscaling {
    min_node_count = 1
    max_node_count = 3
  }

  node_config {
    machine_type    = "e2-standard-4"
    disk_type       = "pd-balanced"
    disk_size_gb    = 100
    service_account = google_service_account.gke_node_sa.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

# ==============================================================================
# 6A. Option 1 / Burst Priority 1: GKE Spot Node Pool with G4 GPUs + Fast Start
#     - Machine: var.burst_g4_machine_type (default g4-standard-48: 1x RTX PRO 6000 96GB)
#     - Fast Start: Image Streaming (GCFS) + Secondary Boot Disk (Container/Weight Cache)
#                   + Hyperdisk Balanced (no Local SSDs by default for Fast-Starting Nodes)
#     - Optional GPU Sharing: NVIDIA MPS (2 or 3 clients/GPU) or MIG (2g.48gb / 1g.24gb)
# ==============================================================================
resource "google_container_node_pool" "g4_spot" {
  count          = local.enable_spot_pool_effective ? 1 : 0
  provider       = google-beta
  name           = "g4-spot-faststart"
  location       = var.region
  cluster        = google_container_cluster.dgem_cluster.name
  node_locations = var.zones

  lifecycle {
    precondition {
      condition     = !(var.gpu_sharing_strategy == "mps" && contains(local.fractional_g4_shapes, var.burst_g4_machine_type))
      error_message = "GKE does not allow NVIDIA MPS (gpu_sharing_strategy = \"mps\") on fractional <1 GPU G4 machine types (g4-standard-6, g4-standard-12, g4-standard-24). Use g4-standard-48 or larger for burst_g4_machine_type."
    }
    precondition {
      condition     = !(var.gpu_sharing_strategy == "mig" && contains(local.fractional_g4_shapes, var.burst_g4_machine_type))
      error_message = "GKE does not allow Multi-Instance GPU (gpu_sharing_strategy = \"mig\") on fractional <1 GPU G4 machine types (g4-standard-6, g4-standard-12, g4-standard-24). Use g4-standard-48 or larger for burst_g4_machine_type."
    }
  }

  autoscaling {
    total_min_node_count = 0
    total_max_node_count = var.spot_max_nodes
    location_policy      = "ANY"
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = var.burst_g4_machine_type
    spot            = true
    image_type      = "COS_CONTAINERD"
    disk_type       = "hyperdisk-balanced"
    disk_size_gb    = 200
    service_account = google_service_account.gke_node_sa.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type               = var.g4_gpu_type
      count              = local.burst_gpu_count
      gpu_partition_size = var.gpu_sharing_strategy == "mig" ? var.mig_partition_size : null

      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }

      dynamic "gpu_sharing_config" {
        for_each = var.gpu_sharing_strategy == "mps" ? [1] : []
        content {
          gpu_sharing_strategy       = "MPS"
          max_shared_clients_per_gpu = var.mps_max_shared_clients
        }
      }
    }

    # Enable GKE Image Streaming for fast container layer streaming
    gcfs_config {
      enabled = true
    }

    # Optional Local NVMe SSD (defaults to 0 to preserve GKE Fast-Starting Node eligibility on G4)
    dynamic "ephemeral_storage_local_ssd_config" {
      for_each = var.g4_local_ssd_count > 0 ? [var.g4_local_ssd_count] : []
      content {
        local_nvme_ssd_block_config {
          count = ephemeral_storage_local_ssd_config.value
        }
      }
    }

    # Attach pre-hydrated Secondary Boot Disk when generated by generate_fast_start_snapshots.sh
    dynamic "secondary_boot_disks" {
      for_each = var.secondary_boot_disk_image != "" ? [var.secondary_boot_disk_image] : []
      content {
        disk_image = secondary_boot_disks.value
        mode       = "CONTAINER_IMAGE_CACHE"
      }
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    labels = {
      "dgem.cloud.google.com/role"         = "gpu-engine"
      "dgem.cloud.google.com/provisioning" = "spot"
      "dgem.cloud.google.com/pool-role"    = "elastic-burst"
      "dgem.cloud.google.com/gpu-sharing"  = var.gpu_sharing_strategy
      "cloud.google.com/compute-class"     = "dgem-g4-priority"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    taint {
      key    = "cloud.google.com/compute-class"
      value  = "dgem-g4-priority"
      effect = "NO_SCHEDULE"
    }
  }
}

# ==============================================================================
# 6B. Option 2 / Burst Priority 2: DWS Flex-Start Node Pool with G4 GPUs
#     - Uses Dynamic Workload Scheduler (DWS) Flex-Start to queue and allocate
#       discount G4 GPUs for up to 7 days with non-preemptible duration guarantees
# ==============================================================================
resource "google_container_node_pool" "g4_dws_flex" {
  count          = local.enable_dws_flex_pool_effective ? 1 : 0
  provider       = google-beta
  name           = "g4-dws-flex"
  location       = var.region
  cluster        = google_container_cluster.dgem_cluster.name
  node_locations = var.zones

  lifecycle {
    precondition {
      condition     = !(var.gpu_sharing_strategy == "mps" && contains(local.fractional_g4_shapes, var.burst_g4_machine_type))
      error_message = "GKE does not allow NVIDIA MPS (gpu_sharing_strategy = \"mps\") on fractional <1 GPU G4 machine types (g4-standard-6, g4-standard-12, g4-standard-24). Use g4-standard-48 or larger for burst_g4_machine_type."
    }
    precondition {
      condition     = !(var.gpu_sharing_strategy == "mig" && contains(local.fractional_g4_shapes, var.burst_g4_machine_type))
      error_message = "GKE does not allow Multi-Instance GPU (gpu_sharing_strategy = \"mig\") on fractional <1 GPU G4 machine types (g4-standard-6, g4-standard-12, g4-standard-24). Use g4-standard-48 or larger for burst_g4_machine_type."
    }
  }

  initial_node_count = 0

  autoscaling {
    total_min_node_count = 0
    total_max_node_count = var.dws_max_nodes
    location_policy      = "ANY"
  }

  # DWS Queued Provisioning requires auto_repair = false
  management {
    auto_repair  = false
    auto_upgrade = true
  }

  queued_provisioning {
    enabled = true
  }

  node_config {
    machine_type    = var.burst_g4_machine_type
    flex_start      = true
    image_type      = "COS_CONTAINERD"
    disk_type       = "hyperdisk-balanced"
    disk_size_gb    = 200
    service_account = google_service_account.gke_node_sa.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type               = var.g4_gpu_type
      count              = local.burst_gpu_count
      gpu_partition_size = var.gpu_sharing_strategy == "mig" ? var.mig_partition_size : null

      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }

      dynamic "gpu_sharing_config" {
        for_each = var.gpu_sharing_strategy == "mps" ? [1] : []
        content {
          gpu_sharing_strategy       = "MPS"
          max_shared_clients_per_gpu = var.mps_max_shared_clients
        }
      }
    }

    gcfs_config {
      enabled = true
    }

    dynamic "ephemeral_storage_local_ssd_config" {
      for_each = var.g4_local_ssd_count > 0 ? [var.g4_local_ssd_count] : []
      content {
        local_nvme_ssd_block_config {
          count = ephemeral_storage_local_ssd_config.value
        }
      }
    }

    dynamic "secondary_boot_disks" {
      for_each = var.secondary_boot_disk_image != "" ? [var.secondary_boot_disk_image] : []
      content {
        disk_image = secondary_boot_disks.value
        mode       = "CONTAINER_IMAGE_CACHE"
      }
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    labels = {
      "dgem.cloud.google.com/role"         = "gpu-engine"
      "dgem.cloud.google.com/provisioning" = "dws-flex"
      "dgem.cloud.google.com/pool-role"    = "elastic-burst"
      "dgem.cloud.google.com/gpu-sharing"  = var.gpu_sharing_strategy
      "cloud.google.com/compute-class"     = "dgem-g4-priority"
    }

    taint {
      key    = "cloud.google.com/gke-queued"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    taint {
      key    = "cloud.google.com/compute-class"
      value  = "dgem-g4-priority"
      effect = "NO_SCHEDULE"
    }
  }
}

# ==============================================================================
# 6C. Option 3 / Pattern 4 Always-On Baseline: On-Demand Node Pool with G4 GPUs
#     - When deployment_pattern == "hybrid_baseline_plus_burst", sets
#       initial_node_count >= 1 & total_min_node_count >= 1 and labels nodes with
#       "dgem.cloud.google.com/pool-role" = "baseline-ondemand" so the model is
#       100% always available with zero cold-start delay.
#     - Can use a cost-optimized fractional vGPU shape (g4-standard-24 48GB at
#       ~50% lower On-Demand cost) or a full GPU shape (g4-standard-48 96GB).
# ==============================================================================
resource "google_container_node_pool" "g4_ondemand" {
  count          = local.enable_ondemand_pool_effective ? 1 : 0
  provider       = google-beta
  name           = "g4-ondemand"
  location       = var.region
  cluster        = google_container_cluster.dgem_cluster.name
  node_locations = var.zones

  lifecycle {
    precondition {
      condition     = !(var.gpu_sharing_strategy == "mps" && contains(local.fractional_g4_shapes, var.baseline_g4_machine_type))
      error_message = "GKE does not allow NVIDIA MPS (gpu_sharing_strategy = \"mps\") on fractional <1 GPU G4 machine types (g4-standard-6, g4-standard-12, g4-standard-24). Set baseline_g4_machine_type = \"g4-standard-48\" (or larger) when using MPS, or set gpu_sharing_strategy = \"none\"."
    }
    precondition {
      condition     = !(var.gpu_sharing_strategy == "mig" && contains(local.fractional_g4_shapes, var.baseline_g4_machine_type))
      error_message = "GKE does not allow Multi-Instance GPU (gpu_sharing_strategy = \"mig\") on fractional <1 GPU G4 machine types (g4-standard-6, g4-standard-12, g4-standard-24). Set baseline_g4_machine_type = \"g4-standard-48\" (or larger) when using MIG, or set gpu_sharing_strategy = \"none\"."
    }
  }

  initial_node_count = local.ondemand_initial_nodes_effective

  autoscaling {
    total_min_node_count = local.ondemand_min_nodes_effective
    total_max_node_count = local.ondemand_max_nodes_effective
    location_policy      = "ANY"
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = var.baseline_g4_machine_type
    spot            = false
    image_type      = "COS_CONTAINERD"
    disk_type       = "hyperdisk-balanced"
    disk_size_gb    = 200
    service_account = google_service_account.gke_node_sa.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type               = var.g4_gpu_type
      count              = local.baseline_gpu_count
      gpu_partition_size = var.gpu_sharing_strategy == "mig" ? var.mig_partition_size : null

      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }

      dynamic "gpu_sharing_config" {
        for_each = var.gpu_sharing_strategy == "mps" ? [1] : []
        content {
          gpu_sharing_strategy       = "MPS"
          max_shared_clients_per_gpu = var.mps_max_shared_clients
        }
      }
    }

    gcfs_config {
      enabled = true
    }

    dynamic "ephemeral_storage_local_ssd_config" {
      for_each = var.g4_local_ssd_count > 0 ? [var.g4_local_ssd_count] : []
      content {
        local_nvme_ssd_block_config {
          count = ephemeral_storage_local_ssd_config.value
        }
      }
    }

    dynamic "secondary_boot_disks" {
      for_each = var.secondary_boot_disk_image != "" ? [var.secondary_boot_disk_image] : []
      content {
        disk_image = secondary_boot_disks.value
        mode       = "CONTAINER_IMAGE_CACHE"
      }
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    labels = {
      "dgem.cloud.google.com/role"         = "gpu-engine"
      "dgem.cloud.google.com/provisioning" = "ondemand"
      "dgem.cloud.google.com/pool-role"    = var.deployment_pattern == "hybrid_baseline_plus_burst" ? "baseline-ondemand" : "fallback-ondemand"
      "dgem.cloud.google.com/gpu-sharing"  = var.gpu_sharing_strategy
      "cloud.google.com/compute-class"     = "dgem-g4-priority"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    taint {
      key    = "cloud.google.com/compute-class"
      value  = "dgem-g4-priority"
      effect = "NO_SCHEDULE"
    }
  }
}

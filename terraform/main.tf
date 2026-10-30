# ==============================================================================
# 1. Enable Required Google Cloud APIs
# ==============================================================================
locals {
  required_apis = [
    "compute.googleapis.com",
    "container.googleapis.com",
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "secretmanager.googleapis.com",
  ]
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
#    Google Cloud Production Enterprise Security Best Practices:
#    - Satisfies constraints/compute.vmExternalIpAccess (zero public node IPs)
#    - Satisfies constraints/compute.skipDefaultNetworkCreation
#    - Enables Private Google Access for zero-egress-cost GCS & Artifact Registry
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
#    Google Cloud Production Enterprise Security Best Practices:
#    - Enforces uniform_bucket_level_access (constraints/storage.uniformBucketLevelAccess)
# ==============================================================================
resource "google_artifact_registry_repository" "dgem_repo" {
  location      = var.region
  repository_id = "dgem-images"
  description   = "Container images for dgem GPU engine (vLLM + structured_server) and dgem gateway"
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# Model Weights, FlashInfer Blackwell SM120 JIT Cache, and Calibration Receipts Bucket
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
# Reference: https://cloud.google.com/kubernetes-engine/docs/how-to/pod-snapshots-prepare
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
# 4. Least-Privilege Service Accounts & Keyless Workload Identity Federation
#    Google Cloud Production Enterprise Security Best Practices:
#    - Complies with constraints/iam.disableServiceAccountKeyCreation (zero exported JSON keys)
#    - Uses GKE Workload Identity Federation (svc.id.goog) for pod-level IAM
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
# Reference: https://cloud.google.com/kubernetes-engine/docs/how-to/pod-snapshots-prepare
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
# 5. GKE Standard Cluster (G4 Ready, Image Streaming, Workload Identity, GCS FUSE)
#    Google Cloud Production Enterprise Security Best Practices:
#    - enable_private_nodes = true (nodes have RFC 1918 private IPs only)
#    - enable_shielded_nodes = true (constraints/compute.requireShieldedVm)
#    - workload_identity_config enabled for keyless pod authentication
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
# 6A. Option 1: GKE Spot Node Pool with G4 GPUs + Fast Start
#     - Machine: g4-standard-48 (1x nvidia-rtx-pro-6000 96GB GDDR7, 48 vCPU, 180GB RAM)
#     - Fast Start: Image Streaming (GCFS) + Secondary Boot Disk (CONTAINER_IMAGE_CACHE)
#                   + Hyperdisk Balanced boot disk (G4 requires Hyperdisk)
# ==============================================================================
resource "google_container_node_pool" "g4_spot" {
  count          = var.enable_spot_pool ? 1 : 0
  provider       = google-beta
  name           = "g4-spot-faststart"
  location       = var.region
  cluster        = google_container_cluster.dgem_cluster.name
  node_locations = var.zones

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
    machine_type    = var.g4_machine_type
    spot            = true
    image_type      = "COS_CONTAINERD"
    disk_type       = "hyperdisk-balanced"
    disk_size_gb    = 200
    service_account = google_service_account.gke_node_sa.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type  = var.g4_gpu_type
      count = var.g4_gpu_count
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
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

    # Attach pre-hydrated Secondary Boot Disk when generated by scripts/generate_fast_start_snapshots.sh
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
# 6B. Option 2: DWS Flex-Start Node Pool with G4 GPUs (Queued Provisioning)
#     - Uses Dynamic Workload Scheduler (DWS) Flex-Start to queue and allocate
#       discount G4 GPUs for up to 7 days with non-preemptible duration guarantees
# ==============================================================================
resource "google_container_node_pool" "g4_dws_flex" {
  count          = var.enable_dws_flex_pool ? 1 : 0
  provider       = google-beta
  name           = "g4-dws-flex"
  location       = var.region
  cluster        = google_container_cluster.dgem_cluster.name
  node_locations = var.zones

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
    machine_type    = var.g4_machine_type
    flex_start      = true
    image_type      = "COS_CONTAINERD"
    disk_type       = "hyperdisk-balanced"
    disk_size_gb    = 200
    service_account = google_service_account.gke_node_sa.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type  = var.g4_gpu_type
      count = var.g4_gpu_count
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
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
# 6C. Option 3: On-Demand Node Pool with G4 GPUs
#     - Guaranteed baseline capacity or ComputeClass fallback when Spot is preempted
#     - Eligible for GKE Fast-Starting Nodes (hyperdisk-balanced + zero Local SSDs)
# ==============================================================================
resource "google_container_node_pool" "g4_ondemand" {
  count          = var.enable_ondemand_pool ? 1 : 0
  provider       = google-beta
  name           = "g4-ondemand"
  location       = var.region
  cluster        = google_container_cluster.dgem_cluster.name
  node_locations = var.zones

  autoscaling {
    total_min_node_count = var.ondemand_min_nodes
    total_max_node_count = var.ondemand_max_nodes
    location_policy      = "ANY"
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = var.g4_machine_type
    spot            = false
    image_type      = "COS_CONTAINERD"
    disk_type       = "hyperdisk-balanced"
    disk_size_gb    = 200
    service_account = google_service_account.gke_node_sa.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type  = var.g4_gpu_type
      count = var.g4_gpu_count
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
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

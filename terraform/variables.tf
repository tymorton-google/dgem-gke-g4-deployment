variable "project_id" {
  description = "Target GCP Project ID for deploying dgem on GKE with G4 GPUs."
  type        = string
}

variable "region" {
  description = "GCP Region with NVIDIA RTX PRO 6000 (G4) availability (e.g., us-central1, us-east4, europe-west4)."
  type        = string
  default     = "us-central1"
}

variable "zones" {
  description = "Zones within the region where G4 instances are available (in us-central1: us-central1-b, us-central1-c, us-central1-f)."
  type        = list(string)
  default     = ["us-central1-b", "us-central1-c", "us-central1-f"]
}

variable "cluster_name" {
  description = "Name of the GKE Standard cluster."
  type        = string
  default     = "dgem-g4-cluster"
}

variable "gke_release_channel" {
  description = "GKE release channel. G4 GPUs require GKE 1.34.0-gke.1662000+ (RAPID or REGULAR recommended for 1.35.8+ fractional vGPU & ComputeClass gpuSharing features)."
  type        = string
  default     = "RAPID"
}

variable "gke_min_master_version" {
  description = "Minimum GKE control plane version. Fractional G4 shapes (g4-standard-12, g4-standard-24) require >= 1.35.8-gke.1518000 (or >= 1.36.4-gke.1082000) per https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-create-g2-g4."
  type        = string
  default     = "1.35.8-gke.1518000"
}

variable "deployment_pattern" {
  description = <<-EOT
    Target GKE G4 architecture pattern for dgem:
      - "hybrid_baseline_plus_burst" (Default / Recommended Production Standard - Pattern 4):
          Maintains >=1 always-on On-Demand G4 baseline node (dgem-engine-baseline, zero cold-start delay, 100% availability)
          plus an elastic HPA burst pool (dgem-engine-burst) routed via ComputeClass/dgem-g4-priority (Spot G4 -> DWS Flex-Start -> On-Demand).
      - "spot_priority_waterfall" (Pattern 1):
          Scale-from-zero 3-tier ComputeClass waterfall (Spot G4 -> DWS Flex-Start -> On-Demand fallback + active migration).
      - "dws_flex_batch" (Pattern 2):
          Non-preemptible Dynamic Workload Scheduler (DWS) Flex-Start queued provisioning for large batch backfills (up to 7 days).
      - "ondemand_only" (Pattern 3):
          Dedicated On-Demand G4 node pool only for strict zero-preemption SLA environments.
  EOT
  type        = string
  default     = "hybrid_baseline_plus_burst"

  validation {
    condition = contains([
      "hybrid_baseline_plus_burst",
      "spot_priority_waterfall",
      "dws_flex_batch",
      "ondemand_only",
    ], var.deployment_pattern)
    error_message = "deployment_pattern must be one of: hybrid_baseline_plus_burst, spot_priority_waterfall, dws_flex_batch, ondemand_only."
  }
}

variable "baseline_g4_machine_type" {
  description = <<-EOT
    G4 machine type for the On-Demand baseline node pool (g4-ondemand):
      - "g4-standard-24"  : 1/2 vGPU (48 GB GDDR7), 24 vCPUs, 96 GB RAM (~50% lower On-Demand cost; fits 17.53 GiB NVFP4 + 12 GiB KV cache).
      - "g4-standard-12"  : 1/4 vGPU (24 GB GDDR7), 12 vCPUs, 48 GB RAM (tight budget baseline; requires KV_CACHE_GB=3).
      - "g4-standard-48"  : 1x full GPU (96 GB GDDR7), 48 vCPUs, 180 GB RAM (supports NVIDIA MPS 2-3x or MIG 2g.48gb / 1g.24gb).
      - "g4-standard-96"  : 2x full GPUs (192 GB GDDR7), 96 vCPUs, 360 GB RAM.
      - "g4-standard-192" : 4x full GPUs (384 GB GDDR7), 192 vCPUs, 720 GB RAM.
      - "g4-standard-384" : 8x full GPUs (768 GB GDDR7), 384 vCPUs, 1,440 GB RAM.
  EOT
  type        = string
  default     = "g4-standard-24"

  validation {
    condition = contains([
      "g4-standard-12",
      "g4-standard-24",
      "g4-standard-48",
      "g4-standard-96",
      "g4-standard-192",
      "g4-standard-384",
    ], var.baseline_g4_machine_type)
    error_message = "baseline_g4_machine_type must be one of: g4-standard-12, g4-standard-24, g4-standard-48, g4-standard-96, g4-standard-192, g4-standard-384."
  }
}

variable "burst_g4_machine_type" {
  description = "G4 machine type for the elastic Spot (g4-spot-faststart) and DWS Flex-Start (g4-dws-flex) burst node pools. Defaults to g4-standard-48 (1x NVIDIA RTX PRO 6000 96 GB GDDR7)."
  type        = string
  default     = "g4-standard-48"

  validation {
    condition = contains([
      "g4-standard-12",
      "g4-standard-24",
      "g4-standard-48",
      "g4-standard-96",
      "g4-standard-192",
      "g4-standard-384",
    ], var.burst_g4_machine_type)
    error_message = "burst_g4_machine_type must be one of: g4-standard-12, g4-standard-24, g4-standard-48, g4-standard-96, g4-standard-192, g4-standard-384."
  }
}

variable "g4_machine_type" {
  description = "Default G4 machine type fallback (g4-standard-48 provides 1x NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB GDDR7, 48 vCPUs, and 180 GB RAM)."
  type        = string
  default     = "g4-standard-48"

  validation {
    condition = contains([
      "g4-standard-12",
      "g4-standard-24",
      "g4-standard-48",
      "g4-standard-96",
      "g4-standard-192",
      "g4-standard-384",
    ], var.g4_machine_type)
    error_message = "g4_machine_type must be one of: g4-standard-12, g4-standard-24, g4-standard-48, g4-standard-96, g4-standard-192, g4-standard-384."
  }
}

variable "gpu_sharing_strategy" {
  description = <<-EOT
    GPU sharing strategy on full-GPU G4 nodes (g4-standard-48/96/192/384):
      - "none" : Single dgem replica per GPU (or per fractional vGPU on g4-standard-12 / g4-standard-24) using vLLM continuous batching (MAX_SEQS=32).
      - "mps"  : NVIDIA Multi-Process Service (MPS) sharing 1 physical 96 GB GPU across 2 or 3 dgem-engine replicas for 1.4x-1.7x higher batch throughput.
      - "mig"  : Hardware Multi-Instance GPU (MIG) partitioning 1 physical 96 GB GPU into isolated 2g.48gb (2 slices) or 1g.24gb (4 slices) partitions for strict p99 SLA isolation.
    NOTE: Per GKE documentation (https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-create-g2-g4), MPS and MIG cannot be used on fractional <1 GPU G4 shapes (g4-standard-6, g4-standard-12, g4-standard-24).
  EOT
  type        = string
  default     = "none"

  validation {
    condition     = contains(["none", "mps", "mig"], var.gpu_sharing_strategy)
    error_message = "gpu_sharing_strategy must be one of: none, mps, mig."
  }
}

variable "mps_max_shared_clients" {
  description = "Number of dgem-engine containers sharing each physical 96 GB RTX PRO 6000 GPU when gpu_sharing_strategy = \"mps\" (2 recommended = 48 GB pinned VRAM/client; 3 = 32 GB pinned VRAM/client)."
  type        = number
  default     = 2

  validation {
    condition     = contains([2, 3], var.mps_max_shared_clients)
    error_message = "mps_max_shared_clients must be 2 (48 GB per dgem replica) or 3 (32 GB per dgem replica) on a 96 GB RTX PRO 6000."
  }
}

variable "mig_partition_size" {
  description = "Hardware MIG partition profile on NVIDIA RTX PRO 6000 (96 GB) when gpu_sharing_strategy = \"mig\": \"2g.48gb\" (2 isolated 48 GB slices per GPU, recommended) or \"1g.24gb\" (4 isolated 24 GB slices per GPU) or \"4g.96gb\" (1 slice)."
  type        = string
  default     = "2g.48gb"

  validation {
    condition     = contains(["2g.48gb", "1g.24gb", "4g.96gb"], var.mig_partition_size)
    error_message = "mig_partition_size must be one of the GKE-supported RTX PRO 6000 profiles: 2g.48gb, 1g.24gb, or 4g.96gb."
  }
}

variable "g4_local_ssd_count" {
  description = "Optional Local NVMe SSD count for G4 nodes (0, 1, 2, or 4). Defaults to 0 because GKE Fast-Starting Nodes on G4 require hyperdisk-balanced boot disks and NO Local SSDs (https://docs.cloud.google.com/kubernetes-engine/docs/concepts/fast-starting-nodes#g4-requirements)."
  type        = number
  default     = 0
}

variable "g4_gpu_type" {
  description = "Accelerator type for G4 VMs."
  type        = string
  default     = "nvidia-rtx-pro-6000"
}

variable "g4_gpu_count" {
  description = "Number of NVIDIA RTX PRO 6000 GPUs per node (1 for g4-standard-12/24/48; auto-derived per machine shape in locals when using multi-GPU shapes)."
  type        = number
  default     = 1
}

variable "enable_spot_pool" {
  description = "Enable the GKE Spot G4 node pool with Fast Start (Image Streaming + Secondary Boot Disk + Hyperdisk Snapshot)."
  type        = bool
  default     = true
}

variable "spot_max_nodes" {
  description = "Maximum total nodes in the GKE Spot G4 node pool."
  type        = number
  default     = 4
}

variable "enable_dws_flex_pool" {
  description = "Enable the DWS Flex-Start (Queued Provisioning) G4 node pool."
  type        = bool
  default     = true
}

variable "dws_max_nodes" {
  description = "Maximum total nodes in the DWS Flex-Start G4 node pool."
  type        = number
  default     = 4
}

variable "enable_ondemand_pool" {
  description = "Enable the On-Demand G4 node pool."
  type        = bool
  default     = true
}

variable "ondemand_min_nodes" {
  description = "Minimum total nodes in the On-Demand G4 node pool (automatically defaults to >= 1 when deployment_pattern = \"hybrid_baseline_plus_burst\" or \"ondemand_only\", or 0 for scale-from-zero fallback)."
  type        = number
  default     = 1
}

variable "ondemand_max_nodes" {
  description = "Maximum total nodes in the On-Demand G4 node pool."
  type        = number
  default     = 2
}

variable "secondary_boot_disk_image" {
  description = "Optional full resource URI of the pre-built Secondary Boot Disk image for Fast Start container image/weight preloading (e.g., projects/MY_PROJECT/global/images/dgem-g4-image-cache). Leave empty on initial apply before running generate_fast_start_snapshots.sh."
  type        = string
  default     = ""
}

variable "k8s_namespace" {
  description = "Kubernetes namespace for dgem workloads."
  type        = string
  default     = "dgem"
}

variable "cascade_model" {
  description = "Default Stage 2 Gemini model for high-hesitation escalations."
  type        = string
  default     = "gemini-3.5-flash-lite"
}

variable "fitted_temperature" {
  description = "Domain-calibrated post-hoc temperature T* fitted on 50-100 historical tickets/alerts."
  type        = string
  default     = "1.15"
}

variable "cascade_threshold" {
  description = "Normalized Shannon entropy (Hesitation %) threshold in [0, 1] that triggers Stage 2 escalation to Gemini 3.5 Flash-Lite."
  type        = string
  default     = "0.16"
}

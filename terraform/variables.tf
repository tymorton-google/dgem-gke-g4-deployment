variable "project_id" {
  description = "Target Google Cloud Project ID for deploying dgem on GKE with NVIDIA G4 GPUs."
  type        = string
}

variable "region" {
  description = "Google Cloud region with NVIDIA RTX PRO 6000 Blackwell (G4) availability (e.g., us-central1, us-east4, europe-west4)."
  type        = string
  default     = "us-central1"
}

variable "zones" {
  description = "Zones within the region where g4-standard-48 instances are available (in us-central1: us-central1-b, us-central1-c, us-central1-f)."
  type        = list(string)
  default     = ["us-central1-b", "us-central1-c", "us-central1-f"]
}

variable "cluster_name" {
  description = "Name of the GKE Standard cluster."
  type        = string
  default     = "dgem-g4-cluster"
}

variable "gke_release_channel" {
  description = "GKE release channel. G4 GPUs require GKE 1.34.0-gke.1662000+ (RAPID or REGULAR recommended for GKE 1.35+ ComputeClass and Pod Snapshot features)."
  type        = string
  default     = "RAPID"
}

variable "gke_min_master_version" {
  description = "Minimum GKE control plane version (must be >= 1.34.0-gke.1662000 for g4-standard-* support)."
  type        = string
  default     = "1.35"
}

variable "g4_machine_type" {
  description = "G4 machine type. g4-standard-48 provides 1x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7), 48 vCPUs, and 180 GB RAM."
  type        = string
  default     = "g4-standard-48"
}

variable "g4_local_ssd_count" {
  description = "Optional Local NVMe SSD count for g4-standard-48 (0, 1, 2, or 4). Defaults to 0 because GKE Fast-Starting Nodes on G4 require hyperdisk-balanced boot disks and zero Local SSDs (https://cloud.google.com/kubernetes-engine/docs/concepts/fast-starting-nodes#g4-requirements)."
  type        = number
  default     = 0
}

variable "g4_gpu_type" {
  description = "Accelerator type for G4 VMs."
  type        = string
  default     = "nvidia-rtx-pro-6000"
}

variable "g4_gpu_count" {
  description = "Number of NVIDIA RTX PRO 6000 Blackwell GPUs per node (1 for g4-standard-48)."
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
  description = "Minimum total nodes in the On-Demand G4 node pool (set to 1 for always-warm baseline or 0 for scale-from-zero fallback)."
  type        = number
  default     = 0
}

variable "ondemand_max_nodes" {
  description = "Maximum total nodes in the On-Demand G4 node pool."
  type        = number
  default     = 2
}

variable "secondary_boot_disk_image" {
  description = "Optional full resource URI of the pre-built Secondary Boot Disk image for Fast Start container image preloading (e.g., projects/YOUR_PROJECT_ID/global/images/dgem-g4-image-cache). Leave empty on initial apply before running scripts/generate_fast_start_snapshots.sh."
  type        = string
  default     = ""
}

variable "k8s_namespace" {
  description = "Kubernetes namespace for dgem workloads."
  type        = string
  default     = "dgem"
}

variable "cascade_model" {
  description = "Default Stage 2 Vertex AI Gemini model for high-hesitation escalations."
  type        = string
  default     = "gemini-3.5-flash-lite"
}

variable "fitted_temperature" {
  description = "Domain-calibrated post-hoc temperature T* fitted on historical labeled items."
  type        = string
  default     = "1.15"
}

variable "cascade_threshold" {
  description = "Normalized Shannon entropy (Hesitation %) threshold in [0, 1] that triggers Stage 2 escalation to Gemini 3.5 Flash-Lite."
  type        = string
  default     = "0.16"
}

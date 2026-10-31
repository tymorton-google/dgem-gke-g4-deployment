output "gke_cluster_name" {
  description = "Name of the provisioned GKE cluster."
  value       = google_container_cluster.dgem_cluster.name
}

output "gke_get_credentials_command" {
  description = "Command to configure kubectl credentials."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.dgem_cluster.name} --region ${var.region} --project ${var.project_id}"
}

output "deployment_pattern" {
  description = "Active GKE G4 deployment pattern (hybrid_baseline_plus_burst, spot_priority_waterfall, dws_flex_batch, or ondemand_only)."
  value       = var.deployment_pattern
}

output "g4_topology_summary" {
  description = "Summary of provisioned G4 baseline/burst machine shapes, GPU sharing strategy, and always-on baseline floor."
  value = {
    deployment_pattern       = var.deployment_pattern
    baseline_g4_machine_type = var.baseline_g4_machine_type
    burst_g4_machine_type    = var.burst_g4_machine_type
    ondemand_min_nodes       = local.ondemand_min_nodes_effective
    spot_max_nodes           = var.spot_max_nodes
    dws_max_nodes            = var.dws_max_nodes
    gpu_sharing_strategy     = var.gpu_sharing_strategy
    mps_max_shared_clients   = var.gpu_sharing_strategy == "mps" ? var.mps_max_shared_clients : 1
    mig_partition_size       = var.gpu_sharing_strategy == "mig" ? var.mig_partition_size : "none"
  }
}

output "artifact_registry_repo" {
  description = "Artifact Registry Docker repository URL."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.dgem_repo.repository_id}"
}

output "weights_gcs_bucket" {
  description = "GCS bucket for NVFP4 model weights, FlashInfer JIT cache, and domain calibration receipts."
  value       = "gs://${google_storage_bucket.dgem_weights.name}"
}

output "pod_snapshots_gcs_bucket" {
  description = "Hierarchical Namespace (HNS) GCS bucket for GKE Pod Snapshots."
  value       = "gs://${google_storage_bucket.dgem_pod_snapshots.name}"
}

output "dgem_engine_gsa_email" {
  description = "Google Service Account email bound to dgem-engine-ksa via Workload Identity."
  value       = google_service_account.dgem_engine_gsa.email
}

output "dgem_gateway_gsa_email" {
  description = "Google Service Account email bound to dgem-gateway-ksa via Workload Identity (with roles/aiplatform.user for Gemini 3.5 Flash-Lite)."
  value       = google_service_account.dgem_gateway_gsa.email
}

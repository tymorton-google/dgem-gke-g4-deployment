#!/usr/bin/env bash
# ==============================================================================
# deploy_all_gcloud.sh
#
# Parameterized end-to-end gcloud + kubectl deployment script for dgem on GKE
# with NVIDIA G4 GPUs (g4-standard-48 / NVIDIA RTX PRO 6000 Blackwell 96GB).
#
# Implements Google Cloud Production Enterprise Security Best Practices:
#   - Private GKE nodes (--enable-private-nodes) behind Cloud NAT
#   - Shielded VMs with Secure Boot & Integrity Monitoring (--shielded-secure-boot)
#   - Keyless IAM via GKE Workload Identity Federation (--workload-pool)
#   - Uniform Bucket-Level Access on all Cloud Storage buckets
#
# Supports 4 deployment modes via --mode:
#   - auto      (Default) Provisions all 3 pools + Custom ComputeClass waterfall
#               that tries Spot -> DWS Flex-Start -> On-Demand in order until
#               a G4 GPU node comes online, then actively migrates back to Spot.
#   - spot      Provisions ONLY the GKE Spot G4 pool with Fast-Start snapshots.
#   - dws-flex  Provisions ONLY the DWS Flex-Start (Queued Provisioning) G4 pool.
#   - on-demand Provisions ONLY the On-Demand G4 pool (GKE Fast-Starting Nodes).
#
# Usage:
#   ./deploy_all_gcloud.sh \
#     --project-id your-gcp-project-id \
#     --mode auto \
#     --priority-order spot,dws-flex,on-demand \
#     --region us-central1 \
#     --zones us-central1-b,us-central1-c,us-central1-f \
#     --temperature 1.15 \
#     --cascade-threshold 0.16 \
#     --cascade-model gemini-3.5-flash-lite
# ==============================================================================
set -euo pipefail

# Defaults (can be overridden by environment variables or CLI flags)
PROJECT_ID="${PROJECT_ID:-}"
MODE="${MODE:-auto}" # auto | spot | dws-flex | on-demand
PRIORITY_ORDER="${PRIORITY_ORDER:-spot,dws-flex,on-demand}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-b}"
ZONES="${ZONES:-us-central1-b,us-central1-c,us-central1-f}"
CLUSTER_NAME="${CLUSTER_NAME:-dgem-g4-cluster}"
FITTED_TEMPERATURE="${FITTED_TEMPERATURE:-1.15}"
CASCADE_THRESHOLD="${CASCADE_THRESHOLD:-0.16}"
CASCADE_MODEL="${CASCADE_MODEL:-gemini-3.5-flash-lite}"
SECONDARY_DISK_IMAGE="${SECONDARY_DISK_IMAGE:-dgem-g4-image-cache}"
DWS_WAIT_SECONDS="${DWS_WAIT_SECONDS:-120}"
MAX_NODES="${MAX_NODES:-4}"

usage() {
  cat <<EOF
Usage: $(basename "$0") --project-id <PROJECT_ID> [OPTIONS]

Required:
  -p, --project-id <ID>          Google Cloud Project ID

Target Provisioning Options:
  -m, --mode <MODE>              Deployment target mode:
                                   auto      : (Default) Try all 3 options via Custom ComputeClass
                                               until a G4 node is found & online, with active
                                               migration back to Spot when capacity recovers
                                   spot      : GKE Spot G4 only (with Fast-Start snapshots)
                                   dws-flex  : DWS Flex-Start (Queued Provisioning) G4 only
                                   on-demand : On-Demand G4 only (GKE Fast-Starting Nodes)
  -o, --priority-order <ORDER>   Fallback order when --mode=auto
                                 (default: "spot,dws-flex,on-demand")
      --dws-wait-seconds <SEC>   Seconds ComputeClass waits for DWS Flex-Start before falling
                                 back to the next priority rule (default: 120)
      --max-nodes <N>            Maximum G4 nodes per pool (default: 4)

Cluster & Location Options:
  -r, --region <REGION>          Google Cloud region (default: us-central1)
  -z, --zones <Z1,Z2,Z3>         Comma-separated G4 zones (default: us-central1-b,us-central1-c,us-central1-f)
  -c, --cluster-name <NAME>      GKE cluster name (default: dgem-g4-cluster)
      --secondary-disk <NAME>    Secondary boot disk image name (default: dgem-g4-image-cache)

dgem Calibration & Gemini 3.5 Flash-Lite Cascade Options:
  -t, --temperature <FLOAT>      Domain-fitted Stage 1 temperature T* (default: 1.15)
      --cascade-threshold <TAU>  Hesitation % threshold in [0,1] for Stage 2 escalation (default: 0.16)
      --cascade-model <MODEL>    Stage 2 Vertex AI Gemini model (default: gemini-3.5-flash-lite)
  -h, --help                     Show this help message
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project-id)        PROJECT_ID="$2"; shift 2 ;;
    -m|--mode)              MODE="$2"; shift 2 ;;
    -o|--priority-order)    PRIORITY_ORDER="$2"; shift 2 ;;
    --dws-wait-seconds)     DWS_WAIT_SECONDS="$2"; shift 2 ;;
    --max-nodes)            MAX_NODES="$2"; shift 2 ;;
    -r|--region)            REGION="$2"; shift 2 ;;
    -z|--zones)             ZONES="$2"; ZONE="${2%%,*}"; shift 2 ;;
    -c|--cluster-name)      CLUSTER_NAME="$2"; shift 2 ;;
    --secondary-disk)       SECONDARY_DISK_IMAGE="$2"; shift 2 ;;
    -t|--temperature)       FITTED_TEMPERATURE="$2"; shift 2 ;;
    --cascade-threshold)    CASCADE_THRESHOLD="$2"; shift 2 ;;
    --cascade-model)        CASCADE_MODEL="$2"; shift 2 ;;
    -h|--help)              usage ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
done

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: --project-id (or PROJECT_ID env var) is required."
  usage
fi

export PROJECT_ID MODE PRIORITY_ORDER REGION ZONE ZONES CLUSTER_NAME
export FITTED_TEMPERATURE CASCADE_THRESHOLD CASCADE_MODEL SECONDARY_DISK_IMAGE DWS_WAIT_SECONDS MAX_NODES

echo "=============================================================================="
echo " dgem GKE G4 Enterprise Deployment"
echo "   Project ID        : ${PROJECT_ID}"
echo "   Region / Zones    : ${REGION} (${ZONES})"
echo "   Provisioning Mode : ${MODE} (Priority Order: ${PRIORITY_ORDER})"
echo "   Stage 1 Temp (T*) : ${FITTED_TEMPERATURE}"
echo "   Stage 2 Cascade   : ${CASCADE_MODEL} (Hesitation threshold tau = ${CASCADE_THRESHOLD})"
echo "=============================================================================="

echo "==> [1/7] Enabling required Google Cloud APIs..."
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  secretmanager.googleapis.com \
  --project="${PROJECT_ID}"

echo "==> [2/7] Creating Custom VPC, Private Subnet, and Cloud NAT..."
gcloud compute networks create "${CLUSTER_NAME}-vpc" \
  --project="${PROJECT_ID}" \
  --subnet-mode=custom || true

gcloud compute networks subnets create "${CLUSTER_NAME}-subnet" \
  --project="${PROJECT_ID}" \
  --network="${CLUSTER_NAME}-vpc" \
  --region="${REGION}" \
  --range="10.20.0.0/20" \
  --secondary-range="pods-range=10.24.0.0/14,services-range=10.28.0.0/20" \
  --enable-private-ip-google-access || true

gcloud compute routers create "${CLUSTER_NAME}-router" \
  --project="${PROJECT_ID}" \
  --network="${CLUSTER_NAME}-vpc" \
  --region="${REGION}" || true

gcloud compute routers nats create "${CLUSTER_NAME}-nat" \
  --project="${PROJECT_ID}" \
  --router="${CLUSTER_NAME}-router" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges || true

echo "==> [3/7] Creating Artifact Registry & GCS Buckets (Weights + HNS Pod Snapshots)..."
gcloud artifacts repositories create dgem-images \
  --project="${PROJECT_ID}" \
  --repository-format=docker \
  --location="${REGION}" || true

gcloud storage buckets create "gs://${PROJECT_ID}-dgem-weights" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --uniform-bucket-level-access || true

# Hierarchical Namespace (HNS) bucket required for GKE Pod Snapshots
# Reference: https://cloud.google.com/kubernetes-engine/docs/how-to/pod-snapshots-prepare
gcloud storage buckets create "gs://${PROJECT_ID}-dgem-pod-snapshots" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --uniform-bucket-level-access \
  --enable-hierarchical-namespace \
  --soft-delete-duration=0d || true

echo "==> [4/7] Creating Least-Privilege Service Accounts & Keyless Workload Identity Bindings..."
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
for SA in dgem-gke-node-sa dgem-engine-gsa dgem-gateway-gsa; do
  gcloud iam service-accounts create "${SA}" --project="${PROJECT_ID}" || true
done

for ROLE in roles/logging.logWriter roles/monitoring.metricWriter roles/artifactregistry.reader; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:dgem-gke-node-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="${ROLE}" --condition=None --quiet
done

gcloud storage buckets add-iam-policy-binding "gs://${PROJECT_ID}-dgem-weights" \
  --member="serviceAccount:dgem-engine-gsa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer" --quiet

# Grant Pod Snapshot bucket permissions to node SA, workload SA, and GKE Service Agent
for SNAP_ROLE in roles/storage.bucketViewer roles/storage.objectUser; do
  gcloud storage buckets add-iam-policy-binding "gs://${PROJECT_ID}-dgem-pod-snapshots" \
    --member="serviceAccount:dgem-gke-node-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="${SNAP_ROLE}" --quiet
  gcloud storage buckets add-iam-policy-binding "gs://${PROJECT_ID}-dgem-pod-snapshots" \
    --member="serviceAccount:dgem-engine-gsa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="${SNAP_ROLE}" --quiet
done
gcloud storage buckets add-iam-policy-binding "gs://${PROJECT_ID}-dgem-pod-snapshots" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@container-engine-robot.iam.gserviceaccount.com" \
  --role="roles/storage.objectUser" --quiet || true

# Grant Vertex AI User to dgem-gateway-gsa for Stage 2 Gemini 3.5 Flash-Lite escalations
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:dgem-gateway-gsa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user" --condition=None --quiet

echo "==> [5/7] Creating GKE Standard Cluster (${CLUSTER_NAME}, release channel RAPID, 1.35+)..."
gcloud container clusters create "${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --node-locations="${ZONES}" \
  --release-channel=rapid \
  --cluster-version=1.35 \
  --network="${CLUSTER_NAME}-vpc" \
  --subnetwork="${CLUSTER_NAME}-subnet" \
  --cluster-secondary-range-name="pods-range" \
  --services-secondary-range-name="services-range" \
  --enable-ip-alias \
  --enable-private-nodes \
  --no-enable-master-authorized-networks \
  --enable-image-streaming \
  --enable-shielded-nodes \
  --workload-pool="${PROJECT_ID}.svc.id.goog" \
  --addons=GcePersistentDiskCsiDriver,GcsFuseCsiDriver,HttpLoadBalancing \
  --machine-type=e2-standard-4 \
  --num-nodes=1 \
  --service-account="dgem-gke-node-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --shielded-secure-boot \
  --shielded-integrity-monitoring

echo "==> [6/7] Creating G4 GPU Node Pools for mode=${MODE}..."
SECONDARY_DISK_FLAG=""
if gcloud compute images describe "${SECONDARY_DISK_IMAGE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  SECONDARY_DISK_FLAG="--secondary-boot-disk=disk-image=projects/${PROJECT_ID}/global/images/${SECONDARY_DISK_IMAGE},mode=CONTAINER_IMAGE_CACHE"
fi

# 6a. GKE Spot G4 Node Pool with Fast Start
if [[ "${MODE}" == "auto" || "${MODE}" == "spot" ]]; then
  gcloud container node-pools create g4-spot-faststart \
    --project="${PROJECT_ID}" \
    --cluster="${CLUSTER_NAME}" \
    --region="${REGION}" \
    --node-locations="${ZONES}" \
    --machine-type=g4-standard-48 \
    --accelerator=type=nvidia-rtx-pro-6000,count=1,gpu-driver-version=latest \
    --spot \
    --enable-autoscaling \
    --total-min-nodes=0 \
    --total-max-nodes="${MAX_NODES}" \
    --location-policy=ANY \
    --disk-type=hyperdisk-balanced \
    --disk-size=200 \
    --enable-image-streaming \
    --service-account="dgem-gke-node-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --shielded-secure-boot \
    --shielded-integrity-monitoring \
    --node-labels="dgem.cloud.google.com/role=gpu-engine,dgem.cloud.google.com/provisioning=spot,cloud.google.com/compute-class=dgem-g4-priority" \
    --node-taints="nvidia.com/gpu=present:NoSchedule,cloud.google.com/compute-class=dgem-g4-priority:NoSchedule" \
    ${SECONDARY_DISK_FLAG}
fi

# 6b. DWS Flex-Start (Queued Provisioning) G4 Node Pool
if [[ "${MODE}" == "auto" || "${MODE}" == "dws-flex" ]]; then
  gcloud container node-pools create g4-dws-flex \
    --project="${PROJECT_ID}" \
    --cluster="${CLUSTER_NAME}" \
    --region="${REGION}" \
    --node-locations="${ZONES}" \
    --machine-type=g4-standard-48 \
    --accelerator=type=nvidia-rtx-pro-6000,count=1,gpu-driver-version=latest \
    --flex-start \
    --enable-queued-provisioning \
    --enable-autoscaling \
    --num-nodes=0 \
    --total-min-nodes=0 \
    --total-max-nodes="${MAX_NODES}" \
    --location-policy=ANY \
    --no-enable-autorepair \
    --disk-type=hyperdisk-balanced \
    --disk-size=200 \
    --enable-image-streaming \
    --service-account="dgem-gke-node-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --shielded-secure-boot \
    --shielded-integrity-monitoring \
    --node-labels="dgem.cloud.google.com/role=gpu-engine,dgem.cloud.google.com/provisioning=dws-flex,cloud.google.com/compute-class=dgem-g4-priority" \
    --node-taints="cloud.google.com/gke-queued=true:NoSchedule,nvidia.com/gpu=present:NoSchedule,cloud.google.com/compute-class=dgem-g4-priority:NoSchedule" \
    ${SECONDARY_DISK_FLAG}
fi

# 6c. On-Demand G4 Node Pool (Eligible for GKE Fast-Starting Nodes: hyperdisk-balanced + zero Local SSDs)
if [[ "${MODE}" == "auto" || "${MODE}" == "on-demand" ]]; then
  gcloud container node-pools create g4-ondemand \
    --project="${PROJECT_ID}" \
    --cluster="${CLUSTER_NAME}" \
    --region="${REGION}" \
    --node-locations="${ZONES}" \
    --machine-type=g4-standard-48 \
    --accelerator=type=nvidia-rtx-pro-6000,count=1,gpu-driver-version=latest \
    --enable-autoscaling \
    --total-min-nodes=0 \
    --total-max-nodes="${MAX_NODES}" \
    --location-policy=ANY \
    --disk-type=hyperdisk-balanced \
    --disk-size=200 \
    --enable-image-streaming \
    --service-account="dgem-gke-node-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --shielded-secure-boot \
    --shielded-integrity-monitoring \
    --node-labels="dgem.cloud.google.com/role=gpu-engine,dgem.cloud.google.com/provisioning=ondemand,cloud.google.com/compute-class=dgem-g4-priority" \
    --node-taints="nvidia.com/gpu=present:NoSchedule,cloud.google.com/compute-class=dgem-g4-priority:NoSchedule" \
    ${SECONDARY_DISK_FLAG}
fi

echo "==> [7/7] Binding Workload Identity & Applying Kubernetes Manifests..."
gcloud iam service-accounts add-iam-policy-binding \
  "dgem-engine-gsa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[dgem/dgem-engine-ksa]" --quiet

gcloud iam service-accounts add-iam-policy-binding \
  "dgem-gateway-gsa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[dgem/dgem-gateway-ksa]" --quiet

gcloud container clusters get-credentials "${CLUSTER_NAME}" --region="${REGION}" --project="${PROJECT_ID}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="${SCRIPT_DIR}/../k8s"

for MANIFEST in 01-namespace-rbac.yaml 02-compute-class.yaml 03-weights-pvc-snapshot.yaml 05-dgem-engine-deployment.yaml 07-dgem-gateway-deployment.yaml; do
  envsubst < "${K8S_DIR}/${MANIFEST}" | kubectl apply -f -
done

echo "Deployment applied! Check rollout status with: kubectl get pods -n dgem -w"

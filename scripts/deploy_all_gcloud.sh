#!/usr/bin/env bash
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
# deploy_all_gcloud.sh
#
# Parameterized end-to-end gcloud + kubectl deployment script for dgem on GKE
# with NVIDIA G4 GPUs (RTX PRO 6000 Blackwell: fractional vGPU g4-standard-12/24,
# single-GPU g4-standard-48, and multi-GPU g4-standard-96/192/384).
#
# Implements Google Cloud Production Enterprise Security Best Practices:
#   - Private GKE nodes (--enable-private-nodes) behind Cloud NAT
#   - Shielded VMs with Secure Boot & Integrity Monitoring (--shielded-secure-boot)
#   - Keyless IAM via GKE Workload Identity Federation (--workload-pool)
#   - Uniform Bucket-Level Access on all Cloud Storage buckets
#
# Supports 5 deployment modes via --mode:
#   - hybrid-baseline-burst (Default / Recommended Pattern 4)
#               Provisions an always-on On-Demand baseline G4 node pool
#               (min_nodes=1, e.g. g4-standard-24 48GB vGPU or g4-standard-48 96GB)
#               for 100% zero-cold-start availability + elastic HPA burst pools
#               (Spot G4 + DWS Flex-Start G4) via ComputeClass/dgem-g4-priority
#               using 08-dgem-hybrid-baseline-burst.yaml.
#   - auto      (Pattern 1) Provisions all 3 pools (scale-from-zero) + Custom
#               ComputeClass waterfall (Spot -> DWS Flex-Start -> On-Demand)
#               using 05-dgem-engine-deployment.yaml.
#   - spot      Provisions ONLY the GKE Spot G4 pool with Fast-Start snapshots.
#   - dws-flex  (Pattern 2) Provisions ONLY the DWS Flex-Start (Queued Provisioning) G4 pool.
#   - on-demand (Pattern 3) Provisions ONLY the On-Demand G4 pool (min_nodes=1).
#
# Supports GKE GPU Sharing Strategies on >=1 GPU G4 shapes (g4-standard-48+):
#   - none      (Default) Single dgem replica per GPU / fractional vGPU with vLLM continuous batching.
#   - mps       NVIDIA Multi-Process Service (--mps-clients 2 or 3) for 1.4x-1.7x batch throughput.
#   - mig       Hardware Multi-Instance GPU (--mig-partition 2g.48gb or 1g.24gb) for strict p99 QoS.
# ==============================================================================
set -euo pipefail

# Defaults (can be overridden by environment variables or CLI flags)
PROJECT_ID="${PROJECT_ID:-}"
MODE="${MODE:-hybrid-baseline-burst}" # hybrid-baseline-burst | auto | spot | dws-flex | on-demand
PRIORITY_ORDER="${PRIORITY_ORDER:-spot,dws-flex,on-demand}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-b}"
ZONES="${ZONES:-us-central1-b,us-central1-c,us-central1-f}"
CLUSTER_NAME="${CLUSTER_NAME:-dgem-g4-cluster}"
GKE_VERSION="${GKE_VERSION:-1.35.8-gke.1518000}"
BASELINE_MACHINE_TYPE="${BASELINE_MACHINE_TYPE:-g4-standard-24}"
BURST_MACHINE_TYPE="${BURST_MACHINE_TYPE:-g4-standard-48}"
GPU_SHARING="${GPU_SHARING:-none}" # none | mps | mig
MPS_CLIENTS="${MPS_CLIENTS:-2}"    # 2 or 3
MIG_PARTITION="${MIG_PARTITION:-2g.48gb}" # 2g.48gb | 1g.24gb | 4g.96gb
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
  -p, --project-id <ID>              Google Cloud Project ID

Target Architecture & Provisioning Options:
  -m, --mode <MODE>                  Deployment architecture pattern:
                                       hybrid-baseline-burst : (Default - Pattern 4) Always-on On-Demand baseline
                                                               (min_nodes=1, 0s cold start, 100% availability) +
                                                               HPA elastic burst onto Spot/DWS G4 via ComputeClass
                                       auto                  : (Pattern 1) Scale-from-zero 3-tier ComputeClass
                                                               waterfall (Spot -> DWS Flex -> On-Demand)
                                       spot                  : GKE Spot G4 only (with Fast-Start snapshots)
                                       dws-flex              : (Pattern 2) DWS Flex-Start Queued Provisioning only
                                       on-demand             : (Pattern 3) On-Demand G4 only (min_nodes=1)
      --baseline-machine-type <TYPE> G4 machine type for On-Demand baseline pool (default: g4-standard-24 [48GB vGPU];
                                     options: g4-standard-12, g4-standard-24, g4-standard-48, g4-standard-96/192/384)
      --burst-machine-type <TYPE>    G4 machine type for Spot & DWS burst pools (default: g4-standard-48 [96GB GPU])
      --gpu-sharing <STRATEGY>       GPU sharing strategy on g4-standard-48+ (default: none | mps | mig)
      --mps-clients <2|3>            Max shared containers per 96GB GPU when --gpu-sharing=mps (default: 2)
      --mig-partition <PROFILE>      Hardware MIG slice profile when --gpu-sharing=mig (default: 2g.48gb | 1g.24gb)
  -o, --priority-order <ORDER>       Fallback order for ComputeClass (default: "spot,dws-flex,on-demand")
      --dws-wait-seconds <SEC>       Seconds ComputeClass waits for DWS Flex-Start (default: 120)
      --max-nodes <N>                Maximum G4 nodes per pool (default: 4)

Cluster & Location Options:
  -r, --region <REGION>              Google Cloud region (default: us-central1)
  -z, --zones <Z1,Z2,Z3>             Comma-separated G4 zones (default: us-central1-b,us-central1-c,us-central1-f)
  -c, --cluster-name <NAME>          GKE cluster name (default: dgem-g4-cluster)
      --gke-version <VERSION>        GKE version (default: 1.35.8-gke.1518000, required for fractional G4 vGPUs)
      --secondary-disk <NAME>        Secondary boot disk image name (default: dgem-g4-image-cache)

dgem Calibration & Gemini 3.5 Flash-Lite Cascade Options:
  -t, --temperature <FLOAT>          Domain-fitted Stage 1 temperature T* (default: 1.15)
      --cascade-threshold <TAU>      Hesitation % threshold in [0,1] for Stage 2 escalation (default: 0.16)
      --cascade-model <MODEL>        Stage 2 Vertex AI Gemini model (default: gemini-3.5-flash-lite)
  -h, --help                         Show this help message
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project-id)             PROJECT_ID="$2"; shift 2 ;;
    -m|--mode)                   MODE="$2"; shift 2 ;;
    --baseline-machine-type)     BASELINE_MACHINE_TYPE="$2"; shift 2 ;;
    --burst-machine-type)        BURST_MACHINE_TYPE="$2"; shift 2 ;;
    --gpu-sharing)               GPU_SHARING="$2"; shift 2 ;;
    --mps-clients)               MPS_CLIENTS="$2"; shift 2 ;;
    --mig-partition)             MIG_PARTITION="$2"; shift 2 ;;
    -o|--priority-order)         PRIORITY_ORDER="$2"; shift 2 ;;
    --dws-wait-seconds)          DWS_WAIT_SECONDS="$2"; shift 2 ;;
    --max-nodes)                 MAX_NODES="$2"; shift 2 ;;
    -r|--region)                 REGION="$2"; shift 2 ;;
    -z|--zones)                  ZONES="$2"; ZONE="${2%%,*}"; shift 2 ;;
    -c|--cluster-name)           CLUSTER_NAME="$2"; shift 2 ;;
    --gke-version)               GKE_VERSION="$2"; shift 2 ;;
    --secondary-disk)            SECONDARY_DISK_IMAGE="$2"; shift 2 ;;
    -t|--temperature)            FITTED_TEMPERATURE="$2"; shift 2 ;;
    --cascade-threshold)         CASCADE_THRESHOLD="$2"; shift 2 ;;
    --cascade-model)             CASCADE_MODEL="$2"; shift 2 ;;
    -h|--help)                   usage ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
done

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: --project-id (or PROJECT_ID env var) is required."
  usage
fi

# Normalize mode alias
if [[ "${MODE}" == "hybrid" ]]; then
  MODE="hybrid-baseline-burst"
fi

# Validate GKE hardware rules: MPS and MIG are not supported on fractional <1 GPU G4 shapes
is_fractional_g4() {
  local shape="$1"
  [[ "${shape}" == "g4-standard-6" || "${shape}" == "g4-standard-12" || "${shape}" == "g4-standard-24" ]]
}

if [[ "${GPU_SHARING}" != "none" ]]; then
  if is_fractional_g4 "${BURST_MACHINE_TYPE}"; then
    echo "ERROR: GKE does not support --gpu-sharing=${GPU_SHARING} on fractional <1 GPU shape (${BURST_MACHINE_TYPE})."
    echo "       Use --burst-machine-type=g4-standard-48 (or larger) when enabling MPS or MIG."
    exit 1
  fi
  if [[ "${MODE}" == "hybrid-baseline-burst" || "${MODE}" == "on-demand" ]] && is_fractional_g4 "${BASELINE_MACHINE_TYPE}"; then
    echo "ERROR: GKE does not support --gpu-sharing=${GPU_SHARING} on fractional <1 GPU baseline shape (${BASELINE_MACHINE_TYPE})."
    echo "       Set --baseline-machine-type=g4-standard-48 when enabling MPS or MIG, or set --gpu-sharing=none."
    exit 1
  fi
fi

gpu_count_for_shape() {
  case "$1" in
    g4-standard-12|g4-standard-24|g4-standard-48) echo "1" ;;
    g4-standard-96)  echo "2" ;;
    g4-standard-192) echo "4" ;;
    g4-standard-384) echo "8" ;;
    *) echo "1" ;;
  esac
}

build_accelerator_flag() {
  local shape="$1"
  local count
  count="$(gpu_count_for_shape "${shape}")"
  local accel="type=nvidia-rtx-pro-6000,count=${count},gpu-driver-version=latest"
  if [[ "${GPU_SHARING}" == "mps" ]]; then
    accel="${accel},gpu-sharing-strategy=mps,max-shared-clients-per-gpu=${MPS_CLIENTS}"
  elif [[ "${GPU_SHARING}" == "mig" ]]; then
    accel="${accel},gpu-partition-size=${MIG_PARTITION}"
  fi
  echo "${accel}"
}

BASELINE_ACCEL_FLAG="$(build_accelerator_flag "${BASELINE_MACHINE_TYPE}")"
BURST_ACCEL_FLAG="$(build_accelerator_flag "${BURST_MACHINE_TYPE}")"

export PROJECT_ID MODE PRIORITY_ORDER REGION ZONE ZONES CLUSTER_NAME GKE_VERSION
export BASELINE_MACHINE_TYPE BURST_MACHINE_TYPE GPU_SHARING MPS_CLIENTS MIG_PARTITION
export FITTED_TEMPERATURE CASCADE_THRESHOLD CASCADE_MODEL SECONDARY_DISK_IMAGE DWS_WAIT_SECONDS MAX_NODES

echo "=============================================================================="
echo " dgem GKE G4 Enterprise Deployment"
echo "   Project ID        : ${PROJECT_ID}"
echo "   Region / Zones    : ${REGION} (${ZONES})"
echo "   Architecture Mode : ${MODE} (Priority Order: ${PRIORITY_ORDER})"
echo "   Baseline Shape    : ${BASELINE_MACHINE_TYPE} (${BASELINE_ACCEL_FLAG})"
echo "   Burst Shape       : ${BURST_MACHINE_TYPE} (${BURST_ACCEL_FLAG})"
echo "   GPU Sharing       : ${GPU_SHARING} (MPS clients=${MPS_CLIENTS}, MIG=${MIG_PARTITION})"
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
# Verified against https://docs.cloud.google.com/kubernetes-engine/docs/how-to/pod-snapshots-prepare
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

echo "==> [5/7] Creating GKE Standard Cluster (${CLUSTER_NAME}, release channel RAPID, ${GKE_VERSION})..."
gcloud container clusters create "${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --node-locations="${ZONES}" \
  --release-channel=rapid \
  --cluster-version="${GKE_VERSION}" \
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

# 6a. GKE Spot G4 Node Pool with Fast Start (Elastic Burst Priority 1)
if [[ "${MODE}" == "hybrid-baseline-burst" || "${MODE}" == "auto" || "${MODE}" == "spot" ]]; then
  gcloud container node-pools create g4-spot-faststart \
    --project="${PROJECT_ID}" \
    --cluster="${CLUSTER_NAME}" \
    --region="${REGION}" \
    --node-locations="${ZONES}" \
    --machine-type="${BURST_MACHINE_TYPE}" \
    --accelerator="${BURST_ACCEL_FLAG}" \
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
    --node-labels="dgem.cloud.google.com/role=gpu-engine,dgem.cloud.google.com/provisioning=spot,dgem.cloud.google.com/pool-role=elastic-burst,dgem.cloud.google.com/gpu-sharing=${GPU_SHARING},cloud.google.com/compute-class=dgem-g4-priority" \
    --node-taints="nvidia.com/gpu=present:NoSchedule,cloud.google.com/compute-class=dgem-g4-priority:NoSchedule" \
    ${SECONDARY_DISK_FLAG}
fi

# 6b. DWS Flex-Start (Queued Provisioning) G4 Node Pool (Elastic Burst Priority 2 / Batch)
if [[ "${MODE}" == "hybrid-baseline-burst" || "${MODE}" == "auto" || "${MODE}" == "dws-flex" ]]; then
  gcloud container node-pools create g4-dws-flex \
    --project="${PROJECT_ID}" \
    --cluster="${CLUSTER_NAME}" \
    --region="${REGION}" \
    --node-locations="${ZONES}" \
    --machine-type="${BURST_MACHINE_TYPE}" \
    --accelerator="${BURST_ACCEL_FLAG}" \
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
    --node-labels="dgem.cloud.google.com/role=gpu-engine,dgem.cloud.google.com/provisioning=dws-flex,dgem.cloud.google.com/pool-role=elastic-burst,dgem.cloud.google.com/gpu-sharing=${GPU_SHARING},cloud.google.com/compute-class=dgem-g4-priority" \
    --node-taints="cloud.google.com/gke-queued=true:NoSchedule,nvidia.com/gpu=present:NoSchedule,cloud.google.com/compute-class=dgem-g4-priority:NoSchedule" \
    ${SECONDARY_DISK_FLAG}
fi

# 6c. On-Demand G4 Node Pool (Always-On Baseline in hybrid-baseline-burst / on-demand, or Scale-from-Zero Fallback in auto)
if [[ "${MODE}" == "hybrid-baseline-burst" || "${MODE}" == "auto" || "${MODE}" == "on-demand" ]]; then
  ONDEMAND_MIN_NODES=0
  ONDEMAND_INIT_NODES=0
  ONDEMAND_POOL_ROLE="fallback-ondemand"
  if [[ "${MODE}" == "hybrid-baseline-burst" || "${MODE}" == "on-demand" ]]; then
    ONDEMAND_MIN_NODES=1
    ONDEMAND_INIT_NODES=1
    ONDEMAND_POOL_ROLE="baseline-ondemand"
  fi

  gcloud container node-pools create g4-ondemand \
    --project="${PROJECT_ID}" \
    --cluster="${CLUSTER_NAME}" \
    --region="${REGION}" \
    --node-locations="${ZONES}" \
    --machine-type="${BASELINE_MACHINE_TYPE}" \
    --accelerator="${BASELINE_ACCEL_FLAG}" \
    --num-nodes="${ONDEMAND_INIT_NODES}" \
    --enable-autoscaling \
    --total-min-nodes="${ONDEMAND_MIN_NODES}" \
    --total-max-nodes="${MAX_NODES}" \
    --location-policy=ANY \
    --disk-type=hyperdisk-balanced \
    --disk-size=200 \
    --enable-image-streaming \
    --service-account="dgem-gke-node-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --shielded-secure-boot \
    --shielded-integrity-monitoring \
    --node-labels="dgem.cloud.google.com/role=gpu-engine,dgem.cloud.google.com/provisioning=ondemand,dgem.cloud.google.com/pool-role=${ONDEMAND_POOL_ROLE},dgem.cloud.google.com/gpu-sharing=${GPU_SHARING},cloud.google.com/compute-class=dgem-g4-priority" \
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

ENGINE_MANIFEST="05-dgem-engine-deployment.yaml"
if [[ "${MODE}" == "hybrid-baseline-burst" ]]; then
  ENGINE_MANIFEST="08-dgem-hybrid-baseline-burst.yaml"
fi

for MANIFEST in 01-namespace-rbac.yaml 02-compute-class.yaml 03-weights-pvc-snapshot.yaml "${ENGINE_MANIFEST}" 07-dgem-gateway-deployment.yaml; do
  envsubst < "${K8S_DIR}/${MANIFEST}" | kubectl apply -f -
done

echo "Deployment applied (mode=${MODE}, engine_manifest=${ENGINE_MANIFEST})! Check rollout status with: kubectl get pods -n dgem -w"

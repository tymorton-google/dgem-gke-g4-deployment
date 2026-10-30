#!/usr/bin/env bash
# ==============================================================================
# generate_fast_start_snapshots.sh
#
# End-to-end generator for all GKE Spot G4 Fast-Start Snapshot layers:
#   1. Container Image Build & Push (dgemma GPU engine + dgemma-gateway)
#   2. In-Cluster Kubernetes Staging Job scheduled onto `nvidia-rtx-pro-6000`
#      (`g4-standard-48`) so `hyperdisk-balanced` attaches cleanly, staging:
#      - 17.53 GiB `nvidia/diffusiongemma-26B-A4B-it-NVFP4` safetensors weights
#      - Pre-compiled FlashInfer / Triton Blackwell SM120 JIT cache (`jit-cache.tar.gz`)
#   3. Compute Engine Hyperdisk Snapshot (`dgem-nvfp4-weights-snap`) & K8s
#      `VolumeSnapshot` (`dgem-nvfp4-weights-snapshot`) for instant PVC hydration
#   4. GKE Secondary Boot Disk Image (`dgem-g4-image-cache`) via
#      `gke-disk-image-builder` for instant Spot G4 node container layer boot
#   5. GKE Pod Snapshot (`PodSnapshotManualTrigger`) helper for live GPU+CPU
#      memory checkpointing into Hierarchical Namespace (HNS) Cloud Storage
# ==============================================================================
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-b}"
DGEM_REPO_DIR="${DGEM_REPO_DIR:-$(pwd)/dgem}"
AR_REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/dgem-images"
WEIGHTS_BUCKET="gs://${PROJECT_ID}-dgem-weights"
SNAPSHOT_NAME="${SNAPSHOT_NAME:-dgem-nvfp4-weights-snap}"
SECONDARY_DISK_IMAGE="${SECONDARY_DISK_IMAGE:-dgem-g4-image-cache}"
HF_MODEL_ID="${HF_MODEL_ID:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}"

echo "==> [1/5] Building & pushing dgem GPU Engine and Gateway images to ${AR_REPO}..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

if [[ -d "${DGEM_REPO_DIR}" ]]; then
  gcloud builds submit "${DGEM_REPO_DIR}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --tag="${AR_REPO}/dgemma:latest" \
    --machine-type=e2-highcpu-32 \
    --disk-size=200 || {
      docker build -t "${AR_REPO}/dgemma:latest" -f "${DGEM_REPO_DIR}/deploy/cloudrun/Dockerfile" "${DGEM_REPO_DIR}"
      docker push "${AR_REPO}/dgemma:latest"
    }

  # Build static Linux dgem binary and package lightweight gateway container
  (
    cd "${DGEM_REPO_DIR}"
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o dgem .
  )
  docker build -t "${AR_REPO}/dgemma-gateway:latest" -f "${DGEM_REPO_DIR}/deploy/gateway/Dockerfile" "${DGEM_REPO_DIR}" || true
  docker push "${AR_REPO}/dgemma-gateway:latest" || true
else
  echo "    [Info] DGEM_REPO_DIR (${DGEM_REPO_DIR}) not found locally; skipping image build and using existing images in ${AR_REPO}."
fi

echo "==> [2/5] Staging 17.53 GiB NVFP4 weights + Blackwell SM120 JIT cache onto Hyperdisk Balanced PVC..."
# IMPORTANT HARDWARE NOTE:
#   `hyperdisk-balanced` volumes can only attach to Hyperdisk-compatible machine series
#   (such as `g4-standard-48` / `nvidia-rtx-pro-6000`, `c3`, `c4`, `n4`, `a3`) and CANNOT
#   attach to legacy `e2-*` system CPU nodes.
#   Therefore, the staging PVC + Kubernetes Job explicitly selects `nvidia-rtx-pro-6000`
#   so the GCE PD CSI driver attaches the Hyperdisk volume cleanly on the G4 node.

kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: hyperdisk-balanced-dgem
provisioner: pd.csi.storage.gke.io
parameters:
  type: hyperdisk-balanced
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dgem-weights-staging-pvc
  namespace: dgem
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: hyperdisk-balanced-dgem
  resources:
    requests:
      storage: 50Gi
---
apiVersion: batch/v1
kind: Job
metadata:
  name: dgem-stage-weights-job
  namespace: dgem
spec:
  backoffLimit: 2
  template:
    spec:
      serviceAccountName: dgem-engine-ksa
      restartPolicy: OnFailure
      # Schedule onto G4 (nvidia-rtx-pro-6000) so hyperdisk-balanced attaches cleanly
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-rtx-pro-6000
      tolerations:
        - key: "cloud.google.com/compute-class"
          operator: "Equal"
          value: "dgem-g4-priority"
          effect: "NoSchedule"
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
        - key: "cloud.google.com/gke-spot"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"
        - key: "cloud.google.com/gke-queued"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"
        - key: "sandbox.gke.io/runtime"
          operator: "Equal"
          value: "gvisor"
          effect: "NoSchedule"
      containers:
        - name: stager
          image: google/cloud-sdk:slim
          command: ["/bin/bash", "-ce"]
          args:
            - |
              apt-get update && apt-get install -y --no-install-recommends python3-pip python3-venv curl
              mkdir -p /mnt/weights/diffusiongemma-26B-A4B-it-NVFP4
              if gsutil -q stat "${WEIGHTS_BUCKET}/models/diffusiongemma-26B-A4B-it-NVFP4/config.json" 2>/dev/null; then
                echo "Syncing NVFP4 weights from ${WEIGHTS_BUCKET}..."
                gcloud storage rsync -r "${WEIGHTS_BUCKET}/models/diffusiongemma-26B-A4B-it-NVFP4" /mnt/weights/diffusiongemma-26B-A4B-it-NVFP4
              else
                echo "Downloading ${HF_MODEL_ID} from Hugging Face Hub..."
                python3 -m venv /tmp/hf
                /tmp/hf/bin/pip install --upgrade pip "huggingface_hub[cli]" hf_transfer
                export HF_HUB_ENABLE_HF_TRANSFER=1
                /tmp/hf/bin/huggingface-cli download "${HF_MODEL_ID}" \
                  --local-dir /mnt/weights/diffusiongemma-26B-A4B-it-NVFP4 \
                  --exclude "*.md" ".git*"
                gcloud storage rsync -r /mnt/weights/diffusiongemma-26B-A4B-it-NVFP4 "${WEIGHTS_BUCKET}/models/diffusiongemma-26B-A4B-it-NVFP4"
              fi
              if gsutil -q stat "${WEIGHTS_BUCKET}/jit-cache/jit-cache.tar.gz" 2>/dev/null; then
                gcloud storage cp "${WEIGHTS_BUCKET}/jit-cache/jit-cache.tar.gz" /mnt/weights/jit-cache.tar.gz
              elif gsutil -q stat "${WEIGHTS_BUCKET}/cache/jit-cache.tar.gz" 2>/dev/null; then
                gcloud storage cp "${WEIGHTS_BUCKET}/cache/jit-cache.tar.gz" /mnt/weights/jit-cache.tar.gz
              fi
              ls -lh /mnt/weights/diffusiongemma-26B-A4B-it-NVFP4
              sync
          volumeMounts:
            - name: staging-vol
              mountPath: /mnt/weights
      volumes:
        - name: staging-vol
          persistentVolumeClaim:
            claimName: dgem-weights-staging-pvc
EOF

echo "Waiting for dgem-stage-weights-job to complete on G4 node..."
kubectl wait --for=condition=complete job/dgem-stage-weights-job -n dgem --timeout=900s

PV_NAME="$(kubectl get pvc dgem-weights-staging-pvc -n dgem -o jsonpath='{.spec.volumeName}')"
GCE_DISK_URI="$(kubectl get pv "${PV_NAME}" -o jsonpath='{.spec.csi.volumeHandle}')"
GCE_DISK_NAME="${GCE_DISK_URI##*/}"
GCE_DISK_ZONE="$(echo "${GCE_DISK_URI}" | awk -F'/' '{for(i=1;i<=NF;i++) if($i=="zones") print $(i+1)}')"

echo "==> [3/5] Creating Compute Engine Disk Snapshot (${SNAPSHOT_NAME}) from ${GCE_DISK_NAME} (${GCE_DISK_ZONE})..."
kubectl delete job dgem-stage-weights-job -n dgem --ignore-not-found

gcloud compute snapshots create "${SNAPSHOT_NAME}" \
  --project="${PROJECT_ID}" \
  --source-disk="${GCE_DISK_NAME}" \
  --source-disk-zone="${GCE_DISK_ZONE:-${ZONE}}" \
  --storage-location="${REGION}"

kubectl delete pvc dgem-weights-staging-pvc -n dgem --ignore-not-found

echo "==> [4/5] Building GKE Secondary Boot Disk Image (${SECONDARY_DISK_IMAGE}) via gke-disk-image-builder..."
echo "    This pre-caches the ~15 GB dgemma container layers onto a secondary boot disk attached at Spot VM boot."
# Note: G4 (g4-standard-48) requires Hyperdisk (hyperdisk-balanced); legacy pd-ssd is not supported on G4.
go run github.com/GoogleCloudPlatform/ai-on-gke/tools/gke-disk-image-builder/cli@latest \
  --project-name="${PROJECT_ID}" \
  --image-name="${SECONDARY_DISK_IMAGE}" \
  --zone="${ZONE}" \
  --gcs-path="${WEIGHTS_BUCKET}/image-builder-logs" \
  --machine-type="n2-standard-16" \
  --disk-type="hyperdisk-balanced" \
  --disk-size-gb=100 \
  --container-image="${AR_REPO}/dgemma:latest" \
  --container-image="${AR_REPO}/dgemma-gateway:latest"

echo "==> [5/5] Helper command to trigger a GKE Pod Snapshot once dgem-engine is running on GKE:"
cat <<EOF
To capture a live GPU + CPU memory checkpoint (Pod Snapshot) after deploying k8s/05-dgem-engine-deployment.yaml:

  TARGET_POD=\$(kubectl get pods -n dgem -l app=dgem-engine -o jsonpath='{.items[0].metadata.name}')
  kubectl wait --for=condition=Ready "pod/\${TARGET_POD}" -n dgem --timeout=300s
  export PROJECT_ID="${PROJECT_ID}" TARGET_POD_NAME="\${TARGET_POD}"
  envsubst < k8s/04-pod-snapshots.yaml | kubectl apply -f -
  kubectl get podsnapshots -n dgem -w
EOF

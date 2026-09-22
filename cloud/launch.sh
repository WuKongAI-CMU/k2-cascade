#!/bin/bash
# Launch a self-running GPU job. The VM clones the repo at the current commit, runs cloud/jobs/<job>.sh,
# uploads results to $BUCKET/<run>/, and deletes itself. Nothing depends on this machine staying on.
#   cloud/launch.sh <job> [hours=4] [spot|standard]
set -euo pipefail
JOB=${1:?job name, e.g. train_residual}; HOURS=${2:-4}; MODEL=${3:-spot}
PROJECT=${K2_PROJECT:-less-more-475623}; BUCKET=${K2_BUCKET:-gs://k2-cascade-runs-less-more}
[ -f "cloud/jobs/$JOB.sh" ] || { echo "no cloud/jobs/$JOB.sh"; exit 1; }
COMMIT=$(git rev-parse HEAD)
git fetch -q origin && git merge-base --is-ancestor "$COMMIT" origin/main || { echo "commit $COMMIT is not pushed to origin/main"; exit 1; }
RUN="$(echo "$JOB" | tr '_' '-')-$(date -u +%Y%m%d-%H%M%S)"; NAME="k2-$RUN"
PROV=(--provisioning-model=SPOT --instance-termination-action=DELETE)
[ "$MODEL" = standard ] && PROV=(--provisioning-model=STANDARD --instance-termination-action=DELETE)
for Z in us-central1-a us-central1-b us-central1-c us-central1-f; do
  if gcloud compute instances create "$NAME" --project="$PROJECT" --zone="$Z" --machine-type="${K2_MACHINE:-a2-highgpu-1g}" \
      "${PROV[@]}" --max-run-duration="${HOURS}h" --maintenance-policy="${K2_MAINT:-TERMINATE}" \
      --boot-disk-size="${K2_DISK:-200GB}" --boot-disk-type=pd-balanced \
      --image-family="${K2_IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}" --image-project="${K2_IMAGE_PROJECT:-deeplearning-platform-release}" \
      --scopes=cloud-platform --labels=purpose=k2,run="$RUN" \
      --metadata=install-nvidia-driver=True,k2-job="$JOB",k2-commit="$COMMIT",k2-bucket="$BUCKET",k2-run="$RUN" \
      --metadata-from-file=startup-script=cloud/startup.sh >/tmp/k2-launch.err 2>&1; then
    echo "launched $NAME in $Z ($MODEL, max ${HOURS}h, commit ${COMMIT:0:7})"
    echo "status:  cloud/status.sh $RUN"; exit 0
  fi
  echo "$Z: $(grep -m1 -iE "error|exhausted|quota|invalid" /tmp/k2-launch.err | cut -c1-200)"
  grep -qiE "exhausted|does not have enough resources|capacity" /tmp/k2-launch.err || { cat /tmp/k2-launch.err; exit 1; }
done
echo "could not place the VM in any us-central1 zone"; exit 1

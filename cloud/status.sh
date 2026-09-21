#!/bin/bash
# cloud/status.sh            list runs with their last status
# cloud/status.sh <run>      show status, list files, and download results to runs/cloud/<run>/
set -uo pipefail
PROJECT=${K2_PROJECT:-less-more-475623}; BUCKET=${K2_BUCKET:-gs://k2-cascade-runs-less-more}
if [ $# -eq 0 ]; then
  for r in $(gcloud storage ls "$BUCKET/" 2>/dev/null); do printf "%-50s %s\n" "$(basename "$r")" "$(gcloud storage cat "${r}STATUS" 2>/dev/null)"; done
  echo "== running VMs"; gcloud compute instances list --project="$PROJECT" --filter="labels.purpose=k2" --format="table(name,zone.basename(),status,creationTimestamp)"
  exit 0
fi
RUN=$1
echo "STATUS: $(gcloud storage cat "$BUCKET/$RUN/STATUS" 2>/dev/null || echo none yet)"
mkdir -p "runs/cloud/$RUN" && gcloud storage cp -r "$BUCKET/$RUN/*" "runs/cloud/$RUN/" --quiet 2>/dev/null
find "runs/cloud/$RUN" -type f | sort

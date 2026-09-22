#!/bin/bash
# Runs as root when the VM boots. Everything it needs comes from instance metadata, so the job needs
# nobody's laptop and no open SSH session. It always uploads what it has and deletes its own VM.
#   k2-job      name of a script in cloud/jobs/ (without .sh)
#   k2-commit   repo commit to check out
#   k2-bucket   gs://bucket for results
#   k2-run      run id; results land in gs://bucket/<run>/
#   k2-env      optional space-separated VAR=value pairs exported to the job
set -uo pipefail
md() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
JOB=$(md attributes/k2-job); COMMIT=$(md attributes/k2-commit); BUCKET=$(md attributes/k2-bucket); RUN=$(md attributes/k2-run)
NAME=$(md name); ZONE=$(md zone | awk -F/ '{print $NF}')
OUT="$BUCKET/$RUN"; W=/opt/k2; LOG=/opt/k2-job.log
mkdir -p "$W"; exec > >(tee -a "$LOG") 2>&1
status() { echo "$1 $(date -u +%FT%TZ)" | gcloud storage cp - "$OUT/STATUS" --quiet; echo "STATUS $1"; }

finish() {
  rc=$?
  cd "$W/k2-cascade" 2>/dev/null && {
    gcloud storage cp -r analysis logs "$OUT/" --quiet 2>/dev/null
    find runs -name '*.json' -o -name '*.log' 2>/dev/null | xargs -r -I{} gcloud storage cp {} "$OUT/{}" --quiet
  }
  gcloud storage cp "$LOG" "$OUT/startup.log" --quiet
  if [ "$rc" -eq 0 ]; then status DONE; else status "FAILED rc=$rc"; fi
  gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet
}
trap finish EXIT

status "BOOTING job=$JOB commit=$COMMIT"
export HOME=/root PATH="/root/.local/bin:$PATH" HF_HOME=/opt/hf
cd "$W" && git clone -q https://github.com/WuKongAI-CMU/k2-cascade.git && cd k2-cascade && git checkout -q "$COMMIT"
if [ "$JOB" = smoke ]; then status RUNNING; bash cloud/jobs/smoke.sh; exit $?; fi
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv venv -q --python 3.12
uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -q "transformers>=5.15" accelerate safetensors datasets numpy huggingface_hub scikit-learn
status "ENV $(.venv/bin/python -c 'import torch;print(torch.cuda.get_device_name())')"

# pinned model snapshots (the revisions used in the first GPU run unless overridden)
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
import json
rev = {"IFM/K2-Horizon-3.7B": "6360f705b2e57d542959e6a2e67ebeb95dae0373",
       "IFM/K2-Horizon-7B": "d6a80e21f447768a61f1c976aa8e7d8e82a20d57"}
out = {r: {"sha": s, "path": snapshot_download(r, revision=s, allow_patterns=["*.json", "*.py", "*.safetensors", "*.jinja", "tokenizer*"])}
       for r, s in rev.items()}
json.dump(out, open("model_revisions.json", "w"), indent=1)
PY
export SRC=$(.venv/bin/python -c "import json;print(json.load(open('model_revisions.json'))['IFM/K2-Horizon-3.7B']['path'])")
export TGT=$(.venv/bin/python -c "import json;print(json.load(open('model_revisions.json'))['IFM/K2-Horizon-7B']['path'])")
export HF_HUB_OFFLINE=1 PY="$PWD/.venv/bin/python" OUT
mkdir -p analysis logs runs data
for kv in $(md attributes/k2-env 2>/dev/null); do export "$kv"; done
status "RUNNING $JOB"
bash "cloud/jobs/$JOB.sh"

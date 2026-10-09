# Fit a Jacobian lens on one K2 model (K2_WHICH=src for the 3.7B sender, tgt for the 7B receiver).
# Parts of 16 prompts are synced to the bucket every few minutes, so a dead VM loses at most one part;
# relaunching with the same K2_RUN resumes from the parts already there.
set -euo pipefail
BK="${K2_BUCKET:-gs://k2-cascade-runs-less-more}"
WHICH="${K2_WHICH:?src or tgt}"; N="${K2_N:-96}"; DB="${K2_DIM_BATCH:-16}"
MODEL=$([ "$WHICH" = src ] && echo "$SRC" || echo "$TGT")
D="runs/jlens_$WHICH"
note() { echo "[jlens $(date -u +%H:%MZ)] $*" | tee -a logs/jlens.log; gcloud storage cp logs/jlens.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
uv pip install -q --no-deps "git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e"
gcloud storage cp "$BK/program-20260922b/data/fineweb_1024.jsonl" data/ --quiet
mkdir -p "$D"; gcloud storage rsync "$OUT/$D" "$D" --quiet 2>/dev/null || true
( while true; do sleep 300; gcloud storage rsync "$D" "$OUT/$D" --quiet >/dev/null 2>&1 || true; done ) &
SYNC=$!
note "fitting $WHICH ($MODEL) on $N prompts, dim_batch $DB"
$PY -m k2cascade.projector.jlens_fit --model "$MODEL" --data data/fineweb_1024.jsonl --n "$N" --dim_batch "$DB" --out "$D" 2>&1 | tee -a logs/jlens.log
kill $SYNC || true
gcloud storage rsync "$D" "$OUT/$D" --quiet
note "done: $(ls $D)"

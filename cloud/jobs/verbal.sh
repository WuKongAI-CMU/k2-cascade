# The text channel with the sender's uncertainty written down ("verbal" arm) against the cache channel, on the
# confidence variants. K2_FROM = program run (mlp seed2), K2_CONF = confidence run (variants + sender entropy),
# K2_IMP = improve run (KL projector), optional.
set -euo pipefail
BK="${K2_BUCKET:-gs://k2-cascade-runs-less-more}"; FROM="$BK/${K2_FROM:?}"; CONF="$BK/${K2_CONF:?}"; IMP="$BK/${K2_IMP:-none}"
note() { echo "[verbal $(date -u +%H:%MZ)] $*" | tee -a logs/verbal.log; gcloud storage cp logs/verbal.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
gcloud storage rsync -r "$FROM/runs/mlp_top3_s1500_seed2/step_1500" runs/mlp_seed2 --quiet
gcloud storage cp "$CONF/data/squad_variants_nli.jsonl" data/ --quiet
gcloud storage cp -r "$CONF/analysis/conf" analysis/ --quiet
PROJS="mlp_seed2"
if [ "${K2_IMP:-}" != "" ] && gcloud storage ls "$IMP/runs/mlp_kl1/step_1000/mlp.safetensors" >/dev/null 2>&1; then
  gcloud storage rsync -r "$IMP/runs/mlp_kl1/step_1000" runs/mlp_kl1 --quiet; PROJS="mlp_seed2 mlp_kl1"
fi
head -n 1000 data/squad_variants_nli.jsonl > data/conf.jsonl
for p in $PROJS; do
  mkdir -p "analysis/verbal_$p"
  for v in clean contradicted removed; do
    $PY -m k2cascade.projector.verbal --data data/conf.jsonl --se "analysis/conf/se_$v.jsonl" --out "data/conf_verbal_$v.jsonl" > "logs/verbal_attach_$v.log" 2>&1
    $PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector "runs/$p" --data "data/conf_verbal_$v.jsonl" --n 1000 \
        --variant $v --arms none,text,verbal,project,derange --out "analysis/verbal_$p/qa_$v.json" --per_item "analysis/verbal_$p/qa_items_$v.jsonl" > "logs/verbal_qa_${p}_$v.log" 2>&1
    note "$p $v: $($PY -c "import json;d=json.load(open('analysis/verbal_$p/qa_$v.json'));print({a:(round(d[a]['f1'],3),round(d[a]['p_gold'],3),round(d[a]['entropy'],2)) for a in d if isinstance(d[a],dict)})")"
  done
  cp analysis/conf/se_*.jsonl "analysis/verbal_$p/"
  $PY -m k2cascade.projector.confidence --items "analysis/verbal_$p" --se "analysis/verbal_$p" --out "analysis/confidence_verbal_$p.json" > "logs/confidence_verbal_$p.log" 2>&1
  note "metrics $p: $(tr -d '\n' < logs/confidence_verbal_$p.log | cut -c1-700)"
  gcloud storage cp -r analysis logs "$OUT/" --quiet
done
note "verbal finished"

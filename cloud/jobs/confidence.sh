# Confidence transfer (docs/design-confidence-2026-09-22.md). Needs: K2_FROM (program run: seeds), K2_FROM2 (design2
# run: deranged-sender control), and data/squad_variants.jsonl already uploaded to $BUCKET/shared/.
set -euo pipefail
BK="${K2_BUCKET:-gs://k2-cascade-runs-less-more}"; FROM="$BK/${K2_FROM:?}"; FROM2="$BK/${K2_FROM2:?}"
N=${K2_N:-1000}
note() { echo "[confidence $(date -u +%H:%MZ)] $*" | tee -a logs/confidence.log; gcloud storage cp logs/confidence.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
done_p() { gcloud storage ls "$OUT/phases/$1.done" >/dev/null 2>&1; }
mark() { echo ok | gcloud storage cp - "$OUT/phases/$1.done" --quiet; gcloud storage cp -r analysis logs "$OUT/" --quiet; }
gcloud storage rsync -r "$FROM/runs/mlp_top3_s1500_seed2/step_1500" runs/mlp_seed2 --quiet
gcloud storage rsync -r "$FROM2/runs/ctrl_derange_s1500/step_1500" runs/ctrl_derange --quiet
gcloud storage rsync -r "$FROM/runs/ridge_top3" runs/ridge_top3 --quiet
gcloud storage cp "$BK/shared/squad_variants.jsonl" data/ --quiet
mkdir -p analysis/conf

# NLI filter of the contradiction sentences (downloads DeBERTa once)
if ! done_p C0; then
  HF_HUB_OFFLINE=0 $PY -m k2cascade.projector.nli_filter --data data/squad_variants.jsonl --out data/squad_variants_nli.jsonl > logs/nli.log 2>&1
  note "C0 $(tail -1 logs/nli.log)"; gcloud storage cp data/squad_variants_nli.jsonl "$OUT/data/" --quiet; mark C0
else
  gcloud storage cp "$OUT/data/squad_variants_nli.jsonl" data/ --quiet
fi
head -n "$N" data/squad_variants_nli.jsonl > data/conf.jsonl

# sender semantic entropy, three variants (needs the NLI model; downloaded above)
for v in clean contradicted removed; do
  done_p "C1_$v" && continue
  HF_HUB_OFFLINE=0 $PY -m k2cascade.projector.sender_entropy --model "$SRC" --data data/conf.jsonl --variant $v --out "analysis/conf/se_$v.jsonl" > "logs/se_$v.log" 2>&1
  note "C1 sender entropy $v: mean se $($PY -c "import json;r=[json.loads(l)['se'] for l in open('analysis/conf/se_$v.jsonl')];print(round(sum(r)/len(r),3), len(r))")"
  mark "C1_$v"
done

# receiver per-item measurements: main projector on all arms, control projector on project/derange only
ARMS=none,text,project,derange,zero,random
for v in clean contradicted removed; do
  done_p "C2_$v" && continue
  $PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector runs/mlp_seed2 --data data/conf.jsonl --n "$N" \
      --variant $v --arms $ARMS --out "analysis/conf/qa_$v.json" --per_item "analysis/conf/qa_items_$v.jsonl" > "logs/qa_$v.log" 2>&1
  note "C2 $v: $($PY -c "import json;d=json.load(open('analysis/conf/qa_$v.json'));print({a:(round(d[a]['f1'],3),round(d[a]['p_gold'],3),round(d[a]['entropy'],2)) for a in d if isinstance(d[a],dict)})")"
  mark "C2_$v"
done
for v in clean contradicted removed; do
  done_p "C3_$v" && continue
  $PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector runs/ctrl_derange --data data/conf.jsonl --n "$N" \
      --variant $v --arms text,project --out "analysis/conf/qa_ctrl_$v.json" --per_item "analysis/conf/qa_ctrl_items_$v.jsonl" > "logs/qa_ctrl_$v.log" 2>&1
  note "C3 ctrl $v: $($PY -c "import json;d=json.load(open('analysis/conf/qa_ctrl_$v.json'));print({a:(round(d[a]['f1'],3),round(d[a]['p_gold'],3),round(d[a]['entropy'],2)) for a in d if isinstance(d[a],dict)})")"
  mark "C3_$v"
done
$PY -m k2cascade.projector.confidence --items analysis/conf --se analysis/conf --out analysis/confidence.json > logs/confidence_metrics.log 2>&1
note "metrics: $(cat logs/confidence_metrics.log | tr -d '\n' | cut -c1-600)"
gcloud storage cp -r analysis logs "$OUT/" --quiet
note "confidence finished"

# Two improvements the gap analysis asked for: (1) a small projector (bottleneck 32 / 16) so the adapter is MB-scale,
# (2) a projector trained with KL to the receiver's text-path distribution, to keep the sender's uncertainty.
# Each is 1000 steps (the 300-step run already reached 56%), then Noma (6 names) + SQuAD F1; the KL one also gets the
# confidence test on clean/removed. K2_FROM = program run; needs shared/squad_variants.jsonl for the confidence part.
set -euo pipefail
BK="${K2_BUCKET:-gs://k2-cascade-runs-less-more}"; FROM="$BK/${K2_FROM:?}"
STEPS=${K2_STEPS:-1000}
note() { echo "[improve $(date -u +%H:%MZ)] $*" | tee -a logs/improve.log; gcloud storage cp logs/improve.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
done_p() { gcloud storage ls "$OUT/phases/$1.done" >/dev/null 2>&1; }
mark() { echo ok | gcloud storage cp - "$OUT/phases/$1.done" --quiet; gcloud storage cp -r analysis logs "$OUT/" --quiet; gcloud storage rsync -r runs "$OUT/runs" --quiet >/dev/null 2>&1 || true; }
gcloud storage cp "$FROM/data/fineweb_1024.jsonl" data/ --quiet
gcloud storage rsync -r "$FROM/runs/ridge_top3" runs/ridge_top3 --quiet
gcloud storage cp "$BK/shared/squad_600.jsonl" data/ --quiet 2>/dev/null || HF_HUB_OFFLINE=0 $PY -c "from k2cascade.projector.qa import load_squad; import json; open('data/squad_600.jsonl','w').writelines(json.dumps(e)+'\n' for e in load_squad(600, 0))"
gcloud storage cp "$BK/shared/squad_variants.jsonl" data/ --quiet
run_one() {  # run_one <name> <train flags...>
  local n=$1; shift
  done_p "$n" && return 0
  $PY -m k2cascade.projector.train --source "$SRC" --target "$TGT" --data data/fineweb_1024.jsonl --ridge runs/ridge_top3 \
      --out "runs/$n" --steps "$STEPS" --batch 1 --accum 16 --save_every 500 --seed 0 "$@" > "logs/train_$n.log" 2>&1
  $PY -m k2cascade.projector.noma --source "$SRC" --target "$TGT" --projector "runs/$n/step_$STEPS" --episodes 300 --out "analysis/imp_noma_$n.json" > "logs/imp_noma_$n.log" 2>&1
  $PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector "runs/$n/step_$STEPS" --data data/squad_600.jsonl --n 300 --arms text,project,derange --out "analysis/imp_qa_$n.json" > "logs/imp_qa_$n.log" 2>&1
  note "$n: params $(grep -m1 'projector params' logs/train_$n.log | cut -c1-60) | noma $($PY -c "import json;d=json.load(open('analysis/imp_noma_$n.json'));print(round(d['project']['acc'],3),'derange',round(d['derange']['acc'],3))") | qa $($PY -c "import json;d=json.load(open('analysis/imp_qa_$n.json'));print({a:round(d[a]['f1'],3) for a in d if isinstance(d[a],dict)})")"
  mark "$n"
}
run_one mlp_b32 --bottleneck 32
run_one mlp_b16 --bottleneck 16
run_one mlp_kl1 --kl_text 1.0
# confidence on the KL projector: clean and removed, receiver arms text/project, then metrics vs the stored sender entropy
if ! done_p CONF; then
  gcloud storage cp -r "$BK/confidence-20260923-084423/analysis/conf" analysis/ --quiet
  gcloud storage cp "$BK/confidence-20260923-084423/data/squad_variants_nli.jsonl" data/ --quiet
  head -n 1000 data/squad_variants_nli.jsonl > data/conf.jsonl
  mkdir -p analysis/conf_kl
  for v in clean removed contradicted; do
    $PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector "runs/mlp_kl1/step_$STEPS" --data data/conf.jsonl --n 1000 \
        --variant $v --arms none,text,project,derange --out "analysis/conf_kl/qa_$v.json" --per_item "analysis/conf_kl/qa_items_$v.jsonl" > "logs/conf_kl_$v.log" 2>&1
  done
  cp analysis/conf/se_*.jsonl analysis/conf_kl/ 2>/dev/null || true
  $PY -m k2cascade.projector.confidence --items analysis/conf_kl --se analysis/conf_kl --out analysis/confidence_kl.json > logs/confidence_kl.log 2>&1
  note "confidence(kl): $(tr -d '\n' < logs/confidence_kl.log | cut -c1-500)"
  mark CONF
fi
note "improve finished"

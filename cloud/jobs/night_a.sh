# Overnight A: (1) projector trained to match the sender's per-token entropy (the sender-side target the KL
# attempt lacked), evaluated on Noma, SQuAD and the confidence variants; (2) the numeric verbal baseline
# (sender writes its sample-agreement rate) on the confidence variants with the seed-2 projector.
set -euo pipefail
BK="${K2_BUCKET:-gs://k2-cascade-runs-less-more}"; FROM="$BK/${K2_FROM:?}"; CONF="$BK/${K2_CONF:?}"
STEPS=${K2_STEPS:-1000}
note() { echo "[night_a $(date -u +%H:%MZ)] $*" | tee -a logs/night_a.log; gcloud storage cp logs/night_a.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
done_p() { gcloud storage ls "$OUT/phases/$1.done" >/dev/null 2>&1; }
mark() { echo ok | gcloud storage cp - "$OUT/phases/$1.done" --quiet; gcloud storage cp -r analysis logs "$OUT/" --quiet; gcloud storage rsync -r runs "$OUT/runs" --quiet >/dev/null 2>&1 || true; }
gcloud storage cp "$FROM/data/fineweb_1024.jsonl" data/ --quiet
gcloud storage rsync -r "$FROM/runs/ridge_top3" runs/ridge_top3 --quiet
gcloud storage rsync -r "$FROM/runs/mlp_top3_s1500_seed2/step_1500" runs/mlp_seed2 --quiet
gcloud storage cp "$BK/shared/squad_600.jsonl" data/ --quiet
gcloud storage cp "$CONF/data/squad_variants_nli.jsonl" data/ --quiet
gcloud storage cp -r "$CONF/analysis/conf" analysis/ --quiet
head -n 1000 data/squad_variants_nli.jsonl > data/conf.jsonl
conf_eval() {  # conf_eval <projector dir> <tag> <verbal mode>
  local p=$1 t=$2 m=$3; mkdir -p "analysis/conf_$t"
  for v in clean contradicted removed; do
    $PY -m k2cascade.projector.verbal --data data/conf.jsonl --se "analysis/conf/se_$v.jsonl" --mode "$m" --out "data/conf_${m}_$v.jsonl" > /dev/null 2>&1
    $PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector "$p" --data "data/conf_${m}_$v.jsonl" --n 1000 \
        --variant $v --arms none,text,verbal,project,derange --out "analysis/conf_$t/qa_$v.json" --per_item "analysis/conf_$t/qa_items_$v.jsonl" > "logs/conf_${t}_$v.log" 2>&1
  done
  cp analysis/conf/se_*.jsonl "analysis/conf_$t/"
  $PY -m k2cascade.projector.confidence --items "analysis/conf_$t" --se "analysis/conf_$t" --out "analysis/confidence_$t.json" > "logs/confidence_$t.log" 2>&1
  note "confidence $t: $($PY -c "import json;d=json.load(open('analysis/confidence_$t.json'));print('auroc', {v:{a:round(d['sender_tracking'][v]['auroc_'+a],3) for a in ('text','verbal','project')} for v in d['sender_tracking']}, 'drop', {k:round(x,2) for k,x in d['drop_ratio'].items()})")"
}
if ! done_p ENT; then
  $PY -m k2cascade.projector.train --source "$SRC" --target "$TGT" --data data/fineweb_1024.jsonl --ridge runs/ridge_top3 \
      --out runs/mlp_ent1 --steps "$STEPS" --batch 1 --accum 16 --save_every 500 --seed 0 --ent_match 1.0 > logs/train_mlp_ent1.log 2>&1
  $PY -m k2cascade.projector.noma --source "$SRC" --target "$TGT" --projector "runs/mlp_ent1/step_$STEPS" --episodes 300 --out analysis/imp_noma_mlp_ent1.json > logs/imp_noma_ent1.log 2>&1
  $PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector "runs/mlp_ent1/step_$STEPS" --data data/squad_600.jsonl --n 300 --arms text,project,derange --out analysis/imp_qa_mlp_ent1.json > logs/imp_qa_ent1.log 2>&1
  note "ent1: noma $($PY -c "import json;d=json.load(open('analysis/imp_noma_mlp_ent1.json'));print(round(d['project']['acc'],3))") qa $($PY -c "import json;d=json.load(open('analysis/imp_qa_mlp_ent1.json'));print({a:round(d[a]['f1'],3) for a in d if isinstance(d[a],dict)})")"
  mark ENT
fi
done_p CONF_ENT || { conf_eval "runs/mlp_ent1/step_$STEPS" ent1 words; mark CONF_ENT; }
done_p CONF_NUM || { conf_eval runs/mlp_seed2 numeric numeric; mark CONF_NUM; }
done_p CONF_CANDS || { conf_eval runs/mlp_seed2 candidates candidates; mark CONF_CANDS; }
done_p CONF_BELIEF || { conf_eval runs/mlp_seed2 belief candidates; mark CONF_BELIEF; }
if ! done_p CONF_DIAG; then
  for v in clean removed; do
    $PY -m k2cascade.projector.diag_uncertainty --source "$SRC" --target "$TGT" --projector runs/mlp_seed2 --data data/conf.jsonl --se "analysis/conf/se_$v.jsonl" --variant $v --out "analysis/diag_unc_$v.json" > "logs/diag_unc_$v.log" 2>&1
    note "diag $v: $(tail -1 logs/diag_unc_$v.log | cut -c1-300)"
  done
  mark CONF_DIAG
fi
note "night_a finished"

# Memory lens, blend and onset interventions on the binding test (seed-0 projector, the paper's headline map),
# and SQuAD case studies (seed-2 projector, the one used for the doubt experiments). Outputs for the web article.
set -euo pipefail
BK="${K2_BUCKET:-gs://k2-cascade-runs-less-more}"; FROM="$BK/${K2_FROM:?}"; CONF="$BK/${K2_CONF:?}"
note() { echo "[lens $(date -u +%H:%MZ)] $*" | tee -a logs/lens.log; gcloud storage cp logs/lens.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
gcloud storage rsync -r "$FROM/runs/mlp_top3_s1500_seed0/step_1500" runs/mlp_seed0 --quiet
gcloud storage rsync -r "$FROM/runs/mlp_top3_s1500_seed2/step_1500" runs/mlp_seed2 --quiet
gcloud storage cp "$CONF/data/squad_variants_nli.jsonl" data/ --quiet
gcloud storage cp "$CONF/analysis/conf/se_clean.jsonl" data/ --quiet
head -n 1000 data/squad_variants_nli.jsonl > data/conf.jsonl
$PY -m k2cascade.projector.verbal --data data/conf.jsonl --se data/se_clean.jsonl --mode words --out data/cases.jsonl > logs/cases.log 2>&1
$PY -m k2cascade.projector.lens --source "$SRC" --target "$TGT" --projector runs/mlp_seed0 --out analysis/lens --episodes 200 --n_lens 40 > logs/lens_binding.log 2>&1
note "binding: $(tail -1 logs/lens_binding.log)"
gcloud storage cp -r analysis logs "$OUT/" --quiet
$PY -m k2cascade.projector.lens --source "$SRC" --target "$TGT" --projector runs/mlp_seed2 --out analysis/lens --skip_binding --squad data/cases.jsonl --n_squad 24 > logs/lens_squad.log 2>&1
note "squad: $(tail -1 logs/lens_squad.log)"
gcloud storage cp -r analysis logs "$OUT/" --quiet
note "lens finished"

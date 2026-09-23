# Compression, latency and different-context QA (paper gap checklist). K2_FROM = program run.
set -euo pipefail
BK="${K2_BUCKET:-gs://k2-cascade-runs-less-more}"; FROM="$BK/${K2_FROM:?}"
note() { echo "[compress $(date -u +%H:%MZ)] $*" | tee -a logs/compress.log; gcloud storage cp logs/compress.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
gcloud storage rsync -r "$FROM/runs/mlp_top3_s1500_seed2/step_1500" runs/mlp_seed2 --quiet
gcloud storage rsync -r "$FROM/runs/ridge_top3" runs/ridge_top3 --quiet
gcloud storage cp "$BK/shared/squad_600.jsonl" data/ --quiet 2>/dev/null || HF_HUB_OFFLINE=0 $PY -c "from k2cascade.projector.qa import load_squad; import json; open('data/squad_600.jsonl','w').writelines(json.dumps(e)+'\n' for e in load_squad(600, 0))"
SPECS=$(echo "${K2_SPECS:-none int8 int4 layers=18-35 layers=18-35,int8 heads=4 heads=2 rank=32 rank=16 rank=8}" | tr "|;" " ,")  # K2_ENV cannot carry spaces; gcloud metadata cannot carry commas
if [ "${K2_ONLY_SWEEP:-0}" != 1 ]; then
# latency
$PY -m k2cascade.projector.latency --source "$SRC" --target "$TGT" --projector runs/mlp_seed2 --prefix 512 --out analysis/latency_mlp.json > logs/latency.log 2>&1
$PY -m k2cascade.projector.latency --source "$SRC" --target "$TGT" --projector runs/ridge_top3 --prefix 512 --out analysis/latency_ridge.json >> logs/latency.log 2>&1
note "latency: $(cat analysis/latency_mlp.json | tr -d '\n ' | cut -c1-300)"
fi
# compression sweep on Noma (6 names, n=300) and SQuAD (n=300)
for spec in $SPECS; do
  t=$(echo "$spec" | tr '=,' '__')
  $PY -m k2cascade.projector.noma --source "$SRC" --target "$TGT" --projector runs/mlp_seed2 --episodes 300 --names 6 --colours 8 --arms project,derange --compress "$spec" --out "analysis/cmp_noma_$t.json" > "logs/cmp_noma_$t.log" 2>&1
  $PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector runs/mlp_seed2 --data data/squad_600.jsonl --n 300 --arms project,derange --compress "$spec" --out "analysis/cmp_qa_$t.json" > "logs/cmp_qa_$t.log" 2>&1
  note "$spec: noma $($PY -c "import json;d=json.load(open('analysis/cmp_noma_$t.json'));print(round(d['project']['acc'],3), 'B/tok', d['compress']['bytes_per_token'])") | qa f1 $($PY -c "import json;d=json.load(open('analysis/cmp_qa_$t.json'));print(round(d['project']['f1'],3))")"
  gcloud storage cp -r analysis logs "$OUT/" --quiet
done
[ "${K2_ONLY_SWEEP:-0}" = 1 ] && { note "compress finished (sweep only)"; exit 0; }
# different-context QA (HotpotQA bridge, sender and receiver hold one paragraph each)
HF_HUB_OFFLINE=0 $PY -m k2cascade.projector.data.prepare_hotpot --n 600 --out data/hotpot_split.jsonl > logs/hotpot_data.log 2>&1
gcloud storage cp data/hotpot_split.jsonl "$BK/shared/" --quiet
$PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector none --data data/hotpot_split.jsonl --n 400 --max_new 12 --out analysis/hotpot_none.json > logs/hotpot_none.log 2>&1
for p in ridge_top3 mlp_seed2; do
  $PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector "runs/$p" --data data/hotpot_split.jsonl --n 400 --max_new 12 --arms project,derange,zero --out "analysis/hotpot_$p.json" > "logs/hotpot_$p.log" 2>&1
  note "hotpot $p: $($PY -c "import json;d=json.load(open('analysis/hotpot_$p.json'));print({a:round(d[a]['f1'],3) for a in d if isinstance(d[a],dict)})")"
done
note "hotpot baselines: $($PY -c "import json;d=json.load(open('analysis/hotpot_none.json'));print({a:round(d[a]['f1'],3) for a in d if isinstance(d[a],dict)})")"
gcloud storage cp -r analysis logs "$OUT/" --quiet
note "compress finished"

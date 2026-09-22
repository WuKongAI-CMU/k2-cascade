# Reproduce the 2026-09-21 run: fit same-layer and top-3 ridge maps, then Noma + diagnostics for each.
set -euo pipefail
HF_HUB_OFFLINE=0 $PY -m k2cascade.projector.data.prepare_fineweb --tokenizer "$TGT" --n 544 --out data/fineweb_1024.jsonl > logs/data.log 2>&1
[ "$(wc -l < data/fineweb_1024.jsonl)" -ge 544 ] || { echo "data prep wrote too few sequences"; exit 1; }
$PY -m k2cascade.projector.noma --source "$SRC" --target "$TGT" --projector none --episodes 300 --out analysis/noma_precheck.json > logs/precheck.log 2>&1
for m in "last_aligned 1 ridge" "topk 3 ridge_top3"; do
  set -- $m
  $PY -m k2cascade.projector.ridge --source "$SRC" --target "$TGT" --data data/fineweb_1024.jsonl --out runs/$3 --seqs 512 --eval_seqs 32 --map $1 --k $2 > logs/$3.log 2>&1
  $PY -m k2cascade.projector.diag --source "$SRC" --target "$TGT" --projector runs/$3 --data data/fineweb_1024.jsonl > logs/diag_$3.log 2>&1
  $PY -m k2cascade.projector.noma --source "$SRC" --target "$TGT" --projector runs/$3 --episodes 300 --out analysis/noma_$3.json > logs/noma_$3.log 2>&1
  gcloud storage cp -r analysis logs "$OUT/" --quiet
done

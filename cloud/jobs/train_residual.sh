# Tier 2: ridge (top-3 source layers) -> learned gated residual trained on the 7B's continuation loss.
# Compares ridge-only and trained on Noma and on held-out continuation retention.
set -euo pipefail
STEPS=${K2_STEPS:-300}
HF_HUB_OFFLINE=0 $PY -m k2cascade.projector.data.prepare_fineweb --tokenizer "$TGT" --n 2048 --out data/fineweb_1024.jsonl > logs/data.log 2>&1
$PY -m k2cascade.projector.ridge --source "$SRC" --target "$TGT" --data data/fineweb_1024.jsonl --out runs/ridge_top3 --seqs 512 --eval_seqs 32 --map topk --k 3 > logs/ridge_top3.log 2>&1
$PY -m k2cascade.projector.noma --source "$SRC" --target "$TGT" --projector runs/ridge_top3 --episodes 300 --out analysis/noma_ridge_top3.json > logs/noma_ridge_top3.log 2>&1
gcloud storage cp -r analysis logs "$OUT/" --quiet
$PY -m k2cascade.projector.train --source "$SRC" --target "$TGT" --data data/fineweb_1024.jsonl --ridge runs/ridge_top3 \
    --out runs/mlp_top3 --steps "$STEPS" --batch 2 --accum 8 --save_every 50 > logs/train.log 2>&1
FINAL=runs/mlp_top3/step_$STEPS
$PY -m k2cascade.projector.noma --source "$SRC" --target "$TGT" --projector "$FINAL" --episodes 300 --out analysis/noma_mlp_top3.json > logs/noma_mlp_top3.log 2>&1
$PY -m k2cascade.projector.eval --source "$SRC" --target "$TGT" --projector "$FINAL" --data data/fineweb_1024.jsonl > logs/eval_mlp_top3.log 2>&1

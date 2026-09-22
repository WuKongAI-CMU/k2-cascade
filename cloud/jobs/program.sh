# Unattended research program on one on-demand A100. Each phase uploads its results and writes a marker to
# $OUT/phases/<name>.done, so a relaunch with the same run id skips finished phases. Decisions between phases
# follow fixed rules written below; nothing waits for a human.
set -euo pipefail
MAX_HOURS=${K2_MAX_HOURS:-40}
T0=$SECONDS
note() { echo "[program $(date -u +%H:%MZ) +$(( (SECONDS - T0) / 60 ))m] $*" | tee -a logs/program.log; gcloud storage cp logs/program.log "$OUT/logs/program.log" --quiet >/dev/null 2>&1 || true; }
done_p() { gcloud storage ls "$OUT/phases/$1.done" >/dev/null 2>&1; }
mark() { echo ok | gcloud storage cp - "$OUT/phases/$1.done" --quiet; gcloud storage cp -r analysis logs "$OUT/" --quiet; gcloud storage rsync -r runs "$OUT/runs" --quiet >/dev/null 2>&1 || true; }
hours_left() { echo $(( MAX_HOURS - (SECONDS - T0) / 3600 )); }
acc() { $PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['project']['acc'])" "$1"; }
xfer() { $PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['content_transfer'])" "$1"; }
noma() { $PY -m k2cascade.projector.noma --source "$SRC" --target "$TGT" --projector "$1" --episodes 300 --out "$2" > "logs/$(basename "$2" .json).log" 2>&1; }
retention() { $PY -m k2cascade.projector.eval --source "$SRC" --target "$TGT" --projector "$1" --data data/fineweb_1024.jsonl > "logs/eval_$2.log" 2>&1 || true; }
ridge() {  # ridge <map> <k> <name>
  [ -f "runs/$3/ridge.safetensors" ] || $PY -m k2cascade.projector.ridge --source "$SRC" --target "$TGT" \
      --data data/fineweb_1024.jsonl --out "runs/$3" --seqs 512 --eval_seqs 32 --map "$1" --k "$2" > "logs/$3.log" 2>&1; }
train() {  # train <ridge_dir> <name> <steps> [extra flags...]
  local r=$1 n=$2 s=$3; shift 3
  $PY -m k2cascade.projector.train --source "$SRC" --target "$TGT" --data data/fineweb_1024.jsonl --ridge "$r" \
      --out "runs/$n" --steps "$s" --batch 1 --accum 16 --save_every 100 "$@" > "logs/train_$n.log" 2>&1
  noma "runs/$n/step_$s" "analysis/noma_$n.json"; retention "runs/$n/step_$s" "$n"; }

# ---- P0 data: 8192 sequences (32 held out, 512 for ridge, the rest for training)
if ! done_p P0; then
  HF_HUB_OFFLINE=0 $PY -m k2cascade.projector.data.prepare_fineweb --tokenizer "$TGT" --n 8192 --out data/fineweb_1024.jsonl > logs/data.log 2>&1
  [ "$(wc -l < data/fineweb_1024.jsonl)" -ge 8192 ] || { note "data prep short"; exit 1; }
  gcloud storage cp data/fineweb_1024.jsonl "$OUT/data/" --quiet; mark P0; note "P0 data done"
else
  gcloud storage cp "$OUT/data/fineweb_1024.jsonl" data/ --quiet
fi

# ---- P1 baseline: top-3 ridge on Noma, then a short residual run (300 steps) as a smoke test of training
if ! done_p P1; then
  ridge topk 3 ridge_top3; noma runs/ridge_top3 analysis/noma_ridge_top3.json; retention runs/ridge_top3 ridge_top3
  train runs/ridge_top3 mlp_top3_s300 300 --seed 0
  mark P1; note "P1 ridge acc=$(acc analysis/noma_ridge_top3.json) transfer=$(xfer analysis/noma_ridge_top3.json); mlp300 acc=$(acc analysis/noma_mlp_top3_s300.json) transfer=$(xfer analysis/noma_mlp_top3_s300.json)"
fi
ridge topk 3 ridge_top3

# ---- P2 main run: 1500 steps
if ! done_p P2 && [ "$(hours_left)" -ge 10 ]; then
  train runs/ridge_top3 mlp_top3_s1500_seed0 1500 --seed 0
  mark P2; note "P2 mlp1500 acc=$(acc analysis/noma_mlp_top3_s1500_seed0.json) transfer=$(xfer analysis/noma_mlp_top3_s1500_seed0.json)"
fi

# ---- decision: did learning add at least 5 points of Noma accuracy over the ridge map?
BASE=$(acc analysis/noma_ridge_top3.json); MAIN=$(acc analysis/noma_mlp_top3_s1500_seed0.json 2>/dev/null || echo 0)
GAIN=$($PY -c "print(round($MAIN-$BASE,3))"); LEARNS=$($PY -c "print(int($MAIN-$BASE>=0.05))")
note "decision: ridge=$BASE mlp1500=$MAIN gain=$GAIN learns=$LEARNS"

if [ "$LEARNS" = 1 ]; then
  # ---- P3 robustness: two more seeds, frozen-ridge ablation, same-layer ablation
  for spec in "P3a mlp_top3_s1500_seed1 runs/ridge_top3 --seed 1" "P3b mlp_top3_s1500_seed2 runs/ridge_top3 --seed 2" \
              "P3c mlp_top3_s1500_frozenridge runs/ridge_top3 --seed 0 --freeze_ridge" "P3d mlp_same_s1500 runs/ridge_same --seed 0"; do
    set -- $spec; p=$1 n=$2 r=$3; shift 3
    done_p "$p" && continue
    [ "$(hours_left)" -ge 8 ] || { note "stopping before $p: time budget"; break; }
    [ "$r" = runs/ridge_same ] && ridge last_aligned 1 ridge_same
    train "$r" "$n" 1500 "$@"; mark "$p"; note "$p $n acc=$(acc analysis/noma_$n.json) transfer=$(xfer analysis/noma_$n.json)"
  done
else
  # ---- P3' learning did not help: probe whether capacity or step size is the limit
  for spec in "P3e mlp_top3_s1500_wide runs/ridge_top3 --seed 0 --bottleneck 256 --lr 3e-4" "P3d mlp_same_s1500 runs/ridge_same --seed 0"; do
    set -- $spec; p=$1 n=$2 r=$3; shift 3
    done_p "$p" && continue
    [ "$(hours_left)" -ge 8 ] || { note "stopping before $p: time budget"; break; }
    [ "$r" = runs/ridge_same ] && ridge last_aligned 1 ridge_same
    train "$r" "$n" 1500 "$@"; mark "$p"; note "$p $n acc=$(acc analysis/noma_$n.json) transfer=$(xfer analysis/noma_$n.json)"
  done
fi
note "program finished"

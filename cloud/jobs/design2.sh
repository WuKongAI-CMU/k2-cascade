# Redesigned experiment set (docs/experiment-design-2026-09-21.md). Needs the program run's data and checkpoints:
# K2_FROM=<program run id>. Phases write markers like program.sh so a relaunch resumes.
#   E1 control   : residual trained with a deranged sender (previous batch's text) -> a learned soft prompt, no content
#   E2 wait      : the program's 1500-step seed-0 projector
#   E3 evals     : retention with derange arm on fresh text; Noma at n=1000; other attribute types; fact->question
#                  distance 64..512 tokens; which receiver layers carry the binding
set -euo pipefail
FROM="${K2_BUCKET:-gs://k2-cascade-runs-less-more}/${K2_FROM:?set K2_FROM}"
note() { echo "[design2 $(date -u +%H:%MZ)] $*" | tee -a logs/design2.log; gcloud storage cp logs/design2.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
done_p() { gcloud storage ls "$OUT/phases/$1.done" >/dev/null 2>&1; }
mark() { echo ok | gcloud storage cp - "$OUT/phases/$1.done" --quiet; gcloud storage cp -r analysis logs "$OUT/" --quiet; }
acc() { $PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(round(d['project']['acc'],3), 'derange', round(d['derange']['acc'],3), 'follow', round(d['derange']['follow_rate'],3))" "$1"; }
noma() { local p=$1 o=$2; shift 2; $PY -m k2cascade.projector.noma --source "$SRC" --target "$TGT" --projector "$p" --out "analysis/$o.json" "$@" > "logs/$o.log" 2>&1; note "$o: $(acc analysis/$o.json)"; }

gcloud storage cp "$FROM/data/fineweb_1024.jsonl" data/ --quiet
gcloud storage rsync -r "$FROM/runs/ridge_top3" runs/ridge_top3 --quiet
[ -f runs/ridge_top3/ridge.safetensors ] || { note "no ridge_top3 in $FROM"; exit 1; }

# fresh held-out text (never seen by ridge fit or training)
HF_HUB_OFFLINE=0 $PY -m k2cascade.projector.data.prepare_fineweb --tokenizer "$TGT" --n 300 --seed 2 --out data/fresh2_raw.jsonl > logs/data_fresh2.log 2>&1
$PY - <<'PY'
import json
seen={tuple(json.loads(l)['ids'][:64]) for l in open('data/fineweb_1024.jsonl')}
rows=[l for l in open('data/fresh2_raw.jsonl') if tuple(json.loads(l)['ids'][:64]) not in seen]
open('data/fresh2_1024.jsonl','w').writelines(rows[:128]); print(len(rows),"fresh rows")
PY

# ---- E1 control: same recipe as the main run, but the sender reads the wrong text
if ! done_p E1; then
  $PY -m k2cascade.projector.train --source "$SRC" --target "$TGT" --data data/fineweb_1024.jsonl --ridge runs/ridge_top3 \
      --out runs/ctrl_derange_s1500 --steps 1500 --batch 1 --accum 16 --save_every 500 --seed 0 --derange_source > logs/train_ctrl_derange.log 2>&1
  gcloud storage rsync -r runs/ctrl_derange_s1500/step_1500 "$OUT/runs/ctrl_derange_s1500/step_1500" --quiet
  mark E1; note "E1 control trained"
fi

# ---- E2 wait for the program's seed-0 projector (up to 8 h)
MAIN=runs/mlp_top3_s1500_seed0/step_1500
for i in $(seq 1 96); do
  gcloud storage ls "$FROM/phases/P2.done" >/dev/null 2>&1 && { gcloud storage rsync -r "$FROM/$MAIN" "$MAIN" --quiet; break; }
  sleep 300
done
[ -f "$MAIN/mlp.safetensors" ] || { note "program P2 never appeared"; exit 1; }
note "E2 have $MAIN"

# ---- E3 evals
PROJS="runs/ridge_top3 $MAIN runs/ctrl_derange_s1500/step_1500"
tag() { if [ "$(basename "$1")" = step_1500 ]; then basename "$(dirname "$1")"; else basename "$1"; fi; }
for p in $PROJS; do t=$(tag "$p")
  done_p "E3ret_$t" && continue
  $PY -m k2cascade.projector.eval --source "$SRC" --target "$TGT" --projector "$p" --data data/fresh2_1024.jsonl --eval_seqs 128 --batch 2 --out "analysis/ret_fresh_$t.json" > "logs/ret_fresh_$t.log" 2>&1
  note "retention $t: $(tail -1 logs/ret_fresh_$t.log | cut -c1-300)"; mark "E3ret_$t"
done
for p in $PROJS; do t=$(tag "$p")
  done_p "E3main_$t" && continue
  noma "$p" "noma_main_$t" --episodes 1000 --names 8 --colours 8 --seed 0
  noma "$p" "noma_city_$t" --episodes 500 --names 8 --colours 8 --attr city --seed 0
  noma "$p" "noma_animal_$t" --episodes 500 --names 8 --colours 8 --attr animal --seed 0
  mark "E3main_$t"
done
for p in $PROJS; do t=$(tag "$p")
  done_p "E3pad_$t" && continue
  for pad in 64 128 256 512; do noma "$p" "noma_pad${pad}_$t" --episodes 500 --names 8 --colours 8 --pad $pad --seed 0; done
  mark "E3pad_$t"
done
if ! done_p E3layers; then
  for band in 0-11 12-23 24-35 0-17 18-35 0-5 30-35; do
    noma "$MAIN" "noma_layers${band}_mlp" --episodes 500 --names 8 --colours 8 --layers "$band" --arms project,derange --seed 0
  done
  mark E3layers
fi
$PY -m k2cascade.projector.summarize analysis > analysis/summary.md; gcloud storage cp -r analysis "$OUT/" --quiet
note "design2 finished"

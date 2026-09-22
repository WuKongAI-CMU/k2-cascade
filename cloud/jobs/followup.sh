# Follow-up to program.sh: reads that run's checkpoints from GCS (K2_FROM=<run id>) and answers three questions
# the program leaves open: (1) does continuation retention hold on 256 fresh held-out sequences, (2) does Noma
# transfer survive the full 8-name pool with 500 episodes, (3) does the reverse direction 7B -> 3.7B work at all.
set -euo pipefail
FROM="${K2_BUCKET:-gs://k2-cascade-runs-less-more}/${K2_FROM:?set K2_FROM to the program run id}"
note() { echo "[followup $(date -u +%H:%MZ)] $*" | tee -a logs/followup.log; gcloud storage cp logs/followup.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
gcloud storage cp "$FROM/data/fineweb_1024.jsonl" data/ --quiet
gcloud storage cp -r "$FROM/analysis" . --quiet
gcloud storage rsync -r "$FROM/runs" runs --quiet
# best learned projector by Noma accuracy across the program's seeds
BEST=$($PY - <<'PY'
import json,glob,os
c=[(json.load(open(f))['project']['acc'],f) for f in glob.glob('analysis/noma_mlp_top3_s1500_seed*.json')]
c.sort(); f=c[-1][1]; n=os.path.basename(f)[5:-5]; print(f"runs/{n}/step_1500")
PY
)
note "best learned projector: $BEST ($(ls "$BEST"))"

# (1) fresh held-out: new shuffle seed, drop anything whose first 64 tokens appear in the training file
HF_HUB_OFFLINE=0 $PY -m k2cascade.projector.data.prepare_fineweb --tokenizer "$TGT" --n 400 --seed 1 --out data/fresh_raw.jsonl > logs/data_fresh.log 2>&1
$PY - <<'PY'
import json
seen={tuple(json.loads(l)['ids'][:64]) for l in open('data/fineweb_1024.jsonl')}
rows=[l for l in open('data/fresh_raw.jsonl') if tuple(json.loads(l)['ids'][:64]) not in seen]
open('data/fresh_1024.jsonl','w').writelines(rows[:256]); print(len(rows),"fresh rows kept")
PY
for p in runs/ridge_top3 "$BEST"; do
  n=$(echo "$p" | tr '/' '_')
  $PY -m k2cascade.projector.eval --source "$SRC" --target "$TGT" --projector "$p" --data data/fresh_1024.jsonl --eval_seqs 256 > "logs/fresh_eval_$n.log" 2>&1 || true
  note "fresh retention $p: $(tail -1 logs/fresh_eval_$n.log)"
done
gcloud storage cp -r analysis logs "$OUT/" --quiet

# (2) harder Noma: all 8 names, 500 episodes, two seeds
for p in runs/ridge_top3 "$BEST"; do for s in 0 1; do
  n=$(echo "$p" | tr '/' '_')
  $PY -m k2cascade.projector.noma --source "$SRC" --target "$TGT" --projector "$p" --episodes 500 --names 8 --colours 8 --seed $s --out "analysis/noma8_${n}_seed$s.json" > "logs/noma8_${n}_seed$s.log" 2>&1
  note "noma8 $p seed$s acc=$($PY -c "import json;d=json.load(open('analysis/noma8_${n}_seed$s.json'));print(d['project']['acc'], d['derange']['follow_rate'])")"
done; done
gcloud storage cp -r analysis logs "$OUT/" --quiet

# (3) reverse direction 7B -> 3.7B: ridge top-3, 1500 steps, Noma
$PY -m k2cascade.projector.ridge --source "$TGT" --target "$SRC" --data data/fineweb_1024.jsonl --out runs/rev_ridge_top3 --seqs 512 --eval_seqs 32 --map topk --k 3 > logs/rev_ridge_top3.log 2>&1
$PY -m k2cascade.projector.noma --source "$TGT" --target "$SRC" --projector runs/rev_ridge_top3 --episodes 300 --out analysis/noma_rev_ridge_top3.json > logs/noma_rev_ridge_top3.log 2>&1
note "reverse ridge acc=$($PY -c "import json;print(json.load(open('analysis/noma_rev_ridge_top3.json'))['project']['acc'])")"
$PY -m k2cascade.projector.train --source "$TGT" --target "$SRC" --data data/fineweb_1024.jsonl --ridge runs/rev_ridge_top3 \
    --out runs/rev_mlp_top3 --steps 1500 --batch 1 --accum 16 --save_every 100 --seed 0 > logs/train_rev_mlp_top3.log 2>&1
$PY -m k2cascade.projector.noma --source "$TGT" --target "$SRC" --projector runs/rev_mlp_top3/step_1500 --episodes 300 --out analysis/noma_rev_mlp_top3.json > logs/noma_rev_mlp_top3.log 2>&1
note "reverse mlp1500 acc=$($PY -c "import json;d=json.load(open('analysis/noma_rev_mlp_top3.json'));print(d['project']['acc'], d['derange']['follow_rate'])")"
gcloud storage rsync -r runs/rev_mlp_top3/step_1500 "$OUT/runs/rev_mlp_top3/step_1500" --quiet
note "followup finished"

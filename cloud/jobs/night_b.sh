# Overnight B: seeds for the cross-family pair (Qwen3-4B -> K2-7B, seeds 1 and 2) and HotpotQA different-context
# on all three K2 seeds, so the two new headline tables get confidence intervals.
set -euo pipefail
BK="${K2_BUCKET:-gs://k2-cascade-runs-less-more}"; FROM="$BK/${K2_FROM:?}"; XF="$BK/${K2_XF:?}"; CP="$BK/${K2_CP:?}"
note() { echo "[night_b $(date -u +%H:%MZ)] $*" | tee -a logs/night_b.log; gcloud storage cp logs/night_b.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
done_p() { gcloud storage ls "$OUT/phases/$1.done" >/dev/null 2>&1; }
mark() { echo ok | gcloud storage cp - "$OUT/phases/$1.done" --quiet; gcloud storage cp -r analysis logs "$OUT/" --quiet; gcloud storage rsync -r runs "$OUT/runs" --quiet >/dev/null 2>&1 || true; }
gcloud storage cp "$FROM/data/fineweb_1024.jsonl" data/ --quiet
gcloud storage cp "$BK/shared/squad_600.jsonl" data/ --quiet
gcloud storage cp "$BK/shared/hotpot_split.jsonl" data/ --quiet
gcloud storage rsync -r "$XF/runs/ridge_qwen3-4b" runs/ridge_qwen3-4b --quiet
for s in 0 1 2; do gcloud storage rsync -r "$FROM/runs/mlp_top3_s1500_seed$s/step_1500" "runs/mlp_seed$s" --quiet; done
QW=$(HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-4B', allow_patterns=['*.json','*.py','*.safetensors','*.jinja','tokenizer*','*.txt','merges.txt','vocab*']))")
for seed in 1 2; do
  done_p "Q$seed" && continue
  $PY -m k2cascade.projector.train --source "$QW" --target "$TGT" --data data/fineweb_1024.jsonl --ridge runs/ridge_qwen3-4b \
      --out "runs/mlp_qwen_seed$seed" --steps 1500 --batch 1 --accum 16 --save_every 500 --seed $seed > "logs/train_qwen_seed$seed.log" 2>&1
  $PY -m k2cascade.projector.noma --source "$QW" --target "$TGT" --projector "runs/mlp_qwen_seed$seed/step_1500" --episodes 300 --out "analysis/xf_noma_mlp_qwen3-4b_seed$seed.json" > "logs/xf_noma_qwen_seed$seed.log" 2>&1
  $PY -m k2cascade.projector.qa --source "$QW" --target "$TGT" --projector "runs/mlp_qwen_seed$seed/step_1500" --data data/squad_600.jsonl --n 300 --arms text,project,derange --out "analysis/xf_qa_mlp_qwen3-4b_seed$seed.json" > "logs/xf_qa_qwen_seed$seed.log" 2>&1
  note "qwen seed$seed: noma $($PY -c "import json;d=json.load(open('analysis/xf_noma_mlp_qwen3-4b_seed$seed.json'));print(round(d['project']['acc'],3),'derange',round(d['derange']['acc'],3))") qa $($PY -c "import json;d=json.load(open('analysis/xf_qa_mlp_qwen3-4b_seed$seed.json'));print({a:round(d[a]['f1'],3) for a in d if isinstance(d[a],dict)})")"
  mark "Q$seed"
done
for s in 0 1 2; do
  done_p "H$s" && continue
  $PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector "runs/mlp_seed$s" --data data/hotpot_split.jsonl --n 400 --max_new 12 --arms project,derange --out "analysis/hotpot_mlp_seed$s.json" > "logs/hotpot_seed$s.log" 2>&1
  note "hotpot seed$s: $($PY -c "import json;d=json.load(open('analysis/hotpot_mlp_seed$s.json'));print({a:round(d[a]['f1'],3) for a in d if isinstance(d[a],dict)})")"
  mark "H$s"
done
note "night_b finished"

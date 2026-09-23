# Cross-family / cross-vocabulary senders into K2-Horizon-7B (paper gap: >= 3 pairs incl. one non-K2).
# Pairs: Qwen/Qwen3-4B (36 layers, 8 KV heads, d 128: matched KV, different vocab) and IFM/K2-Horizon-0.9B
# (same family, different vocab). Same recipe as program.sh: top-3 ridge -> 1500-step residual -> Noma + SQuAD.
set -euo pipefail
BK="${K2_BUCKET:-gs://k2-cascade-runs-less-more}"; FROM="$BK/${K2_FROM:?}"
STEPS=${K2_STEPS:-1500}
note() { echo "[xfamily $(date -u +%H:%MZ)] $*" | tee -a logs/xfamily.log; gcloud storage cp logs/xfamily.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
done_p() { gcloud storage ls "$OUT/phases/$1.done" >/dev/null 2>&1; }
mark() { echo ok | gcloud storage cp - "$OUT/phases/$1.done" --quiet; gcloud storage cp -r analysis logs "$OUT/" --quiet; gcloud storage rsync -r runs "$OUT/runs" --quiet >/dev/null 2>&1 || true; }
gcloud storage cp "$FROM/data/fineweb_1024.jsonl" data/ --quiet
gcloud storage cp "$BK/shared/squad_600.jsonl" data/ --quiet 2>/dev/null || HF_HUB_OFFLINE=0 $PY -c "from k2cascade.projector.qa import load_squad; import json; open('data/squad_600.jsonl','w').writelines(json.dumps(e)+'\n' for e in load_squad(600, 0))"
# sender snapshots (the startup script only pins the 3.7B/7B)
SENDERS=$(HF_HUB_OFFLINE=0 $PY - <<'PY'
from huggingface_hub import snapshot_download
import json
out = {}
for r in ("Qwen/Qwen3-4B", "IFM/K2-Horizon-0.9B"):
    out[r] = snapshot_download(r, allow_patterns=["*.json", "*.py", "*.safetensors", "*.jinja", "tokenizer*", "*.txt", "merges.txt", "vocab*"])
json.dump(out, open("sender_paths.json", "w"))
print(" ".join(f"{k.split('/')[-1]}={v}" for k, v in out.items()))
PY
)
note "senders: $SENDERS"
for kv in $SENDERS; do
  name=${kv%%=*}; path=${kv#*=}; tag=$(echo "$name" | tr 'A-Z.' 'a-z_')
  if ! done_p "R_$tag"; then
    $PY -m k2cascade.projector.ridge --source "$path" --target "$TGT" --data data/fineweb_1024.jsonl --out "runs/ridge_$tag" --seqs 512 --eval_seqs 32 --map topk --k 3 > "logs/ridge_$tag.log" 2>&1
    $PY -m k2cascade.projector.noma --source "$path" --target "$TGT" --projector "runs/ridge_$tag" --episodes 300 --out "analysis/xf_noma_ridge_$tag.json" > "logs/xf_noma_ridge_$tag.log" 2>&1
    note "$name ridge noma: $($PY -c "import json;d=json.load(open('analysis/xf_noma_ridge_$tag.json'));print(round(d['project']['acc'],3), 'raw', round(d['raw']['acc'],3), 'derange', round(d['derange']['acc'],3))")"
    mark "R_$tag"
  fi
  if ! done_p "T_$tag"; then
    $PY -m k2cascade.projector.train --source "$path" --target "$TGT" --data data/fineweb_1024.jsonl --ridge "runs/ridge_$tag" \
        --out "runs/mlp_$tag" --steps "$STEPS" --batch 1 --accum 16 --save_every 500 --seed 0 > "logs/train_$tag.log" 2>&1
    $PY -m k2cascade.projector.noma --source "$path" --target "$TGT" --projector "runs/mlp_$tag/step_$STEPS" --episodes 300 --out "analysis/xf_noma_mlp_$tag.json" > "logs/xf_noma_mlp_$tag.log" 2>&1
    $PY -m k2cascade.projector.qa --source "$path" --target "$TGT" --projector "runs/mlp_$tag/step_$STEPS" --data data/squad_600.jsonl --n 300 --arms none,text,project,derange --out "analysis/xf_qa_mlp_$tag.json" > "logs/xf_qa_mlp_$tag.log" 2>&1
    note "$name mlp$STEPS noma: $($PY -c "import json;d=json.load(open('analysis/xf_noma_mlp_$tag.json'));print(round(d['project']['acc'],3), 'derange', round(d['derange']['acc'],3), 'follow', round(d['derange']['follow_rate'],3))") | qa: $($PY -c "import json;d=json.load(open('analysis/xf_qa_mlp_$tag.json'));print({a:round(d[a]['f1'],3) for a in d if isinstance(d[a],dict)})")"
    mark "T_$tag"
  fi
done
note "xfamily finished"

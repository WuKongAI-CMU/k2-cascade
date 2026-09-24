# Milestone 4: sender features + receiver arms on 600 SQuAD items, then the accuracy-vs-cost curves.
# K2_FROM = program run (projector seed 2). Needs shared/squad_600.jsonl.
set -euo pipefail
BK="${K2_BUCKET:-gs://k2-cascade-runs-less-more}"; FROM="$BK/${K2_FROM:?}"
note() { echo "[cascade $(date -u +%H:%MZ)] $*" | tee -a logs/cascade.log; gcloud storage cp logs/cascade.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
gcloud storage ls "$BK/shared/squad_600.jsonl" >/dev/null
gcloud storage cp "$BK/shared/squad_600.jsonl" data/ --quiet
gcloud storage rsync -r "$FROM/runs/mlp_top3_s1500_seed2/step_1500" runs/mlp_seed2 --quiet
mkdir -p analysis/cascade
HF_HUB_OFFLINE=0 $PY -m k2cascade.projector.cascade --model "$SRC" --data data/squad_600.jsonl --out analysis/cascade/sender.npz > logs/cascade_sender.log 2>&1
note "sender: $(tail -1 logs/cascade_sender.log)"
# receiver arms; verbal needs the sender's answer + a confidence word derived from its sample agreement
$PY - <<'PY'
import json, numpy as np
s = np.load("analysis/cascade/sender.npz"); rows = [json.loads(l) for l in open("data/squad_600.jsonl")]
se = s["se"]; lo, hi = np.quantile(se, [1/3, 2/3])
with open("data/squad_600_verbal.jsonl", "w") as f:
    for i, ex in enumerate(rows[:len(se)]):
        conf = "high" if se[i] <= lo else ("medium" if se[i] <= hi else "low")
        f.write(json.dumps({**ex, "sender_answer": str(s["greedy"][i]), "sender_conf": conf}) + "\n")
PY
$PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector runs/mlp_seed2 --data data/squad_600_verbal.jsonl --n 600 \
    --arms text,project,verbal --out analysis/cascade/qa.json --per_item analysis/cascade/qa_items_all.jsonl > logs/cascade_qa.log 2>&1
for arm in text project verbal; do cp analysis/cascade/qa_items_all.jsonl "analysis/cascade/qa_items_$arm.jsonl"; done
note "receiver: $($PY -c "import json;d=json.load(open('analysis/cascade/qa.json'));print({a:round(d[a]['f1'],3) for a in d if isinstance(d[a],dict)})")"
$PY -m k2cascade.projector.policy --sender analysis/cascade/sender.npz --items analysis/cascade --out analysis/cascade/curves.json > logs/policy.log 2>&1
note "policy: $(tr '\n' ' ' < logs/policy.log | cut -c1-900)"
gcloud storage cp -r analysis logs "$OUT/" --quiet
note "cascade finished"

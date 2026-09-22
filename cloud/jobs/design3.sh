# Passage QA transfer (milestone 2 of docs/goal-2026-09-21.md): SQuAD passages read by the sender only.
# K2_FROM = program run (ridge_top3, mlp seeds), K2_FROM2 = design2 run (deranged-sender control).
set -euo pipefail
BK="${K2_BUCKET:-gs://k2-cascade-runs-less-more}"; FROM="$BK/${K2_FROM:?}"; FROM2="$BK/${K2_FROM2:?}"
note() { echo "[design3 $(date -u +%H:%MZ)] $*" | tee -a logs/design3.log; gcloud storage cp logs/design3.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
gcloud storage rsync -r "$FROM/runs/ridge_top3" runs/ridge_top3 --quiet
gcloud storage rsync -r "$FROM/runs/mlp_top3_s1500_seed2/step_1500" runs/mlp_seed2 --quiet
gcloud storage rsync -r "$FROM2/runs/ctrl_derange_s1500/step_1500" runs/ctrl_derange --quiet
HF_HUB_OFFLINE=0 $PY -c "from k2cascade.projector.qa import load_squad; import json; ex=load_squad(600, 0); open('data/squad_600.jsonl','w').writelines(json.dumps(e)+'\n' for e in ex); print(len(ex))" > logs/squad.log 2>&1
gcloud storage cp data/squad_600.jsonl "$OUT/data/" --quiet
# baseline arms once (none/text/self/raw do not depend on the projector)
$PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector none --data data/squad_600.jsonl --n 300 --out analysis/qa_none.json > logs/qa_none.log 2>&1
note "baselines: $($PY -c "import json;d=json.load(open('analysis/qa_none.json'));print({a:round(d[a]['f1'],3) for a in ('none','text','self','raw')})")"
for p in ridge_top3 mlp_seed2 ctrl_derange; do
  $PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector "runs/$p" --data data/squad_600.jsonl --n 300 --arms project,derange --out "analysis/qa_$p.json" > "logs/qa_$p.log" 2>&1
  note "$p: $($PY -c "import json;d=json.load(open('analysis/qa_$p.json'));print('project f1',round(d['project']['f1'],3),'em',round(d['project']['em'],3),'logp',round(d['project']['logp'],3),'| derange f1',round(d['derange']['f1'],3),'logp',round(d['derange']['logp'],3))")"
  gcloud storage cp -r analysis logs "$OUT/" --quiet
done
# second 300 with a different seed for the trained projector only (variance check)
$PY -m k2cascade.projector.qa --source "$SRC" --target "$TGT" --projector runs/mlp_seed2 --data data/squad_600.jsonl --n 600 --arms text,project,derange --out analysis/qa_mlp_seed2_n600.json > logs/qa_mlp_seed2_n600.log 2>&1
note "mlp_seed2 n600: $($PY -c "import json;d=json.load(open('analysis/qa_mlp_seed2_n600.json'));print({a:round(d[a]['f1'],3) for a in ('text','project','derange')})")"
gcloud storage cp -r analysis logs "$OUT/" --quiet
note "design3 finished"

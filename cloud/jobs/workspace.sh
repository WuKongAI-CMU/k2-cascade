# Shared-workspace experiments: word overlap between sender and receiver lenses, and the J-space swap.
# Needs the two fitted lenses (K2_SRC_LENS, K2_TGT_LENS: run ids of the jlens jobs) and the seed-0 map.
set -euo pipefail
BK="${K2_BUCKET:-gs://k2-cascade-runs-less-more}"
note() { echo "[ws $(date -u +%H:%MZ)] $*" | tee -a logs/ws.log; gcloud storage cp logs/ws.log "$OUT/logs/" --quiet >/dev/null 2>&1 || true; }
uv pip install -q --no-deps "git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e"
gcloud storage rsync -r "$BK/design2-20260922-022708/runs/mlp_top3_s1500_seed0/step_1500" runs/mlp_seed0 --quiet
mkdir -p runs/jlens_src runs/jlens_tgt
gcloud storage cp "$BK/${K2_SRC_LENS:?}/runs/jlens_src/lens.pt" runs/jlens_src/ --quiet
gcloud storage cp "$BK/${K2_TGT_LENS:?}/runs/jlens_tgt/lens.pt" runs/jlens_tgt/ --quiet
note "lenses and map ready"
$PY -m k2cascade.projector.workspace --source "$SRC" --target "$TGT" --projector runs/mlp_seed0 \
  --src_lens runs/jlens_src/lens.pt --tgt_lens runs/jlens_tgt/lens.pt --out analysis/workspace ${K2_WS_ARGS:-} 2>&1 | tee logs/ws_run.log
note "$(tail -1 logs/ws_run.log)"
gcloud storage cp -r analysis logs "$OUT/" --quiet

# Lifecycle check on a cheap CPU VM: proves metadata, clone, status, upload and self-delete work.
set -euo pipefail
mkdir -p analysis logs
echo "{\"host\": \"$(hostname)\", \"commit\": \"$(git rev-parse HEAD)\", \"utc\": \"$(date -u +%FT%TZ)\"}" > analysis/smoke.json
echo "smoke ok" > logs/smoke.log

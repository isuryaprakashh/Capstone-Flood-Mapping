#!/usr/bin/env bash
# One orchestrator run on GitHub Actions (see .github/workflows/train.yml).
#
# Runs an optional manual action, then ticks every TICK_EVERY seconds for
# LOOP_SECONDS and finally dispatches its successor, so the pipeline never
# depends on GitHub's cron. It hands over early (no successor) when another
# run of the workflow is waiting, e.g. a manual action or a push, which then
# takes over within a minute.
#
# A tick that raises an alert (the pipeline stopped, or keeps erroring) makes
# this run fail, which is what sends the one GitHub email; the successor is
# dispatched first so the chain continues.
#
# Usage: orchestration/loop.sh <tick|update-code|resume> <phase for resume>
set -uo pipefail

action="${1:-tick}"
phase="${2:-train}"
orch="python orchestration/kaggle_orchestrator.py"
api="https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/workflows/train.yml"
alert=0

run() { "$@" || alert=1; }

waiting_runs() {
  local n=0 status
  for status in pending queued; do
    n=$((n + $(curl -fsS -H "Authorization: Bearer ${GITHUB_TOKEN}" "${api}/runs?status=${status}&per_page=1" \
      | python -c "import json,sys; print(json.load(sys.stdin)['total_count'])" 2>/dev/null || echo 0)))
  done
  echo "$n"
}

dispatch_successor() {
  curl -fsS -X POST -H "Authorization: Bearer ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json" \
    "${api}/dispatches" -d '{"ref": "main", "inputs": {"action": "tick"}}' \
    && echo "Dispatched the next orchestrator run" || echo "Could not dispatch the next run (the backup schedule will restart the chain)"
}

phase_now() {
  $orch status 2>/dev/null | python -c "import json,sys; print(json.load(sys.stdin)['phase'])" 2>/dev/null || echo unknown
}

refresh_code() {  # pick up orchestrator changes pushed to main while this run is alive
  git pull -q --ff-only origin main || true
  export GITHUB_SHA="$(git rev-parse HEAD)"
}

case "$action" in
  update-code) run $orch tick --update-code ;;
  resume)      $orch resume --phase "$phase" && run $orch tick --ignore-hold || alert=1 ;;
  *)           run $orch tick ;;
esac

end=$((SECONDS + ${LOOP_SECONDS:-20400}))
while [ "$alert" -eq 0 ] && [ "$(phase_now)" != "done" ] && [ "$SECONDS" -lt "$end" ]; do
  for _ in $(seq 1 $(( ${TICK_EVERY:-600} / 60 ))); do
    sleep 60
    if [ "$(waiting_runs)" -gt 0 ]; then
      echo "Another run of this workflow is waiting; handing over to it"
      exit "$alert"
    fi
  done
  refresh_code
  run $orch tick
done

if [ "$(phase_now)" != "done" ]; then
  dispatch_successor
fi
exit "$alert"

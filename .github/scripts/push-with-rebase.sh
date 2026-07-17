#!/usr/bin/env bash
# Push HEAD to origin/main, rebasing onto the updated remote between attempts.
# Long-running jobs race with the hourly pushes; a plain `git push` from a stale
# checkout loses the whole run's work (update-endpoints #228 lost a ~$7
# onboarding batch this way). Spend-ledger conflicts auto-resolve via the
# merge=union attribute; an aborted rebase is cleaned up so retries start fresh.
set -uo pipefail

for i in 1 2 3 4 5; do
  git pull --rebase origin main && git push origin main && exit 0
  git rebase --abort 2>/dev/null
  sleep $((i * 10))
done
echo "push failed after 5 attempts" >&2
exit 1

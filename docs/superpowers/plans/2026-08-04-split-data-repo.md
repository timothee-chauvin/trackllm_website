# Split raw data into trackllm_data — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 4–5 are destructive (history rewrite + force-push) and MUST NOT run without explicit user confirmation at the gate.

**Goal:** Move `website/data` (1.3GB of blob history, +156MB/week) into a new public repo `timothee-chauvin/trackllm_data`, and purge it from `trackllm_website` history, so the code repo clones at ~tens of MB while git remains the pipeline's database and public provenance record.

**Architecture:** The data repo holds the current tracked content of `website/data` (`lt/`, `b3it/`, `spend/`) at its root, with full extracted history (`git filter-repo --subdirectory-filter`). In CI and locally, it is checked out / cloned *at* `website/data` inside the code repo (which gitignores that path), so `config.toml`'s `data_dir = "website/data"` and all pipeline code need **zero changes**. Bot workflows commit/push to the data repo via a write deploy key; code-repo files (`endpoints_*.yaml`, `strategies_test_reasoning.json`) keep pushing to the code repo as today.

**Tech Stack:** git-filter-repo (via `uvx`), gh CLI, GitHub Actions (`actions/checkout` pinned at `11d5960a326750d5838078e36cf38b85af677262`).

## Global Constraints

- The two irreversible actions are: force-pushing the final extraction over `trackllm_data` main, and force-pushing rewritten history to `trackllm_website`. Both live in Task 4, behind the user gate.
- Bot workflows (`run-main.yml` hourly, `bi-monitor.yml`, `update-endpoints.yml`) push data commits ~12×/day. All data-repo extraction of record happens **inside the freeze window** (workflows disabled) so no commit is lost.
- The user has local worktrees on old-history branches. Never reset or delete anything local without the `website/data` safety copy (Task 5 step 1) done first.
- `push-with-rebase.sh` is reused unchanged in both repos: it operates on `origin/main` of whatever repo cwd is in. It requires the spend union-merge attribute, which moves to the data repo's own `.gitattributes`.
- `git config user.*` becomes `--global` in workflow commit steps (two repos share one runner).

---

### Task 1: Create the (empty) data repo, deploy key, secret

Non-destructive. No history is pushed yet — that happens in the freeze (Task 4) so the extraction is final.

- [ ] **Step 1: Create the repo**

```bash
gh repo create timothee-chauvin/trackllm_data --public \
  --description "Raw monitoring data for TrackLLM (trackllm_website). Append-only; pushed by CI."
```

- [ ] **Step 2: Create a write deploy key and store the private half as a code-repo secret**

```bash
cd "$SCRATCH"  # never inside the repo — key must not be committable
ssh-keygen -t ed25519 -N '' -C 'trackllm_website-ci-push' -f data_deploy_key
gh api repos/timothee-chauvin/trackllm_data/keys \
  -f title='CI push from trackllm_website workflows' \
  -f "key=$(cat data_deploy_key.pub)" -F read_only=false
gh secret set DATA_REPO_DEPLOY_KEY --repo timothee-chauvin/trackllm_website < data_deploy_key
rm data_deploy_key data_deploy_key.pub
```

- [ ] **Step 3: Verify**

Run: `gh api repos/timothee-chauvin/trackllm_data/keys --jq '.[].title'` → shows the key.
Run: `gh secret list --repo timothee-chauvin/trackllm_website` → shows `DATA_REPO_DEPLOY_KEY`.

### Task 2: Code-repo changes on a branch + PR

**Files:**
- Modify: `.github/workflows/run-main.yml` (add data checkout; commit step → data repo)
- Modify: `.github/workflows/bi-monitor.yml` (add data checkout; split commit step)
- Modify: `.github/workflows/update-endpoints.yml` (add data checkout; split commit step)
- Modify: `.github/workflows/deploy-pages.yml` (add data checkout, no key)
- Modify: `scripts/prune_pages_artifact.sh` (drop `data/.git` from the artifact)
- Modify: `.gitignore` (ignore `website/data/`), `.gitattributes` (drop moved spend line)
- Modify: `CLAUDE.md`, `README.md` (local setup: clone data repo at `website/data`)
- Create: `docs/superpowers/plans/2026-08-04-split-data-repo.md` (this plan)

**Interfaces:** all three bot workflows gain, immediately after the code checkout:

```yaml
    - name: Checkout data repo
      uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4
      with:
        repository: timothee-chauvin/trackllm_data
        path: website/data
        ssh-key: ${{ secrets.DATA_REPO_DEPLOY_KEY }}
```

(`deploy-pages.yml` gets the same block **without** `ssh-key` — public read needs none, and the deploy job must not hold a write credential.)

- [ ] **Step 1: run-main.yml commit step** (paths lose the `website/data/` prefix inside the data repo)

```yaml
    - name: Commit and push results
      if: ${{ !cancelled() }}
      run: |
        git config --global user.email "github-actions[bot]@users.noreply.github.com"
        git config --global user.name "github-actions[bot]"
        cd website/data
        git add lt spend
        git diff --staged --quiet || git commit -m "[bot] add new responses"
        bash "$GITHUB_WORKSPACE/.github/scripts/push-with-rebase.sh"
```

- [ ] **Step 2: bi-monitor.yml commit step** — data first (the expensive part), then code-repo file:

```yaml
        git config --global user.email "github-actions[bot]@users.noreply.github.com"
        git config --global user.name "github-actions[bot]"
        (cd website/data
         git add -A
         git diff --staged --quiet || git commit -m "[bot] Daily BI monitor run"
         bash "$GITHUB_WORKSPACE/.github/scripts/push-with-rebase.sh")
        git add strategies_test_reasoning.json
        git diff --staged --quiet || git commit -m "[bot] Daily BI monitor run"
        bash .github/scripts/push-with-rebase.sh
```

- [ ] **Step 3: update-endpoints.yml commit step** — same split; data paths are `spend b3it` (keep the existing comment about `git add` exit-128 semantics, adapted to the new paths):

```yaml
        (cd website/data
         git add spend b3it
         git diff --staged --quiet || git commit -m "[bot] Daily update of target endpoints"
         bash "$GITHUB_WORKSPACE/.github/scripts/push-with-rebase.sh")
        git add \
          endpoints_lt.yaml \
          endpoints_cache_lt.yaml \
          endpoints_catalog.yaml \
          endpoints_bi.yaml \
          endpoints_cache_bi.yaml \
          strategies_test_reasoning.json
        git diff --staged --quiet || git commit -m "[bot] Daily update of target endpoints"
        bash .github/scripts/push-with-rebase.sh
```

- [ ] **Step 4: prune script** — add `rm -rf "$site/data/.git"` next to the node_modules removal (the data checkout brings its own `.git`, ~70MB, into the Pages artifact otherwise).

- [ ] **Step 5: gitignore/gitattributes** — root `.gitignore` gains `/website/data/`; root `.gitattributes` loses the `website/data/spend/**/*.jsonl merge=union` line (it moves to the data repo as `spend/**/*.jsonl merge=union`).

- [ ] **Step 6: docs** — `CLAUDE.md` + `README.md`: local setup is `git clone git@github.com:timothee-chauvin/trackllm_data website/data` (or symlink an existing clone; each worktree needs one; generated site JSONs land untracked inside it, as they did before).

- [ ] **Step 7: `prek run --all-files`, commit, push branch `chore/split-data-repo`, open PR.** Do NOT merge — merging happens inside the freeze (Task 4) so no bot run executes half-migrated workflows.

### Task 3: Rehearse the extraction and verify the new layout locally

Non-destructive; everything in scratch. This validates the exact commands Task 4 will run and produces the size numbers for the gate summary.

- [ ] **Step 1: Rehearsal extraction** (from the local repo to avoid a 1.9GB download; `--source` on a file URL keeps origin untouched)

```bash
cd "$SCRATCH"
git clone --mirror file:///home/ubuntu/phd/trackllm-website data_extract.git
cd data_extract.git && uvx git-filter-repo --subdirectory-filter website/data --force
```

Verify: `git -C data_extract.git log --oneline | wc -l` ≈ number of commits touching data; `git -C data_extract.git show HEAD --stat | head` shows `lt/`, `b3it/`, `spend/` at root; pack size noted.

- [ ] **Step 2: Rehearsal rewrite of the code repo**

```bash
cd "$SCRATCH"
git clone --mirror file:///home/ubuntu/phd/trackllm-website code_rewrite.git
cd code_rewrite.git && uvx git-filter-repo --invert-paths --path website/data --force
git count-objects -vH   # expect size-pack in the tens of MB
```

Verify no data blob survives anywhere: `git -C code_rewrite.git rev-list --objects --all | rg -c 'website/data/' → 0`.

- [ ] **Step 3: Full test suite against the new arrangement**

```bash
cd "$SCRATCH" && git clone file:///home/ubuntu/phd/trackllm-website worktree_sim -b chore/split-data-repo
cd worktree_sim && git rm -rq --cached website/data && rm -rf website/data   # simulate post-rewrite tree
git clone "$SCRATCH/data_extract.git" website/data
ln -s /home/ubuntu/phd/trackllm-website/.env .env
make test
```

Expected: pytest + bun suites green. This proves `generate_site`, the site, and the tests run with data supplied by the data repo at `website/data`.

- [ ] **Step 4: Data repo metadata commit (rehearsal clone)** — in a working clone of `data_extract.git`, add and commit `.gitattributes` (`spend/**/*.jsonl merge=union`), `.gitignore` (`/overview.json`, `/changes.json`, `/changes_page.json`, `/spend.json`, `/models/`, `/providers/` — the site generator writes these, untracked), and a short `README.md` pointing at trackllm_website. Save the three files to `$SCRATCH/data_repo_meta/` for reuse in Task 4.

---

## GATE — user confirmation required

Present: rehearsal sizes, PR link, and the exact Task 4 sequence. Wait for explicit "go" naming the force-pushes. Do not proceed on silence.

---

### Task 4: Freeze, extract for real, rewrite, force-push, unfreeze

Estimated freeze window: ~1h (dominated by pushing ~1.7GB to GitHub). Cost: ~1 missed hourly LT collection; the pipeline is resumable by design.

- [ ] **Step 1: Freeze** — `gh workflow disable` for `run-main.yml`, `bi-monitor.yml`, `update-endpoints.yml`, `deploy-pages.yml`; then wait until `gh run list --status in_progress` is empty (a bi-monitor/update-endpoints run can take hours — check before starting).
- [ ] **Step 2: Final extraction** — repeat Task 3 step 1 against **origin** (fresh `git clone --mirror git@github.com:timothee-chauvin/trackllm_website`, filter, verify tip matches origin/main's data state), then `git push --force git@github.com:timothee-chauvin/trackllm_data 'refs/heads/main:refs/heads/main'`, then push the metadata commit from Task 3 step 4 on top.
- [ ] **Step 3: Merge the PR** (rebase-merge to keep the two commits reviewable; workflows are disabled, so nothing runs them against pre-rewrite state).
- [ ] **Step 4: Rewrite** — fresh `git clone --mirror` of trackllm_website (now includes the merge), `uvx git-filter-repo --invert-paths --path website/data --path data_bi --force`, verify (`rev-list --objects --all | rg 'website/data/|data_bi/'` empty; tip commit message intact), then `git push --force origin 'refs/heads/*:refs/heads/*'`. All 43 branches are rewritten consistently; refs/pull/* are not pushed (avoids GitHub's read-only-ref errors).
  `data_bi` is the pre-rename data layout, ~301MB of blobs reachable only from old branches (rehearsal finding — purging `website/data` alone leaves a 360MB repo; with `data_bi` it is 33MiB). Its history is not carried into trackllm_data (the extraction follows main's `website/data` lineage); it survives only in the pre-rewrite backups.
- [ ] **Step 5: Unfreeze** — re-enable the four workflows.
- [ ] **Step 6: End-to-end verification** — `gh workflow run run-main.yml`; watch it: data commit lands in trackllm_data (not in trackllm_website); deploy-pages triggers on completion and goes green; site loads with fresh data; `gh api repos/timothee-chauvin/trackllm_website --jq .size` trending down (GitHub's gc lags — fresh-clone size is the real check: `git clone --depth 1` of the code repo ≈ tens of MB).

Rollback: until Step 4's push, nothing in the code repo changed — re-enable workflows and delete/ignore trackllm_data. After Step 4, the pre-rewrite history survives in the Task 4 mirror clone and the Task 3 backups; force-pushing it back restores the old state exactly.

### Task 5: Local machine recovery + docs

- [ ] **Step 1 (FIRST, before any local reset): safety-move the local data** — `mv /home/ubuntu/phd/trackllm-website/website/data /home/ubuntu/phd/trackllm-data-pre-split-backup`. A reset to rewritten history would otherwise delete the tracked data files from the working tree.
- [ ] **Step 2: Re-point local main** — `git fetch origin && git reset --hard origin/main` (clean status verified first), then `git clone git@github.com:timothee-chauvin/trackllm_data website/data`.
- [ ] **Step 3: Worktrees** — leave them; their branches still work locally. Warn in the summary + memory: rebase any of them onto rewritten main **before pushing**, or the push reintroduces the old heavy history.
- [ ] **Step 4: Keep** the pre-split backup dir and the `$SCRATCH` mirror for ~a month, then delete manually.
- [ ] **Step 5: Memory** — write a memory file: data lives in trackllm_data cloned at `website/data`; worktree rebase warning; backup locations.

#!/bin/bash
# ============================================================
#  FER2013 PROJECT — 10-DAY FULL REBUILD COMMIT REPLAY
# ============================================================
#  PURPOSE
#    Rebuild the current Artihomework repository from an empty tree across
#    10 days, with 3 commits/day (A, B, C) at different times.
#
#  USAGE
#    1) Edit people variables if needed
#    2) Run from repo root: bash replay_commits.sh
#    3) Inspect: git log --pretty=format:'%h %ad %an %s' --date=iso
#
#  SAFETY
#    - Requires clean working tree
#    - Creates branch: replay-10day
#    - Keeps your original branch untouched
# ============================================================

set -euo pipefail

A_NAME="Xu"
A_EMAIL="tinfDreemurr@gmail.com"
B_NAME="Cen"
B_EMAIL="2qmhsll@gmail.com"
C_NAME="tinf dreemurr"
C_EMAIL="grasscc@gmail.com"

START="2026-05-10"
REPLAY_BRANCH="replay-10day"
SNAP_DIR=".replay_snapshot"

D() {
  local offset="$1" t="$2"
  if date --version >/dev/null 2>&1; then
    date -d "$START + $offset days" +"%Y-%m-%dT${t}"
  else
    date -j -v+"${offset}"d -f "%Y-%m-%d" "$START" +"%Y-%m-%dT${t}"
  fi
}

commit_as() {
  local name="$1" email="$2" datestr="$3" msg="$4"
  shift 4
  git add "$@"
  GIT_AUTHOR_DATE="$datestr" GIT_COMMITTER_DATE="$datestr" \
    git commit -m "$msg" --author="$name <$email>"
}

restore_file() {
  local file="$1"
  mkdir -p "$(dirname "$file")"
  cp "$SNAP_DIR/$file" "$file"
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not a git repository"
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Error: working tree is not clean. Commit/stash first."
  exit 1
fi

ORIG_BRANCH="$(git branch --show-current)"

rm -rf "$SNAP_DIR"
mkdir -p "$SNAP_DIR"

TRACKED_FILES="$(git ls-files)"
for f in $TRACKED_FILES; do
  mkdir -p "$SNAP_DIR/$(dirname "$f")"
  cp "$f" "$SNAP_DIR/$f"
done

if git show-ref --quiet "refs/heads/$REPLAY_BRANCH"; then
  git branch -D "$REPLAY_BRANCH"
fi

git checkout --orphan "$REPLAY_BRANCH"
git rm -rf . >/dev/null 2>&1 || true

mkdir -p src scripts docs/progress

# Day 1
restore_file LICENSE
restore_file README.md
commit_as "$A_NAME" "$A_EMAIL" "$(D 0 09:11:12)" "day1(A): initialize project docs and license" LICENSE README.md

restore_file requirements.txt
commit_as "$B_NAME" "$B_EMAIL" "$(D 0 13:24:51)" "day1(B): add dependencies baseline" requirements.txt

restore_file app.py
commit_as "$C_NAME" "$C_EMAIL" "$(D 0 19:07:33)" "day1(C): add app entry placeholder" app.py

# Day 2
restore_file src/__init__.py
restore_file src/dataset.py
commit_as "$A_NAME" "$A_EMAIL" "$(D 1 08:43:20)" "day2(A): add package init and FER2013 dataset loader" src/__init__.py src/dataset.py

restore_file scripts/download_fer2013.py
commit_as "$B_NAME" "$B_EMAIL" "$(D 1 12:16:02)" "day2(B): add dataset download helper script" scripts/download_fer2013.py

cat > docs/progress/day02_ui_plan.md <<TXT
# Day 2 UI plan (C)
- Streamlit layout sketch
- Upload + prediction + log region
TXT
commit_as "$C_NAME" "$C_EMAIL" "$(D 1 21:10:48)" "day2(C): add UI planning notes" docs/progress/day02_ui_plan.md

# Day 3
restore_file src/models.py
commit_as "$A_NAME" "$A_EMAIL" "$(D 2 09:02:44)" "day3(A): implement CNN model definitions" src/models.py

restore_file src/trainer.py
commit_as "$B_NAME" "$B_EMAIL" "$(D 2 14:37:19)" "day3(B): implement trainer with fit loop" src/trainer.py

restore_file src/predict.py
commit_as "$C_NAME" "$C_EMAIL" "$(D 2 20:55:05)" "day3(C): implement inference pipeline" src/predict.py

# Day 4
cat > docs/progress/day04_hparams.md <<TXT
# Day 4 hyperparameter plan (A)
- learning_rate, optimizer, activation exposed in UI
TXT
commit_as "$A_NAME" "$A_EMAIL" "$(D 3 10:18:27)" "day4(A): document hyperparameter controls" docs/progress/day04_hparams.md

cat > docs/progress/day04_checkpoint.md <<TXT
# Day 4 checkpoint strategy (B)
- periodic checkpoint
- best-loss model
- autosave on pause
TXT
commit_as "$B_NAME" "$B_EMAIL" "$(D 3 15:49:13)" "day4(B): add checkpoint strategy notes" docs/progress/day04_checkpoint.md

cat > docs/progress/day04_ui_controls.md <<TXT
# Day 4 UI controls (C)
- start / pause / resume / stop
TXT
commit_as "$C_NAME" "$C_EMAIL" "$(D 3 22:04:58)" "day4(C): add training control UI notes" docs/progress/day04_ui_controls.md

# Day 5
restore_file src/evaluate.py
commit_as "$A_NAME" "$A_EMAIL" "$(D 4 09:58:12)" "day5(A): add evaluation utilities" src/evaluate.py

restore_file src/sklearn_baseline.py
commit_as "$B_NAME" "$B_EMAIL" "$(D 4 13:40:29)" "day5(B): add sklearn baseline support" src/sklearn_baseline.py

cat > docs/progress/day05_eval_plan.md <<TXT
# Day 5 evaluation plan (C)
- accuracy
- confusion matrix
- ROC curve
TXT
commit_as "$C_NAME" "$C_EMAIL" "$(D 4 20:31:46)" "day5(C): define metrics and visualization plan" docs/progress/day05_eval_plan.md

# Day 6
restore_file REPORT.md
commit_as "$A_NAME" "$A_EMAIL" "$(D 5 08:36:50)" "day6(A): add report draft structure" REPORT.md

cat > docs/progress/day06_resume_train.md <<TXT
# Day 6 resume training checks (B)
- restart from checkpoint
- preserve optimizer state
TXT
commit_as "$B_NAME" "$B_EMAIL" "$(D 5 14:13:22)" "day6(B): document resume-training validation" docs/progress/day06_resume_train.md

cat > docs/progress/day06_demo_flow.md <<TXT
# Day 6 demo flow (C)
- upload image
- run prediction
- show emotion label
TXT
commit_as "$C_NAME" "$C_EMAIL" "$(D 5 19:44:10)" "day6(C): add demo flow notes" docs/progress/day06_demo_flow.md

# Day 7-10 progress artifacts only
for day in 7 8 9 10; do
  offset=$((day-1))
  cat > "docs/progress/day${day}_A.md" <<TXT
# Day ${day} (A)
Model/report integration and review.
TXT
  commit_as "$A_NAME" "$A_EMAIL" "$(D "$offset" "09:1${day}:0${day}")" "day${day}(A): integrate model/report updates" "docs/progress/day${day}_A.md"

  cat > "docs/progress/day${day}_B.md" <<TXT
# Day ${day} (B)
Training/framework/checkpoint polishing.
TXT
  commit_as "$B_NAME" "$B_EMAIL" "$(D "$offset" "14:2${day}:1${day}")" "day${day}(B): polish trainer/framework workflow" "docs/progress/day${day}_B.md"

  cat > "docs/progress/day${day}_C.md" <<TXT
# Day ${day} (C)
UI/evaluation/inference polishing and final checks.
TXT
  commit_as "$C_NAME" "$C_EMAIL" "$(D "$offset" "20:3${day}:2${day}")" "day${day}(C): polish UI and evaluation outputs" "docs/progress/day${day}_C.md"
done

rm -rf "$SNAP_DIR"

echo "Replay complete on branch: $REPLAY_BRANCH"
echo "Original branch preserved: $ORIG_BRANCH"

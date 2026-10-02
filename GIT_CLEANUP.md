# Git cleanup and history purge instructions

This document explains how to remove committed derived data and caches from the repository and from its history. These steps rewrite history; coordinate with your collaborators and ensure you have backups.

Summary of issues found
- Large derived CSVs committed under data-processed/ (e.g. best_training_data.csv, ctgan_500_records.csv).
- Archive directory contains split artifacts (archive/R01/...).
- Compiled Python cache present in __pycache__.

What this commit does
- Adds patterns to .gitignore to prevent re-committing derived data and the archive.
- Provides instructions below to remove the existing committed files and purge them from history.

Quick local cleanup (keeps history — recommended first step)

1. Pull the latest main and make sure your working tree is clean:

   git checkout main
   git pull origin main

2. Remove files from the index while keeping them locally:

   # Remove specific files or whole directories that should not be tracked
   git rm -r --cached __pycache__ || true
   git rm --cached data-processed/*.csv || true
   git rm -r --cached archive || true

3. Commit the removals (this does not rewrite history):

   git add .gitignore
   git commit -m "git: stop tracking derived data; remove cached files"
   git push origin main

This removes the files from the current branch going forward but leaves them in the repository history. To purge them from history entirely, follow one of the methods below.

A. Purge using git-filter-repo (recommended)

1. Install git-filter-repo:

   pip install git-filter-repo

2. Create a bare mirror clone:

   git clone --mirror git@github.com:KT-Devv/student-dropout-prediction-ghana.git
   cd student-dropout-prediction-ghana.git

3. Run git-filter-repo to remove paths (example):

   git filter-repo --invert-paths --paths archive --paths data-processed/best_training_data.csv --paths data-processed/ctgan_500_records.csv --paths data-processed/ctgan_dropout_500.csv --paths data-processed/X_test_focal.csv --paths data-processed/y_test_focal.csv

   # You can add any other files or globs you want removed.

4. Force-push the cleaned repository back to GitHub:

   git push --force --all
   git push --force --tags

B. Purge using BFG Repo-Cleaner (alternative)

1. Download BFG and create a mirror clone:

   git clone --mirror https://github.com/KT-Devv/student-dropout-prediction-ghana.git
   java -jar bfg.jar --delete-files "*.csv" student-dropout-prediction-ghana.git

2. Follow BFG's recommendations:

   cd student-dropout-prediction-ghana.git
   git reflog expire --expire=now --all && git gc --prune=now --aggressive
   git push --force

C. Final steps after any history rewrite

- Inform collaborators: history was rewritten — they must re-clone or follow BFG/git-filter-repo recovery instructions.
- Re-run any CI that depends on removed files.
- Optionally re-upload small allowed artifacts (e.g. small CSVs required for tests) to a protected location or GitHub Releases.

Notes and cautions
- Rewriting history is destructive for any branches that contain the removed files. Only proceed if you understand the impact.
- If you prefer not to rewrite history, skipping the filter step and keeping the files in history but untracked going forward is acceptable for many projects.

If you want, I can:
- Open a PR that updates .gitignore and adds this cleanup doc.
- Prepare a filter-repo command list tailored to every exact file currently tracked (I can enumerate them first).


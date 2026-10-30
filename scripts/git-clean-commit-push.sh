#!/usr/bin/env bash
set -euo pipefail

COMMIT_MSG="${1:-Fix pipeline regex import issue}"

echo "Current branch:"
BRANCH="$(git branch --show-current)"
echo "$BRANCH"

if [ -z "$BRANCH" ]; then
  echo "ERROR: Detached HEAD. Checkout a real branch first."
  exit 1
fi

echo "Cleaning Python/cache artifacts..."
find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.log" \) -delete

echo "Checking status..."
git status --short

echo "Staging changes..."
git add -A

if git diff --cached --quiet; then
  echo "No staged changes to commit."
  exit 0
fi

echo "Committing..."
git commit -m "$COMMIT_MSG"

echo "Pushing to origin/$BRANCH..."
git push origin "$BRANCH"

echo "Done."

# execute below scripts if you want to run them manually
# chmod +x scripts/git-clean-commit-push.sh
# ./scripts/git-clean-commit-push.sh "Fix pipeline regex import issue"

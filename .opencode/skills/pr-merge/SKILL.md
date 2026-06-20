---
name: pr-merge
description: Use when preparing to merge a pull request — checking CI, review status, and resolving threads.
---

# PR Merge

Use this skill to verify a PR is ready to merge.

## Checklist

1. All CI checks passing (lint, type check, test).
2. All review threads resolved.
3. Branch is up to date with main.
4. No merge conflicts.
5. Commit history is clean (squash merge preferred).

## Rules

- Never merge with failing CI.
- Never merge with open review threads.
- If unsure, ask the user.

---
name: dev-workflow
description: Use when making a non-trivial code, test, configuration, or documentation change that should be planned, implemented, and verified end-to-end — from prompt to PR.
---

# Dev Workflow

End-to-end workflow from planning through PR creation. Uses git worktrees so the local repo checkout is never touched.

## Prerequisites

- `gh` authenticated (`gh auth status`)
- Default branch: `main`

## Variable Setup

At the start of work, capture these paths:

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
WORKTREE_ROOT="$REPO_ROOT/../.worktrees"
```

These variables are used in every step below. (The skill uses `$BRANCH`, `$REPO_ROOT`, and `$WORKTREE_ROOT` — substitute your values.)

## Workflow

### 1. Scope & Plan

1. Inspect relevant files to understand context, conventions, and patterns.
2. Propose a concrete plan to the user:
   - What files will change
   - Branch name (`feature/short-description`)
   - Whether tests are needed
   - Whether delegation to `@python-engineer` is appropriate
3. **Wait for explicit user approval** before proceeding.

### 2. Branch & Worktree

Set the branch name (replace with your branch):

```bash
BRANCH="feature/short-description"
WORKTREE_PATH="$WORKTREE_ROOT/$BRANCH"
```

Check if work already exists:

```bash
git fetch origin "$BRANCH" 2>/dev/null || true
EXISTING=$(gh pr list --head "$BRANCH" --state open --json headRefName 2>/dev/null)
```

**Resume existing work** (branch or PR exists):

```bash
git fetch origin "$BRANCH":"$BRANCH" 2>/dev/null || true
git worktree add "$WORKTREE_PATH" "$BRANCH"
```

**New work** (neither branch nor PR exists):

```bash
git worktree add -b "$BRANCH" "$WORKTREE_PATH" main
```

> **Rule:** All file edits and commands run inside `$WORKTREE_PATH`.
> Use `workdir: $WORKTREE_PATH` for bash commands and absolute paths
> (`$WORKTREE_PATH/path/to/file`) for read/edit/write tools.
> **Never touch the main repo checkout.**

### 3. Implement

#### Python code changes → delegate to `@python-engineer`

Use the `task` tool with `subagent_type: python-engineer`.
Include in the prompt:
- The full worktree path (`$WORKTREE_PATH`)
- The specific files to change (use absolute paths under `$WORKTREE_PATH`)
- The behavior to implement and what tests to write
- To run `uv run ruff format --check .`, `uv run ruff check --fix`, `uv run ty check --exclude tests/`, and `uv run pytest` on completion
- To delegate back to you when done with a summary

#### Config / docs / Markdown → implement directly

- `read`/`edit`/`write` with absolute paths under `$WORKTREE_PATH`
- `bash` with `workdir: $WORKTREE_PATH`
- Write or update tests first (TDD), then implement the change

### 4. Verify

Run inside the worktree (`workdir: $WORKTREE_PATH`):

```bash
uv run ruff format --check .
uv run ruff check --fix
uv run ty check --exclude tests/
uv run pytest
```

If any check fails, fix it and re-verify. Do not proceed until all pass.

### 5. Commit & Push

Inside the worktree:

```bash
git add -A
git commit -m "<scope>: <brief description>"
git push origin "$BRANCH"
```

### 6. Create or Update PR

```bash
EXISTING_PR=$(gh pr list --head "$BRANCH" --state open --json number,url 2>/dev/null)
```

**If PR exists:** Note the URL, nothing more needed.

**If no PR exists:**

```bash
gh pr create --fill
```

Capture the PR URL from the output for the summary.

### 7. Cleanup

```bash
git worktree remove "$WORKTREE_PATH"
```

(If removal fails due to unstaged changes, commit or resolve first.)

The `$WORKTREE_ROOT/.gitignore` should prevent stale files from lingering.

### 8. Summarize

Report to the user:
- What changed and why
- Verification results (all checks passed)
- PR URL

## Standard Checks

```bash
uv run ruff format --check .
uv run ruff check --fix
uv run ty check --exclude tests/
uv run pytest
```

## Rules

- **Always use a worktree.** Never edit files in the main repo checkout.
- **Always create / update a PR.** Never leave work without an open PR.
- **Always verify before PR.** Run all checks — if any fail, fix them.
- Use `uv` for all Python commands.
- Keep commits focused (one logical change per commit).
- Do not add dependencies without a clear reason.
- If evidence contradicts the plan, stop and revise.
- Default branch is `main`.

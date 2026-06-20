---
name: plan-change
description: Use when planning a non-trivial change — breaking down scope, identifying risks, and defining verification steps before implementation.
---

# Plan Change

Use this skill to produce a concrete, reviewable plan before writing code.

## Steps

1. State the goal in one sentence.
2. List the files that will change and why.
3. Identify risks and edge cases.
4. Define verification steps (what tests, what checks).
5. Estimate scope (small / medium / large).
6. Get user approval before proceeding.

## Rules

- Plans must be specific enough to verify.
- If the plan is large, propose breaking it into smaller PRs.
- Do not implement while planning.

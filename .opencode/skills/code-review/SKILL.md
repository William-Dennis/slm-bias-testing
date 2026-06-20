---
name: code-review
description: Use when reviewing a change, pull request, diff, or patch for bugs, regressions, missing tests, and maintainability risks.
---

# Code Review

Use this skill to review changes with a correctness-first mindset.

## Review Order

1. Identify behavior changes and affected call paths.
2. Look for bugs, regressions, unsafe edge cases, and missing validation.
3. Check whether tests cover the changed behavior.
4. Check dependency, packaging, and CI impact.
5. Report findings before summaries.

## Output

- Findings, ordered by severity, with file and line references.
- Open questions or assumptions.
- Brief summary only after findings.

If no findings are found, say so and mention residual testing gaps.

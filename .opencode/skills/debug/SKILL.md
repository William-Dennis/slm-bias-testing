---
name: debug
description: Use when diagnosing an error, failing test, crash, unexpected output, or bug report.
---

# Debug

Use this skill to move from symptom to verified fix.

## Workflow

1. Capture the exact symptom and command that fails.
2. Reproduce the failure if feasible.
3. Read the smallest relevant code path.
4. Form one concrete hypothesis at a time.
5. Make the smallest fix that addresses the root cause.
6. Re-run the failing command and relevant regression checks.

## Rules

- Do not guess past available evidence.
- Prefer fixing root causes over masking symptoms.
- If reproduction is not possible, state what evidence is missing.

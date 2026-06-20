---
name: doc-alignment
description: Use when checking if code, tests, and documentation are consistent with each other.
---

# Doc Alignment

Use this skill to verify that docs match reality.

## Check Order

1. Read the claimed behavior in AGENTS.md.
2. Read the actual code implementation.
3. Check if tests cover the documented behavior.
4. Report mismatches with file and line references.

## Rules

- Code is truth. Docs are claims.
- If docs say X and code does Y, flag it.
- Propose the fix (update docs or fix code) based on which is correct.

---
name: skill-audit
description: Audit SKILL.md files for compliance with the OpenCode skills spec.
---

Audit all SKILL.md files in `.opencode/skills/` against the spec at https://opencode.ai/docs/skills/.

For each skill, check:
- Frontmatter has `name` (required) and `description` (required)
- Name: 1-64 chars, lowercase alphanumeric, single hyphens, matches directory name
- Description: 1-1024 characters
- No unknown frontmatter fields (license, compatibility, metadata are OK)

Report violations as a list with file path and issue. If all pass, say so.

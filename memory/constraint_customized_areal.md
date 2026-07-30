---
name: Constraint - Use customized_areal for Custom Implementations
description: Do not modify core areal/ files; implement all customizations in customized_areal/
type: feedback
---

**Rule:** Do NOT modify files in the `areal/` directory. All custom implementations
(datasets, workflows, rewards, etc.) must be placed in `customized_areal/`.

**Why:** This maintains clean separation between core framework code (areal/) and user
customizations (customized_areal/), making upgrades and maintenance easier.

**How to apply:**

- Create new files under `customized_areal/` instead of modifying `areal/`
- Import and extend areal classes rather than modifying them
- Use custom config paths in your YAML configs to point to customized_areal modules

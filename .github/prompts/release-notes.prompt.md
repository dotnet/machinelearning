# ML.NET Release Notes

Generate classified release notes between two commits.

## Categories

1. **Product** — Bug fixes, features, improvements
2. **Dependencies** — Package/SDK updates
3. **Testing** — Test changes and infrastructure
4. **Documentation** — Docs, samples
5. **Housekeeping** — Build, CI, cleanup

## Process

```bash
# Get commits between two points
git log --pretty=format:"%h - %s (%an)" BRANCH1..BRANCH2 > commits.txt
```

Classify each commit. When uncertain, default to Housekeeping. Group related commits. Flag breaking changes with ⚠️.

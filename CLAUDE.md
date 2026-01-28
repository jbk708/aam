# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Quick Reference

```bash
# Install
pip install -e ".[dev,docs,training]"

# Test & Lint
pytest tests/ -v
ruff check aam/
uvx ty check aam/

# CLI
aam pretrain --help
aam train --help
aam predict --help
```

## Documentation

| Document | Content |
|----------|---------|
| `README.md` | Project overview, quickstart |
| `ARCHITECTURE.md` | Design decisions, model structure |
| `docs/getting-started.md` | Installation, first run |
| `docs/user-guide.md` | Full CLI reference |
| `docs/how-it-works.md` | Concepts + implementation |

## Development

See `.agents/TICKET_OVERVIEW.md` for outstanding work and `.agents/WORKFLOW.md` for development process.

### Commit Messages
- No Claude attribution (no "Co-Authored-By", no "Generated with Claude Code")
- Format: `{PREFIX}-{ticket}: Brief description`
- Include detailed body explaining changes

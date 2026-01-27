# Contributing to AAM

Thank you for your interest in contributing to AAM.

## Development Setup

```bash
# Clone and install with dev dependencies
git clone <repo-url>
cd aam
pip install -e ".[dev,docs,training]"
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=aam --cov-report=html

# Single file
pytest tests/test_trainer.py -v
```

## Code Quality

```bash
# Lint
ruff check aam/ tests/

# Format check
ruff format --check aam/ tests/

# Type check
uvx ty check aam/
```

## Branch Naming

```
{prefix}-{ticket}-{short-name}
```

Prefixes: `pyt`, `doc`, `cos`, `cat`, `cln`, `fus`, `reg`

Example: `doc-5-readme-refactor`

## Commit Messages

```
{PREFIX}-{ticket}: Brief description

Detailed explanation of changes.
```

Example:
```
DOC-5: Refactor README and create docs/ structure

- Create docs/ folder with getting-started, user-guide, how-it-works
- Reduce README to ~130 lines overview
- Add links to detailed documentation
```

## PR Process

1. Create branch from `main`
2. Make changes with tests
3. Run all checks (test, lint, type check)
4. Create PR with description
5. Address review feedback

## Ticket Workflow

See `.agents/WORKFLOW.md` for detailed ticket development process.

Outstanding tickets are tracked in `.agents/TICKET_OVERVIEW.md`.

## Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build HTML docs
cd docs
make html

# View docs (open in browser)
open _build/html/index.html

# Live reload during development
make livehtml
```

## Code Style

- Self-documenting code over comments
- Test-driven development
- Keep changes focused and minimal
- Follow existing patterns in the codebase

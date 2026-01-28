# Agent Workflow

## Dev Branches

Some ticket groups have dedicated dev branches. Check `TICKET_OVERVIEW.md` for the correct base branch.

| Tickets | Dev Branch | PR Target |
|---------|------------|-----------|
| TRN-* | `dev/train-bugfix` | `dev/train-bugfix` |
| Others | `main` | `main` |

## Ticket Development

1. **Branch**: Check dev branch in `TICKET_OVERVIEW.md`, then `git checkout -b {prefix}-{ticket}-{name}` (prefixes: `pyt`, `cos`, `cat`, `cln`, `fus`, `trn`, `reg`)
2. **Stub**: Create file structure, commit: `"{PREFIX}-{ticket}: Stub out {component}"`
3. **Test**: Write tests, commit: `"{PREFIX}-{ticket}: Add tests for {component}"`
4. **Implement**: Fill in stubs, commit: `"{PREFIX}-{ticket}: Implement {component}"`
5. **Verify**: `pytest tests/ -v`, `ruff check aam/ tests/`, `ruff format --check aam/ tests/`, and `uvx ty check aam/`
6. **Simplify**: Run code-simplifier agent on changes before PR
7. **Update**: Mark ticket complete in the relevant tickets file

## Testing Guidelines

**Before writing tests:**
1. Search for existing tests: `grep -r "def test_" tests/ | grep -i "{feature}"`
2. Check test file naming: `ls tests/test_{module}.py`
3. Read existing test patterns in the relevant test file

**Test-driven approach:**
- Write tests before implementation
- Check if tests already exist for similar functionality
- Reuse existing test fixtures and helpers from `tests/conftest.py`
- Avoid duplicating test coverage

## Principles
- Test-driven: tests before implementation
- Self-documenting: minimize inline comments
- See `docs/` for guides and `ARCHITECTURE.md` for design decisions

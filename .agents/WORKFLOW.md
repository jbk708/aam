# Agent Workflow

## Ticket Development

1. **Branch**: `git checkout -b {prefix}-{ticket}-{name}` (prefixes: `pyt`, `cos`, `cat`, `cln`, `fus`)
2. **Stub**: Create file structure, commit: `"{PREFIX}-{ticket}: Stub out {component}"`
3. **Test**: Write tests, commit: `"{PREFIX}-{ticket}: Add tests for {component}"`
4. **Implement**: Fill in stubs, commit: `"{PREFIX}-{ticket}: Implement {component}"`
5. **Verify**: `pytest tests/ -v`, `ruff check aam/ tests/`, `ruff format --check aam/ tests/`, and `uvx ty check aam/`
6. **Simplify**: Run code-simplifier agent on changes before PR
7. **Update**: Mark ticket complete in the relevant tickets file

## Principles
- Test-driven: tests before implementation
- Self-documenting: minimize inline comments
- See `docs/` for guides and `ARCHITECTURE.md` for design decisions

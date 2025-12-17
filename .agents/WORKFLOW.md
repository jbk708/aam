# Agent Workflow

## Ticket Development

1. **Branch**: `git checkout -b pyt-{ticket}-{name}`
2. **Stub**: Create file structure, commit: `"PYT-{ticket}: Stub out {component}"`
3. **Test**: Write tests, commit: `"PYT-{ticket}: Add tests for {component}"`
4. **Implement**: Fill in stubs, commit: `"PYT-{ticket}: Implement {component}"`
5. **Verify**: `pytest tests/ -v` and `ruff check aam/ tests/`
6. **Update**: Mark ticket complete in `PYTORCH_PORTING_TICKETS.md`

## Principles
- Test-driven: tests before implementation
- Self-documenting: minimize inline comments
- Follow `_design_plan/` documents as guides

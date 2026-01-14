# Documentation Tickets

**Last Updated:** 2026-01-13

---

## DOC-1: README & Installation Modernization
**Priority:** LOW | **Effort:** 2-4 hours | **Status:** COMPLETE

Comprehensive documentation pass to modernize installation instructions, ensure README accuracy, and improve developer experience.

### Scope

#### 1. Installation Modernization
- [ ] Review and update `environment.yml` dependencies
- [ ] Consider adding `pyproject.toml` extras for optional dependencies (e.g., `pip install aam[rocm]`)
- [ ] Document Python version requirements (currently 3.11+)
- [ ] Add installation verification commands
- [ ] Consider adding a `make install` or similar for quick setup

#### 2. README Accuracy Audit
- [ ] Verify all CLI examples work with current codebase
- [ ] Update test count (currently says "679 tests")
- [ ] Verify all flags mentioned exist and have correct defaults
- [ ] Check that code examples match current API
- [ ] Update any outdated screenshots or diagrams

#### 3. Quick Start Improvements
- [ ] Add minimal working example with included test data
- [ ] Add common troubleshooting section
- [ ] Document typical workflow from data to predictions
- [ ] Add example output screenshots/logs

#### 4. Developer Documentation
- [ ] Update CLAUDE.md with any new commands or patterns
- [ ] Ensure ARCHITECTURE.md is current
- [ ] Document any new environment variables or config options
- [ ] Add contributing guidelines if missing

#### 5. Platform-Specific Docs
- [ ] Consolidate ROCm setup instructions (currently spread across README and CLAUDE.md)
- [ ] Add Apple Silicon (MPS) notes if applicable
- [ ] Document CPU-only usage for development/testing

### Acceptance Criteria
- [x] Fresh clone + install works on Linux/macOS
- [x] All README examples execute without error
- [x] Test count is accurate (updated to 919)
- [x] ROCm setup has single source of truth
- [x] Quick start takes <5 minutes for new user

### Completed (2026-01-13)
- Replaced conda/mamba installation with pip-only workflow
- Added Python version requirements (3.9-3.12)
- Added PyTorch installation instructions for CUDA/CPU/ROCm
- Added Quick Start section with included test data (781 samples)
- Updated test count from 679 to 919
- Fixed stale environment.yml reference
- Updated CLAUDE.md with complete test data file list

### Files to Review
- `README.md` - Main documentation
- `CLAUDE.md` - Claude Code instructions
- `ARCHITECTURE.md` - System design
- `environment.yml` - Dependencies
- `pyproject.toml` - Package configuration
- `setup.py` (if exists) - Legacy setup

### Notes
- Consider using `mkdocs` or `sphinx` for more comprehensive docs in future
- May want to add badges (CI status, coverage, PyPI version) to README
- Could add a CHANGELOG.md for tracking releases

---

## Future Documentation Work (Backlog)

### DOC-2: API Reference Generation
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Backlog

Auto-generate API documentation from docstrings using sphinx or similar.

### DOC-3: Tutorial Notebooks
**Priority:** LOW | **Effort:** 8-12 hours | **Status:** Backlog

Create Jupyter notebooks demonstrating:
- Basic training workflow
- Custom dataset preparation
- Model interpretation/visualization
- Multi-GPU training setup

### DOC-4: Video Walkthrough
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Backlog

Create a short video tutorial covering installation and basic usage.

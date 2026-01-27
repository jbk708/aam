# Documentation Tickets

**Last Updated:** 2026-01-27
**Status:** COMPLETE - Documentation overhaul finished

---

## Overview

Restructure AAM documentation for both end users (researchers) and contributors/developers. Create a `docs/` folder with organized guides, update ARCHITECTURE.md with design rationale, and set up automated API reference generation.

**Target audience:** Researchers using AAM + developers extending it

---

## Phase 1: Foundation

### DOC-5: Create docs/ Folder Structure and README Refactor
**Priority:** HIGH | **Effort:** 2-3 hours | **Status:** COMPLETE (2026-01-27)

Create the documentation folder structure and refactor README.md to be an overview with links.

**Deliverables:**
- Create `docs/` folder with placeholder files
- Refactor README.md to ~100-150 lines:
  - Project description (what AAM does)
  - Installation (brief, link to full guide)
  - Quick example (5-line usage)
  - Links to all docs
  - Badge section (tests, coverage, etc.)
- Move detailed content to appropriate docs

**Folder structure:**
```
docs/
├── getting-started.md      # Installation + first run
├── user-guide.md           # Full CLI reference
├── how-it-works.md         # Concepts + implementation
├── api/                    # Auto-generated API docs (DOC-2)
└── images/                 # Diagrams, architecture visuals
```

**Acceptance criteria:**
- [ ] `docs/` folder created with structure above
- [ ] README.md reduced to overview (~100-150 lines)
- [ ] All links work (even if pointing to placeholder content)
- [ ] No information lost (moved, not deleted)

---

### DOC-6: Write Getting Started Guide
**Priority:** HIGH | **Effort:** 2-3 hours | **Status:** COMPLETE (2026-01-27)

Write `docs/getting-started.md` - the first document new users should read.

**Deliverables:**
- Prerequisites (Python, conda/mamba, PyTorch)
- Step-by-step installation
- Verify installation works
- Run first example with test data
- Expected output explanation
- Common installation issues / troubleshooting
- Next steps (link to user guide)

**Content to migrate from README:**
- Installation section
- Quick Start section
- PyTorch Installation section

**Acceptance criteria:**
- [ ] New user can go from zero to working example following only this doc
- [ ] Covers CPU, CUDA, and ROCm installation paths
- [ ] Includes troubleshooting for common issues
- [ ] Test data example runs successfully

---

### DOC-7: Write User Guide (CLI Reference)
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** COMPLETE (2026-01-27)

Write `docs/user-guide.md` - comprehensive CLI reference for all commands and options.

**Deliverables:**
- Command overview (pretrain, train, predict, rf-baseline)
- Full option reference tables (migrated from README)
- Workflow examples:
  - Pretraining workflow
  - Fine-tuning workflow
  - Classification vs regression
  - Multi-GPU training
  - Memory optimization
- Categorical metadata guide (currently in README)
- Target normalization guide
- Loss function selection guide
- Monitoring training (TensorBoard)

**Content to migrate from README:**
- All "Key Options" sections
- Categorical Metadata section
- Regression Options section
- Memory Optimization section
- Multi-GPU Training section
- Monitoring Training section
- Baseline Comparison section

**Acceptance criteria:**
- [ ] All CLI options documented with examples
- [ ] Decision trees for choosing options (loss type, normalization, etc.)
- [ ] Copy-paste ready command examples
- [ ] Cross-referenced with getting-started.md

---

### DOC-8: Write Contributing Guide
**Priority:** HIGH | **Effort:** 2 hours | **Status:** COMPLETE (2026-01-27)

Write `CONTRIBUTING.md` in repo root for developers who want to contribute.

**Deliverables:**
- Development setup (install with dev dependencies)
- Running tests (`pytest`, coverage)
- Linting and formatting (`ruff check`, `ruff format`)
- Type checking (`uvx ty check`)
- Branch naming convention (from WORKFLOW.md)
- Commit message format
- PR process
- Ticket workflow (reference .agents/)
- Code style guidelines
- Where to find design docs (_design_plan/)

**Content to consolidate from:**
- `.agents/WORKFLOW.md`
- `CLAUDE.md` (relevant parts)
- `_design_plan/README.md`

**Acceptance criteria:**
- [ ] New contributor can set up dev environment following guide
- [ ] All verification commands documented (test, lint, type check)
- [ ] Links to relevant .agents/ and _design_plan/ docs
- [ ] PR checklist included

---

## Phase 2: Deep Dives

### DOC-9: Update ARCHITECTURE.md with Design Rationale
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** COMPLETE (2026-01-27)

Expand ARCHITECTURE.md to explain not just *what* but *why* - design decisions and trade-offs.

**Current state:** 209 lines, good diagram, brief component descriptions

**Deliverables:**
- Keep and improve existing diagram
- Add "Design Decisions" section:
  - Why composition over inheritance (SequencePredictor composes SequenceEncoder)
  - Why hierarchical processing (nucleotide → ASV → sample)
  - Why pre-computed UniFrac (not computed on-the-fly)
  - Why self-supervised pretraining helps
  - Why batch size must be even
- Add "Trade-offs" section:
  - Memory vs accuracy (token_limit)
  - Freeze-base vs fine-tune jointly
  - Different fusion strategies
- Expand component descriptions with:
  - Input/output shapes
  - Key implementation details
  - Links to source files
- Add "Extension Points" section:
  - How to add new fusion strategies
  - How to add new loss functions
  - How to add new prediction heads

**Acceptance criteria:**
- [ ] "Design Decisions" section with rationale for 5+ major decisions
- [ ] "Trade-offs" section documenting key choices
- [ ] All components have input/output shape documentation
- [ ] Links to source code files (file:line format)

---

### DOC-10: Write How It Works - Part 1: Concepts
**Priority:** HIGH | **Effort:** 4-5 hours | **Status:** COMPLETE (2026-01-27)

Write the "Concepts" section of `docs/how-it-works.md` explaining the science/ML behind AAM.

**Target audience:** Researchers who want to understand what AAM is doing, not just how to run it.

**Deliverables:**
- **Microbiome Data Primer**
  - What is a BIOM table?
  - What are ASVs (Amplicon Sequence Variants)?
  - What is UniFrac distance? Why is it useful?
  - What is phylogenetic information?

- **Transformer Architecture Basics**
  - Attention mechanism intuition
  - Why transformers for sequence data?
  - Position embeddings explained

- **Hierarchical Processing**
  - Why process at nucleotide, ASV, and sample levels?
  - What does each level capture?
  - Analogy: words → sentences → documents

- **Self-Supervised Learning**
  - What is self-supervised learning?
  - Masked language modeling (nucleotide prediction)
  - Contrastive learning with UniFrac
  - Why pretraining improves downstream tasks

- **Multi-Task Learning**
  - Why predict multiple things simultaneously?
  - How auxiliary tasks help the main task
  - Loss weighting strategies

- **Categorical Conditioning**
  - Why condition on metadata?
  - Fusion strategies intuition (concat, GMU, cross-attention)

**Acceptance criteria:**
- [ ] Readable by someone with basic ML knowledge but no microbiome background
- [ ] Includes diagrams/figures for key concepts
- [ ] References to papers/resources for deeper reading
- [ ] No code - purely conceptual

---

### DOC-11: Write How It Works - Part 2: Implementation
**Priority:** HIGH | **Effort:** 4-5 hours | **Status:** COMPLETE (2026-01-27)

Write the "Implementation" section of `docs/how-it-works.md` explaining the code architecture.

**Target audience:** Developers who want to understand and extend the codebase.

**Deliverables:**
- **Data Flow Walkthrough**
  - BIOM file → BIOMLoader → rarefied table
  - Sequences → SequenceTokenizer → token tensors
  - UniFrac file → UniFracLoader → distance matrix
  - All together → ASVDataset → DataLoader batches

- **Model Forward Pass Walkthrough**
  - Input shapes at each stage
  - ASVEncoder: tokens → ASV embeddings
  - SampleSequenceEncoder: ASV embeddings → sample embeddings
  - SequenceEncoder: sample embeddings → base predictions
  - SequencePredictor: adds target/count heads

- **Training Loop Walkthrough**
  - Trainer initialization
  - Epoch structure
  - Loss computation (MultiTaskLoss)
  - Gradient accumulation
  - Checkpointing strategy
  - Early stopping logic

- **Key Code Patterns**
  - Composition pattern (SequencePredictor)
  - Gradient checkpointing usage
  - Attention masking for padding
  - NaN prevention in attention

- **Extension Guide**
  - Adding a new model component
  - Adding a new CLI option
  - Adding a new loss function
  - Adding a new fusion strategy

**Acceptance criteria:**
- [ ] Code snippets with file:line references
- [ ] Dimension annotations at each stage (B, S, L, D)
- [ ] Diagrams showing data flow
- [ ] "Extension Guide" enables adding new components

---

## Phase 3: API & Polish

### DOC-2: Set Up Sphinx API Reference Generation
**Priority:** HIGH | **Effort:** 4-6 hours | **Status:** COMPLETE (2026-01-27)

Set up automated API documentation generation using Sphinx (already in pyproject.toml).

**Deliverables:**
- `docs/conf.py` - Sphinx configuration
- `docs/index.rst` - Main index
- `docs/api/` - Auto-generated API reference
- `docs/Makefile` - Build commands
- GitHub Actions workflow for building docs (optional)
- Instructions in CONTRIBUTING.md for building docs locally

**Configuration:**
- Use `sphinx.ext.autodoc` for docstring extraction
- Use `sphinx.ext.napoleon` for Google-style docstrings
- Use `myst_parser` for Markdown support (already in deps)
- Use `sphinx.ext.viewcode` for source links

**Modules to document:**
- `aam.data` - Data loading and processing
- `aam.models` - Model architectures
- `aam.training` - Training utilities
- `aam.cli` - CLI commands

**Acceptance criteria:**
- [ ] `pip install -e ".[docs]"` installs all doc dependencies
- [ ] `make html` in docs/ generates API reference
- [ ] All public classes/functions have generated docs
- [ ] Source code links work
- [ ] Instructions added to CONTRIBUTING.md

---

### DOC-12: Consolidate _design_plan/ Directory
**Priority:** HIGH | **Effort:** 2-3 hours | **Status:** COMPLETE (2026-01-27)

Clean up and consolidate the `_design_plan/` directory for better navigation.

**Current state:** 17+ files, some outdated, INDEX.md exists but incomplete

**Deliverables:**
- Update INDEX.md with accurate status for all documents
- Archive completed/outdated documents to `_design_plan/archive/`
- Update cross-references between documents
- Add "Status" badges (Complete, In Progress, Planned)
- Remove or update stale content (test counts, outdated references)
- Link from ARCHITECTURE.md and how-it-works.md

**Documents to review:**
- 00-14: Core implementation docs (verify complete)
- 15-17: Feature docs (verify status)
- FUTURE_WORK.md: Update with current state
- archive/: Ensure properly categorized

**Acceptance criteria:**
- [ ] INDEX.md accurately reflects all documents
- [ ] No broken cross-references
- [ ] Stale content updated or archived
- [ ] Clear distinction between complete vs planned features

---

### DOC-13: Final Review and Polish
**Priority:** HIGH | **Effort:** 2-3 hours | **Status:** COMPLETE (2026-01-27)

Final pass to ensure all documentation is consistent, cross-linked, and accurate.

**Deliverables:**
- Verify all cross-links work
- Update test count (currently 1173 in README)
- Consistent formatting across all docs
- Spell check / grammar review
- Verify all code examples run
- Add navigation (prev/next) to docs
- Update CLAUDE.md if needed
- Update .agents/TICKET_OVERVIEW.md

**Checklist:**
- [ ] README.md links all work
- [ ] docs/*.md links all work
- [ ] Test count accurate (run pytest --collect-only | tail -1)
- [ ] Code examples in docs are tested
- [ ] No duplicate content across docs
- [ ] Consistent heading styles
- [ ] TICKET_OVERVIEW.md updated

---

## Backlog (Future Work)

### DOC-3: Tutorial Jupyter Notebooks
**Priority:** LOW | **Effort:** 8-12 hours | **Status:** BACKLOG

Create interactive Jupyter notebooks for hands-on learning.

**Potential notebooks:**
- `01_data_exploration.ipynb` - Explore BIOM data, visualize UniFrac
- `02_training_walkthrough.ipynb` - Step-by-step training
- `03_custom_dataset.ipynb` - Using your own data
- `04_model_interpretation.ipynb` - Attention visualization

---

### DOC-4: Video Walkthrough
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** BACKLOG

Short video tutorial for installation and basic usage.

---

## Completed

### DOC-1: README & Installation Modernization
**Status:** COMPLETE (2026-01-13)

- Pip-only installation workflow
- Python 3.9-3.12 requirements
- Quick Start with test data (781 samples)
- Test count updated

---

## Summary

| Phase | Tickets | Status |
|-------|---------|--------|
| Phase 1: Foundation | DOC-5, DOC-6, DOC-7, DOC-8 | COMPLETE |
| Phase 2: Deep Dives | DOC-9, DOC-10, DOC-11 | COMPLETE |
| Phase 3: API & Polish | DOC-2, DOC-12, DOC-13 | COMPLETE |
| **Total** | **11 tickets** | **ALL COMPLETE** |

**Completed:** 2026-01-27

### Deliverables Created

- `docs/getting-started.md` - Installation and quickstart guide
- `docs/user-guide.md` - Full CLI reference (400+ lines)
- `docs/how-it-works.md` - Concepts and implementation guide
- `docs/api/` - Sphinx API documentation structure
- `ARCHITECTURE.md` - Expanded with design rationale (~400 lines)
- `CONTRIBUTING.md` - Developer workflow guide
- `README.md` - Refactored to concise overview (~130 lines)

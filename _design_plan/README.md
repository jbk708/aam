# Design Plan

**Status:** Implementation Complete

Internal design documents for AAM (Attention All Microbes) PyTorch implementation. These documents capture implementation decisions made during development.

## For Users

- **User documentation:** [docs/](../docs/)
- **Architecture:** [ARCHITECTURE.md](../ARCHITECTURE.md)
- **Contributing:** [CONTRIBUTING.md](../CONTRIBUTING.md)

## For Developers

See [INDEX.md](INDEX.md) for full document listing.

### Quick Links

- [00_overview.md](00_overview.md) - Architecture overview
- [FUTURE_WORK.md](FUTURE_WORK.md) - Outstanding enhancements

### Ticket Tracking

See `.agents/` folder for ticket files:
- `TICKET_OVERVIEW.md` - Summary of outstanding work
- `WORKFLOW.md` - Development workflow

## Training Strategy

**Stage 1 (Pretraining):** Train `SequenceEncoder` on UniFrac + nucleotide prediction

**Stage 2 (Fine-tuning):** Train `SequencePredictor` with pretrained encoder
- `--freeze-base`: Freeze encoder weights
- Without flag: Joint fine-tuning

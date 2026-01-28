# Regressor Optimization Tickets

**Last Updated:** 2026-01-27
**Status:** 2 tickets remaining (~10-14 hours) | 1 HIGH priority

**Completed:** REG-1 to REG-8, REG-BUG-1 (see `ARCHIVED_TICKETS.md`)

---

## Outstanding Tickets

### REG-5: Quantile Regression
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Complete

Predict multiple quantiles for uncertainty estimation.

**Scope:**
- `--loss-type quantile` with pinball loss
- `--quantiles 0.1,0.5,0.9` (default)
- Output dim = original × num_quantiles

**Files:** `aam/training/losses.py`, `aam/models/sequence_predictor.py`, `aam/cli/train.py`

---

### REG-6: Asymmetric Loss
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Complete

Different penalties for over/under prediction.

**Scope:**
- `--loss-type asymmetric`
- `--over-penalty 2.0 --under-penalty 1.0`

**Acceptance Criteria:**
- [x] `--loss-type asymmetric` works
- [x] `--over-penalty` and `--under-penalty` CLI flags
- [x] Validation for positive penalty values
- [x] 14 unit tests

**Files:** `aam/training/losses.py`, `aam/cli/train.py`, `tests/test_losses.py`

---

### REG-7: Residual Regression Head
**Priority:** LOW | **Effort:** 2-3 hours | **Status:** Complete

Skip connection: `output = Linear(x) + MLP(x)`

**Scope:**
- `--residual-regression-head` flag
- Requires `--regressor-hidden-dims`

**Acceptance Criteria:**
- [x] `--residual-regression-head` CLI flag
- [x] ResidualRegressionHead class with skip + mlp branches
- [x] Validation: requires regressor_hidden_dims
- [x] 15 unit tests

**Files:** `aam/models/sequence_predictor.py`, `aam/cli/train.py`

---

### REG-8: Per-Output Loss Config
**Priority:** LOW | **Effort:** 3-4 hours | **Status:** ✅ Complete

Different loss per target column for multi-output regression.

**Scope:**
- `--loss-config '{"pH": "mse", "temp": "huber"}'`

**Acceptance Criteria:**
- [x] `--loss-config` CLI flag with JSON parsing
- [x] `loss_config` parameter in MultiTaskLoss
- [x] Per-column loss computation (mse, mae, huber)
- [x] Fallback to default loss type for unconfigured columns
- [x] 19 unit tests

**Files:** `aam/training/losses.py`, `aam/cli/train.py`

---

### REG-9: Mixture of Experts
**Priority:** LOW | **Effort:** 6-8 hours | **Status:** Not Started

Separate expert heads per category with learned routing.

**Scope:**
- `--moe-experts 4 --moe-routing location`
- Soft routing with load balancing

**Files:** `aam/models/moe.py` (new), `aam/models/sequence_predictor.py`, `aam/cli/train.py`

---

### REG-10: Count Magnitude Embeddings
**Priority:** HIGH | **Effort:** 4-6 hours | **Status:** Not Started

Incorporate ASV count magnitudes as input features, not just for masking.

**Motivation:**
Random Forest baseline achieves MAE=50, R²=0.7 using raw ASV counts as features.
AAM achieves MAE=70, R²=0.42 - a 20-point gap. Key difference: AAM currently only
uses counts to determine which ASVs are present (binary mask), not their abundance.
RF can learn "high count of ASV_X correlates with target Y" directly.

**Current Behavior:**
```python
# sequence_encoder.py:162 - counts only used for masking
asv_mask = (tokens.sum(dim=-1) > 0).long()
```

**Proposed Solution:**
Add count embedding that gets combined with sequence embeddings:
```python
# Option 1: Additive embedding
count_embedding = self.count_embed(log(counts + 1))  # [batch, num_asvs, embed_dim]
asv_embedding = sequence_embedding + count_embedding

# Option 2: Concatenation + projection
combined = torch.cat([sequence_embedding, count_embedding], dim=-1)
asv_embedding = self.combine_proj(combined)

# Option 3: FiLM-style modulation
scale, shift = self.count_film(log(counts + 1)).chunk(2, dim=-1)
asv_embedding = sequence_embedding * scale + shift
```

**Scope:**
- Add `--count-embedding` flag to enable count magnitude embeddings
- Add `--count-embedding-method [add|concat|film]` for fusion strategy
- Modify ASVEncoder or SampleSequenceEncoder to accept counts tensor
- Log-transform counts before embedding: `log(counts + 1)`
- Update collate_fn to pass counts through the model path

**CLI:**
```bash
--count-embedding                    # Enable count magnitude embeddings
--count-embedding-method add         # Fusion method (add, concat, film)
```

**Acceptance Criteria:**
- [ ] `--count-embedding` flag enables count input to embeddings
- [ ] `--count-embedding-method` supports add, concat, film
- [ ] Counts are log-transformed before embedding
- [ ] Works with pretrained encoders (graceful handling if not present)
- [ ] Backward compatible (default behavior unchanged)
- [ ] 15+ unit tests
- [ ] Integration test showing improvement over baseline

**Files:**
- `aam/models/asv_encoder.py` - add count embedding layer
- `aam/models/sample_sequence_encoder.py` - pass counts through
- `aam/models/sequence_encoder.py` - pass counts through
- `aam/models/sequence_predictor.py` - pass counts through
- `aam/data/dataset.py` - ensure counts available in batch
- `aam/cli/train.py` - add CLI flags
- `aam/cli/pretrain.py` - add CLI flags
- `tests/test_asv_encoder.py` - unit tests
- `tests/test_sequence_encoder.py` - integration tests

---

## Summary

| Ticket | Description | Effort | Priority | Status |
|--------|-------------|--------|----------|--------|
| **REG-5** | Quantile regression | 4-6h | MEDIUM | Complete |
| **REG-6** | Asymmetric loss | 2-3h | MEDIUM | Complete |
| **REG-7** | Residual head | 2-3h | LOW | Complete |
| **REG-8** | Per-output loss | 3-4h | LOW | Complete |
| **REG-9** | Mixture of Experts | 6-8h | LOW | Not Started |
| **REG-10** | Count magnitude embeddings | 4-6h | **HIGH** | Not Started |
| **Total** | | **13-18h** | |

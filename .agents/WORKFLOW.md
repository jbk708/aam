# Agent Workflow Instructions

## Standard Ticket Workflow

1. **Review `.agents/PYTORCH_PORTING_TICKETS.md`** to identify the next ticket (first "Not Started" ticket)

2. **Create branch**: `git checkout -b pyt-{ticket-number}-{short-name}`

3. **Stub out work first**:
   - Create file structure with class/function stubs
   - Use `pass` for unimplemented methods
   - Commit: `"PYT-{ticket}: Stub out {component}"`

4. **Write tests**:
   - Create comprehensive test files
   - Cover all acceptance criteria
   - Include edge cases
   - Commit: `"PYT-{ticket}: Add tests for {component}"`

5. **Implement functionality**:
   - Fill in stubs to make tests pass
   - Minimize inline comments
   - Commit: `"PYT-{ticket}: Implement {component}"`

6. **Verify**: Run tests, check coverage, ensure all pass

7. **After completion** (when ticket is merged):
   - Update `.agents/PYTORCH_PORTING_TICKETS.md`: Mark ticket as ✅ Completed, check all acceptance criteria, add implementation notes
   - Update corresponding design plan document in `_design_plan/`: Mark status as completed
   - Commit: `"Update PYT-{ticket} ticket status to completed"`

## Key Principles

- **Check in before each step**: Commit stubs, then tests, then implementation
- **Minimize inline comments**: Code should be self-documenting
- **Test-driven**: Write tests before implementation
- **Follow plan documents**: Use `pytorch_porting_plan/*.md` as implementation guide
- **Update documentation**: Keep tickets and plan docs in sync

## Example

```bash
# 1. Review tickets
cat .agents/PYTORCH_PORTING_TICKETS.md

# 2. Create branch
git checkout -b pyt-2.2-transformer

# 3. Stub work
# ... create files with stubs ...
git add -f aam/models/transformer.py
git commit -m "PYT-2.2: Stub out transformer encoder"

# 4. Write tests
# ... create test files ...
git add tests/test_transformer.py
git commit -m "PYT-2.2: Add tests for transformer encoder"

# 5. Implement
# ... fill in implementation ...
git add -f aam/models/transformer.py
git commit -m "PYT-2.2: Implement transformer encoder"

# 6. Verify
pytest tests/test_transformer.py -v

# 7. After merge, update docs
# ... update tickets and plan ...
git commit -m "Update PYT-2.2 ticket status to completed"
```

#!/usr/bin/env python3
"""Debug script to identify NaN failure point in pretraining with token_limit.

This script sets up a minimal pretraining scenario with token_limit and gradient
accumulation, then traces through the forward pass to identify where NaN first appears.

Usage:
    python debug_nan_token_limit.py [--token-limit TOKEN_LIMIT] [--batch-size BATCH_SIZE]
"""

import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from aam.data.dataset import ASVDataset, collate_fn
from aam.data.tokenizer import SequenceTokenizer
from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac import UniFracComputer
from aam.models.sequence_encoder import SequenceEncoder
from aam.training.losses import MultiTaskLoss
from torch.utils.data import DataLoader
from functools import partial


def check_for_nan(tensor: torch.Tensor, name: str, step: str = "") -> bool:
    """Check for NaN/Inf in tensor and print diagnostics."""
    has_nan = torch.any(torch.isnan(tensor))
    has_inf = torch.any(torch.isinf(tensor))
    
    if has_nan or has_inf:
        print(f"\n{'='*80}")
        print(f"❌ NaN/Inf detected: {name} at {step}")
        print(f"{'='*80}")
        print(f"Shape: {tensor.shape}")
        print(f"Has NaN: {has_nan}")
        print(f"Has Inf: {has_inf}")
        if has_nan:
            nan_count = torch.isnan(tensor).sum().item()
            print(f"NaN count: {nan_count} / {tensor.numel()} ({100*nan_count/tensor.numel():.2f}%)")
        if has_inf:
            inf_count = torch.isinf(tensor).sum().item()
            print(f"Inf count: {inf_count} / {tensor.numel()} ({100*inf_count/tensor.numel():.2f}%)")
        
        # Print statistics
        finite_mask = torch.isfinite(tensor)
        if finite_mask.any():
            finite_values = tensor[finite_mask]
            print(f"Finite values - min: {finite_values.min().item():.6f}, max: {finite_values.max().item():.6f}, mean: {finite_values.mean().item():.6f}")
        else:
            print("No finite values found!")
        
        # Print sample values
        print(f"Sample values (first 10): {tensor.flatten()[:10].tolist()}")
        return True
    
    return False


def check_tensor_stats(tensor: torch.Tensor, name: str, step: str = ""):
    """Print tensor statistics for debugging."""
    print(f"\n[{step}] {name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}, Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")
    print(f"  Has NaN: {torch.any(torch.isnan(tensor))}")
    print(f"  Has Inf: {torch.any(torch.isinf(tensor))}")
    
    # Check for invalid token values (should be 0-5 for vocab_size=6)
    if tensor.dtype == torch.long:
        invalid_tokens = (tensor < 0) | (tensor >= 6)
        if invalid_tokens.any():
            print(f"  ⚠️  Invalid token values: {invalid_tokens.sum().item()} values outside [0, 5]")
            invalid_values = tensor[invalid_tokens].unique()
            print(f"  Invalid values: {invalid_values.tolist()}")


def debug_collate_fn(batch, token_limit, unifrac_distances, unifrac_metric):
    """Debug collate_fn output."""
    print("\n" + "="*80)
    print("DEBUGGING collate_fn")
    print("="*80)
    
    # Check input batch
    print(f"\nInput batch size: {len(batch)}")
    for i, sample in enumerate(batch):
        print(f"\nSample {i}:")
        print(f"  sample_id: {sample['sample_id']}")
        print(f"  tokens shape: {sample['tokens'].shape}")
        print(f"  counts shape: {sample['counts'].shape}")
        
        # Check for START_TOKEN at position 0 of each sequence
        tokens = sample['tokens']
        if tokens.shape[0] > 0:
            first_tokens = tokens[:, 0]  # First token of each sequence
            start_token_count = (first_tokens == 5).sum().item()
            print(f"  Sequences with START_TOKEN at pos 0: {start_token_count}/{tokens.shape[0]}")
            
            # Check for all-padding sequences
            all_padding = (tokens.sum(dim=1) == 0).sum().item()
            print(f"  All-padding sequences: {all_padding}")
            
            # Check token validity
            invalid_tokens = (tokens < 0) | (tokens >= 6)
            if invalid_tokens.any():
                print(f"  ⚠️  Invalid tokens: {invalid_tokens.sum().item()}")
    
    # Run collate_fn
    result = collate_fn(batch, token_limit, unifrac_distances, unifrac_metric)
    
    # Check output
    print(f"\nCollate_fn output:")
    print(f"  tokens shape: {result['tokens'].shape}")
    print(f"  counts shape: {result['counts'].shape}")
    
    tokens = result['tokens']
    batch_size, num_asvs, seq_len = tokens.shape
    
    # Check START_TOKEN preservation after truncation
    if num_asvs > 0:
        first_tokens = tokens[:, :, 0]  # First token of each sequence [batch_size, num_asvs]
        start_token_count = (first_tokens == 5).sum().item()
        print(f"  START_TOKEN at pos 0: {start_token_count}/{batch_size * num_asvs}")
        
        # Check for all-padding sequences
        seq_sums = tokens.sum(dim=2)  # [batch_size, num_asvs]
        all_padding = (seq_sums == 0).sum().item()
        print(f"  All-padding sequences: {all_padding}")
        
        # Check token validity
        invalid_tokens = (tokens < 0) | (tokens >= 6)
        if invalid_tokens.any():
            print(f"  ⚠️  Invalid tokens: {invalid_tokens.sum().item()}")
            invalid_positions = torch.where(invalid_tokens)
            print(f"  Invalid positions: batch={invalid_positions[0][:5].tolist()}, asv={invalid_positions[1][:5].tolist()}, pos={invalid_positions[2][:5].tolist()}")
    
    # Check for NaN/Inf
    check_for_nan(result['tokens'].float(), "collate_fn tokens", "collate_fn")
    check_for_nan(result['counts'], "collate_fn counts", "collate_fn")
    
    return result


def debug_model_forward(model, tokens, step_name="forward"):
    """Debug model forward pass with detailed checks."""
    print("\n" + "="*80)
    print(f"DEBUGGING Model Forward Pass: {step_name}")
    print("="*80)
    
    # Check input tokens
    check_tensor_stats(tokens, "Input tokens", step_name)
    if check_for_nan(tokens.float(), "input tokens", step_name):
        return None
    
    # Check token values
    invalid_tokens = (tokens < 0) | (tokens >= 6)
    if invalid_tokens.any():
        print(f"⚠️  Invalid input tokens detected!")
        return None
    
    # Trace through model components
    batch_size, num_asvs, seq_len = tokens.shape
    
    # 1. ASVEncoder: Token embedding
    print("\n--- ASVEncoder: Token Embedding ---")
    token_emb = model.sample_encoder.asv_encoder.token_embedding(tokens)
    check_tensor_stats(token_emb, "Token embeddings", step_name)
    if check_for_nan(token_emb, "token embeddings", step_name):
        return None
    
    # 2. ASVEncoder: Position embedding
    print("\n--- ASVEncoder: Position Embedding ---")
    tokens_flat = tokens.reshape(batch_size * num_asvs, seq_len)
    pos_emb = model.sample_encoder.asv_encoder.position_embedding(token_emb.reshape(batch_size * num_asvs, seq_len, -1))
    check_tensor_stats(pos_emb, "Position embeddings", step_name)
    if check_for_nan(pos_emb, "position embeddings", step_name):
        return None
    
    # 3. ASVEncoder: Transformer
    print("\n--- ASVEncoder: Transformer ---")
    mask = (tokens_flat > 0).long()
    transformer_out = model.sample_encoder.asv_encoder.transformer(pos_emb, mask=mask)
    check_tensor_stats(transformer_out, "Transformer output", step_name)
    if check_for_nan(transformer_out, "transformer output", step_name):
        return None
    
    # 4. ASVEncoder: Attention pooling
    print("\n--- ASVEncoder: Attention Pooling ---")
    transformer_out_reshaped = transformer_out.reshape(batch_size, num_asvs, seq_len, -1)
    mask_reshaped = mask.reshape(batch_size, num_asvs, seq_len).unsqueeze(-1).float()
    pooled = model.sample_encoder.asv_encoder.attention_pooling(
        transformer_out_reshaped, mask=mask_reshaped
    )
    check_tensor_stats(pooled, "ASV embeddings (pooled)", step_name)
    if check_for_nan(pooled, "ASV embeddings", step_name):
        return None
    
    # 5. ASVEncoder: Nucleotide head
    print("\n--- ASVEncoder: Nucleotide Prediction Head ---")
    nuc_logits = model.sample_encoder.asv_encoder.nucleotide_head(transformer_out_reshaped)
    check_tensor_stats(nuc_logits, "Nucleotide logits", step_name)
    if check_for_nan(nuc_logits, "nucleotide logits", step_name):
        return None
    
    # 6. SampleSequenceEncoder: Sample position embedding
    print("\n--- SampleSequenceEncoder: Position Embedding ---")
    asv_mask = (tokens.sum(dim=-1) > 0).long()
    sample_pos_emb = model.sample_encoder.sample_position_embedding(pooled)
    check_tensor_stats(sample_pos_emb, "Sample position embeddings", step_name)
    if check_for_nan(sample_pos_emb, "sample position embeddings", step_name):
        return None
    
    # 7. SampleSequenceEncoder: Sample transformer
    print("\n--- SampleSequenceEncoder: Transformer ---")
    sample_transformer_out = model.sample_encoder.sample_transformer(sample_pos_emb, mask=asv_mask)
    check_tensor_stats(sample_transformer_out, "Sample transformer output", step_name)
    if check_for_nan(sample_transformer_out, "sample transformer output", step_name):
        return None
    
    # 8. Full forward pass
    print("\n--- Full Forward Pass ---")
    model.eval()  # Use eval mode to match validation
    with torch.no_grad():
        outputs = model(tokens, return_nucleotides=True)
    
    if "nuc_predictions" in outputs:
        check_tensor_stats(outputs["nuc_predictions"], "Final nucleotide predictions", step_name)
        if check_for_nan(outputs["nuc_predictions"], "final nucleotide predictions", step_name):
            return None
    
    if "base_prediction" in outputs:
        check_tensor_stats(outputs["base_prediction"], "Base prediction", step_name)
        if check_for_nan(outputs["base_prediction"], "base prediction", step_name):
            return None
    
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Debug NaN in pretraining with token_limit")
    parser.add_argument("--token-limit", type=int, default=512, help="Token limit for truncation")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum training steps to debug")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    
    # Load data
    print("\n" + "="*80)
    print("Loading data...")
    print("="*80)
    
    data_dir = Path(args.data_dir)
    biom_file = data_dir / "fall_train_only_all_outdoor.biom"
    tree_file = data_dir / "all-outdoors_sepp_tree.nwk"
    
    if not biom_file.exists() or not tree_file.exists():
        print(f"⚠️  Data files not found in {data_dir}")
        print("Creating synthetic data for testing...")
        
        # Create minimal synthetic data
        from biom import Table
        import numpy as np
        
        # Create a small synthetic table with enough ASVs to trigger truncation
        num_samples = 20
        num_asvs = max(200, args.token_limit + 50)  # Ensure we have more ASVs than token_limit
        data = np.random.randint(1, 100, (num_asvs, num_samples))
        observation_ids = [f"ASV_{i:04d}" + "ACGT" * 37 + "A" for i in range(num_asvs)]
        sample_ids = [f"sample_{i:03d}" for i in range(num_samples)]
        table = Table(data, observation_ids=observation_ids, sample_ids=sample_ids)
        
        # Create minimal tree (all ASVs in a simple tree)
        import tempfile
        import os
        tree_lines = ["("]
        for i in range(min(100, num_asvs)):  # Limit tree size
            if i > 0:
                tree_lines.append(",")
            tree_lines.append(f"ASV_{i:04d}:0.1")
        tree_lines.append(");")
        tree_content = "".join(tree_lines)
        
        temp_dir = Path(tempfile.mkdtemp())
        tree_file = temp_dir / "synthetic_tree.nwk"
        with open(tree_file, 'w') as f:
            f.write(tree_content)
        print(f"Created synthetic tree at {tree_file}")
    else:
        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        table = loader.rarefy(table, depth=5000, random_seed=42)
    
    # Compute UniFrac distances
    print("Computing UniFrac distances...")
    computer = UniFracComputer()
    unifrac_distances = computer.compute_unweighted(table, str(tree_file))
    
    # Create dataset
    print("Creating dataset...")
    tokenizer = SequenceTokenizer()
    dataset = ASVDataset(
        table=table,
        tokenizer=tokenizer,
        max_bp=150,
        token_limit=args.token_limit,
        unifrac_distances=unifrac_distances,
        unifrac_metric="unweighted",
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create DataLoader
    collate = partial(
        collate_fn,
        token_limit=args.token_limit,
        unifrac_distances=unifrac_distances,
        unifrac_metric="unweighted",
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )
    
    # Create model
    print("\n" + "="*80)
    print("Creating model...")
    print("="*80)
    
    model = SequenceEncoder(
        vocab_size=6,
        embedding_dim=128,
        max_bp=150,
        token_limit=args.token_limit,
        asv_num_layers=4,
        asv_num_heads=4,
        sample_num_layers=4,
        sample_num_heads=4,
        encoder_num_layers=4,
        encoder_num_heads=4,
        base_output_dim=args.batch_size,  # For pairwise UniFrac
        encoder_type="unifrac",
        predict_nucleotides=True,
    )
    model = model.to(device)
    model.train()
    
    # Create loss function
    loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=1.0, class_weights=None)
    loss_fn = loss_fn.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Token limit: {args.token_limit}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    # Training loop with detailed debugging
    print("\n" + "="*80)
    print("Starting training loop with debugging...")
    print("="*80)
    
    step = 0
    for epoch in range(1):
        for batch_idx, batch in enumerate(dataloader):
            if step >= args.max_steps:
                print(f"\nReached max steps ({args.max_steps}), stopping.")
                break
            
            step += 1
            print(f"\n{'='*80}")
            print(f"STEP {step} (Epoch {epoch}, Batch {batch_idx})")
            print(f"{'='*80}")
            
            # Move batch to device
            tokens = batch["tokens"].to(device)
            counts = batch["counts"].to(device)
            unifrac_target = batch["unifrac_target"].to(device)
            
            # Debug collate_fn output
            print("\n--- Checking collate_fn output ---")
            check_tensor_stats(tokens, "Batch tokens", f"step_{step}")
            check_tensor_stats(counts, "Batch counts", f"step_{step}")
            check_tensor_stats(unifrac_target, "UniFrac target", f"step_{step}")
            
            # Check for invalid tokens
            invalid_tokens = (tokens < 0) | (tokens >= 6)
            if invalid_tokens.any():
                print(f"❌ Invalid tokens detected in batch!")
                invalid_count = invalid_tokens.sum().item()
                print(f"Invalid token count: {invalid_count}")
                break
            
            # Debug model forward pass (detailed trace)
            print("\n--- Debugging model forward pass (detailed trace) ---")
            try:
                outputs = debug_model_forward(model, tokens, f"step_{step}")
                if outputs is None:
                    print(f"\n❌ Model forward pass failed at step {step}")
                    break
            except Exception as e:
                print(f"\n❌ Exception during model forward pass: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # Also run actual forward pass to see if it matches
            print("\n--- Running actual forward pass ---")
            model.train()
            try:
                actual_outputs = model(tokens, return_nucleotides=True)
                if "nuc_predictions" in actual_outputs:
                    if check_for_nan(actual_outputs["nuc_predictions"], "actual nuc_predictions", f"step_{step}"):
                        print(f"\n❌ NaN detected in actual nuc_predictions at step {step}")
                        print("This is the failure point!")
                        break
            except Exception as e:
                print(f"\n❌ Exception during actual forward pass: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # Check outputs
            if "nuc_predictions" in outputs:
                nuc_pred = outputs["nuc_predictions"]
                if check_for_nan(nuc_pred, "nuc_predictions", f"step_{step}"):
                    print(f"\n❌ NaN detected in nuc_predictions at step {step}")
                    print("This is the failure point!")
                    break
            
            # Prepare targets
            targets = {
                "base_target": unifrac_target,
                "tokens": tokens,
            }
            
            # Compute loss (use actual_outputs if available, otherwise run forward again)
            print("\n--- Computing loss ---")
            if actual_outputs is None:
                model.train()
                outputs_for_loss = model(tokens, return_nucleotides=True)
            else:
                outputs_for_loss = actual_outputs
            
            losses = loss_fn(outputs_for_loss, targets, is_classifier=False, encoder_type="unifrac")
            total_loss = losses["total_loss"]
            
            check_tensor_stats(total_loss, "Total loss", f"step_{step}")
            if check_for_nan(total_loss, "total_loss", f"step_{step}"):
                print(f"\n❌ NaN detected in loss at step {step}")
                break
            
            # Backward pass
            print("\n--- Backward pass ---")
            total_loss = total_loss / args.gradient_accumulation_steps
            total_loss.backward()
            
            # Check gradients
            print("\n--- Checking gradients ---")
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.any(torch.isnan(param.grad)):
                        print(f"❌ NaN gradient in {name}")
                        has_nan_grad = True
            
            if has_nan_grad:
                print(f"\n❌ NaN gradients detected at step {step}")
                break
            
            # Optimizer step
            if (step % args.gradient_accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            print(f"\n✅ Step {step} completed successfully")
    
    print("\n" + "="*80)
    print("Debugging complete!")
    print("="*80)


if __name__ == "__main__":
    main()

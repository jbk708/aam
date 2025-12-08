#!/usr/bin/env python3
"""Investigate NaN in attention pooling with all-padding sequences."""

import argparse
import sys
import torch
import torch.nn as nn
import numpy as np

from aam.models.attention_pooling import AttentionPooling


def check_tensor_stats(tensor, name, detailed=False):
    """Print tensor statistics."""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    if tensor.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.long):
        print(f"  Min: {tensor.min().item()}, Max: {tensor.max().item()}")
    else:
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        total = tensor.numel()
        print(f"  Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}")
        print(f"  Mean: {tensor.mean().item():.6f}")
        print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
        if has_nan:
            print(f"  NaN count: {nan_count} / {total} ({100 * nan_count / total:.2f}%)")
        if has_inf:
            print(f"  Inf count: {inf_count} / {total} ({100 * inf_count / total:.2f}%)")
        
        if detailed and len(tensor.shape) >= 2:
            # Show per-sample statistics
            if tensor.shape[0] <= 10:
                for i in range(tensor.shape[0]):
                    sample_tensor = tensor[i]
                    sample_nan = torch.isnan(sample_tensor).any().item()
                    sample_inf = torch.isinf(sample_tensor).any().item()
                    if sample_nan or sample_inf:
                        print(f"    Sample {i}: NaN={sample_nan}, Inf={sample_inf}")


def trace_attention_pooling(embeddings, mask, pooling_layer):
    """Trace attention pooling step by step."""
    print("\n" + "=" * 80)
    print("TRACING ATTENTION POOLING")
    print("=" * 80)
    
    batch_size, seq_len, hidden_dim = embeddings.shape
    
    print(f"\nInput:")
    print(f"  embeddings shape: {embeddings.shape}")
    print(f"  mask shape: {mask.shape if mask is not None else 'None'}")
    
    # Step 1: Compute query scores
    print("\n" + "-" * 80)
    print("STEP 1: Compute query scores")
    print("-" * 80)
    scores = pooling_layer.query(embeddings)  # [batch_size, seq_len, 1]
    scores = scores.squeeze(-1)  # [batch_size, seq_len]
    scores = scores / (hidden_dim ** 0.5)
    check_tensor_stats(scores, "scores", detailed=True)
    
    # Step 2: Handle mask and all-padding detection
    print("\n" + "-" * 80)
    print("STEP 2: Handle mask and all-padding detection")
    print("-" * 80)
    all_padding = None
    if mask is not None:
        mask_sum = mask.sum(dim=-1, keepdim=True)  # [batch_size, 1]
        all_padding = (mask_sum == 0)  # [batch_size, 1]
        num_all_padding = all_padding.sum().item()
        print(f"  mask_sum shape: {mask_sum.shape}")
        print(f"  mask_sum values (first 10): {mask_sum[:10].squeeze().tolist()}")
        print(f"  Number of all-padding sequences: {num_all_padding} / {batch_size}")
        print(f"  all_padding shape: {all_padding.shape}")
        print(f"  all_padding values (first 10): {all_padding[:10].squeeze().tolist()}")
        
        if all_padding.any():
            all_padding_expanded = all_padding.expand(-1, seq_len)
            scores = scores.masked_fill(all_padding_expanded, 0.0)
            print(f"  After setting scores to 0 for all-padding sequences:")
            check_tensor_stats(scores, "scores (after all-padding fix)", detailed=True)
            
            # Mask padding for valid sequences
            valid_mask_expanded = (~all_padding).expand(-1, seq_len)
            scores = scores.masked_fill(valid_mask_expanded & (mask == 0), float("-inf"))
            print(f"  After masking padding with -inf for valid sequences:")
            check_tensor_stats(scores, "scores (after -inf masking)", detailed=True)
        else:
            scores = scores.masked_fill(mask == 0, float("-inf"))
            print(f"  After masking padding with -inf:")
            check_tensor_stats(scores, "scores (after -inf masking)", detailed=True)
    
    # Step 3: Softmax
    print("\n" + "-" * 80)
    print("STEP 3: Softmax")
    print("-" * 80)
    attention_weights = torch.softmax(scores, dim=-1)
    check_tensor_stats(attention_weights, "attention_weights (after softmax)", detailed=True)
    print(f"  attention_weights sum per sample (first 10): {attention_weights.sum(dim=-1)[:10].tolist()}")
    
    # Step 4: Apply mask and normalize
    print("\n" + "-" * 80)
    print("STEP 4: Apply mask and normalize")
    print("-" * 80)
    if mask is not None:
        # Handle all-padding sequences
        if all_padding is not None and all_padding.any():
            all_padding_expanded = all_padding.expand(-1, seq_len)
            uniform_weights = torch.full_like(attention_weights, 1.0 / seq_len)
            attention_weights_all_padding = uniform_weights
            print(f"  Created uniform weights for all-padding: {1.0 / seq_len}")
            check_tensor_stats(attention_weights_all_padding, "uniform_weights", detailed=True)
        
        # Apply mask
        attention_weights = attention_weights * mask
        print(f"  After applying mask:")
        check_tensor_stats(attention_weights, "attention_weights (after mask)", detailed=True)
        print(f"  attention_weights sum per sample (first 10): {attention_weights.sum(dim=-1)[:10].tolist()}")
        
        # Normalize
        attention_weights_sum = attention_weights.sum(dim=-1, keepdim=True)
        print(f"  attention_weights_sum shape: {attention_weights_sum.shape}")
        print(f"  attention_weights_sum values (first 10): {attention_weights_sum[:10].squeeze().tolist()}")
        
        normalized = attention_weights / (attention_weights_sum + 1e-8)
        print(f"  After normalization:")
        check_tensor_stats(normalized, "normalized attention_weights", detailed=True)
        print(f"  normalized sum per sample (first 10): {normalized.sum(dim=-1)[:10].tolist()}")
        
        # Replace all-padding with uniform weights
        if all_padding is not None and all_padding.any():
            all_padding_expanded = all_padding.expand(-1, seq_len)
            attention_weights = torch.where(all_padding_expanded, attention_weights_all_padding, normalized)
            print(f"  After replacing all-padding with uniform weights:")
            check_tensor_stats(attention_weights, "attention_weights (final)", detailed=True)
            print(f"  attention_weights sum per sample (first 10): {attention_weights.sum(dim=-1)[:10].tolist()}")
        else:
            attention_weights = normalized
    
    # Step 5: Pool embeddings
    print("\n" + "-" * 80)
    print("STEP 5: Pool embeddings")
    print("-" * 80)
    pooled = torch.sum(embeddings * attention_weights.unsqueeze(-1), dim=1)
    check_tensor_stats(pooled, "pooled (before LayerNorm)", detailed=True)
    
    # Step 6: LayerNorm
    print("\n" + "-" * 80)
    print("STEP 6: LayerNorm")
    print("-" * 80)
    pooled = pooling_layer.norm(pooled)
    check_tensor_stats(pooled, "pooled (after LayerNorm)", detailed=True)
    
    return pooled


def test_attention_pooling_scenarios():
    """Test various scenarios that might cause NaN."""
    print("=" * 80)
    print("TESTING ATTENTION POOLING SCENARIOS")
    print("=" * 80)
    
    pooling = AttentionPooling(hidden_dim=32)
    
    # Scenario 1: Mixed valid and all-padding sequences
    print("\n" + "#" * 80)
    print("SCENARIO 1: Mixed valid and all-padding sequences")
    print("#" * 80)
    batch_size = 10
    seq_len = 151
    hidden_dim = 32
    
    embeddings = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.zeros(batch_size, seq_len)
    # First 5 samples have valid positions
    mask[:5, :] = 1
    # Last 5 samples are all padding
    
    print(f"\nSetup:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  Samples 0-4: valid (mask sum = {mask[:5].sum(dim=-1).tolist()})")
    print(f"  Samples 5-9: all padding (mask sum = {mask[5:].sum(dim=-1).tolist()})")
    
    try:
        result = trace_attention_pooling(embeddings, mask, pooling)
        print("\n✅ SUCCESS: No NaN in result")
        print(f"Result shape: {result.shape}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Scenario 2: Large batch with many all-padding (like integration test)
    print("\n" + "#" * 80)
    print("SCENARIO 2: Large batch with many all-padding (like integration test)")
    print("#" * 80)
    batch_size = 4096
    seq_len = 151
    hidden_dim = 32
    
    embeddings = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.zeros(batch_size, seq_len)
    # First 100 samples have valid positions
    mask[:100, :] = 1
    # Rest are all padding
    
    print(f"\nSetup:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  Samples 0-99: valid")
    print(f"  Samples 100-4095: all padding ({batch_size - 100} all-padding sequences)")
    
    try:
        result = trace_attention_pooling(embeddings, mask, pooling)
        print("\n✅ SUCCESS: No NaN in result")
        print(f"Result shape: {result.shape}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Scenario 3: All sequences are valid
    print("\n" + "#" * 80)
    print("SCENARIO 3: All sequences are valid")
    print("#" * 80)
    batch_size = 10
    seq_len = 151
    hidden_dim = 32
    
    embeddings = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.ones(batch_size, seq_len)
    
    print(f"\nSetup:")
    print(f"  batch_size: {batch_size}")
    print(f"  All sequences valid")
    
    try:
        result = trace_attention_pooling(embeddings, mask, pooling)
        print("\n✅ SUCCESS: No NaN in result")
        print(f"Result shape: {result.shape}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Scenario 4: All sequences are all-padding
    print("\n" + "#" * 80)
    print("SCENARIO 4: All sequences are all-padding")
    print("#" * 80)
    batch_size = 10
    seq_len = 151
    hidden_dim = 32
    
    embeddings = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.zeros(batch_size, seq_len)
    
    print(f"\nSetup:")
    print(f"  batch_size: {batch_size}")
    print(f"  All sequences are all-padding")
    
    try:
        result = trace_attention_pooling(embeddings, mask, pooling)
        print("\n✅ SUCCESS: No NaN in result")
        print(f"Result shape: {result.shape}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_with_real_model_forward():
    """Test with actual model forward pass to see where NaN originates."""
    print("\n" + "=" * 80)
    print("TESTING WITH ACTUAL MODEL FORWARD PASS")
    print("=" * 80)
    
    from aam.models.asv_encoder import ASVEncoder
    
    batch_size = 4096
    num_asvs = 1  # Single ASV per sample (like in ASV encoder)
    seq_len = 151
    vocab_size = 6
    embedding_dim = 32
    
    # Create ASV encoder
    asv_encoder = ASVEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_bp=150,
        num_layers=1,
        num_heads=2,
    )
    
    # Create tokens: mix of valid and all-padding
    tokens = torch.randint(1, vocab_size, (batch_size, num_asvs, seq_len))
    # First 100 have valid tokens
    # Rest are all zeros (all padding)
    tokens[100:, :, :] = 0
    
    # Create mask
    mask = (tokens.sum(dim=-1) > 0).long()  # [batch_size, num_asvs]
    mask = mask.squeeze(1)  # [batch_size]
    
    print(f"\nSetup:")
    print(f"  tokens shape: {tokens.shape}")
    print(f"  mask shape: {mask.shape}")
    print(f"  Valid sequences: {mask.sum().item()} / {batch_size}")
    print(f"  All-padding sequences: {(mask == 0).sum().item()} / {batch_size}")
    
    # Trace through ASV encoder step by step
    print("\n" + "-" * 80)
    print("Tracing through ASV encoder...")
    print("-" * 80)
    
    try:
        asv_encoder.eval()
        with torch.no_grad():
            # Check tokens
            check_tensor_stats(tokens, "tokens", detailed=False)
            
            # Embedding layer
            embedded = asv_encoder.token_embedding(tokens.squeeze(1))  # [batch_size, seq_len, embedding_dim]
            check_tensor_stats(embedded, "embedded (after token_embedding)", detailed=False)
            
            # Position embedding
            pos_embedded = asv_encoder.position_embedding(embedded)
            check_tensor_stats(pos_embedded, "pos_embedded (after position_embedding)", detailed=False)
            
            # Transformer
            transformer_output = asv_encoder.transformer(pos_embedded, mask=mask.unsqueeze(1).unsqueeze(2))
            check_tensor_stats(transformer_output, "transformer_output", detailed=False)
            
            # Attention pooling
            pooled = asv_encoder.attention_pooling(transformer_output, mask=mask)
            check_tensor_stats(pooled, "pooled (final output)", detailed=False)
            
        print("\n✅ SUCCESS: No NaN in ASV encoder output")
        print(f"Final output shape: {pooled.shape}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Investigate NaN in attention pooling")
    parser.add_argument("--scenarios", action="store_true", help="Run test scenarios")
    parser.add_argument("--model-forward", action="store_true", help="Test with actual model forward pass")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    if args.all or args.scenarios:
        test_attention_pooling_scenarios()
    
    if args.all or args.model_forward:
        test_with_real_model_forward()
    
    if not (args.scenarios or args.model_forward or args.all):
        # Default: run all
        test_attention_pooling_scenarios()
        test_with_real_model_forward()


if __name__ == "__main__":
    main()

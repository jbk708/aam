#!/usr/bin/env python
"""Investigate stripe-based UniFrac computation APIs.

This script investigates whether the unifrac package or scikit-bio supports
stripe-based UniFrac computation (computing distances for samples against
a reference set rather than all pairwise combinations).
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

import biom
from biom import Table
import skbio
from skbio import TreeNode
import pandas as pd

# Try importing unifrac package
try:
    import unifrac
    UNIFRAC_AVAILABLE = True
except ImportError:
    UNIFRAC_AVAILABLE = False
    print("WARNING: unifrac package not available", file=sys.stderr)

# Check scikit-bio availability (should be available based on dependencies)
try:
    from skbio.diversity import beta_diversity
    from skbio.diversity.beta import unweighted_unifrac
    SKBIO_AVAILABLE = True
except ImportError:
    SKBIO_AVAILABLE = False
    print("WARNING: scikit-bio diversity module not available", file=sys.stderr)


def investigate_unifrac_package_api():
    """Investigate unifrac package API for stripe-based computation."""
    print("\n" + "="*80)
    print("INVESTIGATING: unifrac Python Package")
    print("="*80)
    
    if not UNIFRAC_AVAILABLE:
        print("❌ unifrac package not available - skipping investigation")
        return None
    
    print(f"✅ unifrac package version: {unifrac.__version__ if hasattr(unifrac, '__version__') else 'unknown'}")
    
    # Check available functions
    print("\nAvailable functions in unifrac module:")
    unifrac_functions = [attr for attr in dir(unifrac) if not attr.startswith('_')]
    for func in sorted(unifrac_functions):
        print(f"  - {func}")
    
    # Check if unweighted function exists and inspect its signature
    if hasattr(unifrac, 'unweighted'):
        print("\n📋 unifrac.unweighted() signature:")
        import inspect
        sig = inspect.signature(unifrac.unweighted)
        print(f"  {sig}")
        
        # Check docstring for hints about stripe/batch computation
        doc = unifrac.unweighted.__doc__
        if doc:
            print("\n📄 Docstring preview:")
            lines = doc.split('\n')[:10]
            for line in lines:
                print(f"  {line}")
            if len(doc.split('\n')) > 10:
                print("  ...")
    
    # Check for other potential stripe-related functions
    stripe_keywords = ['stripe', 'batch', 'reference', 'sample', 'subset']
    print("\n🔍 Searching for stripe-related functions:")
    found_stripe = False
    for func_name in unifrac_functions:
        func_lower = func_name.lower()
        if any(keyword in func_lower for keyword in stripe_keywords):
            print(f"  ⚠️  Found potential: {func_name}")
            found_stripe = True
    
    if not found_stripe:
        print("  ❌ No obvious stripe-related functions found")
    
    # Check if unifrac has access to underlying C++ API
    print("\n🔍 Checking for C++ API access:")
    if hasattr(unifrac, '_api') or hasattr(unifrac, 'api'):
        print("  ✅ May have access to underlying API")
    else:
        print("  ❌ No obvious C++ API access")
    
    return {
        'available': True,
        'version': unifrac.__version__ if hasattr(unifrac, '__version__') else 'unknown',
        'has_unweighted': hasattr(unifrac, 'unweighted'),
        'functions': unifrac_functions,
    }


def investigate_scikit_bio_api():
    """Investigate scikit-bio API for stripe-based computation."""
    print("\n" + "="*80)
    print("INVESTIGATING: scikit-bio (skbio.diversity.beta)")
    print("="*80)
    
    if not SKBIO_AVAILABLE:
        print("❌ scikit-bio diversity module not available - skipping investigation")
        return None
    
    print(f"✅ scikit-bio version: {skbio.__version__ if hasattr(skbio, '__version__') else 'unknown'}")
    
    # Check beta_diversity module
    print("\nAvailable functions in skbio.diversity.beta:")
    try:
        from skbio.diversity import beta_diversity
        beta_functions = [attr for attr in dir(beta_diversity) if not attr.startswith('_')]
        for func in sorted(beta_functions):
            print(f"  - {func}")
    except Exception as e:
        print(f"  ❌ Error accessing beta_diversity: {e}")
    
    # Check unweighted_unifrac specifically
    print("\n📋 Investigating skbio.diversity.beta.unweighted_unifrac:")
    try:
        from skbio.diversity.beta import unweighted_unifrac
        import inspect
        
        sig = inspect.signature(unweighted_unifrac)
        print(f"  Signature: {sig}")
        
        doc = unweighted_unifrac.__doc__
        if doc:
            print("\n📄 Docstring preview:")
            lines = doc.split('\n')[:15]
            for line in lines:
                print(f"  {line}")
            if len(doc.split('\n')) > 15:
                print("  ...")
        
        # Check parameters for hints about batch/stripe computation
        params = sig.parameters
        print("\n📋 Parameters:")
        for param_name, param in params.items():
            print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'no annotation'}")
            if param.default != inspect.Parameter.empty:
                print(f"    Default: {param.default}")
        
    except ImportError as e:
        print(f"  ❌ unweighted_unifrac not available: {e}")
    except Exception as e:
        print(f"  ❌ Error investigating unweighted_unifrac: {e}")
    
    # Check if there are other beta diversity functions that might support batching
    print("\n🔍 Checking for batch/stripe capabilities:")
    try:
        # Check if beta_diversity function supports subsetting
        from skbio.diversity import beta_diversity
        sig = inspect.signature(beta_diversity)
        print(f"  beta_diversity signature: {sig}")
    except Exception as e:
        print(f"  ❌ Error checking beta_diversity: {e}")
    
    return {
        'available': True,
        'version': skbio.__version__ if hasattr(skbio, '__version__') else 'unknown',
        'has_unweighted_unifrac': 'unweighted_unifrac' in dir(beta_diversity) if SKBIO_AVAILABLE else False,
    }


def test_stripe_computation_with_unifrac(
    table: Table,
    tree: TreeNode,
    reference_sample_ids: List[str],
    test_sample_ids: List[str]
) -> Optional[np.ndarray]:
    """Test if unifrac package can compute stripe distances."""
    print("\n" + "="*80)
    print("TESTING: Stripe computation with unifrac package")
    print("="*80)
    
    if not UNIFRAC_AVAILABLE:
        print("❌ unifrac package not available")
        return None
    
    try:
        # Method 1: Try filtering table to reference + test samples, then extract stripe
        print("\n📋 Method 1: Compute full matrix, extract stripe")
        all_sample_ids = reference_sample_ids + test_sample_ids
        combined_table = table.filter(all_sample_ids, axis="sample", inplace=False)
        
        distance_matrix = unifrac.unweighted(combined_table, tree)
        
        # Extract stripe: rows = test samples, columns = reference samples
        test_indices = [distance_matrix.ids.index(sid) for sid in test_sample_ids]
        ref_indices = [distance_matrix.ids.index(sid) for sid in reference_sample_ids]
        
        stripe = distance_matrix.data[np.ix_(test_indices, ref_indices)]
        print(f"✅ Successfully extracted stripe: shape {stripe.shape}")
        print(f"   Expected: [{len(test_sample_ids)}, {len(reference_sample_ids)}]")
        
        return stripe
        
    except Exception as e:
        print(f"❌ Error in Method 1: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_stripe_computation_with_skbio(
    table: Table,
    tree: TreeNode,
    reference_sample_ids: List[str],
    test_sample_ids: List[str]
) -> Optional[np.ndarray]:
    """Test if scikit-bio can compute stripe distances."""
    print("\n" + "="*80)
    print("TESTING: Stripe computation with scikit-bio")
    print("="*80)
    
    if not SKBIO_AVAILABLE:
        print("❌ scikit-bio not available")
        return None
    
    try:
        # Method 1: Use beta_diversity with subset
        print("\n📋 Method 1: Using skbio.diversity.beta_diversity")
        from skbio.diversity import beta_diversity
        
        # Convert table to counts matrix format expected by skbio
        all_sample_ids = reference_sample_ids + test_sample_ids
        combined_table = table.filter(all_sample_ids, axis="sample", inplace=False)
        
        # Get observation IDs and counts
        obs_ids = list(combined_table.ids(axis="observation"))
        sample_data = []
        for sid in all_sample_ids:
            counts = [combined_table.get_value_by_ids(obs_id, sid) for obs_id in obs_ids]
            sample_data.append(counts)
        
        counts_matrix = np.array(sample_data)
        
        # Try computing with unweighted_unifrac metric
        from skbio.diversity.beta import unweighted_unifrac
        
        distance_matrix = beta_diversity(
            metric=unweighted_unifrac,
            counts=counts_matrix,
            ids=all_sample_ids,
            tree=tree,
        )
        
        # Extract stripe
        test_indices = [distance_matrix.ids.index(sid) for sid in test_sample_ids]
        ref_indices = [distance_matrix.ids.index(sid) for sid in reference_sample_ids]
        
        stripe = distance_matrix.data[np.ix_(test_indices, ref_indices)]
        print(f"✅ Successfully computed stripe: shape {stripe.shape}")
        print(f"   Expected: [{len(test_sample_ids)}, {len(reference_sample_ids)}]")
        
        return stripe
        
    except Exception as e:
        print(f"❌ Error in Method 1: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_test_data(tmp_path: Path) -> Tuple[Table, TreeNode, List[str], List[str]]:
    """Create test BIOM table and tree for testing."""
    print("\n" + "="*80)
    print("CREATING: Test data")
    print("="*80)
    
    # Create simple test data
    np.random.seed(42)
    n_observations = 10
    n_samples = 20
    
    # Generate random sequences for observations
    bases = "ACGT"
    observation_ids = [''.join(np.random.choice(list(bases), size=50)) for _ in range(n_observations)]
    
    # Generate sample IDs
    sample_ids = [f"sample_{i:02d}" for i in range(n_samples)]
    
    # Create random count data
    data = np.random.randint(0, 100, size=(n_observations, n_samples))
    table = Table(data, observation_ids=observation_ids, sample_ids=sample_ids)
    
    # Create simple tree (star tree for simplicity)
    tree_str = "(" + ",".join([f"{obs_id}:0.1" for obs_id in observation_ids]) + ");"
    tree = skbio.read(tree_str, format="newick", into=TreeNode)
    
    # Define reference and test samples
    reference_sample_ids = sample_ids[:5]  # First 5 samples
    test_sample_ids = sample_ids[5:10]     # Next 5 samples
    
    print(f"✅ Created test data:")
    print(f"   Observations: {n_observations}")
    print(f"   Samples: {n_samples}")
    print(f"   Reference samples: {len(reference_sample_ids)}")
    print(f"   Test samples: {len(test_sample_ids)}")
    
    return table, tree, reference_sample_ids, test_sample_ids


def main():
    parser = argparse.ArgumentParser(
        description="Investigate stripe-based UniFrac computation APIs"
    )
    parser.add_argument(
        "--table",
        type=str,
        help="Path to BIOM table (optional, will create test data if not provided)"
    )
    parser.add_argument(
        "--tree",
        type=str,
        help="Path to phylogenetic tree (optional, will create test data if not provided)"
    )
    parser.add_argument(
        "--reference-samples",
        type=int,
        default=5,
        help="Number of reference samples for testing (default: 5)"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=5,
        help="Number of test samples for testing (default: 5)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("STRIPE-BASED UNIFRAC API INVESTIGATION")
    print("="*80)
    print("\nThis script investigates whether unifrac package or scikit-bio")
    print("supports stripe-based UniFrac computation.\n")
    
    # Step 1: Investigate APIs
    unifrac_info = investigate_unifrac_package_api()
    skbio_info = investigate_scikit_bio_api()
    
    # Step 2: Test with actual computation (if test data available)
    if args.table and args.tree:
        from aam.data.biom_loader import BIOMLoader
        loader = BIOMLoader()
        table = loader.load_table(args.table)
        tree = skbio.read(args.tree, format="newick", into=TreeNode)
        
        # Use first N samples as reference, next M as test
        all_sample_ids = list(table.ids(axis="sample"))
        reference_sample_ids = all_sample_ids[:args.reference_samples]
        test_sample_ids = all_sample_ids[args.reference_samples:args.reference_samples + args.test_samples]
    else:
        # Create test data
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            table, tree, reference_sample_ids, test_sample_ids = create_test_data(tmp_path)
            
            # Step 3: Test stripe computation
            print("\n" + "="*80)
            print("TESTING STRIPE COMPUTATION")
            print("="*80)
            
            unifrac_stripe = test_stripe_computation_with_unifrac(
                table, tree, reference_sample_ids, test_sample_ids
            )
            
            skbio_stripe = test_stripe_computation_with_skbio(
                table, tree, reference_sample_ids, test_sample_ids
            )
            
            # Compare results if both succeeded
            if unifrac_stripe is not None and skbio_stripe is not None:
                print("\n" + "="*80)
                print("COMPARING RESULTS")
                print("="*80)
                print(f"unifrac stripe shape: {unifrac_stripe.shape}")
                print(f"skbio stripe shape: {skbio_stripe.shape}")
                
                if unifrac_stripe.shape == skbio_stripe.shape:
                    diff = np.abs(unifrac_stripe - skbio_stripe)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    print(f"\nMax difference: {max_diff:.6f}")
                    print(f"Mean difference: {mean_diff:.6f}")
                    
                    if max_diff < 1e-6:
                        print("✅ Results match (within numerical precision)")
                    else:
                        print("⚠️  Results differ (may be due to different implementations)")
    
    # Step 4: Summary and recommendations
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    print("\n📋 Findings:")
    if unifrac_info:
        print(f"  ✅ unifrac package: Available (version: {unifrac_info.get('version', 'unknown')})")
        print(f"     - Has unweighted function: {unifrac_info.get('has_unweighted', False)}")
    else:
        print("  ❌ unifrac package: Not available")
    
    if skbio_info:
        print(f"  ✅ scikit-bio: Available (version: {skbio_info.get('version', 'unknown')})")
        print(f"     - Has unweighted_unifrac: {skbio_info.get('has_unweighted_unifrac', False)}")
    else:
        print("  ❌ scikit-bio: Not available")
    
    print("\n💡 Recommendation:")
    print("  See detailed report in: _design_plan/21_stripe_unifrac_api_investigation.md")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

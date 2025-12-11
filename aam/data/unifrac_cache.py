"""Caching utilities for UniFrac distance matrices."""

import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging

import biom
from biom import Table
from skbio import DistanceMatrix
import pandas as pd

logger = logging.getLogger(__name__)


def get_cache_key(
    table: Union[str, Table],
    tree_path: str,
    rarefy_depth: Optional[int] = None,
    rarefy_seed: Optional[int] = None,
    metric: str = "unweighted",
) -> str:
    """Generate a cache key for UniFrac distance matrix.
    
    Args:
        table: BIOM table (Table object) or path to table file
        tree_path: Path to phylogenetic tree file
        rarefy_depth: Rarefaction depth (if applied)
        rarefy_seed: Rarefaction random seed (if applied)
        metric: UniFrac metric type ("unweighted" or "faith_pd")
    
    Returns:
        SHA256 hash string for cache key
    """
    # Handle both Table objects and file paths
    if isinstance(table, Table):
        # For Table objects, use a hash of the table data
        import pickle
        table_hash = hashlib.sha256(pickle.dumps(table)).hexdigest()[:16]
        table_identifier = f"table_hash_{table_hash}"
    else:
        table_identifier = str(Path(table).resolve())
    
    # Create a string representation of all parameters
    key_parts = [
        table_identifier,
        str(Path(tree_path).resolve()),
        str(rarefy_depth) if rarefy_depth is not None else "None",
        str(rarefy_seed) if rarefy_seed is not None else "None",
        metric,
    ]
    
    # Include file modification times to detect changes (if table is a path)
    if not isinstance(table, Table):
        try:
            table_mtime = Path(table).stat().st_mtime
            tree_mtime = Path(tree_path).stat().st_mtime
            key_parts.extend([str(table_mtime), str(tree_mtime)])
        except OSError:
            pass  # Files might not exist yet
    else:
        # For Table objects, include tree mtime
        try:
            tree_mtime = Path(tree_path).stat().st_mtime
            key_parts.append(str(tree_mtime))
        except OSError:
            pass
    
    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()


def get_cache_path(cache_dir: Optional[str] = None, cache_key: Optional[str] = None) -> Path:
    """Get the cache file path for a distance matrix.
    
    Args:
        cache_dir: Directory for cache files (default: ~/.aam_cache/unifrac)
        cache_key: Cache key (if None, returns cache directory)
    
    Returns:
        Path to cache file or cache directory
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".aam_cache" / "unifrac"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if cache_key is None:
        return cache_dir
    
    return cache_dir / f"{cache_key}.npz"


def save_distance_matrix(
    distances: Union[DistanceMatrix, pd.Series],
    cache_path: Path,
    metric: str = "unweighted",
) -> None:
    """Save distance matrix to cache file.
    
    Args:
        distances: DistanceMatrix or Series to save
        cache_path: Path to save cache file
        metric: Metric type ("unweighted" or "faith_pd")
    """
    try:
        if metric == "unweighted":
            if not isinstance(distances, DistanceMatrix):
                raise TypeError(f"Expected DistanceMatrix for unweighted, got {type(distances)}")
            np.savez_compressed(
                cache_path,
                data=distances.data,
                ids=distances.ids,
                metric=metric,
            )
        else:
            if not isinstance(distances, pd.Series):
                raise TypeError(f"Expected Series for faith_pd, got {type(distances)}")
            np.savez_compressed(
                cache_path,
                data=distances.values,
                index=distances.index.values,
                metric=metric,
            )
        logger.info(f"Saved distance matrix cache to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save distance matrix cache: {e}")


def load_distance_matrix(
    cache_path: Path,
    metric: str = "unweighted",
) -> Optional[Union[DistanceMatrix, pd.Series]]:
    """Load distance matrix from cache file.
    
    Args:
        cache_path: Path to cache file
        metric: Expected metric type ("unweighted" or "faith_pd")
    
    Returns:
        DistanceMatrix or Series if cache exists and is valid, None otherwise
    """
    if not cache_path.exists():
        return None
    
    try:
        cache_data = np.load(cache_path, allow_pickle=True)
        
        # Verify metric matches
        cached_metric = cache_data.get("metric", "unweighted")
        if cached_metric != metric:
            logger.warning(
                f"Cache metric mismatch: expected {metric}, got {cached_metric}. "
                f"Ignoring cache."
            )
            return None
        
        if metric == "unweighted":
            from skbio import DistanceMatrix
            data = cache_data["data"]
            ids = cache_data["ids"]
            return DistanceMatrix(data, ids=ids)
        else:
            data = cache_data["data"]
            index = cache_data["index"]
            return pd.Series(data, index=index)
    except Exception as e:
        logger.warning(f"Failed to load distance matrix cache: {e}")
        return None


def save_stripe_matrix(
    stripe_distances: np.ndarray,
    cache_path: Path,
    sample_ids: list,
    reference_sample_ids: list,
) -> None:
    """Save stripe distance matrix to cache file.
    
    Args:
        stripe_distances: Stripe distance matrix [N_samples, N_reference_samples]
        cache_path: Path to save cache file
        sample_ids: List of all sample IDs (rows)
        reference_sample_ids: List of reference sample IDs (columns)
    """
    try:
        np.savez_compressed(
            cache_path,
            stripe_distances=stripe_distances,
            sample_ids=sample_ids,
            reference_sample_ids=reference_sample_ids,
            metric="stripe_unweighted",
        )
        logger.info(f"Saved stripe distance matrix cache to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save stripe distance matrix cache: {e}")


def load_stripe_matrix(
    cache_path: Path,
) -> Optional[tuple[np.ndarray, list, list]]:
    """Load stripe distance matrix from cache file.
    
    Args:
        cache_path: Path to cache file
    
    Returns:
        Tuple of (stripe_distances, sample_ids, reference_sample_ids) if cache exists, None otherwise
    """
    if not cache_path.exists():
        return None
    
    try:
        cache_data = np.load(cache_path, allow_pickle=True)
        stripe_distances = cache_data["stripe_distances"]
        sample_ids = cache_data["sample_ids"].tolist()
        reference_sample_ids = cache_data["reference_sample_ids"].tolist()
        return stripe_distances, sample_ids, reference_sample_ids
    except Exception as e:
        logger.warning(f"Failed to load stripe distance matrix cache: {e}")
        return None


def clear_cache(cache_dir: Optional[str] = None, pattern: Optional[str] = None) -> int:
    """Clear cached distance matrices.
    
    Args:
        cache_dir: Cache directory (default: ~/.aam_cache/unifrac)
        pattern: Optional pattern to match cache files (e.g., "*unweighted*")
    
    Returns:
        Number of files deleted
    """
    cache_path = get_cache_path(cache_dir)
    
    if pattern:
        files = list(cache_path.glob(pattern))
    else:
        files = list(cache_path.glob("*.npz"))
    
    deleted = 0
    for file in files:
        try:
            file.unlink()
            deleted += 1
        except Exception as e:
            logger.warning(f"Failed to delete cache file {file}: {e}")
    
    logger.info(f"Cleared {deleted} cache files from {cache_path}")
    return deleted

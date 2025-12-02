"""BIOM table loader for microbial sequencing data."""

from typing import Optional, List
import biom
from biom import Table


class BIOMLoader:
    """Load and process BIOM tables for microbial sequencing data.

    This class provides functionality to:
    - Load BIOM format tables
    - Rarefy samples to consistent depth
    - Extract sequences from observation IDs
    """

    def __init__(self):
        """Initialize BIOMLoader."""
        pass

    def load_table(self, path: str) -> Table:
        """Load a BIOM table from file.

        Args:
            path: Path to BIOM file (.biom format)

        Returns:
            biom.Table object containing the loaded table

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file cannot be read as a BIOM table
        """
        try:
            table = biom.load_table(path)
            return table
        except FileNotFoundError:
            raise FileNotFoundError(f"BIOM file not found: {path}")
        except Exception as e:
            raise ValueError(f"Error loading BIOM table from {path}: {e}")

    def rarefy(
        self,
        table: Table,
        depth: int,
        with_replacement: bool = False,
        random_seed: Optional[int] = None,
        inplace: bool = False,
    ) -> Table:
        """Rarefy samples to a consistent depth.

        Uses BIOM table's built-in subsample method to rarefy samples.
        Samples with fewer than `depth` reads are dropped.

        Args:
            table: biom.Table object to rarefy
            depth: Target depth (number of reads) per sample
            with_replacement: If True, sample with replacement (default: False)
            random_seed: Random seed for reproducibility (default: None)
            inplace: If True, modify table in place (default: False)

        Returns:
            Rarefied biom.Table object

        Raises:
            ValueError: If the rarefied table is empty
        """
        if inplace:
            result_table = table
        else:
            result_table = table.copy()

        if with_replacement:
            result_table = result_table.filter(lambda v, i, m: v.sum() >= depth, inplace=False, axis="sample")

        result_table = result_table.subsample(
            depth, axis="sample", by_id=False, with_replacement=with_replacement, seed=random_seed
        )

        if result_table.is_empty():
            raise ValueError(
                "The rarefied table contains no samples or features. "
                "Verify your table is valid and that you provided a "
                "shallow enough sampling depth."
            )

        return result_table

    def get_sequences(self, table: Table) -> List[str]:
        """Extract sequences for ASVs from observation IDs.

        Since BIOM tables typically don't contain sequence metadata, sequences
        are extracted from observation IDs (ASV IDs), which often contain the
        sequence itself.

        Args:
            table: biom.Table object

        Returns:
            List of sequence strings, one per ASV (in order of observation_ids)
        """
        observation_ids = table.ids(axis="observation")
        return list(observation_ids)

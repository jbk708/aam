from __future__ import annotations

import os
from typing import Iterable, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table
from biom.util import biom_open
from skbio import DistanceMatrix
from sklearn import preprocessing
from unifrac import faith_pd, unweighted

from aam.data_handlers.generator_dataset import GeneratorDataset, add_lock


class CombinedGenerator(GeneratorDataset):
    taxon_field = "Taxon"
    levels = [f"Level {i}" for i in range(1, 8)]
    taxonomy_fn = "taxonomy.tsv"

    def __init__(self, tree_path: str, taxonomy, tax_level, **kwargs):
        super().__init__(**kwargs)
        self.tree_path = tree_path
        if self.batch_size % 2 != 0:
            raise Exception("Batch size must be multiple of 2")

        self.tax_level = f"Level {tax_level}"
        self.taxonomy = taxonomy

        self.encoder_target = self._create_encoder_target(self.rarefy_table)
        self.encoder_dtype = (np.float32, np.float32, np.float32, np.int32)
        self.encoder_output_type = (
            tf.TensorSpec(shape=[self.batch_size, self.batch_size], dtype=tf.float32),
            tf.TensorSpec(shape=[self.batch_size, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[self.batch_size, None], dtype=tf.int32),
        )

    def _create_encoder_target(self, table: Table) -> DistanceMatrix:
        if not hasattr(self, "tree_path"):
            return None

        random = np.random.random(1)[0]
        temp_path = f"/tmp/temp{random}.biom"
        with biom_open(temp_path, "w") as f:
            table.to_hdf5(f, "aam")
        uni = unweighted(temp_path, self.tree_path)
        faith = faith_pd(temp_path, self.tree_path)
        os.remove(temp_path)

        obs = table.ids(axis="observation")
        tax = self.taxonomy.loc[obs, "token"]
        return (uni, faith, tax)

    def _encoder_output(
        self,
        encoder_target: Iterable[object],
        sample_ids: Iterable[str],
        obs_ids: list[str],
    ) -> np.ndarray[float]:
        uni, faith, tax = encoder_target
        uni = uni.filter(sample_ids).data
        faith = faith.loc[sample_ids].to_numpy().reshape((-1, 1))

        tax_tokens = [tax.loc[obs] for obs in obs_ids]
        max_len = max([len(tokens) for tokens in tax_tokens])
        tax_tokens = np.array([np.pad(t, [[0, max_len - len(t)]]) for t in tax_tokens])
        return (uni, faith, tax_tokens)

    @property
    def taxonomy(self) -> pd.DataFrame:
        return self._taxonomy

    @taxonomy.setter
    @add_lock
    def taxonomy(self, taxonomy: Union[str, pd.DataFrame]):
        if hasattr(self, "_taxon_set"):
            raise Exception("Taxon already set")
        if taxonomy is None:
            self._taxonomy = taxonomy
            return

        if isinstance(taxonomy, str):
            taxonomy = pd.read_csv(taxonomy, sep="\t", index_col=0)
            if self.taxon_field not in taxonomy.columns:
                raise Exception("Invalid taxonomy: missing 'Taxon' field")

        taxonomy[self.levels] = taxonomy[self.taxon_field].str.split("; ", expand=True)
        taxonomy = taxonomy.loc[self._table.ids(axis="observation")]

        if self.tax_level not in self.levels:
            raise Exception(f"Invalid level: {self.tax_level}")

        level_index = self.levels.index(self.tax_level)
        levels = self.levels[: level_index + 1]
        taxonomy = taxonomy.loc[:, levels]
        taxonomy.loc[:, "class"] = taxonomy.loc[:, levels].agg("; ".join, axis=1)

        le = preprocessing.LabelEncoder()
        taxonomy.loc[:, "token"] = le.fit_transform(taxonomy["class"])
        taxonomy.loc[:, "token"] += 1  # shifts tokens to be between 1 and n
        print(
            "min token:", min(taxonomy["token"]), "max token:", max(taxonomy["token"])
        )
        self.num_tokens = (
            max(taxonomy["token"]) + 1
        )  # still need to add 1 to account for shift
        self._taxonomy = taxonomy


if __name__ == "__main__":
    import numpy as np

    from aam.data_handlers import CombinedGenerator

    ug = CombinedGenerator(
        table="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom",
        tree_path="/home/kalen/aam-research-exam/research-exam/agp/data/agp-aligned.nwk",
        taxonomy="/home/kalen/aam-research-exam/research-exam/healty-age-regression/taxonomy.tsv",
        tax_level=7,
        metadata="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-healthy.txt",
        metadata_column="host_age",
        shift=0.0,
        scale=100.0,
        gen_new_tables=True,
    )
    data = ug.get_data()
    for i, (x, y) in enumerate(data["dataset"]):
        print(y)
        break

    # data = ug.get_data_by_id(ug.rarefy_tables.ids()[:16])
    # for x, y in data["dataset"]:
    #     print(y)
    #     break

import numpy as np

from aam.data_handlers.generator_dataset import GeneratorDataset


def test_generator_dataset():
    """Generator should output a single
    tensor for y. Both iteration should also
    return same value.
    """

    table = "/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom"
    metadata = "/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-healthy.txt"
    generator = GeneratorDataset(
        table=table,
        metadata=metadata,
        metadata_column="host_age",
        gen_new_tables=False,
        shuffle=False,
        epochs=1,
    )

    table = generator.table
    metadata = generator.metadata
    assert table.shape[1] == metadata.shape[0]
    assert np.all(np.equal(table.ids(), metadata.index))

    batch_size = 8
    s_ids = table.ids()[: 5 * batch_size]
    y_true = metadata.loc[s_ids]

    data1 = generator.get_data()
    data2 = generator.get_data()

    data_ys = []
    for ((token1, count1), y1), ((token2, count2), y2) in zip(
        data1["dataset"].take(5), data2["dataset"].take(5)
    ):
        assert np.all(np.equal(token1.numpy(), token2.numpy()))
        assert np.all(np.equal(count1.numpy(), count2.numpy()))
        assert np.all(np.equal(y1.numpy(), y2.numpy()))
        data_ys.append(y1)

    data_ys = np.concatenate(data_ys)
    assert np.all(np.equal(y_true.to_numpy().reshape(-1), data_ys.reshape(-1)))

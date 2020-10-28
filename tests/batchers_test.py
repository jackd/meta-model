import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from meta_model import batchers


class BatchersTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        batchers.RaggedBatcher, batchers.PaddedRaggedBatcher, batchers.RaggedBatcherV2
    )
    def test_padded_ragged(self, batcher_cls):
        batcher = batcher_cls(batch_size=4)
        ds = tf.data.Dataset.range(8).map(tf.range)
        ds = batcher(ds)
        elements = list(ds)
        assert len(elements) == 2
        for el in elements:
            assert isinstance(el, tf.RaggedTensor)
        np.testing.assert_equal(elements[0].values.numpy(), [0, 0, 1, 0, 1, 2])
        np.testing.assert_equal(elements[0].row_splits.numpy(), [0, 0, 1, 3, 6])

        np.testing.assert_equal(
            elements[1].values.numpy(),
            [*list(range(4)), *list(range(5)), *list(range(6)), *list(range(7))],
        )
        np.testing.assert_equal(elements[1].row_splits.numpy(), [0, 4, 9, 15, 22])


if __name__ == "__main__":
    tf.test.main()

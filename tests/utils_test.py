import tensorflow as tf
from absl.testing import parameterized

from meta_model import utils


def assert_specs_equal(actual, expected):
    assert tuple(actual._shape) == tuple(expected._shape)
    assert actual._dtype == expected._dtype
    assert type(actual) == type(expected)
    if isinstance(actual, tf.RaggedTensorSpec):
        assert actual._ragged_rank == expected._ragged_rank


class UtilsTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        ((None, 3), tf.int32), ((2,), tf.float64), ((2, None), tf.float32),
    )
    def test_tensor_placeholder(self, shape, dtype):
        spec = tf.TensorSpec(shape=shape, dtype=dtype)
        placeholder = utils.placeholder(spec)
        assert tuple(placeholder.shape) == shape
        assert placeholder.dtype == dtype

    @parameterized.parameters(
        ((2, None), tf.int32, 1),
        ((None, None), tf.int64, 1),
        ((None, None, 2), tf.float64, 1),
        ((None, None, None), tf.float64, 2),
    )
    def test_ragged_placeholder(self, shape, dtype, ragged_rank):
        expected = tf.RaggedTensorSpec(shape, ragged_rank=ragged_rank, dtype=dtype)
        placeholder = utils.placeholder(expected)
        assert_specs_equal(utils.type_spec(placeholder), expected)
        assert tuple(placeholder.shape) == shape
        assert placeholder.dtype == dtype

    @parameterized.parameters(
        ((None, None), tf.int32),
        ((None, 3), tf.int32),
        ((2,), tf.float64),
        ((2, None), tf.float32),
    )
    def test_sparse_placeholder(self, shape, dtype):
        spec = tf.SparseTensorSpec(shape, dtype)
        placeholder = utils.placeholder(spec)
        assert tuple(placeholder.shape) == shape
        assert placeholder.dtype == dtype

    # @parameterized.parameters(
    #     (tf.TensorSpec((2,), tf.float64), 3, None, tf.TensorSpec, (3, 2)),
    #     (tf.TensorSpec((None,), tf.float64), 3, False, tf.TensorSpec, (3, None)),
    #     (tf.TensorSpec((None,), tf.float64), None, False, tf.TensorSpec, (None, None)),
    #     (
    #         tf.TensorSpec((None,), tf.float64),
    #         None,
    #         True,
    #         tf.RaggedTensorSpec,
    #         (None, None),
    #     ),
    #     (
    #         tf.RaggedTensorSpec((None, None), tf.float32),
    #         None,
    #         None,
    #         tf.RaggedTensorSpec,
    #         (None, None, None),
    #     ),
    #     (
    #         tf.SparseTensorSpec((2, 3), tf.float64),
    #         None,
    #         None,
    #         tf.SparseTensorSpec,
    #         (None, 2, 3),
    #     ),
    # )
    # def test_batched_spec(self, spec, batch_size, ragged, expected_cls, expected_shape):
    #     actual = utils.batched_spec(spec, batch_size=batch_size, ragged=ragged)
    #     assert tuple(actual._shape) == expected_shape
    #     assert isinstance(actual, expected_cls)
    #     assert actual._dtype == spec._dtype

    @parameterized.parameters(
        (
            tf.keras.Input((3,), batch_size=2, dtype=tf.float64),
            tf.TensorSpec((2, 3), tf.float64),
        ),
        (
            tf.keras.Input(shape=(None,), batch_size=3, ragged=True, dtype=tf.float64),
            tf.RaggedTensorSpec((3, None), tf.float64),
        ),
        (
            tf.keras.Input(shape=(4,), batch_size=3, sparse=True, dtype=tf.float64),
            tf.SparseTensorSpec((3, 4), tf.float64),
        ),
    )
    def test_type_spec(self, x, expected):
        assert_specs_equal(utils.type_spec(x), expected)


if __name__ == "__main__":
    tf.test.main()

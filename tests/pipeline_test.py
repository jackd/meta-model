import numpy as np
import tensorflow as tf

from meta_model import batchers
from meta_model import pipeline as pl


def pre_cache_map(xy, label):
    return xy, label


def pre_batch_map(xy, label):
    x, y = xy
    return x * y, label


def post_batch_map(z, label):
    f = z ** 2
    return (f,), label


def model_fn(z):
    return tf.keras.layers.Dense(2, kernel_initializer="ones")(z)


def build_fn(xy, label):
    xy, label = pre_cache_map(xy, label)
    xy, label = tf.nest.map_structure(pl.cache, (xy, label))
    args = pre_batch_map(xy, label)
    args = tf.nest.map_structure(pl.batch, args)
    features, labels = post_batch_map(*args)
    model_inp = tf.nest.map_structure(pl.model_input, features)
    model_out = model_fn(*model_inp)
    return model_out, labels


class MultiBuilderTest(tf.test.TestCase):
    def test_build_pipelined_model(self):
        batch_size = 3
        x = tf.random.uniform(shape=(100, 5), dtype=tf.float32)
        y = tf.random.uniform(shape=(100, 5), dtype=tf.float32)
        labels = tf.random.uniform(shape=(100,), maxval=10, dtype=tf.int64)
        dataset = tf.data.Dataset.from_tensor_slices(((x, y), labels))

        batcher = batchers.RectBatcher(batch_size=batch_size)
        pipeline, model = pl.build_pipelined_model(
            build_fn, dataset.element_spec, batcher=batcher,
        )
        processed = dataset.map(pipeline.pre_cache_map_func())
        processed = processed.map(pipeline.pre_batch_map_func())
        processed = processed.apply(batcher)
        processed = processed.map(pipeline.post_batch_map_func())
        actual_z = None
        actual_labels = None
        for actual_z, actual_labels in processed.take(1):
            pass
        actual_out = model(actual_z)

        # expected
        processed = (
            dataset.map(pre_cache_map)
            .map(pre_batch_map)
            .apply(batcher)
            .map(post_batch_map)
        )
        expected_z = None
        expected_labels = None
        for expected_z, expected_labels in processed.take(1):
            break
        expected_out = model_fn(*expected_z)

        # compare
        actual_out, actual_label, expected_out, expected_label = self.evaluate(
            (actual_out, actual_labels, expected_out, expected_labels)
        )
        np.testing.assert_allclose(actual_out, expected_out)
        np.testing.assert_allclose(actual_label, expected_label)

    def test_ragged(self):
        batch_size = 2

        def gen():
            return ((np.arange(i, dtype=np.float32),) for i in range(6))

        dataset = tf.data.Dataset.from_generator(gen, (tf.float32,), ((None,),))

        def build_fn(x):
            x = pl.cache(x)
            x = pl.batch(x)
            x = tf.keras.layers.Lambda(lambda x: x + 1)(x)
            return pl.model_input(x), (), ()

        batcher = batchers.RaggedBatcher(batch_size)
        pipeline, model = pl.build_pipelined_model(
            build_fn, dataset.element_spec, batcher=batcher
        )
        dataset = dataset.map(pipeline.pre_cache_map_func())
        dataset = dataset.map(pipeline.pre_batch_map_func())
        dataset = batcher(dataset)
        dataset = dataset.map(pipeline.post_batch_map_func())

        expected_flat = [1], [1, 2, 1, 2, 3], [1, 2, 3, 4, 1, 2, 3, 4, 5]
        for ((actual,), _, __), expected in zip(dataset, expected_flat):
            np.testing.assert_equal(actual.values.numpy(), expected)


if __name__ == "__main__":
    tf.test.main()

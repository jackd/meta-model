import abc

import tensorflow as tf

from meta_model import utils


@utils.register_serializable
class Batcher(abc.ABC):
    def __init__(self, batch_size: int, drop_remainder: bool = False):
        assert isinstance(batch_size, int)
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder
        self._static_batch_size = self._batch_size if self._drop_remainder else None

    def get_config(self):
        return dict(batch_size=self._batch_size, drop_remainder=self._drop_remainder)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @abc.abstractmethod
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        raise NotImplementedError("Abstract method")


@utils.register_serializable
class RectBatcher(Batcher):
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.batch(self._batch_size, self._drop_remainder)


@utils.register_serializable
class RaggedBatcher(Batcher):
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(
                self._batch_size, self._drop_remainder
            )
        )


@utils.register_serializable
class PaddedRaggedBatcher(RaggedBatcher):
    """
    `RaggedBatcher` implemented with `Dataset.padded_batch` and `Dataset.map`s.

    Tensors have the leading dimension recorded prior to batching, and the batched
    version is truncated using `tf.RaggedTensor.from_tensor` following batching.
    """

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        specs = dataset.element_spec

        def pre_batch_map_func(*args):
            if len(args) == 1:
                (args,) = args
            sizes = tf.nest.map_structure(
                lambda x: x.nrows()
                if isinstance(x, tf.RaggedTensor)
                else tf.shape(x)[0]
                if x.shape.ndims > 0
                else tf.zeros((), dtype=tf.int32),
                args,
            )
            return args, sizes

        def post_batch_map_func(args, sizes):
            return tf.nest.map_structure(
                lambda arg, size, spec: tf.RaggedTensor.from_tensor(arg, size)
                if spec.shape[0] is None and isinstance(spec, tf.TensorSpec)
                else arg,
                args,
                sizes,
                specs,
            )

        return (
            dataset.map(pre_batch_map_func)
            .padded_batch(self._batch_size, drop_remainder=self._drop_remainder)
            .map(post_batch_map_func)
        )


@utils.register_serializable
class RaggedBatcherV2(RaggedBatcher):
    """
    `RaggedBatcher` implemented with `Dataset.batch` and `Dataset.map`s.

    Tensors with dynamic leading dimension are mapped to ragged tensors prior
    to batching, then mapped to ragged tensors with ragged_rank==1 after batching.
    """

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        spec = dataset.element_spec
        flat_spec = tf.nest.flatten(spec)
        applicable = [
            isinstance(x, tf.TensorSpec) and x.shape.ndims > 0 and x.shape[0] is None
            for x in flat_spec
        ]

        def pre_batch_ragged(
            tensor: tf.Tensor, row_splits_dtype=tf.int64
        ) -> tf.RaggedTensor:
            return tf.RaggedTensor.from_tensor(
                tf.expand_dims(tensor, axis=0), row_splits_dtype=row_splits_dtype
            )

        def post_batch_ragged(rt: tf.RaggedTensor, validate=True) -> tf.RaggedTensor:
            return tf.RaggedTensor.from_nested_row_splits(
                rt.flat_values, rt.nested_row_splits[1:], validate=validate
            )

        def pre_batch_map_func(*args):
            args = tf.nest.flatten(args)
            args = [
                pre_batch_ragged(x) if app else x for x, app in zip(args, applicable)
            ]
            return args

        def post_batch_map_func(*args):
            args = [
                post_batch_ragged(x) if app else x for x, app in zip(args, applicable)
            ]
            return tf.nest.pack_sequence_as(spec, args)

        return (
            dataset.map(pre_batch_map_func)
            .batch(self._batch_size, drop_remainder=self._drop_remainder)
            .map(post_batch_map_func)
        )


def get(identifier):
    if isinstance(identifier, Batcher):
        return identifier
    out = tf.keras.utils.deserialize_keras_object(identifier)
    if not isinstance(out, Batcher):
        raise ValueError(f"Unknown batcher: {out}")
    return out

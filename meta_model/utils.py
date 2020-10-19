from typing import Callable, Optional

import tensorflow as tf

from meta_model.types import TensorLike, TensorLikeSpec

if tf.version.VERSION >= "2.4":

    def _spec_components(spec: TensorLikeSpec):
        return spec.shape, spec.dtype

    def type_spec(x) -> TensorLikeSpec:
        assert tf.keras.backend.is_keras_tensor(x)
        return x.type_spec

    def batched_spec(
        spec: TensorLikeSpec,
        batch_size: Optional[int] = None,
        drop_remainder: bool = False,
        ragged: bool = False,
    ):
        kwargs = dict(batch_size=batch_size, drop_remainder=drop_remainder)
        dataset = tf.data.Dataset.from_generator(lambda: None, output_signature=spec)
        if ragged:
            dataset = dataset.apply(
                tf.data.experimental.dense_to_ragged_batch(**kwargs)
            )
        else:
            dataset = dataset.batch(**kwargs)
        return dataset.element_spec


else:

    def _spec_components(spec: TensorLikeSpec):
        # ragged tensors don't have shape or dtype properties
        return spec._shape, spec._dtype  # pylint:disable=protected-access

    def type_spec(x: TensorLike) -> TensorLikeSpec:
        """Get the spec associated with the given [Sparse|Ragged]?Tensor."""
        if isinstance(x, (tf.Tensor, tf.Variable)):
            return tf.TensorSpec(shape=x.shape, dtype=x.dtype)
        else:
            # composite tensor
            return x._type_spec  # pylint:disable=protected-access

    def batched_spec(
        spec: TensorLikeSpec,
        batch_size: Optional[int] = None,
        drop_remainder: bool = False,
        ragged: Optional[bool] = None,
    ) -> TensorLikeSpec:
        """Get the [Sparse|Ragged]?TensorSpec associated with the batched spec."""
        shape, dtype = _spec_components(spec)
        if not drop_remainder:
            batch_size = None
        if isinstance(spec, tf.RaggedTensorSpec):
            shape = [batch_size, None, *shape[1:]]
            return tf.RaggedTensorSpec(
                shape,
                dtype,
                row_splits_dtype=spec._row_splits_dtype,  # pylint:disable=protected-access
            )
        if isinstance(spec, tf.SparseTensorSpec):
            return tf.SparseTensorSpec((batch_size, *shape), dtype)
        if isinstance(spec, tf.TensorSpec):
            if len(shape) > 0 and shape[0] is None:
                if ragged is None:
                    raise ValueError(
                        "`ragged` must be provided if leading dimension is not statically "
                        "known"
                    )
                if ragged:
                    return tf.RaggedTensorSpec((batch_size, *shape), dtype=dtype)
            return tf.TensorSpec((batch_size, *shape), dtype)

        raise TypeError(f"Invalid spec {spec}")


def placeholder(spec: TensorLikeSpec, name: Optional[str] = None):
    shape, dtype = _spec_components(spec)
    return tf.keras.backend.placeholder(
        shape=shape,
        dtype=dtype,
        name=name,
        ragged=isinstance(spec, tf.RaggedTensorSpec),
        sparse=isinstance(spec, tf.SparseTensorSpec),
    )


def placeholder_like(x: TensorLike, name: Optional[str] = None):
    return placeholder(type_spec(x), name=name)


def model_fn(inputs, outputs) -> Callable:
    """
    Get a callable based on a `tf.keras.Model` with structured inputs / outputs.

    Args:
        inputs: nested structure of keras inputs.
        outputs: nested structure of output tensors.

    Returns:
        Callable that maps *args -> outputs, where args must have the same structure
            as inputs.
    """

    def assert_compatible(keras_tensor, x):
        spec = type_spec(keras_tensor)
        if not spec.is_compatible_with(x):
            raise ValueError(f"Tensor {x} is not compatible with spec {spec}")

    model = tf.keras.Model(tf.nest.flatten(inputs), tf.nest.flatten(outputs))

    @tf.function
    def fn(*args, validate: bool = False):
        if isinstance(inputs, dict):
            assert len(args) == 1
            (args,) = args
        tf.nest.assert_same_structure(inputs, args)
        if validate:
            tf.nest.map_structure(assert_compatible, inputs, args)
        flat_out = model(tf.nest.flatten(args))

        if isinstance(flat_out, (tf.Tensor, tf.RaggedTensor, tf.SparseTensor)):
            flat_out = (flat_out,)
        return tf.nest.pack_sequence_as(outputs, flat_out)

    return fn


register_serializable = tf.keras.utils.register_keras_serializable(package="MetaModel")

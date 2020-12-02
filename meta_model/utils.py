from typing import Callable, Optional, Tuple

import tensorflow as tf

from meta_model.types import TensorLike, TensorLikeStruct, TypeSpecStruct


def _always_throw(*args, **kwargs):
    raise RuntimeError("This function always throws")


if tf.version.VERSION >= "2.4":

    def _spec_components(spec: tf.TypeSpec) -> Tuple[tf.TensorShape, tf.DType]:
        return spec.shape, spec.dtype

    def _spec_to_dataset(spec: TypeSpecStruct) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(_always_throw, output_signature=spec)

    def type_spec(x: TensorLike) -> tf.TypeSpec:
        if tf.keras.backend.is_keras_tensor(x):
            return x.type_spec
        if isinstance(x, tf.Tensor):
            return tf.TensorSpec.from_tensor(x)
        if isinstance(x, tf.RaggedTensor):
            return tf.RaggedTensorSpec.from_value(x)
        if isinstance(x, tf.SparseTensor):
            return tf.SparseTensorSpec.from_value(x)
        raise TypeError(f"Unrecognized TensorLike {x}")


else:

    def _spec_components(spec: tf.TypeSpec) -> Tuple[tf.TensorShape, tf.DType]:
        # ragged tensors don't have shape or dtype properties
        return spec._shape, spec._dtype  # pylint:disable=protected-access

    def _spec_to_dataset(spec: TypeSpecStruct) -> tf.data.Dataset:
        flat_spec = tf.nest.flatten(spec, expand_composites=True)
        return tf.data.Dataset.from_generator(
            _always_throw,
            tuple(s.dtype for s in flat_spec),
            tuple(s.shape for s in flat_spec),
        ).map(
            lambda *args: tf.nest.pack_sequence_as(spec, args, expand_composites=True)
        )

    def type_spec(x: TensorLike) -> tf.TypeSpec:
        """Get the spec associated with the given [Sparse|Ragged]?Tensor."""
        if isinstance(x, (tf.Tensor, tf.Variable)):
            return tf.TensorSpec(shape=x.shape, dtype=x.dtype)
        else:
            # composite tensor
            return x._type_spec  # pylint:disable=protected-access


def placeholder(spec: tf.TypeSpec, name: Optional[str] = None) -> TensorLike:
    shape, dtype = _spec_components(spec)
    return tf.keras.backend.placeholder(
        shape=shape,
        dtype=dtype,
        name=name,
        ragged=isinstance(spec, tf.RaggedTensorSpec),
        sparse=isinstance(spec, tf.SparseTensorSpec),
    )


def placeholder_like(x: TensorLike, name: Optional[str] = None) -> TensorLike:
    return placeholder(type_spec(x), name=name)


def transformed_spec(
    transform: Callable[[tf.data.Dataset], tf.data.Dataset],
    spec: TypeSpecStruct,
) -> TypeSpecStruct:
    """
    Get the spec resulting from applying `transform` to a dataset with spec `spec`.

    Args:
        transform: dataset transform
        spec: TensorSpec (or composite), or possibly nested structure of them.

    Returns:
        TensorSpec (or composite), or nested structure of them.
    """
    return transform(_spec_to_dataset(spec)).element_spec


register_serializable = tf.keras.utils.register_keras_serializable(package="MetaModel")


@register_serializable
class ModelMap:
    """
    Serializable wrapper that converts a `tf.keras.Model` into `fn(*args, training=xx)`.

    This is required for use with `tf.data.Dataset.map`.

    Example usage:
    ```python
    out = dataset.map(ModelMap(model, training=False), num_parallel_calls=...)
    ```
    equivalent to
    ```python
    def map_fn(*args, training=False):
        if len(args) == 1:
            args, = args
        return model(args, training=training)

    out = dataset.map(functools.partial(map_fn, training=xx))
    ```
    """

    def __init__(self, model: tf.keras.Model, training: bool = False):
        if not isinstance(model, tf.keras.Model):
            model = tf.keras.utils.deserialize_keras_object(
                model, module_object={"Functional": tf.keras.Model}
            )
        assert isinstance(model, tf.keras.Model)
        self._model = model
        self._training = training

    def __call__(self, *args) -> TensorLikeStruct:
        if len(args) == 1:
            (args,) = args
        return self._model(tf.nest.flatten(args), training=self._training)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return dict(
            model=tf.keras.utils.serialize_keras_object(self._model),
            training=self._training,
        )

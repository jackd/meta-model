from typing import TypeVar, Union

import tensorflow as tf

KerasTensor = TypeVar("KerasTensor")

TensorLike = Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor, tf.Variable]
TensorLikeSpec = Union[tf.TensorSpec, tf.RaggedTensorSpec, tf.SparseTensorSpec]
TensorOrVariable = Union[tf.Tensor, tf.Variable]

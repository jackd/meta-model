from typing import Dict, List, Tuple, TypeVar, Union

import tensorflow as tf

KerasTensor = TypeVar("KerasTensor")

TensorLike = Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor, tf.Variable]
TensorOrVariable = Union[tf.Tensor, tf.Variable]

TensorLikeStruct = Union[
    TensorLike,
    List["TensorLikeStruct"],
    Tuple["TensorLikeStruct", ...],
    Dict[str, "TensorLikeStruct"],
]

TypeSpecStruct = Union[
    tf.TypeSpec,
    List["TypeSpecStruct"],
    Tuple["TypeSpecStruct"],
    Dict[str, "TypeSpecStruct"],
]

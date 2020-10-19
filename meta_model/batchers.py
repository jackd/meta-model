import abc
from typing import Optional

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

    def epoch_length(self, example_epoch_length: Optional[int]) -> Optional[int]:
        if example_epoch_length is None:
            return None
        epoch_length = example_epoch_length // self._batch_size
        if self._drop_remainder or example_epoch_length % self._batch_size == 0:
            return epoch_length
        return epoch_length + 1


@utils.register_serializable
class RectBatcher(Batcher):
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.batch(self._batch_size, self._drop_remainder)

    def batched_spec(self, spec):
        return utils.batched_spec(
            spec,
            batch_size=self._batch_size,
            drop_remainder=self._drop_remainder,
            ragged=False,
        )


@utils.register_serializable
class RaggedBatcher(Batcher):
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(
                self._batch_size, self._drop_remainder
            )
        )

    def batched_spec(self, spec):
        return utils.batched_spec(
            spec,
            batch_size=self._batch_size,
            drop_remainder=self._drop_remainder,
            ragged=True,
        )

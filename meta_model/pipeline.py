from typing import Callable, NamedTuple, Optional

import tensorflow as tf

from meta_model import utils
from meta_model.batchers import Batcher
from meta_model.types import TensorLike


class Pipeline(NamedTuple):
    pre_cache_map: Callable
    pre_batch_map: Callable
    post_batch_map: Callable
    batcher: Batcher


class PipelinedModel(NamedTuple):
    pipeline: Pipeline
    model: tf.keras.Model


class PipelinedModelBuilder:
    def __init__(self, spec, batcher: Batcher):
        self._pre_cache_inputs = tf.nest.map_structure(utils.placeholder, spec)
        self._batcher = batcher
        self._pre_cache_outputs = []
        self._pre_batch_inputs = []
        self._pre_batch_outputs = []
        self._post_batch_inputs = []
        self._post_batch_outputs = []
        self._model_inputs = []

    @property
    def pre_cache_inputs(self):
        return self._pre_cache_inputs

    _stack = []

    @staticmethod
    def get_default() -> "PipelinedModelBuilder":
        if len(PipelinedModelBuilder._stack) == 0:
            raise RuntimeError("No PipelinedModelBuilder contexts open.")
        return PipelinedModelBuilder._stack[-1]

    def __enter__(self) -> "PipelinedModelBuilder":
        PipelinedModelBuilder._stack.append(self)
        return self

    def __exit__(self, *args, **kwargs):
        top = PipelinedModelBuilder._stack.pop()
        assert top is self

    def cache(self, x: TensorLike, name: Optional[str] = None) -> TensorLike:
        """Connect a tensor in the pre_cache graph to the pre_batch graph."""
        self._pre_cache_outputs.append(x)
        out = utils.placeholder_like(x, name=name)
        self._pre_batch_inputs.append(out)
        return out

    def batch(self, x: TensorLike, name: Optional[str] = None) -> TensorLike:
        """Connect a tensor in the pre_batch graph to the post_batch graph."""
        self._pre_batch_outputs.append(x)
        out = utils.placeholder(
            self._batcher.batched_spec(utils.type_spec(x)), name=name
        )
        self._post_batch_inputs.append(out)
        return out

    def model_input(self, x: TensorLike, name: Optional[str] = None) -> TensorLike:
        """Connect a tensor in the post_batch graph to the model graph."""
        self._post_batch_outputs.append(x)
        out = utils.placeholder_like(x, name=name)
        self._model_inputs.append(out)
        return out

    def build_pre_cache_map(self, cached_outputs=None):
        if cached_outputs is None:
            cached_outputs = self._pre_cache_outputs
        return utils.model_fn(self._pre_cache_inputs, cached_outputs)

    def build_pipeline(
        self, post_batch_outputs, batched_labels, batched_weights=None
    ) -> Pipeline:
        batched_outputs = tf.keras.utils.pack_x_y_sample_weight(
            post_batch_outputs, batched_labels, batched_weights
        )

        pre_cache_map = self.build_pre_cache_map()
        pre_batch_map = utils.model_fn(
            tuple(self._pre_batch_inputs), self._pre_batch_outputs
        )
        post_batch_map = utils.model_fn(tuple(self._post_batch_inputs), batched_outputs)
        return Pipeline(pre_cache_map, pre_batch_map, post_batch_map, self._batcher)

    def build(
        self, model_outputs, batched_labels, batched_weights=None,
    ) -> PipelinedModel:
        pipeline = self.build_pipeline(
            tuple(self._post_batch_outputs), batched_labels, batched_weights
        )
        model = tf.keras.Model(tuple(self._model_inputs), model_outputs)
        return PipelinedModel(pipeline, model)


get_default = PipelinedModelBuilder.get_default


def _default_docs(fn):
    fn.__doc__ = getattr(PipelinedModelBuilder, fn.__name__).__doc__
    return fn


@_default_docs
def cache(x: TensorLike, name=None) -> TensorLike:
    return get_default().cache(x, name=name)


@_default_docs
def batch(x: TensorLike, name: Optional[str] = None) -> TensorLike:
    return get_default().batch(x, name=name)


@_default_docs
def model_input(x: TensorLike, name=None) -> TensorLike:
    return get_default().model_input(x, name=name)


def build_pipelined_model(build_fn, element_spec, batcher: Batcher) -> PipelinedModel:
    with PipelinedModelBuilder(element_spec, batcher=batcher) as builder:
        inp = builder.pre_cache_inputs
        if isinstance(inp, (list, tuple)):
            outputs = build_fn(*inp)
        else:
            outputs = build_fn(inp)
        if len(outputs) == 2:
            model_outputs, labels = outputs
            weights = None
        else:
            model_outputs, labels, weights = outputs
    return builder.build(model_outputs, labels, weights)

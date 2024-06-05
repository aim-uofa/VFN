# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import argparse
import importlib
import os

from .distributed_unicore_model import DistributedUnicoreModel
from .unicore_model import (
    BaseUnicoreModel,
)

MODEL_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}
ARCH_MODEL_NAME_REGISTRY = {}
ARCH_MODEL_INV_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}


__all__ = [
    "BaseUnicoreModel",
    "DistributedUnicoreModel",
]


def build_model(args, task):
    return ARCH_MODEL_REGISTRY[args.arch].build_model(args, task)


def register_model(name):
    """
    New model types can be added to unicore with the :func:`register_model`
    function decorator.

    For example::

        @register_model("lstm")
        class LSTM(UnicoreEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseUnicoreModel` interface.
        Typically you will extend :class:`UnicoreEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`UnicoreLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        if not issubclass(cls, BaseUnicoreModel):
            raise ValueError("Model ({}: {}) must extend BaseUnicoreModel".format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


def register_model_architecture(model_name, arch_name):
    """
    New model architectures can be added to unicore with the
    :func:`register_model_architecture` function decorator. After registration,
    model architectures can be selected with the ``--arch`` command-line
    argument.

    For example::

        @register_model_architecture("lstm", "lstm_luong_wmt_en_de")
        def lstm_luong_wmt_en_de(args):
            args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1000)
            (...)

    The decorated function should take a single argument *args*, which is a
    :class:`argparse.Namespace` of arguments parsed from the command-line. The
    decorated function should modify these arguments in-place to match the
    desired architecture.

    Args:
        model_name (str): the name of the Model (Model must already be
            registered)
        arch_name (str): the name of the model architecture (``--arch``)
    """

    def register_model_arch_fn(fn):
        if model_name not in MODEL_REGISTRY:
            raise ValueError("Cannot register model architecture for unknown model type ({})".format(model_name))
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model architecture ({})".format(arch_name))
        if not callable(fn):
            raise ValueError("Model architecture must be callable ({})".format(arch_name))
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_MODEL_INV_REGISTRY.setdefault(model_name, []).append(arch_name)
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn

    return register_model_arch_fn


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith("_") and not file.startswith(".") and (file.endswith(".py") or os.path.isdir(path)):
        model_name = file[:file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("unicore.models." + model_name)

        # extra `model_parser` for sphinx
        if model_name in MODEL_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_archs = parser.add_argument_group("Named architectures")
            group_archs.add_argument("--arch", choices=ARCH_MODEL_INV_REGISTRY[model_name])
            group_args = parser.add_argument_group("Additional command-line arguments")
            MODEL_REGISTRY[model_name].add_args(group_args)
            globals()[model_name + "_parser"] = parser

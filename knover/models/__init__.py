#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Define model."""

import paddle

from knover.core.model import Model, ModelInterface

MODEL_REGISTRY = {}


__all__ = [
    "MODEL_REGISTRY",
    "register_model",
    "create_model",
    "add_cmdline_args"
]


def register_model(name):
    """Register a new model class."""

    def __wrapped__(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model ({name})")
        if not issubclass(cls, Model):
            raise ValueError(f"Model ({name}: {cls.__name__}) must extend Model")
        MODEL_REGISTRY[name] = cls
        return cls

    return __wrapped__


def create_model(args, place) -> ModelInterface:
    """Create a model."""
    model = MODEL_REGISTRY[args.model](args, place)
    return ModelInterface(args, model)


def add_cmdline_args(parser):
    """ Add cmdline argument of Model. """
    group = parser.add_argument_group("Model")

    # Model
    group.add_argument("--model", type=str, required=True,
                       help="The model type.",
                       choices=list(MODEL_REGISTRY.keys()))

    # Config
    group.add_argument("--config_path", type=str, required=True,
                       help="The path of model configuration.")

    # Model related.
    args, _ = parser.parse_known_args()
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {args.model}")
    MODEL_REGISTRY[args.model].add_cmdline_args(parser)
    return group


import knover.models.unified_transformer
import knover.models.recursive

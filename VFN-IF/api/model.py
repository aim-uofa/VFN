import torch.nn as nn
from .modules.denoise import Denosie
from .config import model_config


class DeModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        config = model_config(
            self.args.model_name,
            train=True,
        )
        self.model = Denosie(config)
        self.config = config

    def half(self):
        self.model = self.model.half()
        return self

    def bfloat16(self):
        self.model = self.model.bfloat16()
        return self

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args)

    def forward(self, batch, **kwargs):
        outputs = self.model.forward(batch)
        return outputs, self.config.loss

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--model-name",
            help="choose the model config",
        )


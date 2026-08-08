import torch.nn as nn

from MedTS_TTT import MedTSTTT


class Model(nn.Module):
    """
    Thin adapter for the Medformer / Time-Series-Library classification API.

    Medformer-style loaders usually provide x_enc with shape [B, T, C].
    The clean MedTS-TTT implementation expects [B, C, T], so this wrapper
    only handles the input transpose and config mapping.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.model = MedTSTTT(
            dim=configs.d_model,
            max_channel=getattr(configs, "max_channel", 128),
            num_heads=configs.n_heads,
            num_layers=configs.e_layers,
            patch_size=getattr(configs, "patch_len", 8),
            num_classes=configs.num_class,
        )

    def classification(self, x_enc):
        # Medformer benchmark: [B, T, C]
        # MedTS-TTT model:     [B, C, T]
        return self.model(x_enc.permute(0, 2, 1))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name != "classification":
            raise NotImplementedError("MedTS-TTT adapter currently supports classification only.")
        return self.classification(x_enc)

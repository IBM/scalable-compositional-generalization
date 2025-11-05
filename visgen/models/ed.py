import os
import torch

from visgen.utils.general import plot_box, plot_codebooks_similarity
from .base import BaseModel

class ExpDisentanglement(BaseModel):
    """
    Disentangled Model
    """
    def __init__(
        self,
        preprocessing: torch.nn.Module,
        feature_extractors: list,
        readouts: list,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.preprocessing = preprocessing
        self.feature_extractors = torch.nn.ModuleList(feature_extractors)
        self.readouts = torch.nn.ModuleList(readouts)

    def forward(self, x):
        output = []
        # Process in parallel all the disentangled paths.
        with torch.no_grad():
            x = self.preprocessing(x)
        for resnet, readout in zip(self.feature_extractors, self.readouts):
            feature = resnet(x)
            out = readout(feature)
            if isinstance(out, list):
                out = out[0]
            output.append(out)
        return output

    @torch.no_grad()
    def plot_debug(self, x, path, **kwargs):
        self.train()
        original = plot_box(
            img=x[22].permute(1, 2, 0).cpu().numpy(),
            path=os.path.join(path, "original.png"),
        )
        x_train = self.preprocessing(x)
        train_augm = plot_box(
            img=x_train[22].permute(1, 2, 0).cpu().numpy(),
            path=os.path.join(path, "train_augm.png"),
        )
        self.eval()
        x_test = self.preprocessing(x)
        test_augm = plot_box(
            img=x_test[22].permute(1, 2, 0).cpu().numpy(),
            path=os.path.join(path, "test_augm.png"),
        )
        # plot codebook similarities
        if hasattr(self.readouts[0], "codebooks"):
            intra_cb, inter_cb = plot_codebooks_similarity(
                [r.codebooks[0] for r in self.readouts],
                [r.attributes[0] for r in self.readouts],
            )
            return [original, train_augm, test_augm, intra_cb, inter_cb], [
                "original",
                "train_augm",
                "test_augm",
                "intra_codebook",
                "inter_codebook",
            ]
        else:
            return [original, train_augm, test_augm], [
                "original",
                "train_augm",
                "test_augm",
            ]


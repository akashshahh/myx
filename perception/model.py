"""PANNs Cnn14 perception model, re-headed for 8-dim mastering-issue regression.

The architecture is a faithful reproduction of qiuqiangkong/audioset_tagging_cnn's
`pytorch/models.py::Cnn14` so that the pretrained `Cnn14_mAP=0.431.pth` checkpoint
(Zenodo record 3987831) loads with an exact key match. We then swap the final
`fc_audioset` layer (527 AudioSet classes) for an 8-unit head matching
`data.degradation.ISSUE_DIMENSIONS`, keeping the model's built-in sigmoid so each
output is a per-issue severity in [0, 1] — the same target space the synthetic
labels live in.

Spectrogram hyperparameters are hard-coded to the values the checkpoint was
trained with; do not change them or the pretrained conv weights become invalid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

from data.degradation import ISSUE_DIMENSIONS

# --- Frozen frontend hyperparameters (must match the pretrained checkpoint) ---
SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN = 50
FMAX = 14000
AUDIOSET_CLASSES = 527  # original head width; used only to load pretrained weights
NUM_ISSUES = len(ISSUE_DIMENSIONS)  # 8


def init_layer(layer: nn.Module) -> None:
    """Xavier-uniform init for a linear/conv layer (matches upstream PANNs)."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module) -> None:
    """Init a BatchNorm layer to identity-ish (matches upstream PANNs)."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    """Two 3x3 conv + BN + ReLU, then pooling. Identical to upstream PANNs."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self) -> None:
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x: torch.Tensor, pool_size=(2, 2), pool_type="avg") -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise ValueError(f"unknown pool_type {pool_type!r}")
        return x


class Cnn14(nn.Module):
    """Faithful reproduction of PANNs Cnn14. Returns a dict like upstream so the
    pretrained state_dict loads key-for-key."""

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window="hann", center=True, pad_mode="reflect", freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True,
        )
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self) -> None:
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        """waveform: (batch, samples) mono @ 32 kHz."""
        x = self.spectrogram_extractor(waveform)  # (B, 1, T, freq_bins)
        x = self.logmel_extractor(x)              # (B, 1, T, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)              # pool freq
        (x1, _) = torch.max(x, dim=2)         # pool time (max)
        x2 = torch.mean(x, dim=2)             # pool time (mean)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class PerceptionModel(nn.Module):
    """Cnn14 backbone with the AudioSet head replaced by an 8-dim issue regressor.

    forward(waveform) -> (batch, 8) tensor in [0, 1], aligned to
    `ISSUE_DIMENSIONS`. Loads the pretrained Cnn14 weights when given a
    checkpoint, then re-heads. Backbone conv blocks are frozen by default so the
    first training epoch only fits the new head (per the plan)."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        num_issues: int = NUM_ISSUES,
        freeze_backbone: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.num_issues = num_issues

        self.backbone = Cnn14(
            sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, hop_size=HOP_SIZE,
            mel_bins=MEL_BINS, fmin=FMIN, fmax=FMAX, classes_num=AUDIOSET_CLASSES,
        )

        if checkpoint_path is not None:
            # Zenodo checkpoint is a dict {'model': state_dict, 'iteration': int}.
            # weights_only=True uses torch's restricted unpickler (tensors + basic
            # types only) — no arbitrary code execution from the downloaded file.
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            missing, unexpected = self.backbone.load_state_dict(state, strict=True)
            if missing or unexpected:  # strict=True already raises; belt-and-suspenders
                raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")

        # Re-head: 527 AudioSet classes -> 8 mastering issues. Keep sigmoid (in forward).
        self.backbone.fc_audioset = nn.Linear(2048, num_issues, bias=True)
        init_layer(self.backbone.fc_audioset)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze conv blocks + bn0; leave fc1 and the issue head trainable."""
        for name, p in self.backbone.named_parameters():
            p.requires_grad = name.startswith(("fc1", "fc_audioset"))

    def unfreeze_backbone(self) -> None:
        """Make all backbone parameters trainable (for the fine-tune epochs)."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (batch, samples) mono @ 32 kHz -> (batch, num_issues) in [0,1]."""
        return self.backbone(waveform)["clipwise_output"]


def build_perception_model(
    checkpoint_path: str | None = None,
    freeze_backbone: bool = True,
    device: str | torch.device = "cpu",
) -> PerceptionModel:
    """Convenience constructor matching the plan's API.

    `checkpoint_path` here is the PRETRAINED AudioSet Cnn14 (.pth with 527-class
    head). For a checkpoint produced by our own training, use
    `load_finetuned_perception` instead."""
    model = PerceptionModel(
        checkpoint_path=checkpoint_path,
        num_issues=NUM_ISSUES,
        freeze_backbone=freeze_backbone,
        device=device,
    )
    return model.to(device)


def load_finetuned_perception(
    checkpoint_path: str,
    device: str | torch.device = "cpu",
) -> PerceptionModel:
    """Load a checkpoint produced by `perception/train.py` (8-dim head).

    The saved state_dict has 'backbone.*' keys with an 8-unit fc_audioset, so it
    loads into a fresh PerceptionModel (not the 527-class pretrained path)."""
    model = PerceptionModel(checkpoint_path=None, freeze_backbone=False, device=device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()

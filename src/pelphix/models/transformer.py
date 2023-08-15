from typing import Any, List, Tuple, Optional
import torch
from torch import nn
import logging
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import lightning.pytorch as pl
from torchvision.models import resnet18
from hydra.utils import instantiate
from .unet import UNet

log = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    """Container module with an encoder, a transformer module, and a decoder.

    Transformer model

    """

    def __init__(
        self,
        d_label: int,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 6,
        dropout=0.5,
    ):
        """Transformer model with a positional encoder.

        This model does not have an encoding layer. As such, the input tokens should already be
        have shape `(sequence_length, batch_size, d_model)`. The outputs are of shape
        `(sequence_length, batch_size, d_label)`.

        Args:
            d_label: Output dimension of the decoder. That is, the (combined) number of classes.
            d_model (int): Embedding dimension.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network model.
            num_layers (int): Number of sub-encoder-layers in the encoder.

        """
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.ninp = d_model
        self.decoder = nn.Linear(d_model, d_label)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src: torch.Tensor, has_mask=True, src_key_padding_mask=None):
        """Pass the input through the encoder layers in turn.

        Args:
            src (torch.Tensor): The sequence to the encoder (required). (seq, batch, feature)

        """
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src, self.src_mask, src_key_padding_mask=src_key_padding_mask
        )
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


class RecognitionModel(nn.Module):
    def __init__(
        self,
        supercategory_num_classes: list[int],
        d_input: Optional[int] = None,
        transformer: dict[str, Any] = dict(),
        input_images: bool = False,
    ):
        """Recognition model.

        Args:
            d_input: (int): Dimension of the input features. So the input should have shape (seq, batch, d_input)
                Required if input_images is False.

            supercategory_nun_classes (list[int]): List of number of classes for each sequence label
                *including the background class*. For example, for surgical phase recognition, there
                are [8, 3, 8, 2] non-bg classes for task, activity, acquisition, and frame
                classification, respectively, so we should pass in [9, 4, 9, 3].
            backbone_model (Optional[nn.Module], optional): Pre-trained image encoder model, from
                which we only use backbone_model.backbone.
            keypoint_model (Optional[nn.Module], optional): Pre-trained keypoint detection model. If present, we use
                this to extract keypoints from the input image, which are then encoded and passed into the .
            input_images: Whether the input is an image or a feature vector.

        """
        super(RecognitionModel, self).__init__()

        self.supercategory_num_classes = list(supercategory_num_classes)
        self.num_classes = sum(supercategory_num_classes)
        self.input_images = input_images
        self.transformer = Transformer(self.num_classes, **transformer)

        if input_images:
            # input size
            # output size
            resnet = resnet18(pretrained=True, progress=True)
            self.encoder = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        else:
            assert d_input is not None
            self.encoder = nn.Sequential(
                nn.Linear(d_input, transformer["d_model"]),
                nn.ReLU(),
            )

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder[0].weight, -initrange, initrange)

    def forward(
        self, x: torch.Tensor, has_mask=False, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> list[torch.Tensor]:
        """Forward pass of the recognition model.

        Args:
            x (torch.Tensor): Input feature tensor of shape (S, N, E).
                If input_images is True, then shape is (S, N, C, H, W)
            has_mask (bool, optional): Whether to use a mask for the transformer. Defaults to True.

        Returns:
            tuple[torch.Tensor]: Tuple of output tensors, one for each supercategory, each of shape
                (S, N, C).
        """

        # TODO add batchnorm to the encoder, and do the permutation here rather than elsewhere.

        if self.input_images:
            S, N, C, H, W = x.shape
            x = x.reshape(S * N, C, H, W)
            x = self.encoder(x)
            x = x.reshape(S, N, -1)
        else:
            x = self.encoder(x)
        y = self.transformer(x, has_mask, src_key_padding_mask)  # (seq, batch, num_classes)

        # Split the output into the different supercategories
        ys = torch.split(y, self.supercategory_num_classes, dim=-1)
        return ys


class UNetTransformer(nn.Module):
    def __init__(
        self,
        num_seg_classes: int,
        num_keypoints: int,
        supercategory_num_classes: list[int],
        transformer: dict[str, Any] = dict(),
        unet: dict[str, Any] = dict(),
    ):
        """Recognition model.

        Args:
            d_input: (int): Dimension of the input features. So the input should have shape (seq, batch, d_input)
                Required if input_images is False.

            supercategory_nun_classes (list[int]): List of number of classes for each sequence label
                *including the background class*. For example, for surgical phase recognition, there
                are [8, 3, 8, 2] non-bg classes for task, activity, acquisition, and frame
                classification, respectively, so we should pass in [9, 4, 9, 3].
            backbone_model (Optional[nn.Module], optional): Pre-trained image encoder model, from
                which we only use backbone_model.backbone.
            keypoint_model (Optional[nn.Module], optional): Pre-trained keypoint detection model. If present, we use
                this to extract keypoints from the input image, which are then encoded and passed into the .
            input_images: Whether the input is an image or a feature vector.

        """
        super().__init__()

        self.supercategory_num_classes = list(supercategory_num_classes)
        self.num_classes = sum(supercategory_num_classes)
        self.transformer = Transformer(self.num_classes, **transformer)
        d_model = self.transformer.d_model

        self.num_seg_classes = num_seg_classes
        self.num_keypoints = num_keypoints
        output_channels = num_seg_classes + num_keypoints
        self.unet = instantiate(unet, num_classes=output_channels)
        # self.unet = UNet(num_classes=output_channels, **unet)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.unet.feats, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # else:
        #     self.encoder = nn.Sequential(
        #         nn.Linear(self.unet.feats, d_model, 1),
        #         nn.BatchNorm1d(d_model),
        #         nn.ReLU(),
        #     )

    def forward(
        self,
        images: torch.Tensor,
        has_mask=False,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """Forward pass of the recognition model.

        Args:
            x (torch.Tensor): (S, N, C, H, W) batch of image sequences.
            has_mask (bool, optional): Whether to use a mask for the transformer. Defaults to True.

        Returns:
            dict containing:
                segs (torch.Tensor): (S, N, num_seg_classes, H, W) batch of segmentation maps.
                heatmaps (torch.Tensor): (S, N, num_keypoints, H, W) batch of keypoint heatmaps.
                labels (tuple[torch.Tensor]): Tuple of output tensors, one for each supercategory, each of shape
                    (S, N, supercategory_num_classes[i]).
        """

        # TODO add batchnorm to the encoder, and do the permutation here rather than elsewhere.

        S, N, C, H, W = images.shape
        images = images.reshape(S * N, C, H, W)
        output = self.unet(images)

        logits = output["logits"]  # (S * N, C, H, W)
        logits = logits.reshape(S, N, -1, H, W)  # (S, N, C, H, W)
        masks = logits[:, :, : self.num_seg_classes]  # (S, N, C, H, W)
        heatmaps = logits[:, :, self.num_seg_classes :]  # (S, N, C, H, W)

        feats = output["features"]  # (S * N, C=256, H', W')
        feats = self.encoder(feats)  # (S * N, d_model, 1, 1)
        feats = feats.reshape(S, N, -1)  # (S, N, E)

        y = self.transformer(feats, has_mask, src_key_padding_mask)  # (seq, batch, num_classes)

        # Split the output into the different supercategories
        ys = torch.split(y, self.supercategory_num_classes, dim=-1)
        return dict(
            masks=masks,
            heatmaps=heatmaps,
            labels=ys,
        )


# TODO: make a version of recognition model that takes images as inputs and uses a U-Net backbone
# so the segmentation can be supervised by both.

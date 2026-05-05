"""
models.py
─────────
Neural-network components for BeMamba:

  MambaBlock          — Pure-PyTorch selective SSM (no C extension needed)
  TimeSequenceMamba   — Intra-modal temporal fusion (TSM)
  ModalSequenceMamba  — Cross-modal feature fusion  (MSM)

  Modality extractors (one per sensing modality):
    ImageExtractor    — MobileNetV3-Small CNN backbone
    GPSExtractor      — 2-layer MLP
    LiDARExtractor    — PointPillar-style voxel encoder
    RadarExtractor    — 2-D CNN on range-Doppler maps

  BeMamba             — Full pipeline; accepts whatever modalities the
                        dataset detected and wires them in automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

from config import (
    D_MODEL, D_STATE, D_CONV, EXPAND, NUM_LAYERS, NUM_BEAMS,
    LIDAR_VOXEL_H, LIDAR_VOXEL_W, LIDAR_VOXEL_D,
    RADAR_H, RADAR_W,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MAMBA BLOCK  (Selective State Space Model)
# ══════════════════════════════════════════════════════════════════════════════

class MambaBlock(nn.Module):
    """
    One Mamba block:
      x ──► LayerNorm ──► [SSM branch] × [gate branch] ──► out  +  residual

    The SSM branch uses input-dependent (selective) B, C, Δ parameters,
    discretised with Zero-Order Hold before the sequential scan.
    Linear complexity O(L·d_inner·d_state) in sequence length L.
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.norm    = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal depthwise conv — provides local context before SSM scan
        self.conv1d  = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # Selective (input-dependent) SSM parameter projections
        # Outputs: B (d_state), C (d_state), log-Δ (1)  → total: 2·d_state + 1
        self.x_proj  = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Fixed A (log-parameterised so it stays negative after exp)
        A = torch.arange(1, d_state + 1, dtype=torch.float32) \
                  .unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)  →  (B, L, d_model)"""
        B, L, _ = x.shape
        residual = x
        x = self.norm(x)

        # ── Project and split ──────────────────────────────────────────────
        xz           = self.in_proj(x)                    # (B, L, 2·d_inner)
        x_ssm, gate  = xz.chunk(2, dim=-1)               # each (B, L, d_inner)

        # ── Causal local conv ─────────────────────────────────────────────
        x_ssm = x_ssm.transpose(1, 2)                    # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[..., :L]              # trim right padding
        x_ssm = x_ssm.transpose(1, 2)                    # (B, L, d_inner)
        x_ssm = F.silu(x_ssm)

        # ── Selective parameters ──────────────────────────────────────────
        params  = self.x_proj(x_ssm)                     # (B, L, 2·d_state+1)
        B_p     = params[..., :self.d_state]              # input-dep B
        C_p     = params[..., self.d_state:2*self.d_state]# input-dep C
        dt      = F.softplus(self.dt_proj(params[..., -1:])) # (B, L, d_inner)

        A = -torch.exp(self.A_log)                        # (d_inner, d_state)

        # ── ZOH discretisation ────────────────────────────────────────────
        dA = torch.exp(dt.unsqueeze(-1) * A)              # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_p.unsqueeze(2)          # (B, L, d_inner, d_state)

        # ── Sequential SSM scan ───────────────────────────────────────────
        h  = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            h     = dA[:, t] * h + dB[:, t] * x_ssm[:, t].unsqueeze(-1)
            y_t   = (h * C_p[:, t].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y_t)

        y   = torch.stack(ys, dim=1)                      # (B, L, d_inner)
        y   = y + x_ssm * self.D                          # D skip

        # ── Gating + output proj ──────────────────────────────────────────
        out = y * F.silu(gate)
        return self.out_proj(out) + residual


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TIME SEQUENCE MAMBA  (TSM)  — intra-modal temporal fusion
# ══════════════════════════════════════════════════════════════════════════════

class TimeSequenceMamba(nn.Module):
    """
    Stacks `num_layers` MambaBlocks and returns the last-timestep hidden state,
    i.e. a single (B, d_model) summary of the whole temporal sequence.
    """

    def __init__(self, d_model: int, d_state: int = D_STATE,
                 d_conv: int = D_CONV, expand: int = EXPAND,
                 num_layers: int = NUM_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)  →  (B, d_model)"""
        for layer in self.layers:
            x = layer(x)
        return x[:, -1]   # last timestep = temporally fused feature


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODAL SEQUENCE MAMBA  (MSM)  — cross-modal feature fusion
# ══════════════════════════════════════════════════════════════════════════════

class ModalSequenceMamba(nn.Module):
    """
    Treats each modality's feature vector as a token in a short sequence,
    runs Mamba over the modal axis, then mean-pools to one fused embedding.

    If only one modality is present the sequence length is 1 and the block
    degenerates gracefully to a simple residual transform.
    """

    def __init__(self, d_model: int, d_state: int = D_STATE,
                 d_conv: int = D_CONV, expand: int = EXPAND,
                 num_layers: int = NUM_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])

    def forward(self, modal_features: list) -> torch.Tensor:
        """
        modal_features : list of (B, d_model) tensors — one per active modality.
        Returns        : (B, d_model) fused representation.
        """
        x = torch.stack(modal_features, dim=1)   # (B, M, d_model)
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=1)                       # (B, d_model)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODALITY-SPECIFIC FEATURE EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

class ImageExtractor(nn.Module):
    """
    MobileNetV3-Small CNN backbone (ImageNet-pretrained) with the classifier
    head replaced by a linear projection to d_model.

    Input : (B, 3, H, W)  — normalised RGB frame
    Output: (B, d_model)
    """

    def __init__(self, d_model: int):
        super().__init__()
        backbone      = tv_models.mobilenet_v3_small(
            weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.proj     = nn.Sequential(
            nn.Linear(576, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.features(img)
        x = self.avgpool(x).flatten(1)
        return self.proj(x)


class GPSExtractor(nn.Module):
    """
    Two-layer MLP for normalised (longitude, latitude) coordinates.

    Input : (B, 2)
    Output: (B, d_model)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, gps: torch.Tensor) -> torch.Tensor:
        return self.net(gps)


class LiDARExtractor(nn.Module):
    """
    PointPillar-style voxel encoder for LiDAR point clouds.

    The dataset loader pre-voxelises raw point clouds into a fixed-size
    occupancy / intensity grid of shape (D, H, W).  This extractor processes
    that grid with a lightweight 3-D→2-D CNN and projects to d_model.

    Input : (B, D, H, W)  — voxelised LiDAR cube
    Output: (B, d_model)

    Grid dimensions are set in config.py:
      LIDAR_VOXEL_D = depth  (z-axis bins)
      LIDAR_VOXEL_H = height (y-axis bins)
      LIDAR_VOXEL_W = width  (x-axis bins)
    """

    def __init__(self, d_model: int,
                 voxel_d: int = LIDAR_VOXEL_D,
                 voxel_h: int = LIDAR_VOXEL_H,
                 voxel_w: int = LIDAR_VOXEL_W):
        super().__init__()
        # Collapse depth axis with a 1-D conv, then process bird's-eye view
        self.depth_collapse = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GELU(),
            nn.Conv3d(8, 1, kernel_size=(voxel_d, 1, 1)),   # → (B,1,1,H,W)
        )
        # 2-D BEV encoder
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(1,  16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """voxels: (B, D, H, W)  →  (B, d_model)"""
        x = voxels.unsqueeze(1)                    # (B, 1, D, H, W)
        x = self.depth_collapse(x).squeeze(2)      # (B, 1, H, W)  — squeeze depth
        x = self.bev_encoder(x)
        return self.proj(x)


class RadarExtractor(nn.Module):
    """
    2-D CNN encoder for range-Doppler radar maps.

    Range-Doppler maps are 2-D power spectrograms where one axis is range
    (distance) and the other is radial velocity (Doppler).  They are stored
    as single-channel float tensors of shape (H, W).

    This extractor uses a compact convolutional stack followed by a linear
    projection to d_model.

    Input : (B, 1, H, W)  — range-Doppler map (log-magnitude, normalised)
    Output: (B, d_model)

    Map dimensions are set in config.py: RADAR_H, RADAR_W.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(1,  16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, rdmap: torch.Tensor) -> torch.Tensor:
        """rdmap: (B, 1, H, W)  →  (B, d_model)"""
        return self.proj(self.encoder(rdmap))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  BEMAMBA  — full end-to-end model
# ══════════════════════════════════════════════════════════════════════════════

class BeMamba(nn.Module):
    """
    BeMamba: Efficient Multimodal Sensing-Aided Beamforming via SSM.

    The constructor receives a set of active modality flags inferred directly
    from the dataset (see dataset.py → detect_modalities).  Only the
    sub-networks for present modalities are instantiated, keeping the
    parameter count minimal.

    Forward pass
    ────────────
    For each active modality m with T time-steps:
      1.  Per-frame feature extraction  → (B, T, d_model)
      2.  TimeSequenceMamba (TSM)       → (B, d_model)
    Then:
      3.  ModalSequenceMamba (MSM)      → (B, d_model)  [cross-modal fusion]
      4.  Classifier                    → (B, NUM_BEAMS) logits
    """

    def __init__(
        self,
        modalities: dict,          # e.g. {"image": True, "gps": True, "lidar": False, "radar": True}
        d_model:    int = D_MODEL,
        d_state:    int = D_STATE,
        d_conv:     int = D_CONV,
        expand:     int = EXPAND,
        num_layers: int = NUM_LAYERS,
        num_beams:  int = NUM_BEAMS,
    ):
        super().__init__()
        self.modalities = modalities
        self.active     = [k for k, v in modalities.items() if v]

        if not self.active:
            raise ValueError("BeMamba needs at least one active modality.")

        # ── Extractors ────────────────────────────────────────────────────
        if modalities.get("image"):
            self.image_ext = ImageExtractor(d_model)
            self.image_tsm = TimeSequenceMamba(d_model, d_state, d_conv, expand, num_layers)

        if modalities.get("gps"):
            self.gps_ext = GPSExtractor(d_model)
            self.gps_tsm = TimeSequenceMamba(d_model, d_state, d_conv, expand, num_layers)

        if modalities.get("lidar"):
            self.lidar_ext = LiDARExtractor(d_model)
            self.lidar_tsm = TimeSequenceMamba(d_model, d_state, d_conv, expand, num_layers)

        if modalities.get("radar"):
            self.radar_ext = RadarExtractor(d_model)
            self.radar_tsm = TimeSequenceMamba(d_model, d_state, d_conv, expand, num_layers)

        # ── Cross-modal fusion ────────────────────────────────────────────
        self.msm = ModalSequenceMamba(d_model, d_state, d_conv, expand, num_layers)

        # ── Beam classifier ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_beams),
        )

    # ── Helper: extract one modality over a time window ───────────────────
    def _extract_sequence(self, extractor, tsm, data, is_1d=False):
        """
        data    : (B, T, ...) where ... is the per-frame feature shape
        is_1d   : True for GPS  (B, T, 2)  — no spatial dims to reshape
        Returns : (B, d_model)
        """
        B, T = data.shape[:2]
        if is_1d:
            flat  = data.view(B * T, -1)
        else:
            rest  = data.shape[2:]
            flat  = data.view(B * T, *rest)

        feats = extractor(flat).view(B, T, -1)   # (B, T, d_model)
        return tsm(feats)                          # (B, d_model)

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, batch: dict) -> torch.Tensor:
        """
        batch : dict produced by DeepSense6GDataset.__getitem__
                Keys present depend on which modalities were detected.
        Returns: (B, num_beams) logits
        """
        modal_feats = []

        if self.modalities.get("image") and "images" in batch:
            # images: (B, T, 3, H, W)
            B, T, C, H, W = batch["images"].shape
            flat  = batch["images"].view(B * T, C, H, W)
            feats = self.image_ext(flat).view(B, T, -1)
            modal_feats.append(self.image_tsm(feats))

        if self.modalities.get("gps") and "gps" in batch:
            # gps: (B, T, 2)
            modal_feats.append(
                self._extract_sequence(self.gps_ext, self.gps_tsm,
                                       batch["gps"], is_1d=True)
            )

        if self.modalities.get("lidar") and "lidar" in batch:
            # lidar: (B, T, D, H, W)
            B, T, D, H, W = batch["lidar"].shape
            flat  = batch["lidar"].view(B * T, D, H, W)
            feats = self.lidar_ext(flat).view(B, T, -1)
            modal_feats.append(self.lidar_tsm(feats))

        if self.modalities.get("radar") and "radar" in batch:
            # radar: (B, T, 1, H, W)
            B, T, _, H, W = batch["radar"].shape
            flat  = batch["radar"].view(B * T, 1, H, W)
            feats = self.radar_ext(flat).view(B, T, -1)
            modal_feats.append(self.radar_tsm(feats))

        if not modal_feats:
            raise RuntimeError("No modality data found in batch.")

        fused  = self.msm(modal_feats)
        return self.classifier(fused)
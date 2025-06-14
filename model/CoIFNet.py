import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from omegaconf import open_dict

from .attend import Attend


class CoIFNet(nn.Module):
    def __init__(self, cfg, seq_len=None, pred_len=None):
        super(CoIFNet, self).__init__()

        self.m_cfg = cfg.model
        self.hidden = self.m_cfg.hidden

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.revin = self.m_cfg.revin
        self.init_revin(cfg.dataset.channels, self.m_cfg.revin_affine)

        if self.m_cfg.use_head:
            self.shared_model = SharedModule(
                self.m_cfg,
                self.seq_len,
                self.m_cfg.hidden,
                cfg.dataset.channels,
            )

            self.aux_head = nn.Linear(self.m_cfg.hidden, pred_len)
        else:
            self.shared_model = SharedModule(
                self.m_cfg,
                self.seq_len,
                pred_len,
                cfg.dataset.channels,
            )
        with open_dict(cfg):
            self.m_cfg.device = cfg.device

    def init_revin(self, channels, use_affine=True):
        if self.revin:
            if self.m_cfg.is_RevON:
                from .RevON import RevON

                self.revin_layer = RevON(
                    num_features=channels,
                    affine=use_affine,
                )
            else:
                from .RevIN import RevIN

                self.revin_layer = RevIN(
                    num_features=channels,
                    affine=use_affine,
                )

    def get_shared_parameters(self):
        return [p for n, p in self.shared_model.named_parameters()]

    def share_module(self, x, mask, feat=None):

        h = self.shared_model(x, mask, feat)
        return h

    def impute(self, x, mask, feat=None):
        """
        x: [batch_size, seq_len + pred_len, channel]
        mask: [batch_size, seq_len + pred_len, channel]

        Return:
            [batch_size, seq_len + pred_len, channel]
        """
        if self.revin:
            if self.m_cfg.is_RevON:
                x = self.revin_layer(x, "norm", mask)
            else:
                x = self.revin_layer(x, "norm")

        x = self.share_module(x, mask, feat)

        if self.m_cfg.use_head:
            x = self.aux_head(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.revin:
            x = self.revin_layer(x, "denorm")
        return x.contiguous()


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class TSBlock(nn.Module):
    def __init__(self, input_dim, output_dim, mid_hidden, dropout):
        super(TSBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, mid_hidden * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_hidden, output_dim),
        )

    def forward(self, x):
        return self.block(x)


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, mid_hidden, dropout):
        super(LinearBlock, self).__init__()

        self.block = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.block(x)


def get_block(block_name, input_dim, output_dim, mid_hidden, dropout):
    if block_name == "TSBlock":
        return TSBlock(input_dim, output_dim, mid_hidden, dropout)
    elif block_name == "LinearBlock":
        return LinearBlock(input_dim, output_dim, mid_hidden, dropout)
    elif block_name == "AttentionBlock":
        return AttentionBlock(input_dim, output_dim)
    else:
        raise NotImplementedError(f"block {block_name} not implemented")


class SharedModule(nn.Module):
    def __init__(self, cfg, in_seq, out_seq, channel_dim):
        super(SharedModule, self).__init__()

        self.use_mask = cfg.use_mask
        self.use_feat = cfg.use_feat
        self.use_feat_emb = cfg.use_feat_emb

        self.time_of_day_size = 24
        self.day_of_week_size = 7
        temp_dim_tid = cfg.temp_dim_tid
        temp_dim_tiw = cfg.temp_dim_tiw

        in_channel_dim = channel_dim
        if self.use_mask:
            in_channel_dim += channel_dim  # mask

        feat_emb_dim = 0
        if self.use_feat:
            if self.use_feat_emb:
                feat_emb_dim = temp_dim_tid + temp_dim_tiw
                self.time_in_day_emb = nn.Embedding(
                    self.time_of_day_size, temp_dim_tid
                )
                self.day_in_week_emb = nn.Embedding(
                    self.day_of_week_size, temp_dim_tiw
                )
            else:
                feat_emb_dim = 4

        in_channel_dim += feat_emb_dim

        self.intra_model = get_block(
            cfg.intra_type,
            in_seq,
            out_seq,
            cfg.hidden,
            cfg.dropout,
        )
        self.inter_model = get_block(
            cfg.inter_type,
            in_channel_dim,
            channel_dim,
            cfg.hidden,
            cfg.dropout,
        )

    def forward(self, x, mask, feat=None):
        if self.use_mask:
            x = torch.concat([x, mask], dim=-1)  # [B, L, C] -> [B, L, C*2]

        if self.use_feat:
            if self.use_feat_emb:
                feat_emb = []
                feat_emb.append(self.day_in_week_emb(feat[..., 0].long()))
                feat_emb.append(self.time_in_day_emb(feat[..., 1].long()))
            else:
                feat_emb = [feat]
            x = torch.cat([x] + feat_emb, dim=-1)

        x = self.intra_model(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.inter_model(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self, dim_in, dim_out, dim_head=32, heads=4, dropout=0.0, flash=True
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads

        self.to_qkv = nn.Sequential(
            nn.Linear(dim_in, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h n d", qkv=3, h=heads),
        )

        self.to_v_gates = nn.Sequential(
            nn.Linear(dim_in, heads, bias=False),
            nn.Sigmoid(),
            Rearrange("b n h -> b h n 1", h=heads),
        )

        self.attend = Attend(flash=flash, dropout=dropout)

        self.to_out = nn.Sequential(
            Rearrange("b h n d -> b n (h d)"),
            nn.Linear(dim_inner, dim_out, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        q, k, v = self.to_qkv(x)

        out = self.attend(q, k, v)

        out = out * self.to_v_gates(x)
        return self.to_out(out)

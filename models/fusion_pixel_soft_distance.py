import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register


def _gray_from_rgb(x):
    if x.shape[1] == 1:
        return x
    weight = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x[:, :3] * weight).sum(dim=1, keepdim=True)


class ConvBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


@register('fusion-pixel-soft-distance')
class FusionPixelSoftDistance(nn.Module):

    def __init__(self, encoder_spec, hidden_dim=64, temperature=0.7,
                 residual_scale=0.25, pretrained_liif=None,
                 freeze_encoder=False):
        super().__init__()
        self.is_pixel_fusion = True
        self.temperature = temperature
        self.residual_scale = residual_scale

        self.vi_encoder = models.make(encoder_spec)
        self.ir_encoder = models.make(encoder_spec)
        self.out_dim = self.vi_encoder.out_dim

        dist_in_dim = self.out_dim * 3 + 3
        self.distance_net = nn.Sequential(
            ConvBlock(dist_in_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.decoder = nn.Sequential(
            ConvBlock(self.out_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
        )

        self.last_d_ir = None
        self.last_d_vi = None
        self.last_base = None
        self.last_residual = None

        if pretrained_liif is not None:
            self.load_pretrained_liif(pretrained_liif)
        if freeze_encoder:
            self.freeze_encoders()

    def freeze_encoders(self):
        for encoder in [self.vi_encoder, self.ir_encoder]:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False

    def _adapt_encoder_sd(self, state_dict):
        enc_sd = {}
        for k, v in state_dict.items():
            if not k.startswith('encoder.'):
                continue
            name = k[len('encoder.'):]
            if name == 'head.0.weight' and v.dim() == 4 and v.shape[1] == 3:
                v = v.mean(dim=1, keepdim=True)
            enc_sd[name] = v
        return enc_sd

    def load_pretrained_liif(self, path):
        ckpt = torch.load(path, map_location='cpu')
        model_sd = ckpt.get('model', ckpt).get('sd', ckpt) if isinstance(ckpt, dict) else ckpt
        enc_sd = self._adapt_encoder_sd(model_sd)
        self.vi_encoder.load_state_dict(enc_sd, strict=False)
        self.ir_encoder.load_state_dict(enc_sd, strict=False)

    def _encode(self, vi, ir):
        if not any(p.requires_grad for p in self.vi_encoder.parameters()):
            self.vi_encoder.eval()
            self.ir_encoder.eval()
            with torch.no_grad():
                f_vi = self.vi_encoder(vi)
                f_ir = self.ir_encoder(ir)
        else:
            f_vi = self.vi_encoder(vi)
            f_ir = self.ir_encoder(ir)
        return f_vi, f_ir

    def forward(self, vi, ir):
        vi = _gray_from_rgb(vi).clamp(0, 1)
        ir = _gray_from_rgb(ir).clamp(0, 1)

        f_vi, f_ir = self._encode(vi, ir)
        if f_vi.shape[-2:] != vi.shape[-2:]:
            f_vi = F.interpolate(f_vi, size=vi.shape[-2:], mode='bilinear', align_corners=False)
            f_ir = F.interpolate(f_ir, size=ir.shape[-2:], mode='bilinear', align_corners=False)

        dist_inp = torch.cat([f_vi, f_ir, (f_vi - f_ir).abs(),
                              vi, ir, (vi - ir).abs()], dim=1)
        d_ir = torch.sigmoid(self.distance_net(dist_inp) / max(self.temperature, 1e-6))
        d_vi = 1.0 - d_ir

        f_fuse = d_vi * f_vi + d_ir * f_ir
        base = d_vi * vi + d_ir * ir
        residual = self.residual_scale * torch.tanh(self.decoder(f_fuse))
        pred = torch.clamp(base + residual, 0, 1)

        self.last_d_ir = d_ir.detach()
        self.last_d_vi = d_vi.detach()
        self.last_base = base.detach()
        self.last_residual = residual.detach()
        return pred

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


class ReliabilityNet(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branch_d1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )
        self.branch_d2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.branch_d4 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = torch.cat([self.branch_d1(x), self.branch_d2(x), self.branch_d4(x)], dim=1)
        return torch.softmax(self.head(x), dim=1)


def _grad_mag(x):
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    grad_x = F.pad(grad_x.abs(), (0, 1, 0, 0))
    grad_y = F.pad(grad_y.abs(), (0, 0, 0, 1))
    return grad_x + grad_y


def _local_contrast(x, kernel_size=9):
    pad = kernel_size // 2
    mean = F.avg_pool2d(x, kernel_size, stride=1, padding=pad)
    return (x - mean).abs()


@register('fusion-pixel-soft-distance')
class FusionPixelSoftDistance(nn.Module):

    def __init__(self, encoder_spec, hidden_dim=64, temperature=0.7,
                 residual_scale=0.25, reliability_temperature=1.0,
                 glare_threshold=0.85, glare_delta=0.12,
                 glare_sharpness=20.0, use_reliability=True,
                 pretrained_liif=None, freeze_encoder=False):
        super().__init__()
        self.is_pixel_fusion = True
        self.temperature = temperature
        self.residual_scale = residual_scale
        self.reliability_temperature = reliability_temperature
        self.glare_threshold = glare_threshold
        self.glare_delta = glare_delta
        self.glare_sharpness = glare_sharpness
        self.use_reliability = use_reliability

        self.vi_encoder = models.make(encoder_spec)
        self.ir_encoder = models.make(encoder_spec)
        self.out_dim = self.vi_encoder.out_dim

        dist_in_dim = self.out_dim * 3 + 5
        self.distance_net = nn.Sequential(
            ConvBlock(dist_in_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        rel_in_dim = 11
        self.reliability_net = ReliabilityNet(rel_in_dim, hidden_dim)
        self.decoder = nn.Sequential(
            ConvBlock(self.out_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
        )

        self.last_d_ir = None
        self.last_d_vi = None
        self.last_r_ir = None
        self.last_r_vi = None
        self.last_w_ir = None
        self.last_w_vi = None
        self.last_glare = None
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

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

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

    def forward(self, vi, ir, depth=None):
        vi = _gray_from_rgb(vi).clamp(0, 1)
        ir = _gray_from_rgb(ir).clamp(0, 1)
        if depth is None:
            depth = torch.zeros_like(vi)
        else:
            depth = _gray_from_rgb(depth).clamp(0, 1)
            if depth.shape[-2:] != vi.shape[-2:]:
                depth = F.interpolate(depth, size=vi.shape[-2:], mode='bilinear', align_corners=False)
        depth_grad = _grad_mag(depth).clamp(0, 1)

        f_vi, f_ir = self._encode(vi, ir)
        if f_vi.shape[-2:] != vi.shape[-2:]:
            f_vi = F.interpolate(f_vi, size=vi.shape[-2:], mode='bilinear', align_corners=False)
            f_ir = F.interpolate(f_ir, size=ir.shape[-2:], mode='bilinear', align_corners=False)

        dist_inp = torch.cat([f_vi, f_ir, (f_vi - f_ir).abs(),
                              vi, ir, (vi - ir).abs(), depth, depth_grad], dim=1)
        d_ir = torch.sigmoid(self.distance_net(dist_inp) / max(self.temperature, 1e-6))
        d_vi = 1.0 - d_ir

        vi_grad = _grad_mag(vi)
        ir_grad = _grad_mag(ir)
        vi_contrast = _local_contrast(vi)
        ir_contrast = _local_contrast(ir)
        glare = (
            torch.sigmoid(self.glare_sharpness * (vi - self.glare_threshold))
            * torch.sigmoid(self.glare_sharpness * (vi - ir - self.glare_delta))
        )
        if self.use_reliability:
            rel_inp = torch.cat([
                vi, ir, (vi - ir).abs(),
                vi_grad, ir_grad,
                vi_contrast, ir_contrast,
                glare, glare * (ir + ir_grad + ir_contrast).clamp(0, 1),
                depth, depth_grad,
            ], dim=1)
            reliability = self.reliability_net(rel_inp / max(self.reliability_temperature, 1e-6))
            r_vi = reliability[:, :1]
            r_ir = reliability[:, 1:]
            w_vi_raw = d_vi * r_vi
            w_ir_raw = d_ir * r_ir
            denom = w_vi_raw + w_ir_raw + 1e-6
            w_vi = w_vi_raw / denom
            w_ir = w_ir_raw / denom
        else:
            r_vi = torch.full_like(d_vi, 0.5)
            r_ir = torch.full_like(d_ir, 0.5)
            w_vi = d_vi
            w_ir = d_ir

        f_fuse = w_vi * f_vi + w_ir * f_ir
        base = w_vi * vi + w_ir * ir
        residual = self.residual_scale * torch.tanh(self.decoder(f_fuse))
        pred = torch.clamp(base + residual, 0, 1)

        self.last_d_ir = d_ir
        self.last_d_vi = d_vi
        self.last_r_ir = r_ir
        self.last_r_vi = r_vi
        self.last_w_ir = w_ir
        self.last_w_vi = w_vi
        self.last_glare = glare
        self.last_base = base
        self.last_residual = residual
        return pred

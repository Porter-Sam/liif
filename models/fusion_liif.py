import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


def _gray_from_rgb(x):
    if x.shape[1] == 1:
        return x
    weight = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x[:, :3] * weight).sum(dim=1, keepdim=True)


@register('fusion-liif')
class FusionLIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, metric_dim=32,
                 temperature=0.1, local_ensemble=True, feat_unfold=True,
                 cell_decode=True, pretrained_liif=None, freeze_encoder=False):
        super().__init__()
        self.is_fusion_liif = True
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.temperature = temperature
        self.sample_mode = 'nearest'
        self.last_modality_distance = None

        self.vi_encoder = models.make(encoder_spec)
        self.ir_encoder = models.make(encoder_spec)
        self.out_dim = self.vi_encoder.out_dim

        self.vi_metric = nn.Conv2d(self.out_dim, metric_dim, 1)
        self.ir_metric = nn.Conv2d(self.out_dim, metric_dim, 1)
        self.query_metric = nn.Sequential(
            nn.Linear(metric_dim * 2 + 2, metric_dim),
            nn.ReLU(inplace=True),
            nn.Linear(metric_dim, metric_dim)
        )

        if imnet_spec is not None:
            imnet_in_dim = self.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

        self.last_modality_weight = None
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

    def _adapt_imnet_sd(self, state_dict):
        imnet_sd = {}
        own_sd = self.imnet.state_dict() if self.imnet is not None else {}
        for k, v in state_dict.items():
            if not k.startswith('imnet.'):
                continue
            name = k[len('imnet.'):]
            if name in own_sd and own_sd[name].shape != v.shape:
                if (
                    v.dim() > 0 and own_sd[name].shape[0] == 1
                    and v.shape[0] == 3 and own_sd[name].shape[1:] == v.shape[1:]
                ):
                    v = v.mean(dim=0, keepdim=True)
                else:
                    continue
            imnet_sd[name] = v
        return imnet_sd

    def load_pretrained_liif(self, path):
        ckpt = torch.load(path, map_location='cpu')
        model_sd = ckpt.get('model', ckpt).get('sd', ckpt) if isinstance(ckpt, dict) else ckpt

        enc_sd = self._adapt_encoder_sd(model_sd)
        self.vi_encoder.load_state_dict(enc_sd, strict=False)
        self.ir_encoder.load_state_dict(enc_sd, strict=False)

        if self.imnet is not None:
            imnet_sd = self._adapt_imnet_sd(model_sd)
            self.imnet.load_state_dict(imnet_sd, strict=False)

    def gen_feat(self, vi, ir=None):
        if ir is None:
            ir = vi
        vi = _gray_from_rgb(vi)
        ir = _gray_from_rgb(ir)

        if not any(p.requires_grad for p in self.vi_encoder.parameters()):
            self.vi_encoder.eval()
            self.ir_encoder.eval()
            with torch.no_grad():
                self.vi_feat = self.vi_encoder(vi)
                self.ir_feat = self.ir_encoder(ir)
        else:
            self.vi_feat = self.vi_encoder(vi)
            self.ir_feat = self.ir_encoder(ir)
        self.vi_metric_feat = self.vi_metric(self.vi_feat)
        self.ir_metric_feat = self.ir_metric(self.ir_feat)
        return self.vi_feat, self.ir_feat

    def _sample(self, feat, coord):
        return F.grid_sample(
            feat, coord.flip(-1).unsqueeze(1),
            mode=self.sample_mode, align_corners=False)[:, :, 0, :].permute(0, 2, 1)

    def query_rgb(self, coord, cell=None):
        vi_feat = self.vi_feat
        ir_feat = self.ir_feat

        if self.imnet is None:
            vi_q = self._sample(vi_feat, coord)
            ir_q = self._sample(ir_feat, coord)
            return 0.5 * (vi_q + ir_q)

        if self.feat_unfold:
            vi_feat = F.unfold(vi_feat, 3, padding=1).view(
                vi_feat.shape[0], vi_feat.shape[1] * 9, vi_feat.shape[2], vi_feat.shape[3])
            ir_feat = F.unfold(ir_feat, 3, padding=1).view(
                ir_feat.shape[0], ir_feat.shape[1] * 9, ir_feat.shape[2], ir_feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / vi_feat.shape[-2] / 2
        ry = 2 / vi_feat.shape[-1] / 2

        feat_coord = make_coord(vi_feat.shape[-2:], flatten=False).to(coord.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(vi_feat.shape[0], 2, *vi_feat.shape[-2:])

        preds = []
        areas = []
        weight_records = []
        distance_records = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_vi = self._sample(vi_feat, coord_)
                q_ir = self._sample(ir_feat, coord_)
                q_vi_m = self._sample(self.vi_metric_feat, coord_)
                q_ir_m = self._sample(self.ir_metric_feat, coord_)
                q_coord = self._sample(feat_coord, coord_)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= vi_feat.shape[-2]
                rel_coord[:, :, 1] *= vi_feat.shape[-1]

                metric_inp = torch.cat([q_vi_m, q_ir_m, rel_coord], dim=-1)
                q_metric = self.query_metric(metric_inp)
                d_vi = (q_metric - q_vi_m).pow(2).mean(dim=-1)
                d_ir = (q_metric - q_ir_m).pow(2).mean(dim=-1)
                logits = torch.stack([-d_vi, -d_ir], dim=-1) / max(self.temperature, 1e-6)
                weights = torch.softmax(logits, dim=-1)
                weight_records.append(weights)
                distance_records.append(torch.stack([d_vi, d_ir], dim=-1))

                q_feat = (
                    weights[..., :1] * q_vi
                    + weights[..., 1:] * q_ir
                )
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= vi_feat.shape[-2]
                    rel_cell[:, :, 1] *= vi_feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            areas[0], areas[3] = areas[3], areas[0]
            areas[1], areas[2] = areas[2], areas[1]
            weight_records[0], weight_records[3] = weight_records[3], weight_records[0]
            weight_records[1], weight_records[2] = weight_records[2], weight_records[1]
            distance_records[0], distance_records[3] = distance_records[3], distance_records[0]
            distance_records[1], distance_records[2] = distance_records[2], distance_records[1]

        ret = 0
        avg_weight = 0
        avg_distance = 0
        for pred, weight, distance, area in zip(preds, weight_records, distance_records, areas):
            area = (area / tot_area).unsqueeze(-1)
            ret = ret + pred * area
            avg_weight = avg_weight + weight * area
            avg_distance = avg_distance + distance * area
        self.last_modality_weight = avg_weight.detach()
        self.last_modality_distance = avg_distance.detach()
        return ret

    def forward(self, vi, ir=None, coord=None, cell=None):
        if cell is None:
            cell = coord
            coord = ir
            ir = None
        self.gen_feat(vi, ir)
        return self.query_rgb(coord, cell)

"""Attention-based multimodal fusion for intracranial aneurysm RUPTURE prediction.

Implemented in plain PyTorch (no torch_geometric / torch_scatter) so the model
runs unmodified inside the stock NGC pytorch container on the DGX. Meshes are
decimated to a fixed node count, so a dense normalised adjacency is cheap and
avoids sparse-scatter dependencies entirely.

NOTE on the entropy regulariser: the submitted manuscript defines
    R_alpha = -lambda * sum_m alpha_m log alpha_m      and     L = L_BCE + ... + R_alpha
That is +lambda * H(alpha) added to the loss, which MINIMISES attention entropy and
therefore *encourages* the collapse it was introduced to prevent. The sign is wrong.
Here we subtract the entropy (maximise it), which is what the text describes.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- geometry
class GCNLayer(nn.Module):
    """H' = act(  D^-1/2 (A+I) D^-1/2  H  W )  with a dense batched adjacency."""

    def __init__(self, din, dout):
        super().__init__()
        self.lin = nn.Linear(din, dout)

    def forward(self, H, Ahat):
        return self.lin(torch.bmm(Ahat, H))


class GATLayer(nn.Module):
    """Single-head dense graph attention (Velickovic et al., 2018)."""

    def __init__(self, din, dout, slope=0.2):
        super().__init__()
        self.W = nn.Linear(din, dout, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, 1, dout))
        self.a_dst = nn.Parameter(torch.empty(1, 1, dout))
        nn.init.xavier_uniform_(self.a_src); nn.init.xavier_uniform_(self.a_dst)
        self.slope = slope

    def forward(self, H, mask):
        Wh = self.W(H)                                            # B,N,d
        e = F.leaky_relu((Wh * self.a_src).sum(-1).unsqueeze(2)
                         + (Wh * self.a_dst).sum(-1).unsqueeze(1), self.slope)
        e = e.masked_fill(mask == 0, float("-inf"))
        return torch.bmm(torch.softmax(e, dim=-1).nan_to_num(0.0), Wh)


def global_shape_descriptors(x):
    """Scale-bearing global descriptors, computed from the raw node coordinates.

    Message passing here is followed by BatchNorm, which standardises node
    features across the batch and therefore destroys ABSOLUTE SIZE. Size is the
    dominant geometric signal in this cohort: on the same sampled point clouds,
    r_max/eig1/extent each reach univariate AUC 0.62-0.64, and a linear model on
    these descriptors alone scores 0.597 versus 0.433 for the message-passing
    encoder without them. They are concatenated after pooling so the classifier
    keeps access to real millimetre scale.
    """
    p = x[..., :3]                                   # B,N,3, centred, in units of 10 mm
    r = p.norm(dim=-1)                               # B,N
    ext = p.amax(1) - p.amin(1)                      # B,3
    q = p - p.mean(1, keepdim=True)
    cov = torch.einsum("bni,bnj->bij", q, q) / q.shape[1]
    ev = torch.linalg.eigvalsh(cov.float())          # B,3 ascending
    return torch.cat([r.mean(1, keepdim=True), r.std(1, keepdim=True),
                      r.amax(1, keepdim=True), ext, ev], -1)      # B,9


class GeomEncoder(nn.Module):
    def __init__(self, din=6, hid=64, dout=128, p=0.2, n_global=9):
        super().__init__()
        self.g1, self.g2 = GCNLayer(din, hid), GCNLayer(hid, hid)
        self.gat = GATLayer(hid, hid)
        self.bn1, self.bn2 = nn.BatchNorm1d(hid), nn.BatchNorm1d(hid)
        self.gbn = nn.BatchNorm1d(n_global)
        self.head = nn.Sequential(nn.Linear(2 * hid + n_global, dout),
                                  nn.ReLU(), nn.Dropout(p))

    @staticmethod
    def _bn(bn, x):
        B, N, C = x.shape
        return bn(x.reshape(-1, C)).reshape(B, N, C)

    def forward(self, x, Ahat, mask, nmask):
        h = F.relu(self._bn(self.bn1, self.g1(x, Ahat)))
        h = F.relu(self._bn(self.bn2, self.g2(h, Ahat)))
        h = F.relu(self.gat(h, mask))
        nm = nmask.unsqueeze(-1)                                  # B,N,1 valid-node mask
        mean = (h * nm).sum(1) / nm.sum(1).clamp(min=1)
        mx = h.masked_fill(nm == 0, float("-inf")).max(1).values.nan_to_num(0.0)
        g = self.gbn(global_shape_descriptors(x))
        return self.head(torch.cat([mx, mean, g], -1))


# ---------------------------------------------------------------------- imaging
class ImgEncoder(nn.Module):
    """Compact 3D CNN over a 64^3 aneurysm-centred CTA subvolume."""

    def __init__(self, dout=128, p=0.2, ch=(16, 32, 64, 128)):
        super().__init__()
        blocks, cin = [], 1
        for c in ch:
            blocks += [nn.Conv3d(cin, c, 3, padding=1, bias=False),
                       nn.BatchNorm3d(c), nn.ReLU(inplace=True), nn.MaxPool3d(2)]
            cin = c
        self.f = nn.Sequential(*blocks, nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(cin, dout), nn.ReLU(), nn.Dropout(p))

    def forward(self, v):
        return self.head(self.f(v))


# --------------------------------------------------------------- clinical/morph
class TabEncoder(nn.Module):
    def __init__(self, din, dout=128, p=0.3):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(din, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, dout), nn.ReLU())

    def forward(self, x):
        return self.f(x)


# ------------------------------------------------------------------------ fusion
class AttnFusion(nn.Module):
    """s_m = w2^T tanh(W1 e_m + b1) + b2 ;  alpha = softmax(s) ;  z = sum alpha_m e_m."""

    def __init__(self, d=128, hid=64):
        super().__init__()
        self.W1, self.w2 = nn.Linear(d, hid), nn.Linear(hid, 1)

    def forward(self, embs):
        E = torch.stack(embs, 1)                                  # B,M,d
        alpha = torch.softmax(self.w2(torch.tanh(self.W1(E))).squeeze(-1), -1)
        return (alpha.unsqueeze(-1) * E).sum(1), alpha


class FusionNet(nn.Module):
    """modalities: any non-empty subset of ('geom','img','tab') -> drives ablations."""

    def __init__(self, tab_dim, modalities=("geom", "img", "tab"), d=128, p=0.2):
        super().__init__()
        self.modalities = tuple(modalities)
        self.enc = nn.ModuleDict()
        if "geom" in self.modalities: self.enc["geom"] = GeomEncoder(dout=d, p=p)
        if "img"  in self.modalities: self.enc["img"]  = ImgEncoder(dout=d, p=p)
        if "tab"  in self.modalities: self.enc["tab"]  = TabEncoder(tab_dim, d, p=0.3)
        self.fuse = AttnFusion(d) if len(self.modalities) > 1 else None
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, 1))

    def forward(self, batch):
        embs = []
        for m in self.modalities:
            if m == "geom":
                embs.append(self.enc["geom"](batch["x"], batch["Ahat"], batch["mask"], batch["nmask"]))
            elif m == "img":
                embs.append(self.enc["img"](batch["vol"]))
            else:
                embs.append(self.enc["tab"](batch["tab"]))
        if self.fuse is None:
            z, alpha = embs[0], torch.ones(embs[0].shape[0], 1, device=embs[0].device)
        else:
            z, alpha = self.fuse(embs)
        return self.head(z).squeeze(-1), alpha


def loss_fn(logits, y, alpha, pos_weight, lam_ent=0.01):
    """Class-balanced BCE minus lambda*H(alpha)  ->  entropy is MAXIMISED."""
    bce = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pos_weight)
    if alpha.shape[1] < 2:
        return bce, torch.zeros((), device=logits.device)
    H = -(alpha.clamp_min(1e-8).log() * alpha).sum(1).mean()
    return bce - lam_ent * H, H

"""Fusion-operator study: why attention-over-embeddings fails, and what fixes it.

MOTIVATION (measured, not assumed, in the reference run):
  imaging alone      0.714      geometry alone 0.473 (chance)
  geometry+imaging   0.683      imaging+clinical 0.567
  all three          0.614
Every branch added to imaging makes it worse, and attention entropy sits at
H/Hmax = 0.95 -- essentially uniform -- with lambda_ent = 0.01 AND with it turned
off entirely (0.614 vs 0.613). So the near-uniformity is NOT the regulariser: it
is the operator. z = sum_m alpha_m e_m is a convex combination in a shared
embedding space, and the softmax that produces alpha is computed from the
embeddings themselves, with no notion of whether a branch can actually predict.

CLAIM UNDER TEST: attention over embeddings cannot suppress an uninformative
modality; per-sample reliability weighting can.

Operators implemented:
  attn        z = sum alpha_m e_m, shared head          (the current method)
  concat      head(concat(e_m))                          (naive baseline)
  late_mean   mean of per-branch probabilities           (does late > early?)
  late_prec   inverse-variance weighting of per-branch   (the proposal)
              logits, with heteroscedastic per-branch loss
  evidential  Dirichlet evidence + Dempster-Shafer style  (stronger prior)
              combination

late_prec is the principled one: each branch emits a logit AND a log-variance,
trained so that exp(-s_m) * BCE_m + s_m is minimised -- a branch that cannot
predict is pushed to large variance and is then automatically silenced by the
precision weights w_m proportional to exp(-s_m).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (GeomEncoder, ImgEncoder, TabEncoder, GATLayer, GCNLayer,
                   global_shape_descriptors)


# ------------------------------------------------------------------ geometry
class PointEncoder(nn.Module):
    """PointNet-style geometry encoder, swappable with GeomEncoder.

    The GNN reaches 0.473 (chance) while plain logistic regression on simple
    descriptors of the SAME sampled point clouds reaches 0.597. Message passing
    over a kNN graph of 256 surface samples is evidently not helping at n=105,
    so this uses a shared per-point MLP + max-pool (permutation invariant, far
    fewer effective parameters) and concatenates the scale-bearing global
    descriptors that carry most of the signal.
    """

    def __init__(self, din=6, hid=64, dout=128, p=0.2, n_global=9):
        super().__init__()
        self.pt = nn.Sequential(
            nn.Linear(din, 64), nn.ReLU(),
            nn.Linear(64, hid), nn.ReLU())
        self.gbn = nn.BatchNorm1d(n_global)
        self.head = nn.Sequential(nn.Linear(hid + n_global, dout),
                                  nn.ReLU(), nn.Dropout(p))

    def forward(self, x, Ahat, mask, nmask):
        h = self.pt(x)                                   # B,N,hid
        mx = h.max(1).values
        g = self.gbn(global_shape_descriptors(x))
        return self.head(torch.cat([mx, g], -1))


class NoiseEncoder(nn.Module):
    """Encoder over a per-sample FIXED random vector: a modality with, by
    construction, zero information about the label. The control condition for
    the central claim -- whatever weight a fusion operator assigns to this is
    weight it should not have assigned."""

    def __init__(self, din=32, dout=128, p=0.2):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(din, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, dout), nn.ReLU())

    def forward(self, v):
        return self.f(v)


# -------------------------------------------------------------------- heads
class BranchHead(nn.Module):
    """Per-branch logit + log-variance (used by late_prec)."""

    def __init__(self, d=128, hid=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, hid), nn.ReLU())
        self.logit = nn.Linear(hid, 1)
        self.logvar = nn.Linear(hid, 1)

    def forward(self, e):
        h = self.body(e)
        # clamp keeps precision weights finite early in training
        return self.logit(h).squeeze(-1), self.logvar(h).squeeze(-1).clamp(-6, 6)


class EvidenceHead(nn.Module):
    """Per-branch Dirichlet evidence for K=2 (Sensoy et al., 2018)."""

    def __init__(self, d=128, hid=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, 2))

    def forward(self, e):
        return F.softplus(self.f(e))            # evidence >= 0


def ds_combine(a, b):
    """Dempster-Shafer combination of two Dirichlet opinions (K=2)."""
    Sa, Sb = a.sum(1, keepdim=True), b.sum(1, keepdim=True)
    ba, bb = (a - 1) / Sa, (b - 1) / Sb
    ua, ub = 2.0 / Sa, 2.0 / Sb
    # conflict between the two opinions
    C = (ba[:, 0:1] * bb[:, 1:2] + ba[:, 1:2] * bb[:, 0:1])
    denom = (1 - C).clamp(min=1e-6)
    bc = (ba * bb + ba * ub + bb * ua) / denom
    uc = (ua * ub) / denom
    S = 2.0 / uc.clamp(min=1e-6)
    return bc * S + 1.0


# ------------------------------------------------------------------- network
ENCODER_KEYS = {"geom": ("x", "Ahat", "mask", "nmask"), "img": ("vol",),
                "tab": ("tab",), "noise": ("noise",)}


class MultiModalNet(nn.Module):
    def __init__(self, tab_dim, modalities=("geom", "img", "tab"),
                 fusion="attn", geom_arch="point", d=128, p=0.2,
                 noise_dim=32, modality_dropout=0.0):
        super().__init__()
        self.modalities = tuple(modalities)
        self.fusion = fusion
        self.modality_dropout = modality_dropout
        # late_prec only: while False the branch weights are held uniform. The
        # first run of this operator (the reference run) gave a pure-noise modality
        # weight 0.258 against a 0.250 uniform share -- i.e. it never learned to
        # discriminate at all. The log-variance heads cannot separate until the
        # branches themselves predict something, so weighting from step 0 makes
        # the variances meaningless. The trainer sets this True after a warmup.
        self.use_prec = True
        self.enc = nn.ModuleDict()
        for m in self.modalities:
            if m == "geom":
                self.enc[m] = (PointEncoder(dout=d, p=p) if geom_arch == "point"
                               else GeomEncoder(dout=d, p=p))
            elif m == "img":
                self.enc[m] = ImgEncoder(dout=d, p=p)
            elif m == "tab":
                self.enc[m] = TabEncoder(tab_dim, d, p=0.3)
            elif m == "noise":
                self.enc[m] = NoiseEncoder(noise_dim, d, p=p)

        M = len(self.modalities)
        if fusion == "attn":
            self.W1, self.w2 = nn.Linear(d, 64), nn.Linear(64, 1)
            self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                      nn.Dropout(p), nn.Linear(64, 1))
        elif fusion == "concat":
            self.head = nn.Sequential(nn.Linear(d * M, 128), nn.ReLU(),
                                      nn.Dropout(p), nn.Linear(128, 1))
        elif fusion in ("late_mean", "late_prec"):
            self.branch = nn.ModuleDict({m: BranchHead(d) for m in self.modalities})
        elif fusion == "evidential":
            self.branch = nn.ModuleDict({m: EvidenceHead(d) for m in self.modalities})
        else:
            raise ValueError("unknown fusion: " + fusion)

    def encode(self, b):
        embs = []
        for m in self.modalities:
            if m == "geom":
                embs.append(self.enc[m](b["x"], b["Ahat"], b["mask"], b["nmask"]))
            elif m == "img":
                embs.append(self.enc[m](b["vol"]))
            elif m == "tab":
                embs.append(self.enc[m](b["tab"]))
            else:
                embs.append(self.enc[m](b["noise"]))
        return embs

    def forward(self, b):
        """Returns (logit, weights, aux) where weights are per-modality and
        comparable ACROSS operators -- that comparability is the whole point."""
        embs = self.encode(b)
        B = embs[0].shape[0]
        dev = embs[0].device
        M = len(self.modalities)

        # modality dropout: zero a branch's embedding during training so the
        # model cannot lean on any single one (renormalised at eval).
        if self.training and self.modality_dropout > 0 and M > 1:
            keep = (torch.rand(B, M, device=dev) > self.modality_dropout).float()
            keep = torch.where(keep.sum(1, keepdim=True) > 0, keep,
                               torch.ones_like(keep))
            embs = [e * keep[:, i:i + 1] for i, e in enumerate(embs)]

        if self.fusion == "attn":
            E = torch.stack(embs, 1)
            w = torch.softmax(self.w2(torch.tanh(self.W1(E))).squeeze(-1), -1)
            z = (w.unsqueeze(-1) * E).sum(1)
            return self.head(z).squeeze(-1), w, None

        if self.fusion == "concat":
            w = torch.full((B, M), 1.0 / M, device=dev)
            return self.head(torch.cat(embs, -1)).squeeze(-1), w, None

        if self.fusion in ("late_mean", "late_prec"):
            lg, lv = [], []
            for m, e in zip(self.modalities, embs):
                a, s = self.branch[m](e)
                lg.append(a); lv.append(s)
            L = torch.stack(lg, 1)                        # B,M
            S = torch.stack(lv, 1)                        # B,M log-variance
            if self.fusion == "late_mean" or not self.use_prec:
                w = torch.full_like(L, 1.0 / M)
            else:
                prec = torch.exp(-S)                      # inverse variance
                w = prec / prec.sum(1, keepdim=True)
            return (w * L).sum(1), w, (L, S)

        # evidential
        alphas = [self.branch[m](e) + 1.0 for m, e in zip(self.modalities, embs)]
        a = alphas[0]
        for nxt in alphas[1:]:
            a = ds_combine(a, nxt)
        prob1 = a[:, 1] / a.sum(1)
        logit = torch.log(prob1.clamp(1e-6, 1 - 1e-6) /
                          (1 - prob1).clamp(1e-6, 1 - 1e-6))
        # weight proxy: per-branch certainty 1-u, normalised
        cert = torch.stack([1.0 - 2.0 / al.sum(1) for al in alphas], 1)
        w = cert / cert.sum(1, keepdim=True).clamp(min=1e-6)
        return logit, w, alphas


# --------------------------------------------------------------------- loss
def fusion_loss(out, y, pos_weight, fusion, lam_aux=2.0, lam_ent=0.0):
    logit, w, aux = out
    yf = y.float()
    if fusion == "evidential":
        loss = 0.0
        for al in aux:
            S = al.sum(1, keepdim=True)
            oh = torch.stack([1 - yf, yf], 1)
            # expected cross-entropy under the Dirichlet
            loss = loss + (oh * (torch.digamma(S) - torch.digamma(al))).sum(1).mean()
        return loss / len(aux)

    bce = F.binary_cross_entropy_with_logits(logit, yf, pos_weight=pos_weight)

    if fusion == "late_prec" and aux is not None:
        L, S = aux
        # heteroscedastic per-branch loss: a branch that cannot predict is
        # pushed toward large log-variance, and precision weighting then
        # silences it. Without this the variances are unconstrained.
        # lam_aux raised 0.5 -> 2.0: at 0.5 the variances never separated
        # (noise modality still received 0.258 of the weight).
        per = F.binary_cross_entropy_with_logits(
            L, yf.unsqueeze(1).expand_as(L), reduction="none")
        bce = bce + lam_aux * (torch.exp(-S) * per + S).mean()
    elif fusion == "late_mean" and aux is not None:
        L, _ = aux
        per = F.binary_cross_entropy_with_logits(
            L, yf.unsqueeze(1).expand_as(L), reduction="none")
        bce = bce + lam_aux * per.mean()

    if lam_ent and w.shape[1] > 1:
        H = -(w.clamp_min(1e-8).log() * w).sum(1).mean()
        bce = bce - lam_ent * H
    return bce

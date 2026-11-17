"""Cheap checks that catch a doomed training run before it costs hours.

Every check here answers a question that, left unasked, sank weeks of training in earlier
attempts at this problem:

* :func:`degeneracy_report` - is a constant output a good solution of the loss? The loss of the
  true labels must beat "everything is background", "every bin has the prior charge mix" and
  "the comb's guess" by a clear margin on real batches, or training will find the constant.
* :func:`charge_metrics` - accuracy that a constant cannot fake: intensity-weighted, broken down
  by charge band, with the share of harmonic (2z, z/2) and neighbour (z +/- 1) errors and the
  predicted against true class frequencies, so a class the network has stopped predicting
  shows up as a zero rather than hiding in an average.
* :func:`overfit_check` - can the model fit a handful of fixed samples at all? If not, the
  plumbing (labels, features, loss) is broken and more data will not help.
"""

from __future__ import annotations

import numpy as np

BANDS = [("z=1", 1, 1), ("z=2-5", 2, 5), ("z=6-15", 6, 15), ("z=16-40", 16, 40), ("z>40", 41, 1000)]


def _logits_from_probs(p, eps: float = 1e-6):
    import torch

    return torch.log(p.clamp(min=eps))


def degeneracy_report(batch: dict, loss_fn) -> dict:
    """Loss of the true labels against constant and heuristic predictors on one batch.

    ``batch`` holds ``features`` [N, 4, L], ``topk_z``/``topk_w`` [N, L, k] and ``heat`` [N, L].
    Returns each predictor's charge cross-entropy and the margin ``constant / truth``; a margin
    near 1 means the loss barely distinguishes the answer from a constant, which is the
    situation in which networks collapse to that constant.
    """
    import torch

    from .losses import soft_charge_ce

    feats = batch["features"]
    topk_z, topk_w = batch["topk_z"], batch["topk_w"]
    n, _c, length = feats.shape
    n_classes = int(max(int(topk_z.max()) + 1, 2))
    from .losses import bin_weights

    w = bin_weights(feats, topk_z, topk_w, loss_fn.weight_floor, loss_fn.weight_cap,
                    getattr(loss_fn, "background_scale", 1.0))

    def ce_of(probs):  # probs [N, C, L]
        return float(soft_charge_ce(_logits_from_probs(probs), topk_z, topk_w, w))

    # the truth itself as a predictor (soft targets spread over the top-k classes)
    true_p = torch.full((n, n_classes, length), 1e-6)
    true_p.scatter_add_(1, topk_z.clamp(min=0).permute(0, 2, 1), topk_w.permute(0, 2, 1))
    true_p = true_p / true_p.sum(1, keepdim=True)
    # all background
    bg = torch.full((n, n_classes, length), 1e-6)
    bg[:, 0, :] = 1.0
    # weighted class prior, the best input-independent answer
    prior = torch.zeros(n_classes)
    prior.scatter_add_(0, topk_z.clamp(min=0).reshape(-1),
                       (topk_w * w[..., None]).reshape(-1))
    prior = (prior + 1e-6) / (prior + 1e-6).sum()
    pr = prior[None, :, None].expand(n, n_classes, length).clone()
    # the comb's argmax charge where it reports one, background elsewhere
    zc = torch.round(feats[:, 3, :] * 64).long().clamp(0, n_classes - 1)
    comb = torch.full((n, n_classes, length), 1e-3)
    comb.scatter_(1, zc[:, None, :], 1.0)
    comb = comb / comb.sum(1, keepdim=True)

    out = {"truth": ce_of(true_p), "all_background": ce_of(bg), "class_prior": ce_of(pr),
           "comb_argmax": ce_of(comb)}
    base = max(out["truth"], 1e-9)
    out["margins"] = {k: out[k] / base for k in ("all_background", "class_prior", "comb_argmax")}
    return out


def charge_metrics(charge_logits, batch: dict, snr_min: float = 0.5) -> dict:
    """Charge assignment quality on ion-bearing bins.

    Bins count when their true dominant class is an ion and their log-SNR feature exceeds
    ``snr_min``; each is weighted by that feature, as the loss is. Accuracy by charge band,
    harmonic and neighbour error shares, background bins called ions, and predicted against
    true class frequencies (a class predicted at 0 % while present is a collapse alarm).
    """
    import torch

    with torch.no_grad():
        pred = charge_logits.argmax(1)  # [N, L]
        true = batch["topk_z"][:, :, 0]
        w = batch["features"][:, 0, :]
        ion = (true > 0) & (w > snr_min)
        bgm = (true == 0) & (w > snr_min)
        wt = torch.where(ion, w, torch.zeros_like(w))
        tot = float(wt.sum()) or 1.0
        correct = float((wt * (pred == true)).sum())
        p, t = pred[ion].float(), true[ion].float()
        ww = w[ion]
        harmonic = ((p == 2 * t) | (2 * p == t)) & (t > 1)
        neighbour = (p - t).abs() == 1
        called_bg = p == 0
        # which charge, given that the bin is an ion: the decoder decides "is this signal" from
        # the intensity, so the charge read off the ion classes alone is what it uses
        pred_ion = charge_logits[:, 1:, :].argmax(1) + 1
        correct_ion = float((wt * (pred_ion == true)).sum())
        multi = ion & (true >= 2)
        wm = torch.where(multi, w, torch.zeros_like(w))
        tot_m = float(wm.sum()) or 1.0
        res = {
            "weighted_accuracy": correct / tot,
            "charge_accuracy_given_ion": correct_ion / tot,
            # the same on bins whose charge is 2 or more (comparable whether or not z = 1 is
            # trained on)
            "charge_accuracy_given_ion_z2": float((wm * (pred_ion == true)).sum()) / tot_m,
            "harmonic_error": float((ww * harmonic).sum()) / tot,
            "neighbour_error": float((ww * neighbour).sum()) / tot,
            "missed_as_background": float((ww * called_bg).sum()) / tot,
            "background_called_ion": float(((pred != 0) & bgm).sum()) / max(float(bgm.sum()), 1.0),
            "bands": {},
        }
        res["bands_given_ion"] = {}
        for name, lo, hi in BANDS:
            m = ion & (true >= lo) & (true <= hi)
            mw = float((w * m).sum())
            if mw > 0:
                res["bands"][name] = float((w * m * (pred == true)).sum()) / mw
                res["bands_given_ion"][name] = float((w * m * (pred_ion == true)).sum()) / mw
        tc = torch.bincount(true[ion].reshape(-1), minlength=charge_logits.shape[1]).float()
        pc = torch.bincount(pred[ion].reshape(-1), minlength=charge_logits.shape[1]).float()
        present = tc > 0
        res["classes_present"] = int(present.sum())
        res["classes_never_predicted"] = int((present & (pc == 0)).sum())
    return res


def overfit_check(model, samples: list, loss_fn, steps: int = 150, lr: float = 3e-3) -> dict:
    """Fit a handful of fixed samples; returns the first and last loss and accuracy."""
    import torch

    from ..nn.train import _batch_dict

    batch = _batch_dict(tuple(torch.stack([s[i] for s in samples]) for i in range(4)), "cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    first = None
    for _ in range(steps):
        out = model(batch["features"])
        loss = loss_fn(out, batch)["loss"]
        if first is None:
            first = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(batch["features"])
        last = float(loss_fn(out, batch)["loss"])
        acc = charge_metrics(out["charge_logits"], batch)["weighted_accuracy"]
    return {"first_loss": first, "last_loss": last, "weighted_accuracy": acc,
            "ok": bool(np.isfinite(last) and last < 0.5 * first)}

"""Instrument-level test-time adaptation of TabICL.

One pretrained TabICL is fine-tuned on meta-tasks drawn from every PREDICT item
of the instrument at once: each step picks an item, takes a random chunk of
training rows where it is observed, splits it into context and query, and
takes a cross-entropy step on the query. The adapted weights then serve every
item through the ordinary in-context predictor. The items share respondents
and GIVEN columns, so what the model learns about this instrument's feature
space transfers between them; that is the point, and the difference from
fine-tuning one item at a time.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F


def adapt(model, Xnum, Y, rows, *, steps=300, lr=1e-5, chunk=3000, query_ratio=0.2,
          max_classes=10, seed=0, device="cuda", time_limit=None):
    """Return a fine-tuned copy of `model`.

    Xnum: [n, d] float array (TabICL's numeric encoding of the GIVEN columns).
    Y: [n, J] int codes, -1 = unknown. rows: indices usable for training.
    """
    import time

    from tabicl._finetune.data import _build_meta_batch

    t0 = time.time()
    m = copy.deepcopy(model).to(device)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    J = Y.shape[1]
    # item weights: observed rows, items with too many classes left out
    usable, wts, enc = [], [], {}
    for j in range(J):
        y = Y[rows, j]
        ok = y >= 0
        cl = np.unique(y[ok])
        if 2 <= len(cl) <= max_classes and ok.sum() >= 200:
            usable.append(j)
            wts.append(ok.sum())
            lut = np.full(Y[:, j].max() + 2, -1)
            lut[cl] = np.arange(len(cl))
            enc[j] = lut
    wts = np.asarray(wts, float) / np.sum(wts)
    losses = []
    for step in range(steps):
        if time_limit and time.time() - t0 > time_limit:
            break
        j = usable[rng.choice(len(usable), p=wts)]
        y = Y[rows, j]
        idx = rows[y >= 0]
        if len(idx) > chunk:
            idx = rng.choice(idx, chunk, replace=False)
        yc = enc[j][Y[idx, j]]
        try:
            b = _build_meta_batch(Xnum[idx], yc, classification=True, n_estimators=1,
                                  query_size=max(1, int(len(idx) * query_ratio)),
                                  epoch_seed=seed + step, chunk_idx=0, norm_methods=None,
                                  feat_shuffle_method="latin", class_shuffle_method="shift",
                                  outlier_threshold=4.0, preprocessing_seed=seed + step)
        except Exception:  # noqa: BLE001 -- e.g. a class too rare to stratify
            continue
        ctx = torch.unique(b.y_train.reshape(-1))
        qry = torch.unique(b.y_query.reshape(-1))
        if not bool(torch.isin(qry, ctx).all()):
            continue
        X, ytr, yq = b.X.to(device), b.y_train.to(device), b.y_query.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = m(X, ytr.float())
            n_cls = int(ytr.max().item()) + 1
            loss = F.cross_entropy(logits[..., :n_cls].reshape(-1, n_cls).float(),
                                   yq.long().reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss))
    m.eval()
    return m, losses

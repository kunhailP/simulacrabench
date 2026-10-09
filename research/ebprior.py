"""Model-prior empirical Bayes: shrink cell frequencies toward a cross-fitted model.

For item j and a partition of respondents into cells c (a set of GIVEN columns),

    q_j(. | x) = (n_jc(.) + alpha_j * m_j(x)) / (n_jc + alpha_j),     x in c,

(mode "add") or, preserving the model's variation inside a cell (mode "mult"),

    q_j(. | x) ∝ m_j(x) * pi_jc / mbar_jc,   pi_jc = (n_jc + alpha_j mbar_jc) / (n_jc + alpha_j),

where m_j is a base model's out-of-fold prediction, mbar_jc its cell mean and alpha_j is chosen by the
Dirichlet-multinomial marginal likelihood of the training cell counts around
the cell means of m_j. alpha_j is the number of respondents the model is worth
in a cell: large when the model already explains the cell, small when the cell
holds structure the model missed.

Everything for a visible row is computed without that row's fold (counts and
prior), so the corrected OOF tables can be blended and tempered like any other.
"""

import numpy as np

from oof import V7


def cell_codes(ins, cols):
    code = np.zeros(len(ins.Y), np.int64)
    for g in cols:
        code = code * ins.gcard[g] + ins.gcodes[g]
    return code


def correct(ins, prior_oof, prior_hid, cols_for_item, folds, mode="mult"):
    """prior_oof[j]: [n_vis, K], prior_hid[j]: [n_hid, K]. Returns (oof, hid, alphas)."""
    vis, hid = ins.vis, ins.hid
    J = len(ins.targets)
    out_o, out_h, alphas = [], [], []
    for j in range(J):
        code = cell_codes(ins, cols_for_item(j))
        cv, ch = code[vis], code[hid]
        y = ins.Y[vis, j]
        K = ins.K[j]
        O = prior_oof[j].copy()
        Hh = np.zeros_like(prior_hid[j])
        a_list = []
        for part in folds + [None]:
            if part is None:
                trm = y >= 0
                rows_code, rows_prior = ch, prior_hid[j]
            else:
                trm = np.ones(len(vis), bool)
                trm[part] = False
                trm &= y >= 0
                rows_code, rows_prior = cv[part], prior_oof[j][part]
            cells, inv = np.unique(cv[trm], return_inverse=True)
            tab = np.zeros((len(cells), K))
            np.add.at(tab, (inv, y[trm]), 1.0)
            base = np.zeros((len(cells), K))
            np.add.at(base, inv, prior_oof[j][trm])
            n = tab.sum(1, keepdims=True)
            base /= n
            alpha = V7._eb_alpha(tab, base)
            a_list.append(alpha)
            pos = np.minimum(np.searchsorted(cells, rows_code), len(cells) - 1)
            hit = cells[pos] == rows_code
            cnt = np.where(hit[:, None], tab[pos], 0.0)
            nn = cnt.sum(1, keepdims=True)
            if mode == "add":
                post = (cnt + alpha * rows_prior) / (nn + alpha)
            else:  # recalibrate the model by its cell-level posterior/mean ratio
                mbar = np.where(hit[:, None], base[pos], rows_prior)
                pi = (cnt + alpha * mbar) / (nn + alpha)
                post = rows_prior * pi / np.maximum(mbar, 1e-9)
                post /= post.sum(1, keepdims=True)
            if part is None:
                Hh = post
            else:
                O[part] = post
        out_o.append(O)
        out_h.append(Hh)
        alphas.append(a_list[-1])
    return out_o, out_h, np.array(alphas)

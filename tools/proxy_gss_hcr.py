"""UNHCR-style recording on top of the real GSS proxy.

The UNHCR schema documents one recording quirk precisely: in its multi-select
families, round 1 wrote the ordinary "0" (not selected) for respondents the
routing skipped, so the gate sentinel appears on round-2 rows only. Here the
same transformation is applied to real GSS answers: `wave` = 1 for 2016 and 2
otherwise, and in every binary item routed by a hidden (PREDICT) parent, wave-1 rows
that were not asked get the item's "no" level instead of NA_GATED. Nothing else
is invented: answers, routing and non-response stay as GSS recorded them.

    .venv/bin/python tools/proxy_gss_hcr.py --src data/proxy/gss --out data/proxy/gss_hcr
"""

import argparse
import json
import os

import pandas as pd

GV = "NA_GATED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/proxy/gss")
    ap.add_argument("--out", default="data/proxy/gss_hcr")
    a = ap.parse_args()
    schema = json.load(open(os.path.join(a.src, "schema.json")))
    df = pd.read_parquet(os.path.join(a.src, "respondents.parquet"))
    items = schema["items"]
    df["wave"] = ["1" if y == "2016" else "2" for y in df["year"]]
    w1 = df["wave"] == "1"
    fam = []
    for k, r in items.items():
        g = r.get("gate")
        if r["class"] != "PREDICT" or not g or items[g["parent"]]["class"] != "PREDICT":
            continue
        subst = [v for v in r["values"] if v[:1].isdigit()]
        if len(subst) != 2:
            continue
        no = next((v for v in subst if "no" in v.lower()), subst[1])
        skipped = w1 & (df[k] == GV)
        df.loc[skipped, k] = no
        fam.append((k, no, int(skipped.sum())))
    new = {"wave": {"question": "Survey round.", "class": "GIVEN", "values": ["1", "2"], "gate": None}}
    new.update(items)
    schema["items"] = new
    schema["dataset"]["description"] += (
        "; UNHCR-style coding: in binary items with a hidden parent, wave-1 skips are recorded as 'no'")
    os.makedirs(a.out, exist_ok=True)
    json.dump(schema, open(os.path.join(a.out, "schema.json"), "w"), indent=1)
    cols = ["respondent_id"] + list(new) + ["role"]
    df[cols].to_parquet(os.path.join(a.out, "respondents.parquet"), index=False)
    print(f"wrote {a.out}: {len(fam)} recoded items, e.g. {fam[:3]}")


if __name__ == "__main__":
    main()

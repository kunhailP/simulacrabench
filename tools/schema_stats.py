import json,sys,math,collections
for name in ['unicef','world_bank','unhcr','sample']:
    s=json.load(open(f'/root/simulacrabench/raw/repo/data/{name}.json'))
    it=s['items']; d=s['dataset']
    cls=collections.Counter(r['class'] for r in it.values())
    pred=[k for k,r in it.items() if r['class']=='PREDICT']
    giv=[k for k,r in it.items() if r['class']=='GIVEN']
    K=lambda k: len(it[k]['values'])+(1 if it[k].get('gate') else 0)
    U=sum(math.log(K(k)) for k in pred)/len(pred)
    gated=[k for k in pred if it[k].get('gate')]
    gp_given=[k for k in gated if it[it[k]['gate']['parent']]['class']=='GIVEN']
    gp_excl=[k for k in gated if it[it[k]['gate']['parent']]['class']=='EXCLUDE']
    print(f"=== {name} v{d['version']} n_rows={d['n_rows']} split={s['split']} classes={dict(cls)}")
    print(f"  PREDICT={len(pred)} gated={len(gated)} (parent GIVEN={len(gp_given)}, parent EXCLUDE={len(gp_excl)}) U={U:.4f} meanK={sum(K(k) for k in pred)/len(pred):.2f}")
    print("  GIVEN:")
    for k in giv: print(f"    {k} [{len(it[k]['values'])}] {it[k]['values'][:12]}{' gate='+str(it[k]['gate']) if it[k].get('gate') else ''}")
    if name!='unhcr':
        print("  PREDICT:")
        for k in pred: print(f"    {k} K={K(k)} gate={it[k].get('gate')} | {it[k]['question'][:90]} | {it[k]['values'][:10]}")

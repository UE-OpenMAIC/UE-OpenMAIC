
from __future__ import annotations
import math
from collections import Counter, defaultdict

def adjusted_rand_index(y,z):
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(y,z))
    except Exception:
        return float("nan")

def normalized_mutual_information(y,z):
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(y,z, average_method="geometric"))
    except Exception:
        return float("nan")

def adjusted_mutual_information(y,z):
    try:
        from sklearn.metrics import adjusted_mutual_info_score
        return float(adjusted_mutual_info_score(y,z, average_method="arithmetic"))
    except Exception:
        return float("nan")

def cps(seq):
    seq=list(map(int,seq))
    return [i for i in range(1,len(seq)) if seq[i]!=seq[i-1]]

def segs(seq):
    seq=list(map(int,seq))
    if not seq: return []
    out=[]; st=0
    for i in range(1,len(seq)):
        if seq[i]!=seq[st]:
            out.append((st,i,seq[st])); st=i
    out.append((st,len(seq),seq[st]))
    return out

def covering(y,z):
    n=len(y)
    if n==0: return 0.0
    ts=segs(y); ps=segs(z)
    if not ts or not ps: return 0.0
    total=0.0
    for a,b,_ in ts:
        best=0.0
        for c,d,_ in ps:
            inter=max(0,min(b,d)-max(a,c))
            if inter>0:
                best=max(best, inter/(max(b,d)-min(a,c)))
        total+=(b-a)*best
    return float(total/n)

def cp_f1(y,z,margin):
    t=cps(y); p=cps(z)
    if not t and not p: return 1.0
    if not t or not p: return 0.0
    used=set(); tp=0
    for x in p:
        bj=None; bd=None
        for j,u in enumerate(t):
            if j in used: continue
            dist=abs(x-u)
            if dist<=margin and (bd is None or dist<bd):
                bd=dist; bj=j
        if bj is not None:
            used.add(bj); tp+=1
    prec=tp/len(p) if p else 0.0
    rec=tp/len(t) if t else 0.0
    return 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)

def compute_metrics(labels,pred,cp_margin_ratio):
    import numpy as np
    y=np.asarray(labels,dtype=int); z=np.asarray(pred,dtype=int)
    n=min(len(y),len(z)); y=y[:n]; z=z[:n]
    margin=max(1,int(round(n*float(cp_margin_ratio)))) if n else 1
    return dict(ARI=adjusted_rand_index(y,z), NMI=normalized_mutual_information(y,z),
        AMI=adjusted_mutual_information(y,z), covering_score=covering(y,z),
        f1_score=cp_f1(y,z,margin), cp_margin=margin, true_cps_count=len(cps(y)), pred_cps_count=len(cps(z)))

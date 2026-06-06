
from __future__ import annotations
import os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MOCAP_INFO={
"amc_86_01.4d":{"n_states":4,"label":{588:0,1200:1,2006:0,2530:2,3282:0,4048:3,4579:2}},
"amc_86_02.4d":{"n_states":8,"label":{1009:0,1882:1,2677:2,3158:3,4688:4,5963:0,7327:5,8887:6,9632:7,10617:0}},
"amc_86_03.4d":{"n_states":7,"label":{872:0,1938:1,2448:2,3470:0,4632:3,5372:4,6182:5,7089:6,8401:0}},
"amc_86_07.4d":{"n_states":6,"label":{1060:0,1897:1,2564:2,3665:1,4405:2,5169:3,5804:4,6962:0,7806:5,8702:0}},
"amc_86_08.4d":{"n_states":9,"label":{1062:0,1904:1,2661:2,3282:3,3963:4,4754:5,5673:6,6362:4,7144:7,8139:8,9206:0}},
"amc_86_09.4d":{"n_states":5,"label":{921:0,1275:1,2139:2,2887:3,3667:4,4794:0}},
"amc_86_10.4d":{"n_states":4,"label":{2003:0,3720:1,4981:0,5646:2,6641:3,7583:0}},
"amc_86_11.4d":{"n_states":4,"label":{1231:0,1693:1,2332:2,2762:1,3386:3,4015:2,4665:1,5674:0}},
"amc_86_14.4d":{"n_states":3,"label":{671:0,1913:1,2931:0,4134:2,5051:0,5628:1,6055:2}},
}

@dataclass
class ClaspCase:
    dataset:str; case_id:str; data:Any; labels:Any; n_states:int; window_size:int; n_change_points:int; offset:float; source_path:str; protocol:str

def default_data_root(repo_root:Path)->Path: return Path(repo_root)/"Time2State"/"data"

def normalize_dataset_key(x):
    k=str(x).strip().lower().replace("_","-").replace(" ","")
    m={"synthetic":"synthetic","synthetic2":"synthetic","mocap":"mocap","actrectut":"actrectut",
       "ucrseg":"ucr-seg","ucr-seg":"ucr-seg","tssb":"ucr-seg","uschad":"usc-had","usc-had":"usc-had",
       "pamap2-zero":"pamap2-zero","pamap2zero":"pamap2-zero","pamap2":"pamap2-zero"}
    if k not in m: raise ValueError(f"Unsupported dataset={x!r}")
    return m[k]

def parse_dataset_list(v):
    if isinstance(v,(list,tuple,set)):
        parts=[]
        for i in v: parts += str(i).replace(","," ").replace(";"," ").split()
    else:
        parts=str(v or "").replace(","," ").replace(";"," ").split()
    return [normalize_dataset_key(p.strip("'\"[]()")) for p in parts if p.strip()]

def import_runtime(repo_root:Path):
    for cand in [repo_root/"TSpy-dev", Path(r"D:\code\teacherT2S\TSpy-dev"), repo_root.parent/"TSpy-dev"]:
        if cand.exists():
            s=str(cand)
            if s in sys.path: sys.path.remove(s)
            sys.path.insert(0,s); break
    import numpy as np, pandas as pd, scipy.io
    try:
        from TSpy.utils import normalize as tspy_normalize
    except Exception: tspy_normalize=None
    try:
        from TSpy.dataset import load_USC_HAD
    except Exception: load_USC_HAD=None
    return {"np":np,"pd":pd,"scipy_io":scipy.io,"tspy_normalize":tspy_normalize,"load_USC_HAD":load_USC_HAD}

def safe_zscore(data,np):
    a=np.asarray(data,dtype=float)
    mean=np.nanmean(a,axis=0,keepdims=True); std=np.nanstd(a,axis=0,keepdims=True)
    std[~np.isfinite(std)]=1.0; std[std<1e-8]=1.0
    return np.nan_to_num((a-mean)/std,nan=0.0,posinf=0.0,neginf=0.0)

def norm(data,rt):
    f=rt.get("tspy_normalize")
    if f:
        try: return rt["np"].nan_to_num(f(data),nan=0.0,posinf=0.0,neginf=0.0)
        except Exception: pass
    return safe_zscore(data,rt["np"])

def seg_to_label(info,np):
    lab=[]; st=0
    for end in sorted(info):
        end=int(end); lab += [int(info[end])]*max(0,end-st); st=end
    return np.asarray(lab,dtype=int)

def count_lines(path):
    with Path(path).open("r",encoding="utf-8",errors="ignore") as f: return sum(1 for _ in f)

def fill_nan(data,np):
    a=np.asarray(data,dtype=float).copy()
    if a.ndim==1: a=a.reshape(-1,1)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if np.isnan(a[i,j]): a[i,j]=a[i-1,j] if i>0 else 0.0
    return np.nan_to_num(a,nan=0.0,posinf=0.0,neginf=0.0)

def trunc(data,labels,max_rows,np):
    data=np.asarray(data); labels=np.asarray(labels,dtype=int)
    n=min(len(data),len(labels)); data=data[:n]; labels=labels[:n]
    if max_rows and max_rows>0 and n>max_rows: data=data[:max_rows]; labels=labels[:max_rows]
    return data,labels

def load_synthetic(root,rt,cfg,max_cases=None):
    np,pd=rt["np"],rt["pd"]; base=root/"synthetic_data_for_segmentation3"; out=[]
    for i in range(100):
        p=base/f"test{i}.csv"
        if not p.exists(): continue
        df=pd.read_csv(p)
        if df.shape[1]<5: df=pd.read_csv(p,header=None)
        data=df.iloc[:,0:4].to_numpy(float); labels=df.iloc[:,4].to_numpy(int)
        data,labels=trunc(data,labels,cfg.max_rows,np)
        out.append(ClaspCase("Synthetic",str(i),data,labels,len(set(labels)),cfg.window_size or 100,cfg.n_change_points,cfg.offset,str(p),"ClaSP_synthetic_trueK"))
        if max_cases and len(out)>=max_cases: break
    if not out: raise FileNotFoundError(base)
    return out

def load_mocap(root,rt,cfg,max_cases=None):
    np,pd=rt["np"],rt["pd"]; base=root/"MoCap"/"4d"; out=[]
    for p in sorted(base.glob("*.4d"),key=lambda x:x.name):
        if p.name not in MOCAP_INFO: continue
        data=pd.read_csv(p,sep=r"\s+",usecols=range(4),engine="python").to_numpy(float)
        labels=seg_to_label(MOCAP_INFO[p.name]["label"],np)[:-1]
        data,labels=trunc(data,labels,cfg.max_rows,np)
        out.append(ClaspCase("MoCap",p.name,data,labels,MOCAP_INFO[p.name]["n_states"],cfg.window_size or 50,cfg.n_change_points,cfg.offset,str(p),"ClaSP_mocap_trueK"))
        if max_cases and len(out)>=max_cases: break
    if not out: raise FileNotFoundError(base)
    return out

def load_actrectut(root,rt,cfg,max_cases=None):
    np,scipy_io=rt["np"],rt["scipy_io"]; base=root/"ActRecTut"; out=[]
    for name in ["subject1_walk","subject2_walk"]:
        p=base/name/"data.mat"; m=scipy_io.loadmat(p)
        labels=np.asarray(m["labels"].flatten(),dtype=int); data=norm(m["data"][:,0:10],rt)
        data,labels=trunc(data,labels,cfg.max_rows,np)
        out.append(ClaspCase("ActRecTut",name,data,labels,len(set(labels)),cfg.window_size or 50,cfg.n_change_points,cfg.offset,str(p),"ClaSP_actrectut_trueK"))
        if max_cases and len(out)>=max_cases: break
    return out

def load_ucrseg(root,rt,cfg,max_cases=None):
    np,pd=rt["np"],rt["pd"]; base=root/"UCR-SEG"/"UCR_datasets_seg"; out=[]
    for p in sorted([x for x in base.iterdir() if x.is_file()],key=lambda x:x.name):
        if p.suffix.lower() not in {".csv",".txt",".tsv"}: continue
        info=p.name[:-4].split("_")
        if len(info)<3: continue
        try:
            win=int(info[1]); seg={int(s):i for i,s in enumerate(info[2:])}
        except Exception: continue
        seg[count_lines(p)]=len(info[2:])
        labels=seg_to_label(seg,np); data=pd.read_csv(p,header=None).to_numpy(float)
        if data.shape[1]==1: data=data.flatten()
        data,labels=trunc(data,labels,cfg.max_rows,np)
        out.append(ClaspCase("UCR-SEG",p.name[:-4],data,labels,len(seg),win,cfg.n_change_points,cfg.offset,str(p),"ClaSP_ucrseg_filename_win_trueK"))
        if max_cases and len(out)>=max_cases: break
    if not out: raise FileNotFoundError(base)
    return out

def load_pamap2_zero(root,rt,cfg,max_cases=None):
    np,pd=rt["np"],rt["pd"]; proto=None
    for p in [root/"PAMAP2"/"Protocol", root/"PAMAP2"/"PAMAP2_Dataset"/"Protocol"]:
        if p.exists(): proto=p; break
    if proto is None: raise FileNotFoundError("PAMAP2 Protocol")
    out=[]; ds=int(cfg.pamap2_downsample)
    for i in range(1,9):
        p=proto/f"subject10{i}.dat"
        if not p.exists(): continue
        raw=pd.read_csv(p,sep=" ",header=None).apply(pd.to_numeric,errors="coerce").to_numpy(float)
        labels_all=np.asarray(np.nan_to_num(raw[:,1],nan=0.0),dtype=int)
        data=fill_nan(raw[:,2:],np); valid=labels_all>0
        data=safe_zscore(data[valid],np)[::ds]; labels=labels_all[valid][::ds]
        data,labels=trunc(data,labels,cfg.max_rows,np)
        out.append(ClaspCase("PAMAP2_zero",f"subject10{i}",data,labels,len(set(labels)),cfg.window_size or 50,cfg.n_change_points,cfg.offset,str(p),"ClaSP_pamap2_zero_trueK"))
        if max_cases and len(out)>=max_cases: break
    if not out: raise FileNotFoundError(proto)
    return out

def load_uschad(root,rt,cfg,max_cases=None):
    np=rt["np"]; loader=rt.get("load_USC_HAD")
    if loader is None: raise RuntimeError("TSpy.dataset.load_USC_HAD unavailable")
    out=[]; data_path=str(root)+os.sep
    for s in range(1,15):
        for t in range(1,6):
            data,labels=loader(s,t,data_path); data=norm(data,rt); labels=np.asarray(labels,dtype=int)
            data,labels=trunc(data,labels,cfg.max_rows,np)
            out.append(ClaspCase("USC-HAD",f"s{s}_t{t}",data,labels,len(set(labels)),cfg.window_size or 50,cfg.n_change_points,cfg.offset,f"load_USC_HAD({s},{t})","ClaSP_uschad_trueK"))
            if max_cases and len(out)>=max_cases: return out
    return out

LOADERS={"synthetic":load_synthetic,"mocap":load_mocap,"actrectut":load_actrectut,"ucr-seg":load_ucrseg,"pamap2-zero":load_pamap2_zero,"usc-had":load_uschad}
def load_cases(keys,root,rt,cfg,max_cases=None):
    out=[]
    for k in keys: out += LOADERS[normalize_dataset_key(k)](root,rt,cfg,max_cases)
    return out


from __future__ import annotations
import argparse,csv,json,random,sys,time
from dataclasses import dataclass
from pathlib import Path
THIS_DIR=Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path: sys.path.insert(0,str(THIS_DIR))
from clasp_impl import extract_clasp_cps_from_multivariate_ts
from clasp_metrics import compute_metrics
from clasp_preprocessing import default_data_root, import_runtime, load_cases, parse_dataset_list

@dataclass
class Cfg:
    window_size:int; n_change_points:int; offset:float; max_rows:int; pamap2_downsample:int

def seed_all(seed):
    random.seed(int(seed))
    try:
        import numpy as np; np.random.seed(int(seed))
    except Exception: pass

def read_completed(path):
    if not path.exists(): return set()
    out=set()
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status")=="ok": out.add((r.get("dataset",""),r.get("case_id","")))
    return out

def write_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True,exist_ok=True); exists=path.exists()
    with path.open("a",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction="ignore")
        if not exists: w.writeheader()
        for r in rows: w.writerow(r)

def summary(case_csv,out_dir):
    import pandas as pd
    if not case_csv.exists(): return {"total_cases":0,"ok_cases":0,"failed_cases":0}
    df=pd.read_csv(case_csv); ok=df[df["status"]=="ok"].copy() if "status" in df else df.iloc[0:0].copy()
    rows=[]
    if not ok.empty:
        for ds,g in ok.groupby("dataset",dropna=False):
            rows.append({"algorithm":"ClaSP-TKMeans","dataset":ds,"case_count":len(g),"ARI_mean":g["ARI"].mean(),"NMI_mean":g["NMI"].mean(),"AMI_mean":g["AMI"].mean(),"covering_mean":g["covering_score"].mean(),"f1_mean":g["f1_score"].mean(),"seconds_sum":g["seconds"].sum()})
        rows.append({"algorithm":"ClaSP-TKMeans","dataset":"ALL_DATASETS_EQUAL_CASE_WEIGHT","case_count":len(ok),"ARI_mean":ok["ARI"].mean(),"NMI_mean":ok["NMI"].mean(),"AMI_mean":ok["AMI"].mean(),"covering_mean":ok["covering_score"].mean(),"f1_mean":ok["f1_score"].mean(),"seconds_sum":ok["seconds"].sum()})
        pd.DataFrame(rows).to_csv(out_dir/"algorithm_summary.csv",index=False,encoding="utf-8-sig")
    return {"total_cases":len(df),"ok_cases":len(ok),"failed_cases":len(df)-len(ok)}

def pad(x,maxlen,np):
    a=np.asarray(x,dtype=float)
    if a.ndim==1: a=a.reshape(-1,1)
    out=np.zeros((maxlen,a.shape[1])); out[:len(a)]=a
    return out

def segment_tensor(data,cps,np):
    data=np.asarray(data,dtype=float); cps=sorted([int(c) for c in cps if 0<int(c)<len(data)])
    segs=[]; st=0
    for cp in cps:
        if cp>st: segs.append(data[st:cp])
        st=cp
    if st<len(data): segs.append(data[st:])
    if not segs: segs=[data]
    ml=max(len(s) for s in segs)
    return np.stack([pad(s,ml,np) for s in segs]), [len(s) for s in segs]

def cluster_segments(data,cps,k,seed,np):
    segs,lens=segment_tensor(data,cps,np); k=max(1,min(int(k),segs.shape[0]))
    try:
        from tslearn.clustering import TimeSeriesKMeans
        labs=TimeSeriesKMeans(n_clusters=k,metric="euclidean",random_state=int(seed),n_init=1,max_iter=50).fit_predict(segs)
    except Exception:
        from sklearn.cluster import KMeans
        labs=KMeans(n_clusters=k,random_state=int(seed),n_init=10).fit_predict(segs.reshape(segs.shape[0],-1))
    return np.concatenate([np.full(int(n),int(l),dtype=int) for l,n in zip(labs,lens)])

def run_case(case,args,np,pred_dir,idx):
    win=int(args.window_size) if args.window_size>0 else int(case.window_size)
    ncp=int(args.num_change_points) if args.num_change_points>0 else int(case.n_change_points)
    off=float(args.offset) if args.offset>=0 else float(case.offset)
    k=int(args.num_states) if args.num_states>0 else int(case.n_states)
    data=np.asarray(case.data,dtype=float); labels=np.asarray(case.labels,dtype=int)
    n=min(len(data),len(labels)); data=data[:n]; labels=labels[:n]
    t0=time.time()
    _, cps, _ = extract_clasp_cps_from_multivariate_ts(data, window_size=win, n_change_points=ncp, offset=off)
    pred=cluster_segments(data,cps,k,args.seed+idx,np)
    sec=time.time()-t0; n2=min(len(labels),len(pred)); labels=labels[:n2]; pred=pred[:n2]
    met=compute_metrics(labels,pred,args.cp_margin_ratio)
    pp=pred_dir/f"{case.dataset}_{case.case_id}_labels_pred.npy".replace("/","_").replace("\\","_")
    np.save(pp,np.vstack([labels,pred]))
    return {"algorithm":"ClaSP-TKMeans","dataset":case.dataset,"case_id":case.case_id,"status":"ok","error":"","rows_raw":len(case.data),"rows_eval":n2,"features":int(data.shape[1]) if data.ndim>1 else 1,"true_states":len(np.unique(labels)),"pred_states":len(np.unique(pred)),"true_cps_count":met["true_cps_count"],"pred_cps_count":met["pred_cps_count"],"window_size":win,"num_change_points":ncp,"n_states":k,"offset":off,"ARI":met["ARI"],"NMI":met["NMI"],"AMI":met["AMI"],"covering_score":met["covering_score"],"f1_score":met["f1_score"],"cp_margin":met["cp_margin"],"seconds":sec,"length_aligned":n2,"prediction_path":str(pp),"source_path":case.source_path,"protocol":case.protocol}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",required=True); p.add_argument("--out-dir",required=True); p.add_argument("--datasets",nargs="+",required=True); p.add_argument("--data-root",default="")
    p.add_argument("--window-size",type=int,default=0); p.add_argument("--num-change-points",type=int,default=40); p.add_argument("--num-states",type=int,default=0); p.add_argument("--offset",type=float,default=-1.0); p.add_argument("--max-rows",type=int,default=0); p.add_argument("--pamap2-downsample",type=int,default=20)
    p.add_argument("--seed",type=int,default=1379); p.add_argument("--max-cases",type=int,default=0); p.add_argument("--cp-margin-ratio",type=float,default=0.01); p.add_argument("--skip-completed",action="store_true"); p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    repo=Path(args.repo_root).resolve(); data=Path(args.data_root).resolve() if args.data_root else default_data_root(repo).resolve(); out=Path(args.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True); pred_dir=out/"predictions"; pred_dir.mkdir(exist_ok=True)
    seed_all(args.seed); rt=import_runtime(repo); np=rt["np"]
    cfg=Cfg(args.window_size,args.num_change_points,args.offset if args.offset>=0 else 0.05,args.max_rows,args.pamap2_downsample)
    keys=parse_dataset_list(args.datasets); cases=load_cases(keys,data,rt,cfg,max_cases=(args.max_cases or None))
    print("="*60); print("ClaSP-TKMeans adaptation baseline"); print("Repo root :",repo); print("Data root :",data); print("Output    :",out); print("Datasets  :",keys); print("Cases     :",len(cases)); print("Note      : ClaSP detects change points; segments are clustered with true K."); print("="*60)
    if args.dry_run:
        for c in cases[:20]: print(f"DRY {c.dataset}/{c.case_id}: rows={len(c.data)} K={c.n_states} win={c.window_size}")
        return 0
    case_csv=out/"case_results.csv"; done=read_completed(case_csv) if args.skip_completed else set()
    fields=["algorithm","dataset","case_id","status","error","rows_raw","rows_eval","features","true_states","pred_states","true_cps_count","pred_cps_count","window_size","num_change_points","n_states","offset","ARI","NMI","AMI","covering_score","f1_score","cp_margin","seconds","length_aligned","prediction_path","source_path","protocol"]
    ok=fail=total=0; tall=time.time()
    for i,c in enumerate(cases,1):
        if (c.dataset,c.case_id) in done:
            print(f"[{i}/{len(cases)}] SKIP completed {c.dataset}/{c.case_id}",flush=True); continue
        total+=1; print(f"[{i}/{len(cases)}] {c.dataset}/{c.case_id} rows={len(c.data)} K={c.n_states} win={c.window_size}",flush=True)
        try:
            row=run_case(c,args,np,pred_dir,i); ok+=1
            print(f"  OK ARI={row['ARI']:.4f} NMI={row['NMI']:.4f} pred_states={row['pred_states']} seconds={row['seconds']:.1f}",flush=True)
        except Exception as e:
            fail+=1; row={"algorithm":"ClaSP-TKMeans","dataset":c.dataset,"case_id":c.case_id,"status":"error","error":repr(e),"rows_raw":len(c.data),"rows_eval":len(c.labels),"features":0,"true_states":c.n_states,"pred_states":0,"true_cps_count":0,"pred_cps_count":0,"window_size":c.window_size,"num_change_points":c.n_change_points,"n_states":c.n_states,"offset":c.offset,"ARI":float("nan"),"NMI":float("nan"),"AMI":float("nan"),"covering_score":float("nan"),"f1_score":float("nan"),"cp_margin":0,"seconds":float("nan"),"length_aligned":0,"prediction_path":"","source_path":c.source_path,"protocol":c.protocol}
            print("  ERROR:",row["error"],flush=True)
        write_rows(case_csv,[row],fields)
    st=summary(case_csv,out); st.update({"repo_root":str(repo),"data_root":str(data),"out_dir":str(out),"datasets_requested":keys,"this_run_cases":total,"this_run_ok":ok,"this_run_failed":fail,"total_seconds_this_run":time.time()-tall,"note":"ClaSP-TKMeans uses true K for segment clustering; num_change_points is fixed by config."})
    (out/"run_status.json").write_text(json.dumps(st,ensure_ascii=False,indent=2),encoding="utf-8")
    print("="*60); print("DONE"); print("OK    :",ok); print("FAILED:",fail); print("CSV   :",case_csv); print("="*60)
    return 0 if fail==0 else 1
if __name__=="__main__": raise SystemExit(main())

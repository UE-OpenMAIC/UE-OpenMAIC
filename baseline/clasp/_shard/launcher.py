
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
THIS_DIR=Path(__file__).resolve().parent; CLASP_ROOT=THIS_DIR.parent; REPO_ROOT=CLASP_ROOT.parent.parent; DEFAULT_RUNNER=THIS_DIR/"clasp_runner.py"
TRUE={"1","true","yes","y","on"}; FALSE={"0","false","no","n","off",""}
def parse_bool(v):
    t=str(v or "").strip().lower()
    if t in TRUE: return True
    if t in FALSE: return False
    raise ValueError(v)
def parse_config(path):
    d={}
    for i,line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(),1):
        s=line.strip()
        if not s or s.startswith("#"): continue
        if "=" not in s: raise ValueError(f"line {i}: {line}")
        k,v=s.split("=",1); d[k.strip().lower().replace("-","_")]=v.strip()
    return d
def split(v):
    if v is None: return []
    return [p for p in str(v).replace(","," ").replace(";"," ").split() if p]
def exp(v,default,config_dir,repo_root):
    if v is None or str(v).strip()=="": return default.resolve()
    s=str(v).strip().replace("{CLASP_ROOT}",str(CLASP_ROOT)).replace("{SHARD_DIR}",str(THIS_DIR)).replace("{CONFIG_DIR}",str(config_dir)).replace("{REPO_ROOT}",str(repo_root))
    p=Path(s)
    if not p.is_absolute(): p=config_dir/p
    return p.resolve()
def opt(cmd,flag,val):
    if val is not None and str(val).strip(): cmd.extend([flag,str(val).strip()])
def build(path):
    path=Path(path).resolve(); cdir=path.parent; s=parse_config(path)
    repo=exp(s.get("repo_root"),REPO_ROOT,cdir,REPO_ROOT); runner=exp(s.get("runner"),DEFAULT_RUNNER,cdir,repo)
    ds=(split(s.get("datasets")) or [cdir.name])[0]; out=exp(s.get("out_dir"),cdir/f"results_clasp_{ds}",cdir,repo); out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,"-u",str(runner),"--repo-root",str(repo),"--out-dir",str(out),"--datasets",*(split(s.get("datasets")) or [ds])]
    if s.get("data_root","").strip(): cmd += ["--data-root",str(exp(s.get("data_root"),repo/"Time2State"/"data",cdir,repo))]
    for k,f in [("max_cases","--max-cases"),("window_size","--window-size"),("num_change_points","--num-change-points"),("num_states","--num-states"),("offset","--offset"),("max_rows","--max-rows"),("pamap2_downsample","--pamap2-downsample"),("seed","--seed"),("cp_margin_ratio","--cp-margin-ratio")]:
        opt(cmd,f,s.get(k))
    if parse_bool(s.get("skip_completed","0")): cmd.append("--skip-completed")
    if parse_bool(s.get("dry_run","0")): cmd.append("--dry-run")
    return cmd,runner,out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,required=True); a=ap.parse_args()
    cmd,runner,out=build(a.config)
    print("Config :",a.config.resolve()); print("Runner :",runner); print("Output :",out); print("Command:"); print(" ".join(f'"{p}"' if " " in p else p for p in cmd))
    return subprocess.run(cmd).returncode
if __name__=="__main__": raise SystemExit(main())

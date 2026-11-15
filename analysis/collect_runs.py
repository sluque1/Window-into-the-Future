"""What was every run actually configured with, and what did it score?

Recovers a table of (participant, tag, best eval PR-AUC, trainable parameters,
~40 configuration flags) from the snapshot directories themselves, by reading the
flag echo that train_sbatch.sh writes at the top of each job's .out log. This is
the only way to recover the configuration of a finished run: the snapshot holds
the code it ran, the log holds the flags it ran with, and neither is recorded
anywhere else.

A flag missing from a log is reported as its code default with a trailing ``*``,
because older runs predate flags that were added later -- so "absent" means "the
default applied", not "unknown".

Usage:
    python analysis/collect_runs.py
    SNAPSHOT_GLOB='2026-09-*' python analysis/collect_runs.py
    SNAPSHOTS_ROOT=/elsewhere/snapshots python analysis/collect_runs.py

Writes analysis/out/runs_full.csv -- one row per snapshot directory.
"""
import glob, os, re, csv
SNAP = os.environ.get("SNAPSHOTS_ROOT", "/data/video_datasets/HD-EPIC/snapshots")
# Which snapshot directories to scan, and the earliest basename to accept.
# Both are overridable: the defaults reproduce the table used in the thesis,
# which covers the August 2026 architecture sweep.
SNAPSHOT_GLOB = os.environ.get("SNAPSHOT_GLOB", "2026-08-*")
SNAPSHOT_SINCE = os.environ.get("SNAPSHOT_SINCE", "2026-08-11_00-43-37")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
# flag -> default that applied when the flag did not yet exist in train_sbatch.sh
FLAGS={'LR':'1e-5','NUM_LAYERS':'2','MODALITY':'?','OBS_TIME':'30','FUTURE_TIME':'180',
 'ANTICIP_TIME':'5','NUM_EPOCHS':'10','BATCH_SIZE':'16','EVAL_STEPS':'0.025',
 'READOUT':'token0','VOXEL_BIAS_INIT':'zero','DECODER':'linear','VOXEL_QUERY_TYPE':'axis_sum',
 'VOXEL_BIAS':'factorised','FOURIER_SCALE':'10.0','DECODER_HEADS':'8','DECODER_FFN':'false',
 'DECODER_RESIDUAL':'raw','TIME_ENCODING':'none','TIME_ENCODING_SCALE':'10.0',
 'TIME_ENCODING_GAIN':'fixed','TIME_ENCODING_GAIN_INIT':'8.0','TOKEN_LAYERNORM':'false',
 'POSITION_ENCODING':'(n/a)','MODE':'replace','GAIN_INIT':'48.0','LOCATION_TOKENS':'axis',
 'WEIGHT_DECAY':'0.0','ENCODER_DROPOUT':'0.1','DECODER_DROPOUT':'0.0','OBB_TOKEN_DROPOUT':'0.0',
 'GT_INTERACTIONS_RANGE':'<none>','GT_VOXELS_RANGE':'<none>','MAX_OBB_TOKENS':'512',
 'TIME_ANCHOR':'gaze','ENV_ORPHAN_FALLBACK':'?','TRAJECTORY_SIMPLE':'?',
 'ENV_POSITIONAL_ENCODING':'?','TRAJECTORY_EXCLUDE_UNFINISHED':'?','VAL':'(empty)'}
rows=[]
for d in sorted(glob.glob(os.path.join(SNAP, SNAPSHOT_GLOB))):
    b=os.path.basename(d)
    if b < SNAPSHOT_SINCE: continue
    m=re.match(r'[\d_-]+_loocv_(P\d+)_(.+)$', b)
    pid, tag = (m.group(1), m.group(2)) if m else ('-', re.sub(r'^[\d_-]+_','',b))
    logs=glob.glob(f'{d}/out/*.out'); cfg={}; best=None; params=None; echoed=set()
    if logs:
        txt=open(logs[0],errors='ignore').read(); head=txt[:9000]
        for f_,dflt in FLAGS.items():
            mm=re.search(rf'(?:^|\s){f_}=(\S*)', head)
            if mm: cfg[f_]=mm.group(1); echoed.add(f_)
            else:  cfg[f_]=dflt+'*'          # * = not echoed, code default applied
        pm=re.search(r'Total trainable parameters: ([\d,]+)', txt)
        params=pm.group(1) if pm else None
        vals=[float(x) for x in re.findall(r"'eval_pr_auc': ([0-9.]+)", txt)]
        if vals: best=max(vals)
    rows.append(dict(pid=pid,tag=tag,best=best,params=params,**cfg))
os.makedirs(OUT, exist_ok=True)
out_path = os.path.join(OUT, 'runs_full.csv')
with open(out_path,'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=['pid','tag','best','params']+list(FLAGS)); w.writeheader()
    for r in rows: w.writerow(r)
print('runs:',len(rows),'->',out_path)

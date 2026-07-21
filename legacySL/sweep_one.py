import sys, csv, os, numpy as np, torch
from collections import defaultdict
import copilot_dataset as cd, copilot_core as core
P1,P2=3,25
SRC, MAG = sys.argv[1], sys.argv[2]
vm = MAG if MAG=="inv_ticks" else float(MAG)

def train_subject(views, norm, vel_mag, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng=np.random.default_rng(seed); idx=rng.permutation(len(views)); nv=max(1,int(0.15*len(views)))
    val=[views[i] for i in idx[:nv]]; trn=[views[i] for i in idx[nv:]]
    maxT=max(len(v["vel"]) for v in views)
    m=core.LSTMCopilot(input_size=5,hidden_size=64,n_layers=2)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3); sch=torch.optim.lr_scheduler.StepLR(opt,6,0.5)
    best,bv=None,-1
    for ep in range(1,P1+P2+1):
        if ep<=P1: seqs=[core.build_sequence_raw(v["vel"],v["pos"],norm,"basic") for v in trn]
        else: seqs,_,_=core.simulate_batch(m,[v["vel"] for v in trn],norm,vel_mag,"basic","cpu","additive")
        lab=[v["label"] for v in trn]
        X=np.zeros((len(seqs),maxT,5),np.float32); M=np.zeros((len(seqs),maxT),np.float32)
        for i,s in enumerate(seqs):
            k=min(len(s),maxT); X[i,:k]=s[:k]; M[i,:k]=1
        X=torch.tensor(X);Y=torch.tensor(lab);M=torch.tensor(M);perm=torch.randperm(len(X)); m.train()
        for b in range(0,len(X),128):
            bi=perm[b:b+128]; logits=m(X[bi])
            loss=core.masked_weighted_ce(logits,Y[bi],M[bi],"exponential",3.0)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        sch.step()
        _,_,vp=core.simulate_batch(m,[v["vel"] for v in val],norm,vel_mag,"basic","cpu","additive")
        va=(vp==np.array([v["label"] for v in val])).mean()
        if va>bv: bv=va; best={k:t.clone() for k,t in m.state_dict().items()}
    m.load_state_dict(best); return m

src = cd.load_source("eegk_sim",repo_root=".") if SRC=="sim" else cd.load_csv_file("data/surrogate/surrogate_trajectories.csv")
real=[t for t in cd.load_source("eegk_real",repo_root=".") if int(t.keys[2])!=1]
real_by=defaultdict(list)
for t in real: real_by[t.subject_id].append(t)
by=defaultdict(list)
for t in src: by[t.subject_id].append(t)
subj=sorted(real_by)
bci=np.mean([cd.label_from_position(t.final_pos)==t.target_label for t in real])

out="sweep_results.csv"; new=not os.path.exists(out)
f=open(out,"a",newline=""); w=csv.writer(f)
if new: w.writerow(["source","vel_mag","seed","bci","copilot","delta_pp"])
for seed in [0,1,2]:
    models={}
    for s in subj:
        if s not in by: continue
        norm=cd.compute_norm_stats(by[s])
        views=[{"vel":(t.pos[1:]-t.pos[:-1]).astype(np.float32),"pos":t.pos.astype(np.float32),"label":t.target_label} for t in by[s]]
        models[s]=train_subject(views,norm,vm,seed)
    cop=tot=0
    for s in subj:
        rn=cd.compute_norm_stats(real_by[s]); vels=[(t.pos[1:]-t.pos[:-1]).astype(np.float32) for t in real_by[s]]
        _,_,pr=core.simulate_batch(models[s],vels,rn,vm,"basic","cpu","additive")
        la=np.array([t.target_label for t in real_by[s]]); cop+=(pr==la).sum(); tot+=len(la)
    w.writerow([SRC,MAG,seed,f"{bci*100:.2f}",f"{cop/tot*100:.2f}",f"{(cop/tot-bci)*100:.2f}"]); f.flush()
    print(f"{SRC} {MAG} seed{seed}: Δ={(cop/tot-bci)*100:+.2f}pp")
f.close()

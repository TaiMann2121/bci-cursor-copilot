import numpy as np, glob
from collections import defaultdict
from wordfreq import top_n_list, word_frequency
from harness import load_aligned, TA, nd

# ---------- keyboard: symbol <-> (group, finger) ----------
GROUPS={0:[['w','q'],['e'],['r']],1:[['t'],['y'],['u']],2:[['i'],['o'],['p']],
        3:[['f'],['g'],['h','j']],4:[['m'],['k'],['l']],5:[['b'],['n'],[' ']],
        6:[['z','x'],['c'],['v']],7:[['a'],['s'],['d']]}
sym2pos={}; pos2sym={}
for g,fs in GROUPS.items():
    for f,syms in enumerate(fs):
        pos2sym[(g,f)]=syms
        for s in syms: sym2pos[s]=(g,f)
ALPHA=sorted(sym2pos)                      # 27 symbols (26 letters + space)

# ---------- character 4-gram LM (stupid backoff), freq-weighted ----------
def build_lm(N=4, topk=25000):
    ctx=[defaultdict(lambda:defaultdict(float)) for _ in range(N)]  # order 0..N-1 contexts
    for w in top_n_list('en',topk):
        w=w.lower()
        if not all(c in sym2pos for c in w): continue
        f=word_frequency(w,'en'); s=' '+w+' '
        for i in range(1,len(s)):
            for o in range(N):
                if i-o<0: continue
                ctx[o][s[i-o:i]][s[i]]+=f
    return ctx,N
def p_next(lm, context):
    ctx,N=lm
    for o in range(N-1,-1,-1):
        c=context[-o:] if o>0 else ''
        if c in ctx[o] and sum(ctx[o][c].values())>0:
            d=ctx[o][c]; tot=sum(d.values())
            return np.array([d.get(s,0)/tot for s in ALPHA])
    return np.ones(len(ALPHA))/len(ALPHA)

# ---------- motor models from REAL data ----------
def build_motor():
    R=load_aligned()
    def gpost(mp,kappa=4):
        v=np.diff(mp,axis=0); w=np.hypot(v[:,0],v[:,1]); s=(v*w[:,None]).sum(0)
        lp=kappa*np.cos(np.arctan2(s[1],s[0])-TA); p=np.exp(lp-lp.max()); return p/p.sum()
    by_group=defaultdict(list)
    for r in R: by_group[r['tgt']].append(gpost(r['mp']))
    # finger confusion pooled from npz
    C=np.ones((3,3))
    for f in sorted(glob.glob('data/OnlineArmTrajectoryEEGK/*/**/typing_stats.npz',recursive=True)):
        d=np.load(f,allow_pickle=True)
        if 'finger_confusion' in d:
            ft,fp=d['finger_target_stats'],d['finger_pred_stats']
            for a,b in zip(ft,fp): C[int(a),int(b)]+=1
    return by_group, C

def evaluate(n_words=400, seed=0):
    import numpy as np
    rng=np.random.default_rng(seed)
    lm=build_lm(); by_group,C=build_motor()
    Cr=C/C.sum(1,keepdims=True)                      # row-normalized (p(pred|true))
    # dictionary for word decode
    vocab=[w.lower() for w in top_n_list('en',15000) if all(c in sym2pos for c in w) and w.isalpha()]
    freq={w:word_frequency(w,'en') for w in vocab}
    bylen=defaultdict(list)
    for w in vocab: bylen[len(w)].append(w)
    # test words: sample by frequency, length 3-8
    cand=[w for w in vocab if 3<=len(w)<=8]; wf=np.array([freq[w] for w in cand]); wf/=wf.sum()
    test=rng.choice(cand,n_words,p=wf,replace=True)

    gm=gf=cm=cf=0; ntot=0; wm=wl=0
    for word in test:
        symposts=[]                                   # motor per-char symbol posteriors
        for i,ch in enumerate(word):
            ctx=' '+word[:i]; g0,f0=sym2pos[ch]
            mg=by_group[g0][rng.integers(len(by_group[g0]))].copy()   # real arm motor posterior (true group)
            fp=rng.choice(3,p=Cr[f0]); fpost=C[:,fp]/C[:,fp].sum()    # finger motor posterior
            plm=p_next(lm,ctx)
            # group: motor vs motor x LM-group-prior
            gprior=np.array([plm[[ALPHA.index(s) for s in sum([pos2sym[(g,ff)] for ff in range(3)],[])]].sum() for g in range(8)])
            gm+= (mg.argmax()==g0); gf+= ((mg*gprior).argmax()==g0)
            # char: full position+symbol, motor vs fused
            pos=np.zeros((8,3))
            for g in range(8):
                for ff in range(3): pos[g,ff]=mg[g]*fpost[ff]
            psym_m=np.zeros(len(ALPHA))
            for s in ALPHA: gg,ff=sym2pos[s]; psym_m[ALPHA.index(s)]=pos[gg,ff]/len(pos2sym[(gg,ff)])
            psym_m/=psym_m.sum(); symposts.append(psym_m)
            psym_f=psym_m*plm; psym_f/=psym_f.sum()
            cm+= (ALPHA[psym_m.argmax()]==ch); cf+= (ALPHA[psym_f.argmax()]==ch); ntot+=1
        # word: motor per-char argmax (no language) vs dictionary-constrained decode
        motor_word=''.join(ALPHA[p.argmax()] for p in symposts)
        wm+= (motor_word==word)
        best=None;bs=-1e18
        for cw in bylen[len(word)]:
            sc=sum(np.log(symposts[i][ALPHA.index(cw[i])]+1e-12) for i in range(len(cw)))+0.5*np.log(freq[cw]+1e-12)
            if sc>bs: bs=sc; best=cw
        wl+= (best==word)
    print(f'tested {n_words} real words ({ntot} characters)\n')
    print(f'{"level":12s}{"no language":>14s}{"with language":>15s}')
    print(f'{"group (arm)":12s}{gm/ntot:>14.3f}{gf/ntot:>15.3f}')
    print(f'{"character":12s}{cm/ntot:>14.3f}{cf/ntot:>15.3f}')
    print(f'{"whole word":12s}{wm/n_words:>14.3f}{wl/n_words:>15.3f}')

if __name__=='__main__': evaluate()

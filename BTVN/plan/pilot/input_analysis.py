"""Phân tích input: SNR, độ phủ của x, nhiễu, độ nhạy theo seed. Không phải bài nộp. Cần numpy. (~1-2 phút)"""
import numpy as np, warnings
warnings.simplefilter("ignore")
f=lambda x: np.sin(1+x**2); s=0.03; rng=np.random.default_rng(1)
xg=np.linspace(0,1,200001); fg=f(xg)
print("== 1. Signal")
print(f"f range [{fg.min():.4f},{fg.max():.4f}] argmax x={xg[fg.argmax()]:.4f}  f(0)={f(0):.4f} f(1)={f(1):.4f}")
vf=fg.var(); print(f"Var_x f = {vf:.3e}, sigma^2={s**2:.1e}, SNR(var)={vf/s**2:.2f}, std f={np.sqrt(vf):.4f}")
for d in [1,2,3,4,5]:
    c=np.polyfit(xg,fg,d); print(f"  best-in-class bias^2 deg{d} = {np.mean((np.polyval(c,xg)-fg)**2):.2e}  ratio to sigma^2 = {np.mean((np.polyval(c,xg)-fg)**2)/s**2:.3f}")
print("  y range ~ f +/- 3sigma:", round(fg.min()-3*s,3), round(fg.max()+3*s,3))
print("== 2. Coverage of x (train)")
T=20000
for n in [10,100,1000]:
    X=rng.uniform(0,1,(T,n)); lo=X.min(1); hi=X.max(1); gap=np.diff(np.sort(X,1),axis=1).max(1)
    Xt=rng.uniform(0,1,(T,n)); out=((Xt<lo[:,None])|(Xt>hi[:,None]))
    print(f" n={n}: E[min]={lo.mean():.4f} E[1-max]={1-hi.mean():.4f}  E[max inner gap]={gap.mean():.3f}  "
          f"P(test pt outside train range)={out.mean():.4f} (theory {2/(n+1):.4f})  P(>=1 test pt outside)={out.any(1).mean():.3f}  E[#outside]={out.sum(1).mean():.2f}")
    # bins coverage for n=10
    if n==10:
        cnt=np.array([np.histogram(r,bins=5,range=(0,1))[0] for r in X[:5000]])
        print(f"   P(some 0.2-bin empty)={ (cnt==0).any(1).mean():.3f}  P(no pt in [0.8,1])={(cnt[:,4]==0).mean():.3f}  P(no pt in [0,0.2])={(cnt[:,0]==0).mean():.3f}")
print("== 3. Noise realisation (sample var of eps / sigma^2)")
for n in [10,100,1000]:
    e=rng.normal(0,s,(T,n)); r=(e**2).mean(1)/s**2
    print(f" n={n}: mean={r.mean():.3f} sd={r.std():.3f}  5%-95%=[{np.quantile(r,.05):.2f},{np.quantile(r,.95):.2f}]")
print("== 4. Seed sensitivity of one dataset (what a single table looks like), n=10, 5000 datasets")
def gen(n): x=rng.uniform(0,1,n); return x,f(x)+rng.normal(0,s,n)
for n in [10,100,1000]:
    R=5000 if n<1000 else 1000
    E=np.zeros((R,9)); I=np.zeros((R,9)); outside=np.zeros(R,bool)
    for r in range(R):
        xt,yt=gen(n); xs,ys=gen(n); outside[r]=((xs<xt.min())|(xs>xt.max())).any()
        for d in range(1,10):
            c=np.polyfit(xt,yt,d); I[r,d-1]=np.mean((np.polyval(c,xt)-yt)**2); E[r,d-1]=np.mean((np.polyval(c,xs)-ys)**2)
    best=E.argmin(1)+1
    print(f" n={n}: argmin-degree distribution:", {d:round((best==d).mean(),3) for d in range(1,10)})
    print(f"   P(Eout deg9 > 0.01)={np.mean(E[:,8]>0.01):.3f}  P(Eout deg9>1)={np.mean(E[:,8]>1):.3f}  median Eout deg9={np.median(E[:,8]):.3g}")
    print(f"   P(Eout monotone-U shape? Eout[deg9]>Eout[deg3])={np.mean(E[:,8]>E[:,2]):.3f}  P(Eout deg1 > deg3)={np.mean(E[:,0]>E[:,2]):.3f}")
    if n==10:
        print(f"   Eout deg9: if no test pt outside -> median {np.median(E[~outside,8]):.3g}; if >=1 outside -> median {np.median(E[outside,8]):.3g}; frac outside={outside.mean():.3f}")
        print(f"   Eout deg1 quantiles 5/50/95: {np.quantile(E[:,0],[.05,.5,.95]).round(5)}  Ein deg1: {np.quantile(I[:,0],[.05,.5,.95]).round(5)}")
        # slide-like: deg9 eout < 0.005
        print(f"   P(slide-like: Eout9<0.005)={np.mean(E[:,8]<0.005):.3f}")

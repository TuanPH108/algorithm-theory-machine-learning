"""Pilot: kiểm chứng thiết kế trước khi làm notebook (KHÔNG phải bài nộp, seed khác MSHV).
Chạy: python pilot_bias_variance.py  (cần numpy)"""
import numpy as np, warnings
rng=np.random.default_rng(0); f=lambda x: np.sin(1+x**2); s=0.03
def gen(n): x=rng.uniform(0,1,n); return x, f(x)+rng.normal(0,s,n)
for n in [10,100,1000]:
    print(f"--- n={n}")
    for d in [1,2,3,5,9]:
        ei=[];eo=[];preds=[]
        g=np.linspace(0,1,200)
        for r in range(200):
            xt,yt=gen(n); xs,ys=gen(n)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore"); c=np.polyfit(xt,yt,d)
            ei.append(np.mean((np.polyval(c,xt)-yt)**2)); eo.append(np.mean((np.polyval(c,xs)-ys)**2)); preds.append(np.polyval(c,g))
        P=np.array(preds); bias2=np.mean((P.mean(0)-f(g))**2); var=np.mean(P.var(0))
        print(f"d={d} Ein={np.median(ei):.5f} Eout_med={np.median(eo):.5f} Eout_mean={np.mean(eo):.3g} bias2={bias2:.2e} var={var:.2e}")
X=np.vander(rng.uniform(0,1,10),10); print("cond(Vandermonde, n=10, deg9) =",f"{np.linalg.cond(X):.2e}")

"""Bias²/var theo vùng x và độ tin cậy ước lượng var theo R. Không phải bài nộp. Cần numpy."""
import numpy as np, warnings; warnings.simplefilter("ignore")
f=lambda x: np.sin(1+x**2); s=0.03; rng=np.random.default_rng(2); xg=np.linspace(0,1,101)
def fits(n,d,R):
    P=[]
    for _ in range(R):
        x=rng.uniform(0,1,n); y=f(x)+rng.normal(0,s,n); P.append(np.polyval(np.polyfit(x,y,d),xg))
    return np.array(P)
regions={"[0,.1]":xg<=.1,"[.1,.9]":(xg>.1)&(xg<.9),"[.9,1]":xg>=.9}
for n in [10,100]:
  for d in [1,3,9]:
    P=fits(n,d,500); hb=np.median(P,0) if (n==10 and d==9) else P.mean(0)
    b2=(hb-f(xg))**2; v=P.var(0) if not(n==10 and d==9) else (np.subtract(*np.quantile(P,[.75,.25],0))/1.349)**2
    print(f"n={n} d={d} "+"  ".join(f"{k}: b2={b2[m].mean():.1e} var={v[m].mean():.1e}" for k,m in regions.items()))
# how good is R=3 estimate of var vs R=500 (n=10,d=3)
P=fits(10,3,500); true=P.var(0).mean()
est=[fits(10,3,3).var(0,ddof=1).mean() for _ in range(300)]
print("R=3 var estimate / true: 5-50-95%:",np.quantile(np.array(est)/true,[.05,.5,.95]).round(2))
for R in [10,30,100,200]:
    est=[fits(10,3,R).var(0,ddof=1).mean() for _ in range(100)]
    print(f"R={R}: ratio 5-95%:",np.quantile(np.array(est)/true,[.05,.95]).round(2))

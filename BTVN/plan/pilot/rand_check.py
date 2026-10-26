"""Kiểm định tính ngẫu nhiên của dữ liệu sinh ra (KS, tương quan, độ mạnh của kiểm định). Seed minh họa, không phải MSHV. Cần numpy, scipy."""
import numpy as np, random
from scipy import stats
seed=12345678; np.random.seed(seed); random.seed(seed)
f=lambda x: np.sin(1+x**2); S=0.03
def make_split(n):
    x=np.random.uniform(0,1,n); e=np.random.normal(0,S,n); return x, f(x)+e
data={n:[(make_split(n),make_split(n)) for _ in range(3)] for n in [10,100,1000]}
for n in [10,100,1000]:
    for k,((xt,yt),(xs,ys)) in enumerate(data[n],1):
        e=yt-f(xt)
        p_u=stats.kstest(xt,"uniform").pvalue; p_n=stats.kstest(e/S,"norm").pvalue
        r=np.corrcoef(xt,e)[0,1]; ks2=stats.ks_2samp(xt,xs).pvalue
        print(f"n={n} D{k}: KS-uniform p={p_u:.2f}  KS-normal p={p_n:.2f}  std(e)={e.std(ddof=1):.4f}  corr(x,e)={r:+.2f}  KS train-vs-test p={ks2:.2f}")
# false-positive rate check
P=[]
for _ in range(2000):
    x=np.random.uniform(0,1,10); P.append(stats.kstest(x,"uniform").pvalue)
print("frac p<0.05 under H0 (n=10):",np.mean(np.array(P)<0.05))
# power: can KS detect sigma^2 mistake? sigma=0.0009 at n=10
P=[stats.kstest(np.random.normal(0,0.0009,10)/S,"norm").pvalue for _ in range(2000)]
print("power to detect sigma-vs-sigma^2 bug, n=10:",np.mean(np.array(P)<0.05))
P=[stats.kstest(np.random.uniform(0,0.5,10),"uniform").pvalue for _ in range(2000)]
print("power to detect x~U[0,0.5] bug, n=10:",np.mean(np.array(P)<0.05))

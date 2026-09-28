import math
import numpy as np
import matplotlib.pyplot as plt

# Question 1: Exercise 6.11 - Overrelaxation
def f_relax(x, c=2.0): return 1.0-math.exp(-c*x)
def fp_relax(x, c=2.0): return c*math.exp(-c*x)
def solve_relax(c=2.0, x=1.0, omega=0.0, tol=1e-6):
    for n in range(1,10001):
        xn=(1+omega)*f_relax(x,c)-omega*x
        q=(1+omega)*fp_relax(x,c)-omega
        err=abs((x-xn)/(1-1/q)) if abs(q)>1e-15 and abs(1-1/q)>1e-15 else abs(xn-x)
        x=xn
        if err<tol: return x,n,err
    raise RuntimeError('No convergence')

print('QUESTION 1')
for w in [0.0,0.5,0.68,0.685]:
    x,n,e=solve_relax(omega=w); print(w,x,n,e)

# Question 2: Exercise 6.13 - Wien displacement
h=6.62607015e-34; c=2.99792458e8; kB=1.380649e-23
def fw(x): return 5*math.exp(-x)+x-5
def bisect(a,b,tol=1e-6):
    fa=fw(a)
    while b-a>tol:
        m=(a+b)/2; fm=fw(m)
        if fa*fm<=0: b=m
        else: a=m; fa=fm
    return (a+b)/2
xw=bisect(1,10); bw=h*c/(kB*xw); Ts=bw/(502e-9)
print('\nQUESTION 2'); print('x =',xw); print('b =',bw,'m K'); print('T_sun =',Ts,'K')

# Question 3: multidimensional numerical gradient descent
data=np.loadtxt('smf_cosmos.dat'); logM,nobs,sigma=data.T
def schechter(theta,x=logM):
    lp,lms,a=theta; r=10**(x-lms)
    return np.log(10)*10**lp*r**(a+1)*np.exp(-r)
def chi2(theta): return np.sum(((nobs-schechter(theta))/sigma)**2)
def numgrad(theta,hstep=1e-5):
    t=np.array(theta,float); g=np.zeros_like(t)
    for j in range(len(t)):
        d=hstep*max(1,abs(t[j])); p=t.copy(); m=t.copy(); p[j]+=d; m[j]-=d
        g[j]=(chi2(p)-chi2(m))/(2*d)
    return g
def gradient_descent(start,max_iter=20000):
    t=np.array(start,float); hist=[chi2(t)]
    for _ in range(max_iter):
        g=numgrad(t); norm=np.linalg.norm(g)
        if norm==0: break
        direction=-g/norm; step=.1
        while step>1e-12:
            trial=t+step*direction; val=chi2(trial)
            if np.isfinite(val) and val<hist[-1]: break
            step*=.5
        if step<=1e-12: break
        t=trial; hist.append(val)
        if abs(hist[-2]-hist[-1])<1e-12: break
    return t,np.array(hist)
starts=[[-2.3,11,-1.2],[-2,10.8,-1],[-3,11.2,-1.5]]
sol=[]; histories=[]
print('\nQUESTION 3')
for s in starts:
    t,hist=gradient_descent(s); sol.append(t); histories.append(hist); print('start',s,'->',t,'chi2=',hist[-1])
best=sol[int(np.argmin([chi2(t) for t in sol]))]; lp,lms,a=best; phi=10**lp; Mstar=10**lms
print('best:',best,'phi*=',phi,'M*=',Mstar,'chi2=',chi2(best))
plt.figure(figsize=(7,5))
for i,hist in enumerate(histories,1): plt.semilogy(np.arange(len(hist)),hist,label=f'start {i}')
plt.xlabel('Gradient-descent step i'); plt.ylabel(r'$\chi^2$'); plt.legend(); plt.tight_layout(); plt.savefig('chi2_vs_step.png',dpi=200); plt.close()
grid=np.linspace(logM.min(),logM.max(),400); r=10**(grid-lms); gm=np.log(10)*phi*r**(a+1)*np.exp(-r)
plt.figure(figsize=(7,5)); plt.errorbar(10**logM,nobs,yerr=sigma,fmt='o',capsize=3,label='COSMOS data'); plt.loglog(10**grid,gm,label='Best-fit Schechter function'); plt.xlabel(r'$M_{\rm gal}$'); plt.ylabel(r'$n(M_{\rm gal})$'); plt.legend(); plt.tight_layout(); plt.savefig('schechter_fit.png',dpi=200); plt.close()

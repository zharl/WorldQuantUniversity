import pandas as pd, numpy as np, cvxpy as cvx
import matplotlib.pyplot as plt, scipy.linalg as la

class Frontier:
    def __init__(self, stats_daily, corr_daily, freq=252):
        self.N = corr_daily.shape[0]
        self.tckrs = corr_daily.index
        self.freq  = freq        
        self.mu    = stats_daily.Mean/freq
        self.sigma = stats_daily.Vol
        self.Sigma = np.outer(self.sigma.values, self.sigma.values)*corr_daily
        self.wMS = self.getWeights(self.mu, self.Sigma)
        self.wMV = self.getWeights(np.ones(self.N), self.Sigma)
        self.w = None
        self.ms = None
        self.ss = None        
    def getWeights(self,m,S):
        N = S.shape[0]
        w = la.solve(S,m)
        return pd.Series(w/np.ones(N).dot(w), index=S.index)
    def getVols(self,w):
        return np.sqrt((w*self.Sigma.dot(w)).sum(0))*np.sqrt(self.freq)*100
    def getMus(self,w):
        return self.mu.T.dot(w)*self.freq*100
    def plotFrontier(self,ax,label,points=False, securities=False, **kwargs):
        ax.plot(self.ss,self.ms,label=label, **kwargs)
        # MS and MV points
        if points:
            ax.plot(self.getVols(self.wMS), self.getMus(self.wMS),'ro', label='Max Sharpe')
            ax.plot(self.getVols(self.wMV), self.getMus(self.wMV), 'bo', label='Min Variance');
        if securities:
            colors = plt.cm.jet(np.linspace(0,1,self.N))
            x = self.sigma*100*np.sqrt(self.freq)
            y = self.mu*100*self.freq
            for n,tckr in enumerate(x.index):
                ax.plot(x.iloc[n],    y.iloc[n],'o', color=colors[n], markersize=14)
                ax.text(x.iloc[n]+.2, y.iloc[n], tckr);

        ax.set_title('Exante Mean-Variance')
        ax.set_ylabel('Sample Mean Return'); ax.set_xlabel('Sample Volatility');
        ax.set_ylim(-5,20);

def mvOptimize(self, gammas, constrained):
    self.w   = gammas*self.wMS.values[:,np.newaxis] + (1-gammas)*self.wMV.values[:,np.newaxis]*constrained
    self.w   = pd.DataFrame({self.tckrs[n]:self.w[n,:] for n in range(self.N)}).T
    self.ms  = self.getMus(self.w)
    self.ss  = self.getVols(self.w) 

def longOnlyOptimize(self, gammas):
    gamma = cvx.Parameter(pos=True)
    w = cvx.Variable(self.N)
    R = self.mu.values.reshape((1,self.N)) @ w
    risk = cvx.quad_form(w, self.Sigma.values)
    prob = cvx.Problem(cvx.Maximize(R - .5*gamma*risk), 
                      [cvx.sum(w)==1,
                      w>=0])
    #loop through different levels of risk aversion plotting points
    self.w = {}
    for g in gammas:
        gamma.value = g
        prob.solve()
        self.w[g] = pd.Series(np.array(w.value).ravel(), index=self.tckrs, name=g)
    self.w = pd.DataFrame(self.w).sort_index()
    self.ms = self.getMus(self.w)
    self.ss = self.getVols(self.w)          

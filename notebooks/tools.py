import numpy as np
import netCDF4
import scipy.io as sio
from scipy import stats
from scipy.stats import pearsonr
def transfer_Richaud_2d_a(oldvar,olat,olon):
    """
    transfer 4508 horizontal points into a 2d-array
    """
    oo = len(oldvar)
    # new 2d array
    x = np.arange(olat.min(),olat.max()+0.1,0.2)
    y = np.arange(olon.min(),olon.max()+0.1,0.2)
    #
    newvar = np.ma.zeros([len(x),len(y)]);newvar.mask=True
    for ii in range(oo):
        locx = np.where(abs(x-olat[ii])<1e-3)[0][0]
        locy = np.where(abs(y-olon[ii])<1e-3)[0][0]
        newvar[locx,locy] = oldvar[ii]
    return newvar,x,y
#
def transfer_Richaud_2d_b(oldvar,olat,olon):
    """
    transfer 4508 horizontal points into a 2d-array
    """
    oo,uu = np.shape(oldvar)
    # new 2d array
    x = np.arange(olat.min(),olat.max()+0.1,0.2)
    y = np.arange(olon.min(),olon.max()+0.1,0.2)
    #
    newvar = np.ma.zeros([len(x),len(y),uu]);newvar.mask=True
    for ii in range(oo):
        locx = np.where(abs(x-olat[ii])<1e-3)[0][0]
        locy = np.where(abs(y-olon[ii])<1e-3)[0][0]
        newvar[locx,locy,:] = oldvar[ii,:]
    return newvar,x,y
#
def transfer_Richaud_2d_c(oldvar,olat,olon):
    """
    transfer 4508 horizontal points into a 2d-array
    """
    oo,uu,vv = np.shape(oldvar)
    # new 2d array
    x = np.arange(olat.min(),olat.max()+0.1,0.2)
    y = np.arange(olon.min(),olon.max()+0.1,0.2)
    #
    newvar = np.ma.zeros([len(x),len(y),uu,vv]);newvar.mask=True
    for ii in range(oo):
        locx = np.where(abs(x-olat[ii])<1e-3)[0][0]
        locy = np.where(abs(y-olon[ii])<1e-3)[0][0]
        newvar[locx,locy,:,:] = oldvar[ii,:,:]
    return newvar,x,y
#
def prob_r(corr,df):
    """
    calculate probability value equation;
    pval = exp(-0.5*(df*r^2/(1-r^2))) / sqrt(2*pi/df*(1-r^2))
    if the result is < 0.05, the corr is significant at 95% significance level
    """
    fenzi = np.exp(-0.5*(df*corr**2/(1-corr**2)))
    fenmu = np.sqrt(2*np.pi*(1-corr**2)/df)
    pval = fenzi/fenmu
    return pval
#
def neff1(corr1,n):
    """
    calculate effective degree of freedom
    equation: 1/neff = 1/n*(1+2*corr1/(1-corr1))
    """
    neff = n/(1+2*corr1/(1-corr1))
    return neff
#
def neff2(x1,x2,n):
    """
    calculate effective degree of freedom
    equation: neff = n/(1+2*((n-1)/n*r1*r1'+(n-2)/n*r2*r2'+(n-3)/n*r3*r3'+...))
    """
    ff = min(n//3,1000)+1
    ar1 = np.ma.zeros(ff);ar1.mask=True
    ar2 = np.ma.zeros(ff);ar2.mask=True
    for ii in range(ff):
        ar1[ii] = autocorr(x1,ii)
        ar2[ii] = autocorr(x2,ii)
    rat = np.arange(n-1,n-ff,-1)/float(n)
    ss = np.sum((ar1*ar2)[1:] * rat)
    neff = min(n/(1+2*ss),n-2)
    return neff 
#
def neff3(x1,x2,n):
    """
    calcualte effective degree of freedom
    equation: neff = n*(1-r1*r1')/(1+r2*r2')
    """
    ar1 = autocorr(x1,1)
    ar2 = autocorr(x2,1)
    neff = n*(1-ar1*ar2)/(1+ar1*ar2)
    return neff
#
def autocorr(x,tt):
    """
    autocorrelation of variabile x at lag tt
    """
    corr1 = pearsonr(x[:len(x)-tt],x[tt:len(x)])[0]
    return corr1
#
def calculate_parson_corr_critical_value(N, alpha):
    # N: sample size (or effective sample size)
    df = round(N-2)
    tc = stats.t.ppf(1-alpha/2, df)
    rc = np.sqrt((tc**2/df)/(tc**2/df+1))
    return rc
#
def effectiveSampleSize(data, stepSize = 1):
    samples = len(data)
    assert len(data) > 1,"no stats for short sequences"
    # why???
    # I think the maxLag can change...
    maxLag = min(samples//3, 1000)
    gammaStat = [0,]*maxLag
    #varGammaStat = [0,]*maxLag
    varStat = 0.0;
    if type(data) != np.ndarray:
        data = np.array(data)
    normalizedData = data - data.mean()
    for lag in range(maxLag):
        v1 = normalizedData[:samples-lag]
        v2 = normalizedData[lag:]
        v = v1 * v2
        gammaStat[lag] = sum(v) / len(v)
    #varGammaStat[lag] = sum(v*v) / len(v)
    #varGammaStat[lag] -= gammaStat[0] ** 2
    # print lag, gammaStat[lag], varGammaStat[lag]
        if lag == 0 :
            varStat = gammaStat[0]
        elif lag % 2 == 0 :
            s = gammaStat[lag-1] + gammaStat[lag]
            if s > 0 :
                varStat += 2.0*s
            else :
                break
    # standard error of mean
    # stdErrorOfMean = Math.sqrt(varStat/samples);
    # auto correlation time
    act = stepSize * varStat / gammaStat[0]
    # effective sample size
    ess = (stepSize * samples) / act
    return ess

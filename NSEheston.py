m# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 02:29:49 2021

@author: KUMAR YASHASWI
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy import dot
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv
import scipy
from scipy.stats import norm
import math
import pandas as pd
import datetime
from scipy.linalg import sqrtm
import matplotlib.pylab as plt


def normal_dist1(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def normal_dist(x , mean ,var):
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def multivariate_pdf(vector, mean, cov):
    quadratic_form = np.dot(np.dot(vector-mean,np.linalg.inv(cov)),np.transpose(vector-mean))
    return np.exp(-.5 * quadratic_form)/ (2*np.pi * np.linalg.det(cov))


def degneracy(w_K):
    weight_sqr=0
    for i in range(0,N):
       weight_sqr= weight_sqr + (w_K[i]*w_K[i])
    return weight_sqr

(x_part,yk,yK,Q,R,w_k,N,param,r,dt)=(x_particle,HestonPrice.loc[e,'Measure'],HestonPrice.loc[e,'NextMeasure'],Q,R,w_particle,N,param,r,dt)
def Particle_Filter(x_part,yk,yK,Q,R,w_k,N,param,r,dt):
    xK_part=[]
    w_K=[]
    res_list=[]
    stateMean=[]
    state_cov= 0
    weight_sum=0
    
    for i in range(0,N):
       [mean,state_var] = f(x_part[i],yk,yK,param,r,dt)
       state=np.random.normal(mean,state_var, 1)
       state=state[0]
       xK_part.append(state)
       #print(i," BS ",BS(state,S,k,T))
       #print(i," BS ",state[0]*T)
       price_diff= yK - BS(state,yk,r,dt)
       [mean2,state_var2] = f2(x_part[i],yk,yK,param,r,dt)
       term = (normal_dist(state-mean2,0,state_var2))/(normal_dist(state-mean,0,state_var))
     
       w_K.append(w_k[i]*normal_dist(price_diff,0,x_part[i]*dt)*term)
       stateMean.append(w_K[i] * xK_part[i])
       weight_sum= weight_sum + w_K[i]
    #print(i," BS ",weight_sum)
    state_mean = np.array(stateMean).sum(axis=0)
    
    w_K = [x / weight_sum for x in w_K]
    state_mean = state_mean/ weight_sum 
    '''
    for i in range(0,N):
        error=[x1 - x2 for (x1, x2) in zip(xK_part[i], state_mean)]
        error_matrix=np.expand_dims(np.array(error),axis=1)
        state_cov= np.add(state_cov,w_K[i]*np.dot(error_matrix,error_matrix.T))
    '''
    w_sqr= degneracy(w_K)
    Bessel=(1-w_sqr)
    state_cov=(1/(Bessel))*state_cov 
    n_e= N/2
    Neff = 1/w_sqr
    #print(n_e)
    #print(" PArticle ",w_K)
    xK_resample,w_K = resample_SD(xK_part,w_K,N,Neff,n_e)
    for i in range(0,N):
        res_list.append(w_K[i] * xK_resample[i])
    x_K = np.array(res_list).sum(axis=0)
    
            
    #print("State  ",x_K)
    #print(" PArticle ",xK_resample)
    #print(" Matrix ",state_cov)
    #print("Weight  ",w_K[5])
    return (x_K,xK_resample,state_cov,w_K)


def resample_SD(xK,wK,N,Neff,n_e):
    if Neff < n_e:
        (xK,wK) = resample_particle(xK,wK,N)
        return (xK,wK)
    else:
        return (xK,wK)

def resample_particle(x_part,w_K,N):
    #print(w_K)
    #print(" ")
    w_K.insert(0,0)
    w_K=np.cumsum(w_K)
    #print(w_K)
    s = np.random.uniform(0,1,N)
    #print(s)
    ind=[]
    new_particle =[] 
    for i in range(0,N):
        particle_no=s[i]
        for j in range(1,len(w_K)):
            if particle_no<=w_K[j]:
              ind.append(j-1)
              break
    for k in range(0,N):
        new_particle.append(x_part[ind[k]])
    new_weight=[1/N] * N
    return (new_particle,new_weight)



  
def f(Xn,yk,yK,param,r,dt):
     kcap= param[0]
     theta = param[1]
     sigma = param[2]
     rho = param[3]
     XNZero = Xn + (kcap*(theta)*dt) - ((kcap*dt)*Xn)
     term= sigma*rho*(r - (Xn/2))*dt
     term2= sigma*rho*(yK-yk)
     XNZero = XNZero - term + term2 
     Sigma_N = (sigma*sigma)*Xn*dt*(1-(rho*rho))
     XNOne=Sigma_N
     XN= XNZero
     return [XN,XNOne]
 
def f2(Xn,yk,yK,param,r,dt):
     kcap= param[0]
     theta = param[1]
     sigma = param[2]
     rho = param[3]
     XNZero = Xn + (kcap*(theta)*dt) - ((kcap*dt)*Xn)
     term= sigma*rho*(Xn/2)*dt
     term2= 1 + (sigma*rho*(dt/2))
     XNZero = (XNZero + term)/term2
     Sigma_N = (sigma*sigma)*Xn*dt
     XNOne=Sigma_N/(term2*term2)
     XN= XNZero
     return [XN,XNOne]
    

     
def BS(XK_k,yk,r,dt):
     zK= yk+ ((r - (XK_k/2))*dt)
     return zK


def dataset():
    gap=1
    HestonPrice=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/NSEI.csv')
    HestonPrice['Date'] = pd.to_datetime(HestonPrice['Date'])
    HestonPrice=HestonPrice.dropna()
    HestonPrice=HestonPrice.reset_index()
    hist_vol=[]
    std_sigma= [0]
    time_back = 90
    for e in range(1,HestonPrice.shape[0]):
        #hist_vol.append(HestonPrice.loc[e,'Close'])
        hist_vol.append((HestonPrice.loc[e,'Close'] - HestonPrice.loc[e-1,'Close'])/HestonPrice.loc[e-1,'Close'])
        if(e>=time_back):
            S=np.array(hist_vol[-time_back:])
            sigma= S.std()
            sigma = sigma * (252 ** 0.5)
            std_sigma.append(sigma)
        elif(e<time_back):
            std_sigma.append(0) 
            
    HestonPrice['HistVol'] = std_sigma 
    
    VIX=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/NSE_VIX.csv')
    VIX['Date'] = pd.to_datetime(VIX['Date'])
    
    TresRate=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/RiskRate.csv')
    TresRate['Date'] = pd.to_datetime(TresRate['Date'])
    TresRate['risk_perc']=TresRate['risk_perc']/100
    HestonPrice = pd.merge(HestonPrice,TresRate[['Date', 'risk_perc']],on='Date')
    HestonPrice = pd.merge(HestonPrice,VIX[['Date', 'volatility']],on='Date')
    
    base=HestonPrice.loc[231,'Close']
    HestonPrice['CloseRatio']=HestonPrice['Close']/base
    HestonPrice['Measure']=np.log(HestonPrice['CloseRatio'])
    HestonPrice['NextMeasure'] = HestonPrice['Measure'].shift(-gap)
    
    start_date = pd.datetime(2007,12,4)
    end_date = pd.datetime(2020,12,31)
    mask = (HestonPrice['Date'] >= start_date) & (HestonPrice['Date'] <= end_date)
    HestonPrice = HestonPrice.loc[mask]
    HestonPrice=HestonPrice.reset_index()
    HestonPrice = HestonPrice[['Date', 'Close','CloseRatio','Measure','NextMeasure','risk_perc','volatility','HistVol']]
    
    date=[]
    V_prev= []
    V_present=[]
    y_prev=[]
    y_present=[]
    PF_prices=[]
    
    vix_vol=[]
    his_vol=[]
    
    real_prices=[]
    
    vol_PF_train =[]
    theta_PF_train = []
    rho_PF_train = []
    kappa_PF_train = []
    sigma_PF_train = []
    
    xbar_0 = 0.2
    P_0 = 0.009
    x_temp= xbar_0
    P_temp = P_0
    dt=gap/252
    #dt =0.01
    r=HestonPrice.loc[0,'risk_perc']
    Q= dt
    R= dt
    #param=[0.05,4.2,0.25,-0.9]
    param=[7.438950055361843, 0.2377555104605398, 1.131560078387605, -0.8]
    #param=[8.555300710873333, 0.18575547331250436, 1.034020097411243705, -0.9627228975151242]
    state = xbar_0
    N=1000
    x_particle= np.random.normal(xbar_0, P_0, N)
    w_particle= [1/N] * N
    e=0
    for e in range(0,HestonPrice.shape[0],gap):
        V_prev.append(state)
        y_prev.append(HestonPrice.loc[e,'Measure'])
        r=HestonPrice.loc[e,'risk_perc']

        (state,x_particle,particle_cov,w_particle)=Particle_Filter(x_particle,HestonPrice.loc[e,'Measure'],HestonPrice.loc[e,'NextMeasure'],Q,R,w_particle,N,param,r,dt)
        
        V_present.append(state)
        y_present.append(HestonPrice.loc[e,'NextMeasure'])
        
        date.append(HestonPrice.loc[e,'Date'])
        PF_prices.append(BS(state,HestonPrice.loc[e,'Measure'],r,dt))
        real_prices.append(HestonPrice.loc[e,'NextMeasure'])
        vol_PF_train.append(state)
        vix_vol.append(HestonPrice.loc[e,'volatility'])
        his_vol.append(HestonPrice.loc[e,'HistVol'])
        
        theta_PF_train.append(param[1])
        rho_PF_train.append(param[3])
        kappa_PF_train.append(param[0])
        sigma_PF_train.append(param[2])
        
        if(e>=300):
            param1 = NMLE_Alt(V_prev,V_present,y_prev,y_present,r,dt,param[0],param[1],param[2])
            #param = param1
            if(2*param1[0]*param1[1]>(param1[3]*param1[3])):
                param=param1
            print(param)

plt.plot([a*(252 ** 0.5) for a in vol_PF_train])
plt.plot(vol_PF_train)
plt.plot(vix_vol)
plt.plot(his_vol)

plt.plot(kappa_PF_train[301:])
plt.plot(theta_PF_train[301:])
plt.plot(sigma_PF_train[301:])
plt.plot(rho_PF_train[301:])

a=date

df = pd.DataFrame(list(zip(date, vix_vol,his_vol,vol_PF_train)),
               columns =['Date', 'CBOE','history','PCRLB'])



plt.figure(figsize = (15, 4))   
#plt.plot(UKF_prices_test)
#plt.plot(real_prices_test)
#plt.plot(a,real_prices_test,label="RL Portfolio-Return")
plt.plot(a,vol_PF_train,linestyle='-',label="PCRLB based Switching Filter",color='blue')
plt.plot(a,vix_vol,linestyle='-',label="VIX-NSE Index",color='green')
plt.plot(a,his_vol,linestyle='-',label="Historical Volatility",color='red')
#plt.plot(label="PCRLB")
plt.legend(loc="upper left",fontsize=13)
plt.xlabel('Date',fontsize=15)
plt.ylabel('Volatility Estimate', fontsize=15)
plt.title("National Stock Exchange (NSE)", fontsize=15)
plt.grid()
plt.show()  

plt.figure(figsize = (15, 4))   
#plt.plot(UKF_prices_test)
#plt.plot(real_prices_test)
#plt.plot(a,real_prices_test,label="RL Portfolio-Return")
plt.plot(a,vix_vol,linestyle='-',label="PCRLB",color='green')
#plt.plot(label="PCRLB")
#plt.legend(loc="upper left")
plt.xlabel('Date')
plt.ylabel('Volatility Estimate')
plt.title("NSE VIX Index")
plt.grid()
plt.show()

plt.figure(figsize = (15, 4))   
#plt.plot(UKF_prices_test)
#plt.plot(real_prices_test)
#plt.plot(a,real_prices_test,label="RL Portfolio-Return")
plt.plot(a,his_vol,linestyle='-',label="PCRLB",color='red')
#plt.plot(label="PCRLB")
#plt.legend(loc="upper left")
plt.xlabel('Date')
plt.ylabel('Volatility Estimate')
plt.title("Historical Volatility- NSE")
plt.grid()
plt.show()   





MSE(VIX_vol,PF_vol)- 0.0912498333187159
MSE(VIX_vol,His_vol)- 0.11116833396997806
MSE(VIX_vol,real_vol)- 0.08046331263905969

def MSE(real,predict):
    N=len(real)
    error=[(a-b)*(a-b) for (a,b) in zip(real,predict)]
    error=sum(error)
    error=error/N
    #error=error/(strike*strike)
    return np.sqrt(error)



# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 00:28:40 2021

@author: KUMAR YASHASWI
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 02:29:37 2021

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
from pandas import DataFrame

def extended_KF_predict(X,yk,yK,P,r,dt,param,Q,R):
     XK_k = f_Ekf(X,yk,yK,param,r,dt)[0]
     A= Jacobian_x(X,yk,yK,param,r,dt)
     W=Jacobian_x_err(X,yk,yK,param,r,dt)
     P_k = (A*P*A)+(W*W*Q)
     #print(S)
     #print(T)
     #print(XK_k)
     y1_k = BS(XK_k,yk,r,dt)
     #print(y1_k)
     #print(' ')
     H = Jacobian_y(XK_k,yk,yK,param,r,dt)
     U= Jacobian_y_err(XK_k,yk,yK,param,r,dt)
     #print(H)
     #print(y1_k)
     F_K = (H*P_k*H)+(U*U*R)
     return (y1_k,P_k,F_K,XK_k,H)

def Jacobian_x(Xn,yk,yK,param,r,dt):
     kcap= param[0]
     theta = param[1]
     sigma = param[2]
     rho = param[3]
     Hk= 1 - (kcap*dt)+(sigma*rho*(dt/2))
     return Hk

def Jacobian_y(Xn,yk,yK,param,r,dt):
     kcap= param[0]
     theta = param[1]
     sigma = param[2]
     rho = param[3]
     HK= dt/2
     return -HK

def Jacobian_x_err(Xn,yk,yK,param,r,dt):
     Xn=np.max([Xn,0.01])
     kcap= param[0]
     theta = param[1]
     sigma = param[2]
     rho = param[3]
     Hk= (sigma*np.sqrt(Xn*(1-(rho*rho))))
     return Hk

def Jacobian_y_err(Xn,yk,yK,param,r,dt):
     Xn=np.max([Xn,0.01])
     kcap= param[0]
     theta = param[1]
     sigma = param[2]
     rho = param[3]
     HK= np.sqrt(Xn)
     return HK

  
def f_Ekf(Xn,yk,yK,param,r,dt):
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
     return (XN,Sigma_N)
    
def extended_KF_update(y_K,y1_k, P_k, F_k, XK_k,H):
     #print("H matrix   ",H)
     #print("Real Price  ",y_K)
     #print("Estimated Value  ",y1_k)
     b_K = y_K - y1_k
     #F_k=np.array((F_k))
     #F_k= np.expand_dims(F_k, 0)
     #print("State Cov", P_k)
     K_K = (P_k*H)*(1/F_k)
     #print('Difference   ',b_K)
     #print("State  ",XK_k)
     #print("Kalman Gain  ",K_K)  
     #print("Kalman matrix   ",dot(K_K,b_K).T)
     XK_K = XK_k + (K_K*b_K)
     #print(dot(K_K,b_K).T)
     P_final = P_k - (K_K*H*P_k)
     #print("New State  ",XK_K)
     #print(dot(K_K,b_K).T)
     #print(P_final)
     #print(' ')
     return (XK_K,P_final)
     
def BS(XK_k,yk,r,dt):
     zK= yk+ ((r - (XK_k/2))*dt)
     return zK


HestonPrice=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/SP.csv')
HestonPrice['Date'] = pd.to_datetime(HestonPrice['Date'])
hist_vol=[]
std_sigma= [0]
for e in range(1,HestonPrice.shape[0]):
    #hist_vol.append(HestonPrice.loc[e,'Close'])
    hist_vol.append(np.log(HestonPrice.loc[e,'Close']/HestonPrice.loc[e-1,'Close']))
    if(e>=60):
        S=np.array(hist_vol[-60:])
        sigma= S.std()
        sigma = sigma * (252 ** 0.5)
        std_sigma.append(sigma)
    elif(e<60):
        std_sigma.append(0) 
        
HestonPrice['HistVol'] = std_sigma  

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
    EKF_prices=[]
    
    real_prices=[]
    
    vix_vol=[]
    his_vol=[]
    vol_EKF_train =[]
    theta_EKF_train = []
    rho_EKF_train = []
    kappa_EKF_train = []
    sigma_EKF_train = []
    
    
    xbar_0 = 0.2
    #xbar_0 = 0.24/16
    P_0 = 0.008
    x_temp= xbar_0
    P_temp = P_0
    #dt=gap/HestonPrice.shape[0]
    dt= gap/252
    #dt=0.1
    r=HestonPrice.loc[0,'risk_perc']
    Q= dt
    R= dt
    
    param=[7.438950055361843, 0.2377555104605398, 1.731560078387605, -0.8]
    #param=[7.576411884378964, 0.04994945181080531, 0.013764198385755776, -0.8]
    #param=[25.84210054782672, 0.1059546866171389, 0.12097375045321715, -0.8]
    #param=[3.7477386793240974, 0.01255304227947041, 0.06396015867843322, -0.8]
    #param=[8.31,0.3/(252 ** 0.5),0.066,-0.8]
    
    #e=0
    for e in range(0,HestonPrice.shape[0],gap):
        print(e)
        V_prev.append(x_temp)
        y_prev.append(HestonPrice.loc[e,'Measure'])
        r=HestonPrice.loc[e,'risk_perc']
        #PredResult = extended_KF_predict(x_temp,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],optionPrice.loc[e,'PriceChange'],P_temp, Q,R)
        #UpdateResult_EKF = extended_KF_update(optionPrice.loc[e,'OptionPR'],PredResult[0], PredResult[1], PredResult[2], PredResult[3],PredResult[4])
        #(x_temp,P_temp)= UpdateResult_EKF
        
        PredResult=extended_KF_predict(x_temp,HestonPrice.loc[e,'Measure'],HestonPrice.loc[e,'NextMeasure'],P_temp,r,dt,param,Q,R)
        UpdateResult_EKF = extended_KF_update(HestonPrice.loc[e,'NextMeasure'],PredResult[0], PredResult[1], PredResult[2], PredResult[3],PredResult[4])
        (x_temp,P_temp)= UpdateResult_EKF
        
        V_present.append(x_temp)
        y_present.append(HestonPrice.loc[e,'NextMeasure'])
        
        date.append(HestonPrice.loc[e,'Date'])
        EKF_prices.append(BS(x_temp,HestonPrice.loc[e,'Measure'],r,dt))
        real_prices.append(HestonPrice.loc[e,'NextMeasure'])
        vol_EKF_train.append(x_temp)
        vix_vol.append(HestonPrice.loc[e,'volatility'])
        his_vol.append(HestonPrice.loc[e,'HistVol'])
        #historical_vol.append(HestonPrice.loc[e,'Close'])
        theta_EKF_train.append(param[1])
        rho_EKF_train.append(param[3])
        kappa_EKF_train.append(param[0])
        sigma_EKF_train.append(param[2])
        if(e>=300):
            look = 60
            #param1 = NMLE_Alt(V_prev[-look:],V_present[-look:],y_prev[-look:],y_present[-look:],r,dt,param[2])
            param1 = NMLE_Alt(V_prev,V_present,y_prev,y_present,r,dt,param[0],param[1],param[2])
            param = param1
            if(2*param1[0]*param1[1]>(param1[3]*param1[3])):
                param=param1
            print(param)
        
        plt.plot([a*(1) for a in vol_EKF_train])
        plt.plot(vix_vol)
        plt.plot(his_vol)
plt.plot(kappa_EKF_train[301:])
plt.plot(theta_EKF_train[301:])
plt.plot(sigma_EKF_train[301:])

plt.plot(kappa_EKF_train)
plt.plot(kappa_PF_train)


plt.plot(b) 
plt.plot(ans)   
plt.plot(vol_EKF_train)    
plt.plot(kappa_EKF_train)
plt.plot(theta_EKF_train)
plt.plot(sigma_EKF_train)
plt.plot(rho_EKF_train)  

plt.plot(real_prices)  
plt.plot(EKF_prices) 
 
[np.sqrt(a/b) for a, b in zip([1,2,3,4],[4,3,2,1])]


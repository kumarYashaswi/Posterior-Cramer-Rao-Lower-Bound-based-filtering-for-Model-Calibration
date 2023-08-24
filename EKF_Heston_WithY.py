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
    gap=7
    HestonPrice=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/SP.csv')
    HestonPrice['Date'] = pd.to_datetime(HestonPrice['Date'])
    hist_vol=[]
    std_sigma= [0]
    time_back = 7
    for e in range(1,HestonPrice.shape[0]):
        #hist_vol.append(HestonPrice.loc[e,'Close'])
        hist_vol.append((HestonPrice.loc[e,'Close'] - HestonPrice.loc[e-1,'Close'])/HestonPrice.loc[e-1,'Close'])
        if(e>=time_back):
            S=np.array(hist_vol[:])
            sigma= S.std()
            sigma = sigma * (252 ** 0.5)
            std_sigma.append(sigma)
        elif(e<time_back):
            std_sigma.append(0) 
            
    HestonPrice['HistVol'] = std_sigma 
    
    VIX=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/VIX_history.csv')
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
    end_date = pd.datetime(2020,8,30)
    mask = (HestonPrice['Date'] >= start_date) & (HestonPrice['Date'] <= end_date)
    HestonPrice = HestonPrice.loc[mask]
    HestonPrice=HestonPrice.reset_index()
    HestonPrice = HestonPrice[['Date', 'Close','CloseRatio','Measure','NextMeasure','risk_perc','volatility','HistVol']]
    
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
    
    
    xbar_0 = 0.04
    #xbar_0 = 0.24/16
    P_0 = 0.08
    x_temp= xbar_0
    P_temp = P_0
    dt=7/HestonPrice.shape[0]
    dt= 7/252
    dt=0.1
    r=HestonPrice.loc[0,'risk_perc']
    Q= dt
    R= dt
    
    param=[7.5,0.05,0.05,-0.8]
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
        if(e>=1200):
            look = 60
            #param1 = NMLE_Alt(V_prev[-look:],V_present[-look:],y_prev[-look:],y_present[-look:],r,dt,param[2])
            param1 = NMLE_Alt(V_prev,V_present,y_prev,y_present,r,dt,param[0],param[1],param[2])
            #param = param1
            if(2*param1[0]*param1[1]>(param1[3]*param1[3])):
                param=param1
            print(param)
        
        plt.plot([a*(252 ** 0.5) for a in vol_EKF_train])
        plt.plot(vix_vol)
        plt.plot(his_vol)
plt.plot(kappa_EKF_train)
plt.plot(theta_EKF_train)
plt.plot(sigma_EKF_train)
a,b=generate_heston_paths(100, 10, 0.005, 1, 0.25, 0.2, -0.8, 0.1, 
                          1000, 1, return_vol=True)
#a,b=generate_heston_paths(100, 1, 0.005, 1, 0.25, 0.2, -0.8, 0.5, 
                          10000, 10000, return_vol=True)
plt.plot(a)     
plt.plot(100*b)  
a=np.average(a, axis=0)
a=a.tolist()

b=np.average(b, axis=0)
b=b.tolist()
S, T, r, kappa, theta, v_0, rho, xi, 
                          steps, Npaths, return_vol=False):
 1/1500       
def dataset1():
    gap= 10
    ans= []
    ans1= []
    HestonPrice=DataFrame (a,columns=['Close'])
    
    #base=HestonPrice.loc[0,'Close']
    base=1
    HestonPrice['CloseRatio']=HestonPrice['Close']/base
    HestonPrice['Measure']=np.log(HestonPrice['CloseRatio'])
    HestonPrice['NextMeasure'] = HestonPrice['Measure'].shift(-gap)
    
    HestonPrice = HestonPrice[['Close','CloseRatio','Measure','NextMeasure']]
    
    V_prev= []
    V_present=[]
    y_prev=[]
    y_present=[]
    EKF_prices=[]
    
    real_prices=[]
    
    vol_EKF_train =[]
    theta_EKF_train = []
    rho_EKF_train = []
    kappa_EKF_train = []
    sigma_EKF_train = []
    
    
    xbar_0 = 0.2
    x_temp= xbar_0
    
    dt=0.01
    dt1=0.01
    P_0 = 0.005
    P_temp = P_0
    r=0.005
    Q= dt
    R= dt
    
    #param=[0.8,0.2708,0.7,0.8]
    param=[1.5,0.3,0.2,-0.8]
    
    #e=1
    for e in range(1,HestonPrice.shape[0],gap):
        V_prev.append(x_temp)
        y_prev.append(HestonPrice.loc[e,'Measure'])
        
        
        PredResult=extended_KF_predict(x_temp,HestonPrice.loc[e,'Measure'],HestonPrice.loc[e,'NextMeasure'],P_temp,r,dt,param,Q,R)
        UpdateResult_EKF = extended_KF_update(HestonPrice.loc[e,'NextMeasure'],PredResult[0], PredResult[1], PredResult[2], PredResult[3],PredResult[4])
        (x_temp,P_temp)= UpdateResult_EKF
        
        V_present.append(x_temp)
        y_present.append(HestonPrice.loc[e,'NextMeasure'])
        
        EKF_prices.append(BS(x_temp,HestonPrice.loc[e,'Measure'],r,dt))
        real_prices.append(HestonPrice.loc[e,'NextMeasure'])
        vol_EKF_train.append(x_temp)
        theta_EKF_train.append(param[1])
        rho_EKF_train.append(param[3])
        kappa_EKF_train.append(param[0])
        sigma_EKF_train.append(param[2])
        #ans.append(b[e])
        #ans1.append(b[e-1])
        if(e>=200):
            look = 10000
            #param1 = fPMLE_Alt(V_prev[-look:],V_present[-look:],y_prev[-look:],y_present[-look:],r,dt1,param[0],param[1],param[2])
            param1 = NMLE_Alt(V_prev,V_present,y_prev,y_present,r,dt,param[0],param[1],param[2])
            #param1 = fPMLE_Alt(V_prev,V_present,y_prev,y_present,r,dt,param[0],param[1],param[2])
            param = param1
            if(2*param1[0]*param1[1]>(param1[3]*param1[3])):
                param=param1
        print(param)
            #param=[1,0.25,0.5,-0.8]
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

def fPMLE_Alt(V_prev,V_present,y_prev,y_present,r,dt,kcap,theta,sigma):
     n=len(V_present)
     V_prev_term =[1/a for a in V_prev]
     V_mix_term=[a/b for a, b in zip(V_present,V_prev)]
     const=1/(n*n)
     Beta3 =0 
     likelihood =0
     Beta1 = ((const*sum(V_present)*sum(V_prev_term))-(sum(V_mix_term)/n))/((const*sum(V_prev)*sum(V_prev_term))-1)
     Beta2 = ((sum(V_mix_term)/n)-(Beta1))/((1-Beta1)*sum(V_prev_term)/n)
     #Beta2 = ((sum(V_mix_term)/n)-(Beta1))/((1-Beta1)*sum(V_prev)/n)
     V_linear_term=[a*b for a, b in zip(V_present,V_prev_term)]
     for i in range(0,n):
         term1=(V_present[i]) - (Beta1*V_prev[i]) - (Beta2*(1-Beta1)*(1-Beta1))
         Beta3= Beta3 + (term1*V_prev_term[i])
     Beta3=Beta3/n
     if(Beta1>0):
         kcap= -np.log(Beta1)/dt
     if(Beta2>0):
         theta = Beta2
     temp=(2*kcap*Beta3)/(1-(Beta1*Beta1))
     if(temp>0):
         sigma = np.sqrt((2*kcap*Beta3)/(1-(Beta1*Beta1)))
     for i in range(0,n):
         V_k=V_prev[i]
         V_K=V_present[i]
         y1= y_prev[i]
         y2= y_present[i]
         delta_W1 = (y2 - y1 - (r - (V_k/2))*dt)/(np.sqrt(V_k))
         delta_W2 = (V_K - V_k - (theta - V_k)*kcap*dt)/(sigma*np.sqrt(V_k))
         likelihood= likelihood + (delta_W1*delta_W2)
     likelihood = likelihood/(n*dt)
     rho = likelihood
     rho = -0.8
     return [kcap,theta,sigma,rho]


def NMLE_Alt(V_prev,V_present,y_prev,y_present,r,dt,kcap1,theta1,sigma1):
    n=len(V_present)
    const=1/(n*n)
    likelihood=0
    sigma =0 
    V_sqrt_term=[np.sqrt(a*b) for a, b in zip(V_present,V_prev)] 
    V_sqrt_term_div=[np.sqrt(a/b) for a, b in zip(V_present,V_prev)] 
    V_prev_term =[1/a for a in V_prev]
    V_mix_term=[a/b for a, b in zip(V_present,V_prev)]
    term1 = sum(V_sqrt_term)/n
    term2 = (sum(V_sqrt_term_div)*sum(V_prev))*const
    term3 = (dt*const*0.5)*(sum(V_prev_term)*sum(V_prev))
    term3 = (dt*0.5) - term3
    Pcap = (term1-term2)/term3
    
    Beta1 = ((const*sum(V_present)*sum(V_prev_term))-(sum(V_mix_term)/n))/((const*sum(V_prev)*sum(V_prev_term))-1)
    kcap = (2/dt)*(1 + (((Pcap*dt)/(2*n))*sum(V_prev_term)) - (sum(V_sqrt_term_div)/n))
    #if(kcap<0):
    #kcap= kcap1
    for i in range(0,n):
         sigma_est = np.sqrt(V_present[i]) - np.sqrt(V_prev[i]) - ((dt*0.5)*(1/np.sqrt(V_prev[i]))*(Pcap-(kcap*V_prev[i])))
         sigma_est = sigma_est*sigma_est
         sigma= sigma + sigma_est
         
    sigma = 4*sigma/(dt*n)
    if(sigma<0):
        sigma2= sigma1*sigma1
    #sigma =np.sqrt(sigma2)
    theta = (Pcap + (sigma*sigma/4))/kcap
    
    #if(theta<0):
      #  theta = theta1
    for i in range(0,n):
        V_k=V_prev[i]
        V_K=V_present[i]
        y1= y_prev[i]
        y2= y_present[i]
        delta_W1 = (y2 - y1 - (r - (V_k/2))*dt)/(np.sqrt(V_k))
        delta_W2 = (V_K - V_k - ((theta - V_k)*kcap*dt))/(sigma*np.sqrt(V_k))
        likelihood= likelihood + (delta_W1*delta_W2)
    likelihood = likelihood/(n*dt)
    if(abs(likelihood)>1):
        rho = -0.8
    else:
        rho = likelihood
    return [kcap,theta,sigma,rho]
'''
def createHeston():
    S_0 = 100
    V_0 = 0.2 
    Stock_Price_List =[S_0]
    Vol_list = [V_0]
    dt = 0.01
    n = 15000
    r = 0.005
    param = [1,0.25,0.5,-0.9]
    kcap= param[0]
    theta = param[1]
    sigma = param[2]
    rho = param[3]
    i=0
    for i in range(0,n):
        Noise1 = np.random.normal(0,dt, 1).tolist()[0]
        Noise = np.random.normal(0,dt, 1).tolist()[0]
        Noise2 = Noise - (rho*Noise1)
        Noise2 = Noise2/(np.sqrt((1-(rho*rho))))
        #print((sigma*Noise1*np.sqrt(V_0)))
        V_0 = V_0 + (kcap*dt*(theta - V_0))+ (sigma*Noise1*np.sqrt(V_0))
        Vol_list.append(V_0)
        #V_0=V_0*100
        S_1 = np.log(S_0) + ((r - (V_0/2))*dt) + (Noise2*np.sqrt(V_0*(1-(rho*rho))))+ (rho*Noise1*np.sqrt(V_0))
        print(S_1)
        S_0 = math.exp(S_1)
        #V_0= V_0/100
        #print(S_0)
        Stock_Price_List.append(S_0)
        Vol_list.append(V_0)
 '''
a,b=generate_heston_paths(100, 1, 0.005, 1, 0.25, 0.2, -0.8, 0.5, 
                          2000, 100, return_vol=True)

def generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi, 
                          steps, Npaths, return_vol=False):
    dt = T/steps
    size = (Npaths, steps)
    prices = np.zeros(size)
    sigs = np.zeros(size)
    S_t = S
    v_t = v_0
    for t in range(steps):
        WT = np.random.multivariate_normal(np.array([0,0]), 
                                           cov = np.array([[1,rho],
                                                          [rho,1]]), 
                                           size=Npaths) * np.sqrt(dt) 
        
        S_t = S_t*(np.exp( (r- 0.5*v_t)*dt+ np.sqrt(v_t) *WT[:,0] ) ) 
        v_t = np.abs(v_t + kappa*(theta-v_t)*dt + xi*np.sqrt(v_t)*WT[:,1])
        prices[:, t] = S_t
        sigs[:, t] = v_t
    
    if return_vol:
        return prices, sigs
    
    return prices





        
plt.plot(a)
plt.plot(Vol_list)   
plt.plot(b[10])



plt.plot(EKF_prices)
plt.plot(vol_EKF_train)
plt.plot(kappa_EKF_train)
plt.plot(theta_EKF_train)
plt.plot(sigma_EKF_train)
plt.plot(rho_EKF_train)
plt.title('Pf performance')
plt.legend(['True Value','UKF Estimate'])
plt.xlabel('timestep')
plt.ylabel('Concentration') 
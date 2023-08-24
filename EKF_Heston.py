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



(X,P_k,r,dt,param,Q,R)=(x_temp,P_temp,r,dt,param,Q,R)

def extended_KF_predict(X,P_k,r,dt,param,Q,R):
     [XK_k,P_K,L] = f(X,P_k,Q,param,r,dt)
     
     [H,Mk] = M(X,P_k,Q,param,r,dt)
     #print(y1_k)
     #print(' ')
     term2 = (H*H*P_k) + (np.dot(Mk,np.dot(Q,Mk.T)).tolist()[0][0]) + (np.dot(L,np.dot(Q,Mk.T)).tolist()[0][0]*H) + (np.dot(Mk,np.dot(Q,L.T)).tolist()[0][0]*H)
     
     
     return [XK_k,P_K,term2,Mk,H,L]


term = np.dot(L,np.dot(Q,Mk.T)).tolist()[0]
term = term + (P_K*H)

term2 = (H*H*P_k) + (np.dot(Mk,np.dot(Q,Mk.T)).tolist()[0]) + (np.dot(L,np.dot(Q,Mk.T)).tolist()[0]*H) + (np.dot(Mk,np.dot(Q,L.T)).tolist()[0]*H)
1/term2

def M(Xn,Pk,Qk,param,r,dt):
     kcap= param[0]
     theta = param[1]
     sigma = param[2]
     rho = param[3]
     Hk= dt/2
     term1 = np.sqrt((1-(rho*rho))*Xn*dt)
     term2 = rho*np.sqrt(Xn*dt)
     Mk= [term1,term2]
     Mk= np.array(Mk)
     Mk=np.expand_dims(Mk,0)
     return [-Hk,Mk]


(Xn,Pk,Qk,param,r,dt)=(X,P_k,Q,param,r,dt)
def f(Xn,Pk,Qk,param,r,dt):
     kcap= param[0]
     theta = param[1]
     sigma = param[2]
     rho = param[3]
     
     Fk=1 - (kcap*dt)
     Lk= [0,(sigma*np.sqrt(Xn*dt))]
     Lk= np.array(Lk)
     Lk=np.expand_dims(Lk,0)
     XNZero = Xn + (kcap*(theta)*dt) - ((kcap*dt)*Xn)
     term1 = (Fk*Pk*Fk)
     term2 = dot(Lk,dot(Qk,Lk.T)).tolist()[0][0]
     Sigma_N =term1 + term2 
     XNOne=Sigma_N
     XN=[XNZero,XNOne,Lk]
     return XN

[XK_k,y_K,term2, P_K, Q, Mk,L,H]=[PredResult[0],HestonPrice.loc[e,'NextMeasure'],PredResult[1], PredResult[2],Q, PredResult[3],PredResult[4],PredResult[5]]
def extended_KF_update(XK_k,y_K, P_K,term2, Q, Mk,H,L):
     term = np.dot(L,np.dot(Q,Mk.T)).tolist()[0][0]
     term = term + (P_K*H)
     K_K = term*(1/term2)
     b_K = y_K - (XK_k/2)
     XK_K = XK_k + (K_K*b_K)
     r_K = (H*P_K) + np.dot(Mk,L.T).tolist()[0][0]
     P_final = P_K - (K_K*r_K)
     print("New State  ",XK_K)
     #print(dot(K_K,b_K).T)
     #print(P_final)
     print(' ')
     return (XK_K,P_final)
     
def BS(XK_k,S,k,T):
     zK= ((r - (XK_k/2))*dt)
     return zK



a,b=generate_heston_paths(100, 10, 0.005, 1, 0.25, 0.2, -0.8, 0.5, 
                          1000, 1000, return_vol=True)

plt.plot(a)     

a=np.average(a, axis=0)
a=a.tolist()

b=np.average(b, axis=0)
b=b.tolist()




def dataset():
    gap= 10
    ans= []
    ans1= []
    HestonPrice=DataFrame(a,columns=['Close'])
    
    #base=HestonPrice.loc[0,'Close']
    base=1
    HestonPrice['CloseRatio']=HestonPrice['Close']/base
    HestonPrice['Meas']=np.log(HestonPrice['CloseRatio'])
    HestonPrice['Next'] = HestonPrice['Meas'].shift(-gap)
    HestonPrice['NextMeasure'] = HestonPrice['Next']-HestonPrice['Meas']
    #HestonPrice['Measure'] = HestonPrice['Measure'].shift(-gap)
    HestonPrice = HestonPrice[['Close','CloseRatio','Meas','Next','NextMeasure']]
    
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
    P_0 = 0.0004
    P_temp = P_0
    r=0.005
    Q = []
    cor=-0.8
    Q.append([dt,cor])
    Q.append([cor,dt])
    Q=np.array(Q)
    R= dt
    
    #param=[0.8,0.2708,0.7,0.8]
    param=[1.5,0.3,0.2,-0.8]
    
    #e=1
    for e in range(1,HestonPrice.shape[0],gap):
        V_prev.append(x_temp)
        #y_prev.append(HestonPrice.loc[e,'Measure'])
        
        #[XK_k,P_K,term2,H,L,Mk]
        PredResult=extended_KF_predict(x_temp,P_temp,r,dt,param,Q,R)
        UpdateResult_EKF = extended_KF_update(PredResult[0],HestonPrice.loc[e,'NextMeasure'],PredResult[1], PredResult[2],Q, PredResult[3],PredResult[4],PredResult[5])
        (x_temp,P_temp)= UpdateResult_EKF
        
        V_present.append(x_temp)
        #y_present.append(HestonPrice.loc[e,'NextMeasure'])
        
        EKF_prices.append(BS(x_temp,HestonPrice.loc[e,'NextMeasure'],r,dt))
        real_prices.append(HestonPrice.loc[e,'NextMeasure'])
        vol_EKF_train.append(x_temp)
        theta_EKF_train.append(param[1])
        rho_EKF_train.append(param[3])
        kappa_EKF_train.append(param[0])
        sigma_EKF_train.append(param[2])
        ans.append(b[e])
        ans1.append(b[e-1])
        if(e>=200):
            look = 10000
            #param1 = fPMLE_Alt(V_prev[-look:],V_present[-look:],y_prev[-look:],y_present[-look:],r,dt1,param[0],param[1],param[2])
            #param1 = NMLE_Alt(V_prev,V_present,y_prev,y_present,r,dt,param[0],param[1],param[2])
            param1 = fPMLE_Alt1(V_prev,V_present,r,dt,param[0],param[1],param[2])
            param = param1
            if(2*param1[0]*param1[1]>(param1[3]*param1[3])):
                param=param1
        print(param)
        
def fPMLE_Alt1(V_prev,V_present,r,dt,kcap,theta,sigma):
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
     
     rho = -0.8
     return [kcap,theta,sigma,rho]
 
    
plt.plot(b) 
plt.plot(ans)   
plt.plot(vol_EKF_train)    
plt.plot(kappa_EKF_train)
plt.plot(theta_EKF_train)
plt.plot(sigma_EKF_train)
plt.plot(rho_EKF_train)  

plt.plot(real_prices)  
plt.plot(EKF_prices) 
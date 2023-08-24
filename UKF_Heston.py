# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 02:29:20 2021

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





def Unscented_KF_Transform(xa_uns,P_a,L):
    W_m=[]
    W_c=[]
    X_uns=[]
    #L=5
    alpha=0.00001
    kappa= 3-L
    lamda = (alpha*alpha)*(L + kappa) - L
    beta=2
    #print(xa_uns)
    #print(P_a)
    #print('check')
    #print(' ')
    #array_2d = np.array([[1, 4], [9, 16]], dtype=np.float)
    #array_2d = np.array([[1, 4]], dtype=np.float)
    #array_2d=array_2d.tolist()[0]
    #X_uns.append(array_2d)
    X_uns.append(xa_uns.tolist())
    W_m.append(lamda/(lamda+L))
    W_c.append((lamda/(lamda+L)) + 1 - (alpha*alpha) + beta)
    #print(L + lamda)
    #print((L + lamda)*P_a)
    #np.sqrt((L + lamda)*P_a[:,3-1])
    #np.add(xa_uns,np.sqrt((L + lamda)*P_a[:,1]))
    for i in range(1,L+1):
       #print(np.sqrt((L + lamda)*P_a[:,i-1]))
       X_uns.append(np.add(xa_uns,np.linalg.cholesky((L + lamda)*P_a)[:,i-1]).tolist())
       W_m.append(1/(2*(lamda+L)))
       W_c.append(1/(2*(lamda+L)))
    for j in range(1,L+1):
       X_uns.append(np.subtract(xa_uns,np.linalg.cholesky((L + lamda)*P_a)[:,j-1]).tolist())
       W_m.append(1/(2*(lamda+L)))
       W_c.append(1/(2*(lamda+L)))
       
    X_uns_matrix = np.array(X_uns).T 
    #print(X_uns_matrix)
    X_x = X_uns_matrix[0:1,:]
    X_w = X_uns_matrix[1:2,:]
    X_v = X_uns_matrix[2:3,:]
    #print(X_uns) 
    #print(X_uns_matrix)
    #print(X_x)
    #print(X_w)
    #print(X_v) 
    #print(' ')
    return (X_uns,X_uns_matrix,X_x,X_w,X_v,W_m,W_c)

(xa_uns,P_a,L,param,y_prev,y_present,r,dt)=(xa_uns,P_a,L,param,y_k,y_K,r,dt)

def Unscented_KF_predict(xa_uns,P_a,L,param,y_prev,y_present,r,dt,last_vol):
     # Sigma Points of x_k
     X_uns,X_uns_matrix,X_x,X_w,X_v,W_m,W_c = Unscented_KF_Transform(xa_uns,P_a,L)
     ChiK_k=[]
     y1_k=[]
     weighted_state= 0
     weighted_measure=0
     cov = 0
     cov_y = 0
     cov_yx=0
     # Mean of x
     #i=3
     X_x=np.array(X_x)
     X_v=np.array(X_v)
     X_w=np.array(X_w)
     for i in range(0,(2*L)+1):
        #print("state1",X_x[:,i].tolist())
        a= X_x[:,i].tolist()[0]
        b=X_w[:,i].tolist()[0]
        #print("state2",X_w[:,i].tolist())
        #print(b)
        mean=state(a,b,y_prev,y_present,param,r,dt)
        ChiK_k.append(mean)
        #print(weighted_state.shape)
        #print(mean.shape)
        weighted_state= weighted_state+(W_m[i]*mean)
        #print(' ')
        #print(W_m[i]*mean)
        #print(' ')
        
     # Covariance of y
     for i in range(0,(2*L)+1):
         #print(np.expand_dims(np.subtract(ChiK_k[i],weighted_state),axis=1).shape)
         #print(np.expand_dims(np.subtract(ChiK_k[i],weighted_state),axis=1).T.shape)
         P_xx= (ChiK_k[i]-weighted_state)*(ChiK_k[i]-weighted_state)
         #print(P_xx)
         #print(cov.shape)
         cov= cov+(W_c[i]*P_xx)
         #print(' ')
     # Mean of y
     for i in range(0,(2*L)+1):
         #np.array([1])
         #print('State ',ChiK_k[i].tolist())
         y_mean = BS(ChiK_k[i],y_prev,r,dt)+ (np.sqrt(X_x[:,i].tolist()[0])*X_v[:,i].tolist()[0])
         #if(X_v[:,i]<0):
             #print('state')
         y1_k.append(y_mean)
         #print(weighted_measure.shape)
         weighted_measure = weighted_measure+(W_m[i]*(y_mean))
         #print((y_mean).shape)
     #print(' ')
     # Covariance of y
     for i in range(0,(2*L)+1):
         P_yy = (y1_k[i]-weighted_measure)*(y1_k[i]-weighted_measure)
         #print(P_yy)
         P_xy = (ChiK_k[i]-weighted_state)*(y1_k[i]-weighted_measure)
         #print(P_xy.shape)
         #print(cov_yx.shape)
         #print(' ')
         cov_y= cov_y +(W_c[i]*P_yy)
         cov_yx= cov_yx +(W_c[i]*P_xy)
     #print('y',weighted_measure[0])
#np.expand_dims(np.subtract(ChiK_k[i], weighted_state), axis=1).shape   
     return (weighted_state,cov,weighted_measure,cov_y,cov_yx) 

def state(Xn,v,yk,yK,param,r,dt):
     kcap= param[0]
     theta = param[1]
     sigma = param[2]
     rho = param[3]
     XNZero = Xn + (kcap*(theta)*dt) - ((kcap*dt)*Xn)
     term= sigma*rho*(r - (Xn/2))*dt
     term2= sigma*rho*(yK-yk)
     XNZero = XNZero - term + term2 
     
     XN= XNZero+(np.sqrt((1-(rho*rho))*Xn)*sigma*v)
     return XN

def BS(XK_k,yk,r,dt):
     zK= yk+ ((r - (XK_k/2))*dt)
     return zK

prev_vol=stateUKF
(y_k,y_K,xa_uns,P_a,L,r,dt,param,P_w,P_v)=(HestonPrice.loc[e,'Measure'],HestonPrice.loc[e,'NextMeasure'],xuns_temp,Pa_O,L,r,dt,param,P_w,P_v)
def Unscented_KF_update(prev_vol,y_k,y_K,xa_uns,P_a,L,r,dt,param,P_w,P_v):
     XK_k, P_k, y1_k, P_yy, P_xy = Unscented_KF_predict(xa_uns,P_a,L,param,y_k,y_K,r,dt,prev_vol)
     b_K = y_K - y1_k
     #print('Real Price',y_k)
     #print('Pred Price',y1_k)
     #(1/P_yy)
     K_K = P_xy*(1/P_yy)
     #print(P_xy)
     #print(y1_k)
     #print(K_K.shape)
     #print(XK_k.shape)
     #K_K=np.expand_dims(K_K,axis=1)
     XK_K = XK_k+(K_K*b_K)
     #print(XK_K)
     #print(dot(K_K , np.expand_dims(dot(P_yy,K_K.T),axis=1).T).shape)
     P_final = P_k-(K_K * K_K * P_yy)
     
     UnX,UnP = UKF_matrix(XK_K,P_final, P_w,P_v)
     #print(UnX)
     #print(UnP)
     #print(' ')
     #UnX = [XK_K,0,0,0]
     #UnP = [P_final, Pw,Pv]
     return (UnX,UnP,XK_K,P_final,P_w,P_v)


def UKF_matrix(x,P_O, P_w,P_v):
    x_t=[x]
    x_t.extend([0,0])
    Pa_O=np.diag([P_O,P_w,P_v])
    return (np.array(x_t),Pa_O)

def dataset():
    HestonPrice=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/SP.csv')
    HestonPrice['Date'] = pd.to_datetime(HestonPrice['Date'])
    
    TresRate=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/RiskRate.csv')
    TresRate['Date'] = pd.to_datetime(TresRate['Date'])
    TresRate['risk_perc']=TresRate['risk_perc']/100
    HestonPrice = pd.merge(HestonPrice,TresRate[['Date', 'risk_perc']],on='Date')
    
    base=HestonPrice.loc[231,'Close']
    HestonPrice['CloseRatio']=HestonPrice['Close']/base
    HestonPrice['Measure']=np.log(HestonPrice['CloseRatio'])
    HestonPrice['NextMeasure'] = HestonPrice['Measure'].shift(-7)
    
    start_date = pd.datetime(2007,12,4)
    end_date = pd.datetime(2016,8,30)
    mask = (HestonPrice['Date'] >= start_date) & (HestonPrice['Date'] <= end_date)
    HestonPrice = HestonPrice.loc[mask]
    HestonPrice=HestonPrice.reset_index()
    HestonPrice = HestonPrice[['Date', 'Close','CloseRatio','Measure','NextMeasure','risk_perc']]
    
    V_prev= []
    V_present=[]
    y_prev=[]
    y_present=[]
    UKF_prices=[]
    
    real_prices=[]
    
    vol_UKF_train =[]
    theta_UKF_train = []
    rho_UKF_train = []
    kappa_UKF_train = []
    sigma_UKF_train = []
    
    
    xbar_0 = 0.08
    P_0 = 0.01
    x_temp= xbar_0
    P_temp = P_0
    dt=7/HestonPrice.shape[0]
    r=HestonPrice.loc[0,'risk_perc']
    Q= dt
    R= dt
    L=3
    param=[4.0,0.1,0.05,-0.5]
    (xuns_temp,Pa_O)= UKF_matrix(x_temp,P_temp, Q,R)
    P_w=Q
    P_v=R
    stateUKF=xbar_0
    #e=14
    for e in range(0,HestonPrice.shape[0],7):
        V_prev.append(stateUKF)
        y_prev.append(HestonPrice.loc[e,'Measure'])
        r=HestonPrice.loc[e,'risk_perc']
        #print(x_temp)- y_K,xa_uns,P_a,L,T,uK,S,ks,P_w,P_v
        #XK_k, P_k, y1_k, P_yy, P_xy = Unscented_KF_predict(xuns_temp,Pa_O,L,T,uK,S,ks)
        (xuns_temp,Pa_O,stateUKF,state_covUKF,P_w,P_v)=Unscented_KF_update(stateUKF,HestonPrice.loc[e,'Measure'],HestonPrice.loc[e,'NextMeasure'],xuns_temp,Pa_O,L,r,dt,param,P_w,P_v)
        #print(xuns_temp)
        V_present.append(stateUKF)
        y_present.append(HestonPrice.loc[e,'NextMeasure'])
        #UpdateResult = extended_KF_update(optionPrice.loc[e,'OptionPR'],PredResult[0], PredResult[1], PredResult[2], PredResult[3],PredResult[4])
        #x_temp= UpdateResult[2]
        #P_temp = UpdateResult[3]
        #stock_price.append(np.log(optionPrice.loc[e,'StockPR']/optionPrice.loc[e-1,'StockPR']))
        
        UKF_prices.append(BS(stateUKF,HestonPrice.loc[e,'Measure'],r,dt))
        real_prices.append(HestonPrice.loc[e,'NextMeasure'])
        vol_UKF_train.append(stateUKF)
        theta_UKF_train.append(param[1])
        rho_UKF_train.append(param[3])
        kappa_UKF_train.append(param[0])
        sigma_UKF_train.append(param[2])
        if(e>=100):
            param = fPMLE_Alt(V_prev,V_present,y_prev,y_present,r,dt,param[2])
            print(param)

def fPMLE_Alt(V_prev,V_present,y_prev,y_present,r,dt,sigma):
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
     kcap= - np.log(Beta1)/dt
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
     return [kcap,theta,sigma,rho]


        
plt.plot(real_prices)
plt.plot(UKF_prices)
plt.plot(vol_UKF_train)
plt.title('Pf performance')
plt.legend(['True Value','UKF Estimate'])
plt.xlabel('timestep')
plt.ylabel('Concentration')    
    

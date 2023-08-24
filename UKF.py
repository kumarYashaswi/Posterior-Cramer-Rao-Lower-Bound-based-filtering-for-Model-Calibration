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


def degneracy(w_K):
    weight_sqr=0
    for i in range(0,N):
       weight_sqr= weight_sqr + (w_K[i]*w_K[i])
    return weight_sqr


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
    print(X_uns_matrix)
    X_x = X_uns_matrix[0:2,:]
    X_w = X_uns_matrix[2:4,:]
    X_v = X_uns_matrix[4:5,:]
    #print(X_uns) 
    #print(X_uns_matrix)
    #print(X_x)
    #print(X_w)
    #print(X_v) 
    #print(' ')
    return (X_uns,X_uns_matrix,X_x,X_w,X_v,W_m,W_c)
    
def Unscented_KF_predict(xa_uns,P_a,L,T,uK,S,ks):
     # Sigma Points of x_k
     X_uns,X_uns_matrix,X_x,X_w,X_v,W_m,W_c = Unscented_KF_Transform(xa_uns,P_a,L)
     ChiK_k=[]
     y1_k=[]
     weighted_state=np.array([0, 0], dtype=np.float)
     weighted_measure=np.array([0], dtype=np.float)
     cov = np.array([[0, 0], [0, 0]], dtype=np.float)
     cov_y = np.array([0], dtype=np.float)
     cov_yx=np.expand_dims(np.array([0, 0], dtype=np.float),axis=1)
     # Mean of x
     #i=3
     X_x=np.array(X_x)
     X_v=np.array(X_v)
     X_w=np.array(X_w)
     for i in range(0,(2*L)+1):
        #print("state1",X_x[:,i].tolist())
        a= X_x[:,i].tolist()
        b=X_w[:,i].tolist()
        #print("state2",X_w[:,i].tolist())
        mean=np.array(state(a,b,uK))
        ChiK_k.append(mean)
        #print(weighted_state.shape)
        #print(mean.shape)
        weighted_state= np.add(weighted_state,(W_m[i]*mean))
        #print(' ')
        #print(W_m[i]*mean)
        #print(' ')
        
     # Covariance of y
     for i in range(0,(2*L)+1):
         #print(np.expand_dims(np.subtract(ChiK_k[i],weighted_state),axis=1).shape)
         #print(np.expand_dims(np.subtract(ChiK_k[i],weighted_state),axis=1).T.shape)
         P_xx= np.dot(np.expand_dims(np.subtract(ChiK_k[i],weighted_state),axis=1),np.expand_dims(np.subtract(ChiK_k[i],weighted_state),axis=1).T)
         #print(P_xx)
         #print(cov.shape)
         cov= np.add(cov,(W_c[i]*P_xx))
         #print(' ')
     # Mean of y
     for i in range(0,(2*L)+1):
         #np.array([1])
         #print('State ',ChiK_k[i].tolist())
         y_mean = np.add(np.array(BS(ChiK_k[i].tolist(),S,ks,T)),X_v[:,i])
         if(X_v[:,i]<0):
             print('state',X_v[:,i])
         y1_k.append(y_mean)
         #print(weighted_measure.shape)
         weighted_measure = np.add(weighted_measure,(W_m[i]*(y_mean)))
         #print((y_mean).shape)
     #print(' ')
     # Covariance of y
     for i in range(0,(2*L)+1):
         P_yy = np.dot(np.expand_dims(np.subtract(y1_k[i], weighted_measure),axis=1),(np.expand_dims(np.subtract(y1_k[i], weighted_measure),axis=1)).T)[0]
         #print(P_yy)
         P_xy = np.dot(np.expand_dims(np.subtract(ChiK_k[i], weighted_state), axis=1),np.expand_dims((np.subtract(y1_k[i], weighted_measure)),axis=1).T)
         #print(P_xy.shape)
         #print(cov_yx.shape)
         #print(' ')
         cov_y= np.add(cov_y ,(W_c[i]*P_yy))
         cov_yx= np.add(cov_yx ,(W_c[i]*P_xy))
     #print('y',weighted_measure[0])
#np.expand_dims(np.subtract(ChiK_k[i], weighted_state), axis=1).shape   
     return (weighted_state,cov,weighted_measure[0],cov_y,cov_yx) 

def state(X,v,uN):
     omega= 0
     alpha = 0
     beta = 1
     XN=[]
     XN.append(omega + alpha*(uN * uN) + beta*X[0] + v[0])
     XN.append(X[1] + v[1])
     return XN


def Unscented_KF_update(y_K,xa_uns,P_a,L,T,uK,S,ks,P_w,P_v):
     XK_k, P_k, y1_k, P_yy, P_xy = Unscented_KF_predict(xa_uns,P_a,L,T,uK,S,ks)
     b_K = y_K - y1_k
     #print('Real Price',y_k)
     #print('Pred Price',y1_k)
     #(1/P_yy)
     K_K = P_xy*(1/P_yy[0])
     #print(P_xy)
     #print(y1_k)
     #print(K_K.shape)
     #print(XK_k.shape)
     #K_K=np.expand_dims(K_K,axis=1)
     XK_K = np.add(XK_k,(K_K*b_K).T)
     #print(XK_K)
     #print(dot(K_K , np.expand_dims(dot(P_yy,K_K.T),axis=1).T).shape)
     P_final = np.subtract(P_k,dot(K_K , np.expand_dims(dot(P_yy,K_K.T),axis=1).T))
     #print(P_final)
     #print(P_final)
     #print(' ')
     UnX,UnP = UKF_matrix(XK_K.tolist()[0],P_final, P_w,P_v)
     #print(UnX)
     #print(UnP)
     #print(' ')
     #UnX = [XK_K,0,0,0]
     #UnP = [P_final, Pw,Pv]
     return (UnX,UnP,XK_K.tolist()[0],P_final,P_w,P_v)




     
def BS(XK_k,S,k,T):
     X=[r/100 for r in XK_k]
     dk_plus = (np.log(S/k) + (X[1] + (X[0]/2))*T)/np.sqrt(X[0]*T)
     dk_minus = dk_plus - np.sqrt(X[0]*T)
     rhs = math.exp(-X[1]*T)
     c = S*norm.cdf(dk_plus) - k*rhs*norm.cdf(dk_minus)
     return c



def UKF_matrix(x,P_O, P_w,P_v):
    x_t=x.copy()
    x_t.extend([0,0,0])
    Pa_O=np.pad(P_O, ((0,3),(0,3)), mode='constant', constant_values=0)
    Pa_O[2:4,2:4]=  P_w
    Pa_O[4:,4:]=  P_v
    return (np.array(x_t),Pa_O)

def dataset():
    optionPrice=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/BSFilter/NSE.csv')
    market_price=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/BSFilter/NSEI.csv')
    market_price['Date'] = pd.to_datetime(market_price['Date'])
    optionPrice['Date'] = pd.to_datetime(optionPrice['Date'])
    optionPrice['Expiry'] = pd.to_datetime(optionPrice['Expiry'])
    optionPrice = pd.merge(optionPrice,market_price[['Date', 'Close']],on='Date')
    optionPrice['TimeToMaturity']=optionPrice['Expiry'] - optionPrice['Date']
    optionPrice['TimeToMaturity']=optionPrice['TimeToMaturity'].dt.days
    #optionPrice['TimeToMaturity'] = optionPrice['TimeToMaturity'].dt.days.astype('int16')
    optionPrice=optionPrice.loc[optionPrice['StrikePrice'] == 11000]
    optionPrice=optionPrice.reset_index()
    del optionPrice['index']
    optionPrice=optionPrice.sort_values(by=['Date'])
    optionPrice['StockPR']=optionPrice['Close']
    optionPrice=optionPrice.bfill(axis ='rows')
    optionPrice['PriceChange'] = optionPrice['StockPR'].pct_change()
    optionPrice=optionPrice.loc[240:]
    optionPrice=optionPrice.reset_index()
    del optionPrice['index']
    UKF_prices=[]
    UKF_prices_test=[]
    real_prices=[]
    real_prices_test=[]
    vol_UKF_train =[]
    risk_UKF_train = []
    vol_UKF_test =[]
    risk_UKF_test = []
    stock_price=[optionPrice.loc[1,'StockPR']]
    xbar_0 = [0.012,0.008]
    P_0 = []
    P_0.append([0.00000001,0])
    P_0.append([0,0.00000001])
    x_temp= xbar_0
    P_temp = P_0
    Q = []
    Q.append([0.00000001,0])
    Q.append([0,0.00000001])
    R= [1600]
    L=5
    (xuns_temp,Pa_O)= UKF_matrix(x_temp,P_temp, Q,R)
    P_w=Q
    P_v=R
    x_temp=np.array([x_temp])
    N=500
    x_particle= np.random.multivariate_normal(xbar_0, P_0, N)
    w_particle= [1/N] * N
    for e in range(1,optionPrice.shape[0]):
        if(optionPrice.loc[e,'Set']=='Test'):
            price_bfr=optionPrice.loc[e-1,'StockPR']
            price= GBM(np.array(stock_price[-60:]),price_bfr)
            price_chg= (price-price_bfr)/price_bfr
            PredResult = Unscented_KF_predict(xuns_temp,Pa_O,L,optionPrice.loc[e,'TimeToMaturity'],price_chg,price,optionPrice.loc[e,'StrikePrice'])   
            pred_value= PredResult[2]
            pred_state = PredResult[0]
            UKF_prices_test.append(pred_value)
            real_prices_test.append(optionPrice.loc[e,'OptionPR'])
            vol_UKF_test.append(pred_state[0])
            risk_UKF_test.append(pred_state[1])
        print(e)
        #print(x_temp)- y_K,xa_uns,P_a,L,T,uK,S,ks,P_w,P_v
        #XK_k, P_k, y1_k, P_yy, P_xy = Unscented_KF_predict(xuns_temp,Pa_O,L,T,uK,S,ks)
        (xuns_temp,Pa_O,stateUKF,state_covUKF,P_w,P_v)=Unscented_KF_update(optionPrice.loc[e,'OptionPR'],xuns_temp,Pa_O,L,optionPrice.loc[e,'TimeToMaturity'],optionPrice.loc[e,'PriceChange'],optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],P_w,P_v)
        #print(xuns_temp)
        #UpdateResult = extended_KF_update(optionPrice.loc[e,'OptionPR'],PredResult[0], PredResult[1], PredResult[2], PredResult[3],PredResult[4])
        #x_temp= UpdateResult[2]
        #P_temp = UpdateResult[3]
        stock_price.append(np.log(optionPrice.loc[e,'StockPR']/optionPrice.loc[e-1,'StockPR']))
        if(optionPrice.loc[e,'Set']=='Train'):
            UKF_prices.append(BS(stateUKF,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))
            real_prices.append(optionPrice.loc[e,'OptionPR'])
            vol_UKF_train.append(stateUKF[0])
            risk_UKF_train.append(stateUKF[1])
    
   
    

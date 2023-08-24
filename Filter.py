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
#X_x.append([1,2,3,4,5])
#X_x.append([5,4,3,2,1])
#X_x.append([3,4,5,62,5])
#X_x.append([3,4,5,64,5])
#X_x.append([3,4,5,65,5])
#X_x= np.array(X_x)
#X_x[0:2,:]
#X_x.T
(xa_uns,P_a)=(np.array(x_temp),Pa_O)
uK=0.01
S=100
T=5
ks=110
xa_uns=xuns_temp
P_a= Pa_O

xa_uns=np.array([33.67280951  ,3.85552425  ,0.          ,0.          ,0.        ])
P_a=[[ 1.73010000e+00 ,-6.48357085e-13  ,0.00000000e+00  ,0.00000000e+00
   ,0.00000000e+00],
 [-6.48357085e-13 ,-8.58778004e+01  ,0.00000000e+00  ,0.00000000e+00
   ,0.00000000e+00],
 [ 0.00000000e+00  ,0.00000000e+00  ,1.00000000e+00  ,0.00000000e+00
   ,0.00000000e+00],
 [ 0.00000000e+00  ,0.00000000e+00  ,0.00000000e+00  ,1.00000000e+00
   ,0.00000000e+00],
 [ 0.00000000e+00  ,0.00000000e+00  ,2.00000000e+00  ,0.00000000e+00
   ,1.00000000e+00]]
P_a=np.array(P_a)
P_a[:,2]
np.linalg.cholesky(P_a)

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


def degneracy(w_K,N):
    weight_sqr=0
    for i in range(0,N):
       weight_sqr= weight_sqr + (w_K[i]*w_K[i])
    return weight_sqr


(x_part,uN,Q,w_k,N,y_k,S,k,T,e,R)=(x_particle,optionPrice.loc[e,'PriceChange'],Q,w_particle,N,optionPrice.loc[e,'OptionPR'],optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],e,R[0])
def Particle_Filter(x_part,uN,Q,w_k,N,y_k,S,k,T,e,R):
    xK_part=[]
    w_K=[]
    res_list=[]
    stateMean=[]
    state_cov= np.array([[0, 0], [0, 0]], dtype=np.float)
    weight_sum=0
    for i in range(0,N):
       state = f(x_part[i],uN,Q,Q)[0]
       xK_part.append(state)
       #print(i," BS ",BS(state,S,k,T))
       #print(i," BS ",state[0]*T)
       price_diff= y_k - BS(state,S,k,T)
       
       w_K.append(w_k[i]*normal_dist(price_diff,0,R))
       stateMean.append([w_K[i] * r for r in xK_part[i]])
       weight_sum= weight_sum + w_K[i]
    #print(i," BS ",weight_sum)
    state_mean = np.array(stateMean).sum(axis=0).tolist()
    
    w_K = [x / weight_sum for x in w_K]
    state_mean = [x / weight_sum for x in state_mean]
    
    for i in range(0,N):
        error=[x1 - x2 for (x1, x2) in zip(xK_part[i], state_mean)]
        error_matrix=np.expand_dims(np.array(error),axis=1)
        state_cov= np.add(state_cov,w_K[i]*np.dot(error_matrix,error_matrix.T))
    
    w_sqr= degneracy(w_K,N)
    Bessel=(1-w_sqr)
    state_cov=(1/(Bessel))*state_cov 
    n_e= N/2
    Neff = 1/w_sqr
    #print(n_e)
    #print(" PArticle ",w_K)
    xK_resample,w_K = resample1(xK_part,w_K,N,Neff,n_e)
    for i in range(0,N):
        res_list.append([w_K[i] * r for r in xK_resample[i]])
    x_K = np.array(res_list).sum(axis=0).tolist()
    
            
    #print("State  ",x_K)
    #print(" PArticle ",xK_resample)
    #print(" Matrix ",state_cov)
    #print("Weight  ",w_K[5])
    return (x_K,xK_resample,state_cov,w_K)

error=[1,2]
np.dot(np.array(error).T,np.array(error))

np.expand_dims(np.array(error),axis=1).shape

def resample1(xK,wK,N,Neff,n_e):
    if Neff < n_e:
        [xK,wK] = resample_particle(xK,wK,N)
        return [xK,wK]
    else:
        return [xK,wK]

x_part=x_particle[0]
w_K
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
    return [new_particle,new_weight]

def Unscented_KF_Transform(xa_uns,P_a,L):
    W_m=[]
    W_c=[]
    X_uns=[]
    #L=5
    alpha=0.00001
    kappa= 0
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
         #if(X_v[:,i]<0):
             #print('state',X_v[:,i])
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
 e=1
 (y_K,xa_uns,P_a,L,T,uK,S,ks,P_w,P_v)=(optionPrice.loc[e,'OptionPR'],xuns_temp,Pa_O,L,optionPrice.loc[e,'TimeToMaturity'],optionPrice.loc[e,'PriceChange'],optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],P_w,P_v)
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
     i=0
    (X,v,uN)= (X_x[:,i].tolist(),X_w[:,i].tolist(),uK)
    
def state(X,v,uN):
     omega= 0
     alpha = 0
     beta = 1
     XN=[]
     XN.append(omega + alpha*(uN * uN) + beta*X[0] + v[0])
     XN.append(X[1] + v[1])
     return XN

(y_K,xa_uns,P_a,L,T,uK,S,ks,P_w,P_v)=(optionPrice.loc[e,'OptionPR'],xuns_temp,Pa_O,L,optionPrice.loc[e,'TimeToMaturity'],optionPrice.loc[e,'PriceChange'],optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],P_w,P_v)
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

dot(K_K.shape , np.expand_dims(dot(P_yy,K_K.T),axis=1).T)    
np.expand_dims(dot(P_yy,K_K.T),axis=1).shape

(X,S,ks,T,uK,P,Q,R)=(x_temp,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],optionPrice.loc[e,'PriceChange'],P_temp, Q,R)
def extended_KF_predict(X,S,ks,T,uK,P,Q,R):
     Xk=X[0]
     XK_k = f(Xk,uK,P,Q)[0]
     P_k = f(Xk,uK,P,Q)[1]
     if(XK_k[0]<=0):
         XK_k[0]=Xk[0]
     if(XK_k[1]<=0):
         XK_k[1]=Xk[1]
     #print(S)
     #print(T)
     #print(XK_k)
     y1_k = BS(XK_k,S,ks,T)
     #print(y1_k)
     #print(' ')
     H = Jacobian(XK_k,S,ks,T)
     #print(H)
     #print(y1_k)
     H=np.array(H)
     #print(' ')
     H= np.expand_dims(H, 0)
     F_K = dot(H, dot(P_k, H.T)) + R
     return (y1_k,P_k,F_K,XK_k,H)

def Jacobian(XK_k,S,k,T):
     X=[r/100 for r in XK_k]
     dk_plus = (np.log(S/k) + (X[1] + (X[0]/2))*T)/np.sqrt(X[0]*T)
     dk_minus = dk_plus - np.sqrt(X[0]*T)
     Vega= (0.5*S*np.sqrt(T)*scipy.stats.norm(0, 1).pdf(dk_plus))/(np.sqrt(X[0]))
     #print(S)
     #print(math.exp(-XK_k[1]*T))
     rho =  k*T*(math.exp(-X[1]*T))*scipy.stats.norm(0, 1).pdf(dk_minus)
     return (Vega,rho)

  
def f(Xn,uN,Sigma_n,Q):
     #print(x_temp)
     omega= 0
     alpha = 0
     beta = 1
     #print('wwww')
     #print(Xn)
     XNZero = omega + alpha*(uN * uN) + beta*Xn[0]
     A=[]
     A.append([beta*beta,0])
     A.append([0,1])
     Sigma_N =dot(A,Sigma_n) + Q
     XNOne=Xn[1]
     XN=[XNZero,XNOne]
     return (XN,Sigma_N)
    
def extended_KF_update(y_K,y1_k, P_k, F_k, XK_k,H):
     #print("H matrix   ",H)
     #print("Real Price  ",y_K)
     #print("Estimated Value  ",y1_k)
     b_K = y_K - y1_k
     #F_k=np.array((F_k))
     #F_k= np.expand_dims(F_k, 0)
     #print("State Cov", P_k)
     K_K = dot(P_k, H.T)*(1/F_k)
     #print('Difference   ',b_K)
     #print("State  ",XK_k)
     #print("Kalman Gain  ",K_K)  
     #print("Kalman matrix   ",dot(K_K,b_K).T)
     XK_K = XK_k + dot(K_K,b_K).T
     #print(dot(K_K,b_K).T)
     P_final = P_k - dot(K_K , dot(H,P_k))
     #print("New State  ",XK_K)
     #print(dot(K_K,b_K).T)
     #print(P_final)
     #print(' ')
     return (XK_K,P_final)
     
def BS(XK_k,S,k,T):
     if(XK_k[0]<=0):
         XK_k[0]=0.01
     if(XK_k[1]<=0):
         XK_k[1]=0.01
     X=[r/100 for r in XK_k]
     dk_plus = (np.log(S/k) + (X[1] + (X[0]/2))*T)/np.sqrt(X[0]*T)
     dk_minus = dk_plus - np.sqrt(X[0]*T)
     rhs = math.exp(-X[1]*T)
     c = S*norm.cdf(dk_plus) - k*rhs*norm.cdf(dk_minus)
     return c

def kf_predict(X, P, A, Q, B, U):
     X = dot(A, X) + dot(B, U)
     P = dot(A, dot(P, A.T)) + Q
     return(X,P) 


def kf_update(X, P, Y, H, R):
        IM = dot(H, X)
        IS = R + dot(H, dot(P, H.T))
        K = dot(P, dot(H.T, inv(IS)))
        X = X + dot(K, (Y-IM))
        P = P - dot(K, dot(IS, K.T))
        LH = gauss_pdf(Y, IM, IS)
        return (X,P,K,IM,IS,LH)



'''
def gauss_pdf(X, M, S):
     if M.shape()[1] == 1:
         DX = X - tile(M, X.shape()[1])
         E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
         E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
         P = exp(-E)
     elif X.shape()[1] == 1:
         DX = tile(X, M.shape()[1])- M
         E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
         E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
         P = exp(-E)
     else:
         DX = X-M
         E = 0.5 * dot(DX.T, dot(inv(S), DX))
         E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
         P = exp(-E)
     return (P[0],E[0]) 
 '''
P_O=np.array(P_0)
P_w = np.array(Q)
P_v= np.array(R)
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
    optionPrice=optionPrice.loc[optionPrice['StrikePrice'] == 10500]
    optionPrice=optionPrice.reset_index()
    del optionPrice['index']
    optionPrice=optionPrice.sort_values(by=['Date'])
    optionPrice['StockPR']=optionPrice['Close']
    optionPrice=optionPrice.bfill(axis ='rows')
    optionPrice['PriceChange'] = optionPrice['StockPR'].pct_change()
    optionPrice=optionPrice.loc[170:]
    optionPrice=optionPrice.reset_index()
    del optionPrice['index']
    EKF_prices=[]
    UKF_prices=[]
    PF_prices=[]
    real_prices=[]
    xbar_0 = [0.01,0.01]
    P_0 = []
    P_0.append([0.00001,0])
    P_0.append([0,0.0002])
    x_temp= xbar_0
    P_temp = P_0
    Q = []
    Q.append([0.00001,0])
    Q.append([0,0.0002])
    R= [5000]
    L=5
    (xuns_temp,Pa_O)= UKF_matrix(x_temp,P_temp, Q,R)
    P_w=Q
    P_v=R
    x_temp=np.array([x_temp])
    N=200
    x_particle= np.random.multivariate_normal(xbar_0, P_0, N)
    w_particle= [1/N] * N
    for e in range(1,optionPrice.shape[0]-1):
        print(e)
        #print(x_temp)
        PredResult=extended_KF_predict(x_temp,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],optionPrice.loc[e,'PriceChange'],P_temp, Q,R)
        
        UpdateResult = extended_KF_update(optionPrice.loc[e,'OptionPR'],PredResult[0], PredResult[1], PredResult[2], PredResult[3],PredResult[4])
        x_temp= UpdateResult[0]
        P_temp = UpdateResult[1]
        EKF_prices.append(BS(x_temp.tolist()[0],optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))
        real_prices.append(optionPrice.loc[e,'OptionPR'])
    for e in range(1,optionPrice.shape[0]-1):
        print(e)
        #print(x_temp)- y_K,xa_uns,P_a,L,T,uK,S,ks,P_w,P_v
        #XK_k, P_k, y1_k, P_yy, P_xy = Unscented_KF_predict(xuns_temp,Pa_O,L,T,uK,S,ks)
        (xuns_temp,Pa_O,stateUKF,state_covUKF,P_w,P_v)=Unscented_KF_update(optionPrice.loc[e,'OptionPR'],xuns_temp,Pa_O,L,optionPrice.loc[e,'TimeToMaturity'],optionPrice.loc[e,'PriceChange'],optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],P_w,P_v)
        #print(xuns_temp)
        a=BS(stateUKF,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'])
        print(a-optionPrice.loc[e,'OptionPR'])
        UKF_prices.append(BS(stateUKF,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))
        real_prices.append(optionPrice.loc[e,'OptionPR'])
        #UpdateResult = extended_KF_update(optionPrice.loc[e,'OptionPR'],PredResult[0], PredResult[1], PredResult[2], PredResult[3],PredResult[4])
        #x_temp= UpdateResult[2]
        #P_temp = UpdateResult[3]
    for e in range(1,optionPrice.shape[0]-1):
        #R=std(x_particle,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'])
        #print(x_temp)- y_K,xa_uns,P_a,L,T,uK,S,ks,P_w,P_v
        #print(R)
        (state,x_particle,particle_cov,w_particle)=Particle_Filter(x_particle,optionPrice.loc[e,'PriceChange'],Q,w_particle,N,optionPrice.loc[e,'OptionPR'],optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],e,R[0])
        PF_prices.append(BS(state,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))
        real_prices.append(optionPrice.loc[e,'OptionPR'])
        #print(max(w_particle))
        #print(min(w_particle))
        #print(sum(w_particle))
        #print(particle_cov)
        #print(" ")
        EKF_prices.append(BS(state,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))
        real_prices.append(optionPrice.loc[e,'OptionPR'])
        if not state[0]:
            print(e)
            print(state[0])
            print(state[1])
def std(x_particle,S,k,T):
    a=[]
    for i in range(0,500):
        a.append(BS(x_particle[i],S,k,T))
    return np.std(a)*np.std(a)

plt.figure(figsize = (12, 4))
plt.plot(np.array(real_prices[0:263]),label="OHLC-Rl_Portfolio")
real_price
plt.plot(np.array(real_price[0:263]),label="OHLC-Rl_Portfolio")
#plt.plot(np.array(mark_rewards).cumsum(),label="Markowitz Portfolio Return")
plt.plot(np.array(option_price),label="Adaptive")
plt.plot(np.array(PF_prices),label="PF")
plt.plot(np.array(EKF_prices),label="EKF")
plt.plot(np.array(UKF_prices),label="UKF")
#plt.plot(np.array(market_price['Close']),label="UKF")
#plt.plot(np.array(rand_rewards).cumsum(),label="Random Portfolio Return")
plt.legend(loc="upper left")
plt.xlabel('time period in days')
plt.ylabel('Test set returns')
plt.show()

optionPrice['StrikePrice'].unique().tolist()
a=optionPrice['StrikePrice'].unique().tolist()
for i in range(0,len(a)):
    optionPrice1=optionPrice.loc[optionPrice['StrikePrice'] == a[i]]
    plt.figure(figsize = (12, 4))
    plt.plot(np.array(optionPrice1.loc[:,'OptionPR']),label=str(a[i]))
    #plt.plot(np.array(mark_rewards).cumsum(),label="Markowitz Portfolio Return")
    #plt.plot(np.array(rand_rewards).cumsum(),label="Random Portfolio Return")
    plt.legend(loc="upper left")
    plt.xlabel('time period in days')
    plt.ylabel('Test set returns')
    plt.show()

x_temp.tolist()[0]
1910 = 
 
(XK_k,S,k,T)=([0.002,0.008],11971.2,10500,640)
(XK_k,S,k,T)=([0.02,0.02],12343.2,10500,351)-2301
(XK_k,S,k,T)=([0.02,0.02],8925,10500,260)-400

optionPrice.loc[1:]['PriceChange'].std()
market_price['Close'].loc[0:55].pct_change().std()

0.000099*100


BS(XK_k,S,k,T)
a=[]
for i in range(0,500):
    a.append(BS(x_particle[i],S,k,T))
    print(BS(x_particle[i],S,k,T))

max(a)
min(a)
sum(a)/500
np.array(a).mean
np.std(a)
 dk_plus = (np.log(S/k) + (XK_k[1] + (XK_k[0]/2))*T)/np.sqrt(XK_k[0]*T)
     dk_minus = dk_plus - np.sqrt(XK_k[0]*T)
     try:
       rhs = math.exp(-XK_k[1]*T)
     except OverflowError:
       rhs = 1
     c = S*norm.cdf(dk_plus) - k*rhs*norm.cdf(dk_minus)
     return c

(y_K,xa_uns,P_a,L,T,uK,S,ks,P_w,P_v)

Xk=x_temp   
XK_k=x_temp
S= optionPrice.loc[e,'StockPR']
k= optionPrice.loc[e,'StrikePrice']
T= optionPrice.loc[e,'TimeToMaturity']
dk_plus = (np.log(S/k) + (XK_k[1] + (XK_k[0]/2))*T)/np.sqrt(XK_k[0]*T)
     dk_minus = dk_plus - np.sqrt(XK_k[0]*T)
     Vega= (0.5*S*np.sqrt(T)*scipy.stats.norm(0, 1).pdf(dk_plus))/(np.sqrt(XK_k[0]))
     rho =  k*T*(math.exp(-XK_k[1]*T))*scipy.stats.norm(0, 1).pdf(dk_minus)  
    Jacobian(x_temp,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'])
    
 def extended_KF_update(y_K,y1_k, P_k, F_k, XK_k,H):   
    b_K = y_K - y1_k
     K_K = dot(P_k, dot(H.T,inv(F_k)))
     XK_K = XK_k + dot(K_K,b_K)
     P_final = P_k - dot(K_K , dot(H,P_k))
     return (b_K,K_K,XK_K,P_final)
    
 P = P_temp
 ks= optionPrice.loc[e,'StrikePrice']
  y_K=   optionPrice.loc[e,'OptionPR']
  uK= optionPrice.loc[e,'PriceChange']
  y1_k= PredResult[0]
  P_k= PredResult[1]
  F_k = PredResult[2]
  XK_k = PredResult[3]
  H= PredResult[4]
  
  H= np.expand_dims(H, 0)
  
    H.T.shape[1]
    
    
    
    

x_temp,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity'],optionPrice.loc[e,'PriceChange'],P_temp, Q,R
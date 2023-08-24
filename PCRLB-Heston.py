# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:11:02 2020

@author: Kumar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:50:36 2020

@author: Kumar
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:50:36 2020

@author: Kumar
"""
import math
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
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

'''
def cramer_pf_PF(x_part,uN,Q,w_k,N,y_k,S,k,T,e):
    xK_part=[]
    w_K=[]
    weight_sum=0
    res_list = []
    for i in range(0,N):
       state = f(x_part[i],uN,Q,Q)[0]
       xK_part.append(state)
       price_diff= y_k - BS(state,S,k,T)
       if(not price_diff):
           price_diff = 0.02
       w_K.append(w_k[i]*normal_dist(price_diff,0,1))
       weight_sum= weight_sum + w_K[i]
    w_K = [x / weight_sum for x in w_K]
    xK_resample,w_K = resample(xK_part,w_K,N)
    for i in range(0,N):
        res_list.append([w_K[i] * r for r in xK_resample[i]])
    x_K = np.array(xK_resample).sum(axis=0).tolist()
    x_K = [x / N for x in x_K]
    x_particle= np.random.multivariate_normal(f(x_K,uN,Q,Q)[0], Q, N)
    new_weights= [1/N] * N

    return (x_particle,new_weights)
(x_part,uN,P_x,N,y_K,S,k,T,Q,R,data)=(state,uN,state_cov,N,y_K,S_front,k_front,T_front,Q,R,particle_data)
'''

def cramer_pf_PF(x_part,yk,yK,P_x,N,Q,R,r,dt,param,data):
    xk_resample,temp_weight = resample(data[0],data[1],N)
    weight_sum=0
    res_list = []
    w_K=[]
    x_k = np.array(xk_resample).sum(axis=0).tolist()
    x_k = x_k/N
    var=(Q)
    
    x_Kk= np.random.normal(f_PCRLB(x_k,yk,yK,param,r,var),var_PCRLB(x_k,yk,yK,param,r,var), N) 
    w_k= [1/N] * N
    for i in range(0,N):
       state = x_Kk[i]
       price_diff = yK - BS(state,yk,r,dt)   
       w_K.append(w_k[i]*normal_dist(price_diff,0,x_k*R))
       weight_sum= weight_sum + w_K[i]
    w_K = [x / weight_sum for x in w_K]
    xK_resample,w_K = resample_particle(x_Kk,w_K,N)
    wKk = cramer_forward_particles(xK_resample,xk_resample,Q,N,yk,yK,param,r,dt)
    xKk=[[a,b] for a, b in zip(xK_resample,xk_resample)]
    xKk_resample,wKk = resample_particle(xKk,wKk,N)
    
    x_particle_front = x_Kk
    
    x_particle_back =[a[1] for a in xKk_resample]
    
    return (x_particle_front,x_particle_back)

(x_part,yk,yK,Q,R,w_k,N,param,r,dt)
(x_part,yk,yK,P_x,N,Q,R,r,dt,param)=(state,yT,yTplus1,state_cov,N,Q,R,r,dt,param)
def cramer_pf(x_part,yk,yK,P_x,N,Q,R,r,dt,param):
    #print(x_part)
    #x_part=x_part[0]
    xk_resample = np.random.normal(x_part, P_x, N)
    weight_sum=0
    res_list = []
    w_K=[]
    x_k = np.array(xk_resample).sum(axis=0).tolist()
    x_k = x_k/N
    var=(Q)
    
    x_Kk= np.random.normal(f_PCRLB(x_k,yk,yK,param,r,var),var_PCRLB(x_k,yk,yK,param,r,var), N) 
    w_k= [1/N] * N
    
    for i in range(0,N):
       state = x_Kk[i]
       price_diff = yK - BS(state,yk,r,dt)   
       w_K.append(w_k[i]*normal_dist(price_diff,0,x_k*R))
       weight_sum= weight_sum + w_K[i]
    w_K = [x / weight_sum for x in w_K]
    xK_resample,w_K = resample_particle(x_Kk,w_K,N)
    wKk = cramer_forward_particles(xK_resample,xk_resample,Q,N,yk,yK,param,r,dt)
    xKk=[[a,b] for a, b in zip(xK_resample,xk_resample)]
    xKk_resample,wKk = resample_particle(xKk,wKk,N)
    
    x_particle_front = x_Kk
    
    x_particle_back =[a[1] for a in xKk_resample]
    
    return (x_particle_front,x_particle_back)


def cramer_forward_particles(x_front_particles,x_particles,Q,N,yk,yK,param,r,dt):
    tau = []
    tau_sum = 0
    weight_sum =0 
    for i in range(0,N):
        temp_particle=x_front_particles[i]
        for k in range(0,N):   
            diff = temp_particle - f_PCRLB(x_particles[k],yk,yK,param,r,dt)
            res = normal_dist(diff,0,var_PCRLB(x_particles[k],yk,yK,param,r,dt))
            tau_sum = tau_sum + res
        pdf = normal_dist(temp_particle-f_PCRLB(x_particles[i],yk,yK,param,r,dt),0,var_PCRLB(x_particles[i],yk,yK,param,r,dt))
        pdf_denom=pdf/(N*tau_sum)
        tau.append(pdf_denom)
        weight_sum = weight_sum + pdf_denom
    wKk = [x / weight_sum for x in tau]  
    return wKk
    
def f_PCRLB(Xn,yk,yK,param,r,dt):
     kcap= param[0]
     theta = param[1]
     sigma = param[2]
     rho = param[3]
     XNZero = Xn + (kcap*(theta)*dt) - ((kcap*dt)*Xn)
     term= sigma*rho*(r - (Xn/2))*dt
     term2= sigma*rho*(yK-yk)
     XNZero = XNZero - term + term2 
     XN= XNZero
     return XN
     
def var_PCRLB(Xn,yk,yK,param,r,dt):
    kcap= param[0]
    theta = param[1]
    sigma = param[2]
    rho = param[3] 
    Sigma_N = (sigma*sigma)*Xn*dt*(1-(rho*rho))
    return Sigma_N
            
def Fisher_Matrix(J_t,D_t11,D_t12,D_t22):
    temp1= 1/(J_t+D_t11)  
    temp2= D_t12*D_t12*temp1
    ans=D_t22-temp2
    return ans

def PCRLB(J_t,D_t11,D_t12,D_t22):
    temp1= D_t12/(D_t22)  
    temp2= D_t12*D_t12*(1/D_t22)
    temp2 = 1/(temp2-(J_t+D_t11))
    temp3= D_t12/(D_t22)  
    res= temp1*temp2*temp3
    pcrl=(1/(D_t22))-res
    return pcrl
    

def resample(x_part,w_K,N):
    w_K.insert(0,0)
    w_K=np.cumsum(w_K)
    s = np.random.uniform(0,1,N)
    ind=[]
    new_particle =[] 
    for i in range(0,N):
        particle_no=s[i]
        for j in range(1,len(w_K)):
            if particle_no<=w_K[j]:
              ind.append(j-1)
              break
    #print(ind)
    for k in range(0,N):
        new_particle.append(x_part[ind[k]])
    new_weight=[1/N] * N
    return (new_particle,new_weight)
       
def GBM(S,price):
    S_0 = price
    sigma= S.std()
    mu = S.mean() + (sigma**2/2)
    return S_0*np.exp(t*(mu-(sigma**2/2))+sigma*np.random.normal())
    
          
def first_derivative_f(x_part_back,param,dt):
    kcap= param[0]
    theta = param[1]
    sigma = param[2]
    rho = param[3]
    A= 1 - (kcap*dt) + (sigma*rho*dt/2)
    return A
   
def first_derivative_BS(x_part_front,param,dt):
     return -dt/2
  

def Dt11(x_part_back,Q,param,dt):
    first_der = first_derivative_f(x_part_back,param,dt)
    res=first_der*first_der*(1/Q)
    return res
         
def Dt12(x_part_back,Q,param,dt):
    first_der = first_derivative_f(x_part_back,param,dt)
    res=-first_der*(1/Q)
    return res


def Dt22(x_part_front,R,param,dt):
    first_der = first_derivative_BS(x_part_front,param,dt)
    res= first_der*first_der*(1/R)
    return res


(state,prev_y,yk,yK,P_x,N,Q,R,r,dt)
(state,uN,state_cov,N,y_K,S_front,k_front,T_front,Q,R)
(state,yT,yTplus1,state_cov,N,Q,R,r,dt,param)
(J_t,state,state_cov,yT,yTplus1,M,N,Q,R,r,dt,param,isParticle,particle_data)=(J,x_temp,P_temp,HestonPrice.loc[e,'Measure'],HestonPrice.loc[e,'NextMeasure'],1,N,Q,R,r,dt,param,0,x_particle)
def cramer_elements(J_t,state,state_cov,yT,yTplus1,M,N,Q,R,r,dt,param,isParticle,particle_data):
    D_t11 = 0
    D_t12 = 0
    D_t22 = 0
    if(isParticle==0):
        (x_particles_front,x_particles_back) = cramer_pf(state,yT,yTplus1,state_cov,N,Q,R,r,dt,param)
    elif(isParticle==1):
        (x_particles_front,x_particles_back) = cramer_pf_PF(state,yT,yTplus1,state_cov,N,Q,R,r,dt,param,particle_data)
    for j in range (0,M):
        for i in range(0,N):
            D_t11 = D_t11+Dt11(x_particles_back[i],Q,param,dt)
            D_t12 = D_t12+Dt12(x_particles_back[i],Q,param,dt)
            D_t22 = D_t22+Dt22(x_particles_front[i],R,param,dt)
    D_t11 = D_t11/(M*N)
    D_t12 = D_t12/(M*N)
    D_t22 = D_t22/(M*N)
    D_t22 = (1/Q)+D_t22
    J_T=Fisher_Matrix(J_t,D_t11,D_t12,D_t22)
    J_T_inv=PCRLB(J_t,D_t11,D_t12,D_t22)
    return (J_T,J_T_inv)
   (J_inv,P_x,state,P_w,P_v,N,data)=([J_EKF_inv,J_UKF_inv, J_PF_inv],[P_temp,Cov_uns,Cov_part],[x_temp[0].tolist(),state_uns,state_particle],P_w,P_v,N,[x_particle,w_particle])
i=2
def best_estimator(J_inv,P_x,state,P_w,P_v,N,data):
    optimal_bayesian = 0
    max_sum = -math.inf
    for i in range(0,len(J_inv)):
        temp1 = J_inv[i]
        temp2 = P_x[i]
        phi= temp1/temp2
        phi_sum = phi
        print(temp2)
        if(phi_sum>max_sum):
            optimal_bayesian = i
            max_sum = phi_sum
    #print(state[optimal_bayesian])
    (state_unscented,cov_unscented)= UKF_matrix(state[optimal_bayesian],P_x[optimal_bayesian],P_w,P_v)
    if(optimal_bayesian!=2):
        state_particle = np.random.normal(state[optimal_bayesian],P_x[optimal_bayesian], N)
        weight_particle = [1/N] * N
    elif(optimal_bayesian==2):
        state_particle = data[0]
        weight_particle = data[1]
    return (state[optimal_bayesian],P_x[optimal_bayesian],state_unscented,cov_unscented,x_particle,w_particle,optimal_bayesian)
    
      
def dataset():
    gap= 1
    HestonPrice=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/SP.csv')
    HestonPrice['Date'] = pd.to_datetime(HestonPrice['Date'])
    
    TresRate=pd.read_csv('C:/Users/KUMAR YASHASWI/Documents/filter - Copy/RiskRate.csv')
    TresRate['Date'] = pd.to_datetime(TresRate['Date'])
    TresRate['risk_perc']=TresRate['risk_perc']/100
    HestonPrice = pd.merge(HestonPrice,TresRate[['Date', 'risk_perc']],on='Date')
    
    base=HestonPrice.loc[231,'Close']
    HestonPrice['CloseRatio']=HestonPrice['Close']/base
    HestonPrice['Measure']=np.log(HestonPrice['CloseRatio'])
    HestonPrice['NextMeasure'] = HestonPrice['Measure'].shift(-gap)
    
    start_date = pd.datetime(2007,12,4)
    end_date = pd.datetime(2017,8,31)
    mask = (HestonPrice['Date'] >= start_date) & (HestonPrice['Date'] <= end_date)
    HestonPrice = HestonPrice.loc[mask]
    HestonPrice=HestonPrice.reset_index()
    HestonPrice = HestonPrice[['Date', 'Close','CloseRatio','Measure','NextMeasure','risk_perc']]
    
    V_prev= []
    V_present=[]
    y_prev=[]
    y_present=[]
    
    
    ABF_prices=[]    
    real_prices=[]
    
    vol_ABF_train =[]
    filter_used_ABF=[]
    
    theta_ABF_train = []
    rho_ABF_train = []
    kappa_ABF_train = []
    sigma_ABF_train = []
   
    
    xbar_0 = 0.1
    P_0 = 0.001
    x_temp= xbar_0
    P_temp = P_0
    dt=1/252
    r=HestonPrice.loc[0,'risk_perc']
    param=[7.438950055361843, 0.1877555104605398, 0.9, -0.8]
    kcap= param[0]
    theta = param[1]
    sigma = param[2]
    rho = param[3]
    Q= dt
    R= dt
    L=3
    (xuns_temp,Pa_O)= UKF_matrix(x_temp,P_temp, Q,R)
    P_w=Q
    P_v=R
    #Pa_O=xbar_0
    N=500
    N1=500
    M=1
    x_particle= np.random.normal(xbar_0, P_0, N1)
    w_particle= [1/N1] * N1
    J=1/P_0
    
    
    e=0
    (J_0,J_0_inv)=cramer_elements(J,x_temp,P_temp,HestonPrice.loc[e,'Measure'],HestonPrice.loc[e,'NextMeasure'],1,N,Q,R,r,dt,param,0,x_particle)
    (J_EKF,J_UKF, J_PF)= (J_0,J_0,J_0)
    (J_EKF_inv,J_UKF_inv, J_PF_inv)= (J_0_inv,J_0_inv,J_0_inv)
    e=0
    for e in range(0,HestonPrice.shape[0],gap):
        print(e)
        V_prev.append(x_temp)
        y_prev.append(HestonPrice.loc[e,'Measure'])
        r=HestonPrice.loc[e,'risk_perc']
        
        PredResult=extended_KF_predict(x_temp,HestonPrice.loc[e,'Measure'],HestonPrice.loc[e,'NextMeasure'],P_temp,r,dt,param,Q,R)
        UpdateResult_EKF = extended_KF_update(HestonPrice.loc[e,'NextMeasure'],PredResult[0], PredResult[1], PredResult[2], PredResult[3],PredResult[4])
        (x_temp,P_temp)= UpdateResult_EKF
        (xuns_temp,Pa_O,state_uns,Cov_uns,P_w,P_v)=Unscented_KF_update(xuns_temp,HestonPrice.loc[e,'Measure'],HestonPrice.loc[e,'NextMeasure'],xuns_temp,Pa_O,L,r,dt,param,P_w,P_v)
        (state_particle,x_particle,Cov_part,w_particle)=Particle_Filter(x_particle,HestonPrice.loc[e,'Measure'],HestonPrice.loc[e,'NextMeasure'],Q,R,w_particle,N,param,r,dt)
        
        PCRLB_EKF = cramer_elements(J_EKF,x_temp,P_temp,HestonPrice.loc[e+gap,'Measure'],HestonPrice.loc[e+gap,'NextMeasure'],M,N,Q,R,r,dt,param,0,[])
        PCRLB_UKF = cramer_elements(J_UKF,state_uns,Cov_uns,HestonPrice.loc[e+gap,'Measure'],HestonPrice.loc[e+gap,'NextMeasure'],M,N,Q,R,r,dt,param,0,[])
        PCRLB_PF = cramer_elements(J_PF,state_particle,P_temp,HestonPrice.loc[e+gap,'Measure'],HestonPrice.loc[e+gap,'NextMeasure'],M,N,Q,R,r,dt,param,0,[x_particle,w_particle])
        #print(x_temp[0])
        (x_temp,P_temp,xuns_temp,Pa_O,x_particle,w_particle,index) = best_estimator([J_EKF_inv,J_UKF_inv, J_PF_inv],[P_temp,Cov_uns,P_temp],[x_temp,state_uns,state_particle],P_w,P_v,N,[x_particle,w_particle])
        
        V_present.append(x_temp)
        y_present.append(HestonPrice.loc[e,'NextMeasure'])
        
        (J_EKF,J_UKF, J_PF)= (PCRLB_EKF[0],PCRLB_UKF[0],PCRLB_PF[0])
        (J_EKF_inv,J_UKF_inv, J_PF_inv)= (PCRLB_EKF[1],PCRLB_UKF[1],PCRLB_PF[1])
        
        #print('Real Price',optionPrice.loc[e,'OptionPR'])
        #print('Predicted Price',BS(x_temp,optionPrice.loc[e,'StockPR'],optionPrice.loc[e,'StrikePrice'],optionPrice.loc[e,'TimeToMaturity']))
        print('filter',index)
        ABF_prices.append(BS(x_temp,HestonPrice.loc[e,'Measure'],r,dt))
        real_prices.append(HestonPrice.loc[e,'NextMeasure'])
        vol_ABF_train.append(x_temp)
        theta_ABF_train.append(param[1])
        rho_ABF_train.append(param[3])
        kappa_ABF_train.append(param[0])
        sigma_ABF_train.append(param[2])
        if(e>=300):
            look = 10000
            #param1 = fPMLE_Alt(V_prev[-look:],V_present[-look:],y_prev[-look:],y_present[-look:],r,dt1,param[0],param[1],param[2])
            param1 = NMLE_Alt(V_prev,V_present,y_prev,y_present,r,dt,param[0],param[1],param[2])
            #param1 = fPMLE_Alt(V_prev,V_present,y_prev,y_present,r,dt,param[0],param[1],param[2])
            #param = param1
            if(2*param1[0]*param1[1]>(param1[2]*param1[2])):
                param=param1
        print(param)
    
yT,yTplus1,M,N,Q,R,r,dt,param,isParticle,particle_data
        

def fPMLE(V_prev,V_present,yK,param,r,dt):
     n=len(V_present)
     V_prev_term =[1/a for a in V_prev]
     V_mix_term=[a*b for a, b in zip(V_present,V_prev_term)]
     const=1/(n*n)
     Beta1 = ((const*sum(V_present)*sum(V_prev_term))-(sum(V_mix_term)/n))/((const*sum(V_prev)*sum(V_prev_term))-1)
     Beta2 = ((sum(V_mix_term)/n)-(Beta1))/((1-Beta1)*sum(V_prev)/n)
     Beta2 = ((sum(V_mix_term)/n)-(Beta1))/((1-Beta1)*sum(V_prev)/n)
     V_linear_term=[a*b for a, b in zip(V_present,V_prev_term)]
     for i in range(0,n):
         term1=V_present[i] - Beta1*V_prev[i] - (Beta2*(1-Beta1)*(1-Beta1))
         Beta3= Beta3 + (term1*V_prev_term[i])
     Beta3=Beta3/n
     kcap= -np.log(Beta1)/dt
     theta = Beta2
     sigma = np.sqrt(2*kcap*Beta3/(1-(Beta1*Beta1)))
     for i in range(0,n):
         V_k=V_prev[i]
         V_K=V_present[i]
         Var = (1-(rho*rho))*dt*V_k
         m = yK-yk + (r -(V_k/2) - kcap*(rho/sigma)*(theta-V_k))*dt + ((rho/sigma)*(V_K-V_k))
         likelihood = likelihood + ((np.log(Var)/2) + (m*m/var)/2)
     likelihood = -likelihood
     return XN

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
        sigma= sigma1*sigma1
    sigma =np.sqrt(sigma)
    theta = (Pcap + (sigma*sigma/4))/kcap
    
    if(theta<0):
        theta = theta1
    for i in range(0,n):
        V_k=V_prev[i]
        V_K=V_present[i]
        y1= y_prev[i]
        y2= y_present[i]
        delta_W1 = (y2 - y1 - (r - (V_k/2))*dt)/(np.sqrt(V_k))
        delta_W2 = (V_K - V_k - ((theta - V_k)*kcap*dt))/(sigma*np.sqrt(V_k))
        likelihood= likelihood + (delta_W1*delta_W2)
    likelihood = likelihood/(n*dt)
    rho = likelihood
    rho = -0.9
    return [kcap,theta,sigma,rho]
        
def fPMLE_Alt(V_prev,V_present,y_prev,y_present,r,dt):
     n=len(V_present)
     V_prev_term =[1/a for a in V_prev]
     V_mix_term=[a*b for a, b in zip(V_present,V_prev_term)]
     const=1/(n*n)
     Beta1 = ((const*sum(V_present)*sum(V_prev_term))-(sum(V_mix_term)/n))/((const*sum(V_prev)*sum(V_prev_term))-1)
     Beta2 = ((sum(V_mix_term)/n)-(Beta1))/((1-Beta1)*sum(V_prev)/n)
     Beta2 = ((sum(V_mix_term)/n)-(Beta1))/((1-Beta1)*sum(V_prev)/n)
     V_linear_term=[a*b for a, b in zip(V_present,V_prev_term)]
     for i in range(0,n):
         term1=V_present[i] - Beta1*V_prev[i] - (Beta2*(1-Beta1)*(1-Beta1))
         Beta3= Beta3 + (term1*V_prev_term[i])
     Beta3=Beta3/n
     kcap= -np.log(Beta1)/dt
     theta = Beta2
     sigma = np.sqrt(2*kcap*Beta3/(1-(Beta1*Beta1)))
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

plt.plot(y)
plt.plot(x1_)
plt.title('Pf performance')
plt.legend(['True Value','PF Estimate'])
plt.xlabel('timestep')
plt.ylabel('Concentration'); 

plt.plot(x1_)
plt.plot(xTrue[:,0])
plt.title('Pf performance')
plt.legend(['PF Estimate', 'True State'])
plt.xlabel('timestep')
plt.ylabel('Concentration'); 

plt.plot(x2_)
plt.plot(xTrue[:,1])
plt.title('Pf performance on State 2')
plt.legend(['PF Estimate', 'True State'])
plt.xlabel('timestep')
plt.ylabel('Concentration'); 

    
plt.plot([10,100,500,1000,5000,10000,100000],rmse[1:])
plt.title('RMSE PF vs Kalman')
plt.xlabel('N_SIMULATONS_POINTS')
plt.ylabel('RMSE');

plt.plot(x1_)
plt.plot(state_means[1:])
plt.title('Pf performance')
plt.legend(['PF Estimate', 'Kalman State'])
plt.xlabel('timestep')
plt.ylabel('Returns');

plt.plot(y)
plt.plot(state_means)
plt.title('Kalman filter estimate of average')
plt.legend(['returns','Kalman Estimate'])
plt.xlabel('timestep')
plt.ylabel('returns');

plt.plot(y)
plt.plot(x1_)
plt.title('Paricle filter estimate')
plt.legend(['returns','PF Estimate'])
plt.xlabel('timestep')
plt.ylabel('returns');          


rms = sqrt(mean_squared_error(x1_, state_means))
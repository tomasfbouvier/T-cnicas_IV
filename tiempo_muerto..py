# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:13:02 2020

@author: Usuario
"""
import numpy as np
import scipy.stats as stats
def chi2(f, x, y, sigma):
    chi2list=[]
    chi2t=0
    for i in range(len(x)):
        chi2list.append((y[i]-f[i])**2/(sigma[i]**2)) 
        chi2t+=chi2list[i]
    return chi2t
import pandas as pd

#ruido
    
N = 1159; t = 49*60 + 13.32 - 0.053

nb= N/t
unb= np.sqrt(N)/t

print("ruido= ", nb, unb)

ut=0.034
dt=-0.053

#tiempo muerto

N1,t1, N2, t2, N3, t3 = np.loadtxt("tiempo_muerto.txt" , unpack=True)

t1-=dt;t2-=dt;t3-=dt



n1=[]
n2=[]
n3=[]
un1=[]
un2=[]
un3=[]
X=[]
uX=[]
Y=[]
uY=[]
Z=[]
uZ=[]
tau=0
tau2=[]
utau=0
utau2=[]
wt=0
for i in range (len(N1)):
    n1.append(N1[i]/t1[i])
    un1.append(np.sqrt(N1[i]/t1[i]**2+ (N1[i]/t1[i]**2*ut)**2 ))
    n2.append(N2[i]/t2[i])
    un2.append(np.sqrt(N2[i]/t2[i]**2+ (N2[i]/t2[i]**2*ut)**2 ))
    n3.append(N3[i]/t3[i])
    un3.append(np.sqrt(N3[i]/t3[i]**2+ (N3[i]/t3[i]**2*ut)**2 ))   
    X.append(n1[i]*n3[i]-nb*n2[i])
    uX.append(np.sqrt((n3[i]*un1[i])**2+(n1[i]*un3[i])**2+(n2[i]*unb)**2+(un2[i]*nb)**2))
    Y.append(n1[i]*n3[i]*(n2[i]+nb)-nb*n2[i]*(n1[i]+n3[i]))
    uY.append(np.sqrt((un1[i]*n3[i]*(n2[i]+nb))**2 + (n1[i]*un3[i]*(n2[i]+nb))**2 + (un1[i]*n3[i]*un2[i])**2+ (un1[i]*n3[i]*unb)**2 +(unb*n2[i]*(n1[i]+n3[i]))**2+(nb*un2[i]*(n1[i]+n3[i]))**2 +(nb*n2[i]*un1[i])**2 +(nb*n2[i]*n3[i])**2 )  )    
    Z.append((Y[i]*(n1[i]+n3[i]-n2[i]-nb))/X[i]**2)
    uZ.append(np.sqrt(((uY[i]*(n1[i]+n3[i]-n2[i]-nb))/X[i]**2)**2+((Y[i]*(un1[i]))/X[i]**2)**2 + (Y[i]*un3[i]/X[i]**2)**2 +(Y[i]*unb/X[i]**2)**2+(2*Y[i]*uX[i]*(n1[i]+n3[i]-n2[i]-nb)/X[i]**3)**2))
    tau2.append(X[i]*(1-np.sqrt(1-Z[i]))/Y[i])
    utau2.append(np.sqrt((uX[i]*(1-np.sqrt(1-Z[i]))/Y[i])**2+(X[i]*uZ[i]/(2*np.sqrt(1-Z[i])*Y[i]))**2+(X[i]*(1-np.sqrt(1-Z[i]))*uY[i]/Y[i]**2)**2))
    w=1/utau2[i]**2
    tau+=w*tau2[i]
    wt+=w


tau/=wt
f=tau*np.ones(len(tau2))  
utau=1/np.sqrt(wt)*np.sqrt(chi2(f,tau2,tau2,utau2)/len(utau2))

print("Tiempo muerto ", tau, utau  )

X=np.array(X)
uX=np.array(uX)
Y=np.array(Y)
uY=np.array(uY)
N1=np.array(N1); N2=np.array(N2); N3=np.array(N3)

df= pd.DataFrame({"N1": N1.astype(int), "t1 (s)": t1, "n1 ($s^{-1}$) ":np.around(n1,1), "u(n1) ($s^{-1}$)": np.around(un1,1),"N2": N2.astype(int), "t2 (s)": t2, "n2 ($s^{-1}$) ":np.around(n2,1), "u(n2) ($s^{-1}$)": np.around(un2,1),"N3": N3.astype(int), "t3 (s)": t3})
df2= pd.DataFrame({ "n3 ($s^{-1}$) ":np.around(n3,1), "u(n3) ($s^{-1}$)": np.around(un3,1),"X":X.astype(int) ,"u(X)":uX.astype(int),"Y":Y.astype(int),"u(Y)":uY.astype(int),"Z":np.around(Z,3), "u(Z)":np.around(uZ,3), "$\tau $ ":tau2, "u(\tau) ":utau2 })

print(df.to_latex(index=False, escape=False))

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:49:50 2020

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt 
import random as rand
import pandas as pd

def chi2(f, x, y, P, sigma):
    chi2list=[]
    chi2t=0
    for i in range(len(x)):
        chi2list.append((y[i]-P*f[i])**2/(sigma[i]**2))
        chi2t+=chi2list[i]
    return chi2t

def chisquare(obs,teor, sigma):
    chi2=0
    chivec=[]
    for i in range(len(obs)):
        chivec.append((obs[i]-teor[i])**2/(sigma[i]**2))
        chi2+=(obs[i]-teor[i])**2/(sigma[i]**2)
    return chi2, chivec

def racional2(x,a):
    return a/x**2


def geometrica(x,a):
    return a/2*(1-1/(np.sqrt(1+(1/x)**2)))

def random(t):
    return (rand.randrange(0,t*100000000, 1))/10**8


nb=0.392
unb=0.012
T12=3.02*10*365*24*3600 #este es el periodo de semidesintegracion
landa=np.log(2)/T12
A0=350390 #esta es la actividad de la fuente
t=(17+20)*365*24*3600+(4-2)*30*24*3600
A=A0*np.e**(-landa*t)


tau=0.0009739600954014595 #tiempo muerto de tu detector
utau=2.498776270180617e-05

d = [6.235, 5.275, 4.315, 3.355, 2.395, 1.435, 0.475] #las distancias a las que se encuentra tu fuente
N_distancia = [1258, 1627, 1694, 5963, 2247, 1719, 2983] # los contajes a esa distancia
t_distancia = [87.22, 79.91, 54.29, 115.26, 26.61, 11.99, 12.50] # los tiempos de contaje

ud=0.05
n=[]
un=[]


nnc=[]
unnc=[]
for i in range (len(N_distancia)):
    aux = N_distancia[i]/t_distancia[i]
    uaux = np.sqrt((np.sqrt(N_distancia[i])/t_distancia[i])**2+(0.3*N_distancia[i]/t_distancia[i]**2)**2) 
    nnc.append(aux)
    unnc.append(uaux)
    n.append(((aux)/(1-(aux)*tau)-nb)/A)
    un.append((np.sqrt((uaux/(1-tau*aux**2))**2+(utau*aux**2/(1-aux*tau)**2)**2+(unb)**2))/A)
  
    
un[2]=5*10**(-6)
plt.errorbar(d,n, xerr=ud, yerr=un, fmt = 'bx', capsize=3, alpha=.6, label="datos experimentales")
plt.xlabel("d (cm)")
plt.ylabel("$E_{T}$")


def sphericalrandom():
       theta= 2*np.pi*random(1)
       phi= np.arccos(1-2*random(1))
       return(theta, phi)

contador=0
PMC=[]
nm=np.zeros(len(d))
while(contador<3):
    contador+=1
        
    d2=[]    
      
    n2=[]

    nintr=[]
    unintr=[]
    for j in  range (len(d)):
    
        N=10000
    
        r=1.43 #radio de la ventana de tu detector
        R=d[j]
    
        n3=0
        
        limit= np.arctan(r/R)
        for i in range(N):
            theta,phi = sphericalrandom()
            if (abs(theta)<= 2*limit and abs(phi)<= 2*limit ):
                n3 += 1
        
        n2.append(n3/N)      
        d2.append(d[j])
      
        
    nm[:]+=n2[:]
    x=np.linspace(min(d2), max(d2), 1000)
    
    
    
    def chi2MC(P):
        return(chi2(n2,d2,n,P,un))
    
    
    
    
    c=opt.minimize(chi2MC, x0=20)
    
    print("MC", c)
    

    PMC.append(c.get("x")[0])

PMCm=np.mean(PMC)
PMCstd=np.std(PMC)



for i in range(len(n2)):
    n2[i]*=c.get("x")[0]
    
for i in range(len(nm)):
    nm[i]*=PMCm/contador  

print(chisquare(nm, n, un)[0])
chivecMC=map(int, chisquare(nm, n, un)[1])

plt.plot(d2,nm, 'r.', markersize=10, label="simulación MC")

pr,pcovr=opt.curve_fit(racional2, d, n, sigma=un)
pg,pcovg=opt.curve_fit(geometrica, d, n, sigma=un)


x=np.linspace(min(d), max(d), 1000)
plt.plot(x, racional2(x, pr[0]), 'y-', label="$E=1/d^{2}$")
plt.plot(x, geometrica(x, pg[0]), 'g-',label="$E=\dfrac{1}{2} (1-1/\sqrt{1+(\dfrac{R_{D}}{d})^{2}})$")

plt.legend(loc='best', shadow=True, fontsize = 9)
plt.show()

obsr=[]
obsg=[]
for di in d: 
    obsr.append(racional2(di,pr))
    obsg.append(geometrica(di,pg))

chi2r=chisquare(obsr, n, un)[0]
chi2g=chisquare(obsg,n,un)[0]

ef=[]
uef=[]

for i in range(len(obsg)):
    ef.append(n[i]*pg/obsg[i])
    uef.append(un[i]*pg/obsg[i])
    
ef=np.concatenate(ef,0)    
uef=np.concatenate(uef,0)

plt.figure()
plt.errorbar(d, list(ef), list(uef), fmt = 'bx', capsize=3, alpha=.6, label="datos experimentales")

n=np.array(n)
un=np.array(un)

df=pd.DataFrame({"distancia (cm)":d, "N ":N_distancia, "t (s)":t_distancia , "n (s^{-1})":np.round(nnc,2), "u( n )(s^{-1})":np.round(unnc,2) , "n_{corr} (s^{-1})": np.round(n*A,2), "u(n_{corr}) (s^{-1})": np.round(un*A,2), "E_{T}": np.round(n*10**5,2), "u(E_{T})": np.round(un*10**5,2)})
print(df.to_latex(index=False, escape=False))

print("obsr",obsr)
print("pr", pr)



chivecr=map(int, chisquare(obsr, n, un)[1])
chivecg=map(int, chisquare(obsg, n, un)[1] )

for i in range (len(obsr)):
    obsr[i]=obsr[i]/pr[0]
    obsg[i]/=pg[0]

obsr = [ '%.3f' % elem for elem in obsr ]
obsg = [ '%.4f' % elem for elem in obsg ]



df=pd.DataFrame({"E_{G}^{MC} ": nm[:]/PMCm,"$\chi^{2}_{i MC} $: ":chivecMC, "E_{G}^{S_{Lejana}} ":obsr, "$\chi^{2}_{i S_{Lej}} $: ":chivecr ,"E_{G}^{S_{Cercana}} ":obsg,"$\chi^{2}_{i S_{Cerc}} $: ":chivecg })
print(df.to_latex(index=False, escape=False))


"""
racio2=[]
for i in range(len(d2)):
    racio2.append(racional2(d2[i]))

def chi2racional(P):
    return(chi2(racio2,d2,n,P,un))




c=opt.minimize(chi2racional, x0=1)

print("racional2", c)

plt.plot(x, c.get("x")[0]/x**2, label="$E=1/d^{2}$")


print("""""")



geom=[]
for i in range(len(d2)):
    geom.append(geometrica(d2[i]))

def chi2geom(P):
    return(chi2(geom,d2,n,P,un))

c=opt.minimize(chi2geom, x0=20)
p, pcov= opt.curve_fit()

print("Geometrica ", c)


plt.plot(x, c.get("x")[0]/2*(1-1/np.sqrt(1+(r/x)**2)), label="$E=\dfrac{1}{2} (1-1/\sqrt{1+(\dfrac{R_{D}}{d})^{2}})$")



plt.legend(loc='best', shadow=True, fontsize = 9)


plt.show()

for j in range (len(d)):
    nintr.append(n[j]/geom[j])  
    unintr.append(un[j]/geom[j])  
    
plt.figure()
plt.plot(d2, nintr, 'b.')
plt.errorbar(d2,nintr, yerr=unintr, fmt = 'bx', capsize=3, alpha=.6)
plt.xlabel("d (cm)")
plt.ylabel("$E_{I} $")
plt.show()    

"""
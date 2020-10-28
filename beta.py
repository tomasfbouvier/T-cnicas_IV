# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:47:24 2020

@author: Usuario
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt 
import scipy.special as spe
import random as rand
import pandas as pd

def exp(x,a, b):
    return a*np.e**(-b*x)

def exp2(x, a, b):
    return a*np.e**(-b*x)


def racional2(x, y0, a, b):
    return (y0 + a)*b/x**2
def racional(x, y0, a, b):
    return (y0 + a)*b/x**(1/2)
def recta(x,a, b):
    return a+b*x

def geometrica(x, a, b):
    return b/2*(1-1/(np.sqrt(1+(a/x)**2)))

def random(t):
    return (rand.randrange(0,t*100000000, 1))/10**8

def count_elements(seq) -> dict:
    """Tally elements from `seq`."""
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist
def chi2(f, x, y, sigma):
    chi2list=[]
    chi2t=0
    for i in range(len(x)):
        chi2list.append((y[i]-f[i])**2/(sigma[i]**2)) 
        chi2t+=chi2list[i]
    return chi2t


#ruido
    
N = 1159; t = 49*60 + 13.32

nb= N/t
unb= np.sqrt(N)/t

dt=-0.05
ut= 0.03

print("ruido= ", nb, unb)

tau= 0.0009739600954014595
utau=2.498776270180617e-05

#Beta

"""

minutos, segundos, cuentas, espesor = np.loadtxt("beta_filtros.txt", unpack=True)

plt.figure()
t = 55*60 + 5.34; N = 1258
n_bkg = N/t
ub = np.sqrt((np.sqrt(N)/t)**2+(N/t**2*0.3)**2)

tiempo=[]
for i in range (len(minutos)):
    tiempo.append(minutos[i]*60+segundos[i])
    
tiempo=np.array(tiempo)-dt



neff=[]
uneff=[]
ulog=[]
nnc=[]
unnc=[]


for i in range (0,len(espesor)):
    nnc.append(((cuentas[i])/tiempo[i]))
    unnc.append(np.sqrt((np.sqrt(cuentas[i])/tiempo[i])**2+((cuentas[i])/tiempo[i]**2*ut)**2))
    neff.append((cuentas[i]/tiempo[i])/(1-tau*cuentas[i]/tiempo[i]) - n_bkg)
    uneff.append(np.sqrt((tiempo[i]*np.sqrt(cuentas[i])/(tiempo[i]-tau*cuentas[i])**2)**2+((cuentas[i]*ut)/(tiempo[i]-cuentas[i]*tau)**2)**2+ (utau*(cuentas[i]/tiempo[i])**2/(1-tau*(cuentas[i]/tiempo[i]))**2)**2))
    
    

plt.plot(espesor, neff, 'b.')
plt.errorbar(espesor, neff, yerr=uneff , fmt = 'bx', capsize=3, alpha=.6 ,label='Data')

p, pcov= opt.curve_fit(exp, espesor, neff, sigma=uneff, p0=[1,0.005], maxfev=2000000000)


x=np.linspace(min(espesor), max(espesor)+100, 1000)
plt.plot(x, exp(x,p[0],p[1]), label="Ajuste y=$ae^{-bx} $")
plt.plot(x, exp(x, p[0]+np.sqrt(pcov[0,0]), p[1]-np.sqrt(pcov[1,1])), 'r--')
plt.plot(x, exp(x, p[0]-np.sqrt(pcov[0,0]), p[1]+np.sqrt(pcov[1,1])), 'r--', label= "1 $\sigma$")                
plt.tick_params(axis='both',direction='in', top='on', right='on')

plt.xlabel("w $(mg/cm^{2})$")
plt.ylabel("n ($s^{-1}$)")
plt.legend(loc='best', shadow=True, fontsize = 12)

f=[]
f[:]= exp(espesor[:], p[0], p[1])

chi2= chi2(f, espesor, neff, uneff)



plt.show()

df= pd.DataFrame({"w $(mg/cm^{2})": espesor, "N ":cuentas.astype(int), "t(s)":tiempo, "n ($s^{-1})$ ":np.around(nnc,3),  "u(n) ($s^{-1}$)":np.around(unnc,3)   })
#df= pd.DataFrame({"N ":N.astype(int) , "t(s)":t, "n ($s^{-1})$":[374.8,   3.60,   1.585], "u(n) ($s^{-1})$":[3.4, 0.14 , 0.073], "$n_{corr}$ $s^{-1}$": [590,   3.22,   1.195], "$u(n_{corr})$ $s^{-1}$": [15,  0.14,  0.074], "log($n_{corr} $)":np.around(logn,3), "u(log($n_{corr}$))": np.around(lognerr,3) } )
print(df.to_latex(index=False, escape=False))

Q=0.7086
MCl=35.96830698
MAr=35.967545105
me=5.485*10**(-4)

emax= Q-931.494*(MCl-MAr-me)

print("emax", emax)


"""
#beta con papel de aluminio

w, tiempo, cuentas= np.loadtxt("beta_papel_de_aluminio.txt", unpack=True, skiprows=0)

tiempo-=dt

plt.figure()


neff=[]
uneff=[]
ulog=[]
nnc=[]
unnc=[]


for i in range (0,len(w)):
    nnc.append(((cuentas[i])/tiempo[i]))
    unnc.append(np.sqrt((np.sqrt(cuentas[i])/tiempo[i])**2+((cuentas[i])/tiempo[i]**2*ut)**2))
    neff.append((cuentas[i]/tiempo[i])/(1-tau*cuentas[i]/tiempo[i]) - nb)
    uneff.append(np.sqrt((tiempo[i]*np.sqrt(cuentas[i])/(tiempo[i]-tau*cuentas[i])**2)**2+((cuentas[i]*ut)/(tiempo[i]-cuentas[i]*tau)**2)**2+ (utau*(cuentas[i]/tiempo[i])**2/(1-tau*(cuentas[i]/tiempo[i]))**2)**2))
    
    

plt.plot(w, neff, 'b.')
plt.errorbar(w, neff, yerr=uneff , fmt = 'bx', capsize=3, alpha=.6 ,label='Data')

p, pcov= opt.curve_fit(exp, w, neff, sigma=uneff, p0=[1,0.005], maxfev=2000000000)

x=np.linspace(min(w), max(w), 1000)
plt.plot(x, exp(x,p[0],p[1]), label="Ajuste y=$ae^{-bx} $")
plt.plot(x, exp(x, p[0]+np.sqrt(pcov[0,0]), p[1]-np.sqrt(pcov[1,1])), 'r--')
plt.plot(x, exp(x, p[0]-np.sqrt(pcov[0,0]), p[1]+np.sqrt(pcov[1,1])), 'r--', label= "1 $\sigma$")                

df= pd.DataFrame({"w $(mg/cm^{2})": w.astype(int), "N ":cuentas.astype(int), "t(s)":tiempo, "n ($s^{-1})$ ":np.around(nnc,2),  "u(n) ($s^{-1}$)":np.around(unnc,2)   })
#df= pd.DataFrame({"N ":N.astype(int) , "t(s)":t, "n ($s^{-1})$":[374.8,   3.60,   1.585], "u(n) ($s^{-1})$":[3.4, 0.14 , 0.073], "$n_{corr}$ $s^{-1}$": [590,   3.22,   1.195], "$u(n_{corr})$ $s^{-1}$": [15,  0.14,  0.074], "log($n_{corr} $)":np.around(logn,3), "u(log($n_{corr}$))": np.around(lognerr,3) } )
print(df.to_latex(index=False, escape=False))

plt.xlabel("w $(mg/cm^{2})$")
plt.ylabel("n ($s^{-1}$)")
plt.legend(loc='best', shadow=True, fontsize = 12)
plt.tick_params(axis='both',direction='in', top='on', right='on')

f=[]
f[:]= exp(w[:], p[0], p[1])

chi2= chi2(f, w, neff, uneff)


plt.show()



# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:51:46 2020

@author: Usuario
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt 
import scipy.special as spe
import random as rand
import pandas as pd

def exp(x,a, b, y0):
    return y0 +a*np.e**(-b*x)

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

#beta filtros
"""
minutos, segundos, cuentas, espesor = np.loadtxt("beta_filtros.txt", unpack=True)

plt.figure()
t = 55*60 + 5.34; N = 1258
n_bkg = N/t
ub = np.sqrt((np.sqrt(N)/t)**2+(N/t**2*0.3)**2)

tiempo=[]
for i in range (len(minutos)):
    tiempo.append(minutos[i]*60+segundos[i])

neff=[]
uneff=[]
ulog=[]

for i in range (0,len(espesor)):
    neff.append((cuentas[i]/tiempo[i])/(1-tau*cuentas[i]/tiempo[i]) - n_bkg)
    uneff.append(np.sqrt((tiempo[i]*np.sqrt(cuentas[i])/(tiempo[i]-tau*cuentas[i])**2)**2+((cuentas[i]*0.3)/(tiempo[i]-cuentas[i]*tau)**2)**2+ (utau*(cuentas[i]/tiempo[i])**2/(1-tau*(cuentas[i]/tiempo[i]))**2)**2))
    
    

plt.plot(espesor, neff, 'b.')
plt.errorbar(espesor, neff, yerr=uneff , fmt = 'bx', capsize=3, alpha=.6 ,label='Data')

plt.xlabel("w (g/cm^{2})")
plt.ylabel("n ($s^{-1}$)")
plt.legend(loc='best', shadow=True, fontsize = 12)


p, pcov= opt.curve_fit(exp, espesor, neff, sigma=uneff)







plt.show()




#beta con papel de aluminio

w, tiempo, cuentas= np.loadtxt("beta_papel_de_aluminio.txt", unpack=True, skiprows=0)

plt.figure()


neff=[]
uneff=[]
nlameff=[]
logn=[]
ulog=[]

for i in range (0,len(w)):
    neff.append((cuentas[i]/tiempo[i])/(1-tau*cuentas[i]/tiempo[i]))
    uneff.append(np.sqrt((tiempo[i]*np.sqrt(cuentas[i])/(tiempo[i]-tau*cuentas[i])**2)**2+((cuentas[i]*0.3)/(tiempo[i]-cuentas[i]*tau)**2)**2+ (utau*(cuentas[i]/tiempo[i])**2/(1-tau*(cuentas[i]/tiempo[i]))**2)**2))
    nlameff.append(w[i])
    logn.append(np.log(neff[i]))


plt.plot(w, neff, 'b.')
plt.errorbar(w, neff, yerr=uneff , fmt = 'bx', capsize=3, alpha=.6 ,label='Data')



plt.show()

for i in range (len (uneff)):
    ulog.append(uneff[i]/neff[i])
    
plt.figure()

plt.plot(nlameff, np.log(neff), 'b.')
plt.errorbar(nlameff, np.log(neff), yerr=ulog , fmt = 'bx', capsize=3, alpha=.6 ,label='Data')



x=np.linspace(min(nlameff), max(nlameff), 10000)
fondo=[]


for i  in range(len(x)):
    fondo.append(np.log(n_bkg))

plt.plot(x, fondo)


param, pcov= opt.curve_fit(recta, nlameff, np.log(neff), sigma=ulog)

f=[]
for i in range(len(nlameff)):
    f.append(recta(nlameff[i], param[0],param[1]))
    

chi2n = chi2(f, nlameff, np.log(neff),ulog)


x=np.linspace(min(nlameff), max(nlameff), 100)
plt.plot(x, recta(x, param[0], param[1]), label="ajuste lineal")
plt.plot(x, recta(x, param[0]+np.sqrt(pcov[0,0]), param[1]-np.sqrt(pcov[1,1])), 'r--', label="$\sigma $")
plt.plot(x, recta(x, param[0]-np.sqrt(pcov[0,0]), param[1]+np.sqrt(pcov[1,1])), 'r--')
plt.xlabel("# láminas")
plt.ylabel("log(n) ")
plt.legend(loc='best', shadow=True, fontsize = 12)

a=param[0]
b=param[1]/((3.4*10**(-3)))


ua=np.sqrt(pcov[0,0])
ubb=np.sqrt(pcov[1,1])/((3.4*10**(-3)))

plt.show()

grosor=(np.log(n_bkg)-param[0])/param[1]
print("logbckg", np.log(n_bkg))
print("grosor impenetrable",grosor)

uw=np.sqrt((ub/b)**2+(ua/b)**2+((n_bkg-a)*ubb/b**2)**2)



Emax= 0.239979 #MeV
uEmax= uw/357.283

Q= (5.4857990946*10**(-4) + 35.967545105-35.96830698)*931.494

massneutrino=(Emax+Q) #MeV/c^2

print("masa del neutrino: ", massneutrino)




"""



#alfa
t, N, w = np.loadtxt("alfa.txt", unpack=True)
rho=30
urho=5

print(w)

for i in range(len(w)):
    w[i]*=rho

n=[]
nnc=[]
logn=[]
nerr=[]
nerrnc=[]
lognerr=[]
for i in range(len(N)):
    n.append((N[i]/t[i])/(1-tau*(N[i]/t[i]))-nb)
    nnc.append(N[i]/t[i])
    nerrnc.append(np.sqrt((np.sqrt(N[i])/t[i])**2+(N[i]/t[i]**2*ut)**2))
    nerr.append(np.sqrt((t[i]*np.sqrt(N[i])/(t[i]-tau*N[i])**2)**2+((N[i]*0.3)/(t[i]-N[i]*tau)**2)**2+ (utau*(N[i]/t[i])**2/(1-tau*(N[i]/t[i]))**2)**2 + unb**2))
    logn.append(np.log(n[i]))
    lognerr.append(1/n[i]*nerr[i])
    
p, pcov=opt.curve_fit(recta, w, logn, sigma=lognerr)   
sigma0=np.sqrt(pcov[0,0])
sigma1=np.sqrt(pcov[1,1])

x=np.linspace(min(w), max(w), 100)    

f=[]
r=[]
for i in range (len(w)):
    f.append(recta(w[i], p[0], p[1]))

chialfa=chi2(f,w, logn, lognerr)

plt.plot(w, logn, 'b.')
plt.errorbar(w, logn, yerr=lognerr , fmt = 'bx', capsize=3, alpha=.6 ,label='Data')
#plt.plot(x,recta(x, p[0], p[1]), label= "ajuste lineal")
#plt.plot(x, recta(x, p[0]+sigma0, p[1]-sigma1), 'r--')
#plt.plot(x, recta(x, p[0]-sigma0, p[1]+sigma1),'r--', label="1 $\sigma$")
plt.xlabel("w (g/$m^{2}$)")
plt.ylabel("logn ($s^{-1}$)")
plt.legend(loc='best', shadow=True, fontsize = 12)
plt.show()

n=np.array(n)
nerr=np.array(nerr)
nnc=np.array(nnc)

df= pd.DataFrame({"N ":N.astype(int) , "t(s)":t, "n ($s^{-1})$":[374.8,   3.60,   1.585], "u(n) ($s^{-1})$":[3.4, 0.14 , 0.073], "$n_{corr}$ $s^{-1}$": [590,   3.22,   1.195], "$u(n_{corr})$ $s^{-1}$": [15,  0.14,  0.074], "log($n_{corr} $)":np.around(logn,3), "u(log($n_{corr}$))": np.around(lognerr,3) } )

print(df.to_latex(index=False, escape=False))
"""

 
param2, u2 =opt.curve_fit(recta, d2, n, sigma=un)

x=np.linspace(min(d2), max(d2), 100)

plt.plot(x,recta(x,param2[0],param2[1]), 'g--')


plt.show()

plt.figure()
param, u =opt.curve_fit(exp, d, n)

x=np.linspace(min(d), max(d), 100)

print(param)

plt.plot(x,exp(x,param[0],param[1], param[2]), 'g--')
plt.plot(d, n, 'b.', markersize=5)
plt.errorbar(d, n, yerr=un , fmt = 'bx', capsize=3, alpha=.6 ,label='Data')

plt.xlabel("exponencial")


plt.show()

#intento racional

plt.figure()
param, u =opt.curve_fit(racional2, d, n)

x=np.linspace(min(d), max(d), 100)

print(param)

plt.plot(x,racional2(x,param[0],param[1], param[2]), 'g--')
plt.plot(d, n, 'b.', markersize=5)
plt.errorbar(d, n, yerr=un , fmt = 'bx', capsize=3, alpha=.6 ,label='Data')

plt.xlabel("racional")
plt.show()

#haciedo como me dijo manuel

a=28.6/28

plt.plot(d, n, 'b.', markersize=5)
plt.errorbar(d, n, yerr=un , fmt = 'bx', capsize=3, alpha=.6 ,label='Data')
plt.show()
plt.figure()
plt.plot(x, geometrica(x,a,1))
plt.show()

n2=[]

for i in range(len(n)):
    n2.append(n[i]/geometrica(d[i], a,1))

plt.figure()    
plt.plot(d, n2, 'b.') 
plt.show()   

"""

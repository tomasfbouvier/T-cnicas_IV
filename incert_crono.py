# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:45:46 2020

@author: Usuario
"""

import numpy as np
import pandas as pd

t1,t2=np.loadtxt("incert_crono.txt", unpack=True)

deltat=t1[:]-t2[:]

media=np.mean(deltat)
desv= np.std(deltat)

df=pd.DataFrame({"t1 (s)":t1, "t2 (s)": t2, "$\Delta t $ (s)":deltat})
print(df.to_latex(index=False, escape=False))

print("media=", media)
print("desv=", desv)
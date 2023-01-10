#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:46:31 2022

@author: laura
"""


%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import curve_fit
import scipy.stats as stats
import scipy.odr.odrpack as odr


notes = []
means = []
stds = []
theory_freqs = [220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30]
precisions = [1, 1.2, 1.4, 1.5, 1.7, 1.9, 2.3, 2.5, 3, 3, 4.6, 1.4]


# -------- A -----------


A = pd.read_csv("Data/A/STAT.CSV", skiprows=7)

del A['Unnamed: 1']
del A['Unnamed: 3']
del A['Unnamed: 4']

A_mean = A.iloc[12][1]
A_std = A.iloc[13][1]

notes.append('A')
means.append(A_mean)
stds.append(A_std)


# -------- Asharp -----------


Asharp = pd.read_csv("Data/A#/STAT.CSV", skiprows=7)

del Asharp['Unnamed: 1']
del Asharp['Unnamed: 3']
del Asharp['Unnamed: 4']

Asharp_mean = Asharp.iloc[12][1]
Asharp_std = Asharp.iloc[13][1]

notes.append('Asharp')
means.append(Asharp_mean)
stds.append(Asharp_std)



# -------- B -----------


B = pd.read_csv("Data/B/STAT.CSV", skiprows=7)

del B['Unnamed: 1']
del B['Unnamed: 3']
del B['Unnamed: 4']

B_mean = B.iloc[12][1]
B_std = B.iloc[13][1]

notes.append('B')
means.append(B_mean)
stds.append(B_std)



# -------- C -----------


C = pd.read_csv("Data/C/STAT.CSV", skiprows=7)

del C['Unnamed: 1']
del C['Unnamed: 3']
del C['Unnamed: 4']

C_mean = C.iloc[12][1]
C_std = C.iloc[13][1]

notes.append('C')
means.append(C_mean)
stds.append(C_std)


# -------- Csharp -----------


Csharp = pd.read_csv("Data/C#/STAT.CSV", skiprows=7)

del Csharp['Unnamed: 1']
del Csharp['Unnamed: 3']
del Csharp['Unnamed: 4']

Csharp_mean = Csharp.iloc[12][1]
Csharp_std = Csharp.iloc[13][1]

notes.append('Csharp')
means.append(Csharp_mean)
stds.append(Csharp_std)




# -------- D -----------


D = pd.read_csv("Data/D/STAT.CSV", skiprows=7)

del D['Unnamed: 1']
del D['Unnamed: 3']
del D['Unnamed: 4']

D_mean = D.iloc[12][1]
D_std = D.iloc[13][1]

notes.append('D')
means.append(D_mean)
stds.append(D_std)


# -------- Dsharp -----------

%matplotlib auto

Dsharp = pd.read_csv("Data/D#/STAT.CSV", skiprows=7)

del Dsharp['Unnamed: 1']
del Dsharp['Unnamed: 3']
del Dsharp['Unnamed: 4']

Dsharp_mean = Dsharp.iloc[12][1]
Dsharp_std = Dsharp.iloc[13][1]

notes.append('Dsharp')
means.append(Dsharp_mean)
stds.append(Dsharp_std)



# -------- E -----------


E = pd.read_csv("Data/E/STAT.CSV", skiprows=7)

del E['Unnamed: 1']
del E['Unnamed: 3']
del E['Unnamed: 4']

E_mean = E.iloc[12][1]
E_std = E.iloc[13][1]

notes.append('E')
means.append(E_mean)
stds.append(E_std)



# -------- F -----------


F = pd.read_csv("Data/F/STAT.CSV", skiprows=7)

del F['Unnamed: 1']
del F['Unnamed: 3']
del F['Unnamed: 4']

F_mean = F.iloc[12][1]
F_std = F.iloc[13][1]

notes.append('F')
means.append(F_mean)
stds.append(F_std)



# -------- Fsharp -----------

%matplotlib inline

Fsharp = pd.read_csv("Data/F#/STAT.CSV", skiprows=7)

del Fsharp['Unnamed: 1']
del Fsharp['Unnamed: 3']
del Fsharp['Unnamed: 4']

Fsharp_mean = Fsharp.iloc[12][1]
Fsharp_std = Fsharp.iloc[13][1]

notes.append('Fsharp')
means.append(Fsharp_mean)
stds.append(Fsharp_std)




# -------- G -----------

%matplotlib auto

G = pd.read_csv("Data/G/STAT.CSV", skiprows=7)

del G['Unnamed: 1']
del G['Unnamed: 3']
del G['Unnamed: 4']

G_mean = G.iloc[12][1]
G_std = G.iloc[13][1]

notes.append('G')
means.append(G_mean)
stds.append(G_std)




# -------- Gsharp -----------

%matplotlib inline

Gsharp = pd.read_csv("Data/G#/STAT.CSV", skiprows=7)

del Gsharp['Unnamed: 1']
del Gsharp['Unnamed: 3']
del Gsharp['Unnamed: 4']

Gsharp_mean = Gsharp.iloc[12][1]
Gsharp_std = Gsharp.iloc[13][1]

notes.append('Gsharp')
means.append(Gsharp_mean)
stds.append(Gsharp_std)


#%%


%matplotlib inline


stds = np.array([float(i) for i in stds])
means = np.array([float(i) for i in means])


plt.figure(1)
plt.plot(theory_freqs, means, '-o')
plt.grid()
plt.errorbar(theory_freqs, means, yerr=stds, capsize=3.8, capthick=2, linestyle='', elinewidth=3, color='red')
plt.ylabel('Mean measured frequency [Hz]')
plt.xlabel('Target frequency [Hz]')

freq_DF = pd.DataFrame(means, theory_freqs)
print(freq_DF)





plt.figure(2)
diffs = (theory_freqs - means)**2
plt.scatter(means, diffs, s=50, color='black')
plt.grid()
plt.xlabel('Mean measured frequency [Hz]')
plt.ylabel('Difference squared from target [$Hz^2$]')


fit1 = np.polyfit(means, diffs, 1)
x1 = np.linspace(means[0], means[-1], 100)
fit_vals1 = np.poly1d(fit1)
plt.plot(x1, fit_vals1(x1), lw=2.3, color='purple')







plt.figure(3)
plt.scatter(means, precisions, s=40, color='black')
plt.grid()
plt.xlabel('Mean measured frequency [Hz]')
plt.ylabel('Frequency increase with -0.125us delay [Hz]')

fit2 = np.polyfit(means[:10], precisions[:10], 1)
x2 = np.linspace(means[0]-5, means[-3]+8, 100)
fit_vals2 = np.poly1d(fit2)
plt.plot(x2, fit_vals2(x2), color='forestgreen', lw=2)







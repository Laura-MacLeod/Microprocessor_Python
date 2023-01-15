#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:38:08 2022

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

def sine(x, p1, p2, p3, p4):
    out = p1 + p2*np.sin(p3*x + p4)
    return out

# Function in form for ODR

def sine_odr(p, x):
    out = p[1] + p[2]*np.sin(p[3]*x + p[4])
    return out

plt.rcParams['font.size'] = 18

#%% -------- A -----------

%matplotlib auto

plt.rcParams['font.size'] = 18

A = pd.read_csv("Data/A/WFM.CSV")

A_time = A["in s"]
A_PWM = A["C1 in V"]
A_sine = A["C2 in V"]

# plt.figure(1)
# plt.plot(A_time[1000:8000], A_PWM[1000:8000])
# plt.plot(A_time, A_sine)


# ----- FIT -----

A_guess = [1, 10, 1400, 1]

A_time_range = np.linspace(A_time[0], A_time[len(A_time)-1], len(A_time))

A_para, A_cov = sp.optimize.curve_fit(sine, A_time, A_sine, A_guess, maxfev=100000)

A_range = A_time[len(A_time)-1] - A_time[0]
A_peaks = (A_para[2] / (2*np.pi)) * A_range
A_ind = 5*(int(len(A_time)/A_peaks))

plt.plot(A_time[:A_ind], A_sine[:A_ind], color='#0e3b84', lw=3.5)

plt.plot(A_time_range[:A_ind], sine(A_time_range[:A_ind], A_para[0], A_para[1], A_para[2], A_para[3]), lw=3.5, color='#f7cf09')
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")


# ----- CHI SQUARED VALUE -----

A_sine_fit = sine(A_time_range[:A_ind], A_para[0], A_para[1], A_para[2], A_para[3])
A_residual = A_sine[:A_ind] - A_sine_fit
A_chi = (sum((A_residual**2))/(A_sine_fit * len(A_sine_fit)))
print("A chi squared value = %0.6s" % A_chi)


# ----- PARAMETER UNCERTAINTIES -----

A_p1 = A_para[0]
A_e1 = np.sqrt(A_cov[0][0])
print("%0.8s ± %0.8s" % (A_p1, A_e1))
print("Percentage error %0.5s" % ((A_e1/A_p1)*100))
print(" ")

A_p2 = A_para[1]
A_e2 = np.sqrt(A_cov[1][1])
print("%0.8s ± %0.8s" % (A_p2, A_e2))
print("Percentage error %0.5s" % ((A_e2/A_p2)*100))
print(" ")

A_p3 = A_para[2]
A_e3 = np.sqrt(A_cov[2][2])
print("%0.8s ± %0.8s" % (A_p3, A_e3))
print("Percentage error %0.7s" % ((A_e3/A_p3)*100))
print(" ")

A_p4 = A_para[3]
A_e4 = np.sqrt(A_cov[3][3]) 
print("%0.8s ± %0.8s" % (A_p4, A_e4))
print("Percentage error %0.5s" % ((A_e4/A_p4)*100))
print(" ")


A_sum = ((A_e1/A_p1)*100)**2 + ((A_e2/A_p2)*100)**2 + ((A_e3/A_p3)*100)**2 + ((A_e4/A_p4)*100)**2
print("Sum of percentage errors: ", A_sum)








# odr_model = odr.Model(sine_odr)
# odr_vals = odr.RealData(A_time[:A_ind], A_sine[:A_ind])
# odr_vals2 = odr.ODR(odr_vals, odr_model, beta0=[0, 1, 10, 1400, 1])
# odr_out = odr_vals2.run()
# odr_out.pprint()
# odr_bet = odr_out.beta
# odr_cov = odr_out.cov_beta


# plt.plot(A_time_range[:A_ind], sine(A_time_range[:A_ind], odr_bet[1], odr_bet[2], odr_bet[3], odr_bet[4]), label='ODR')






#%% -------- Asharp -----------


Asharp = pd.read_csv("Data/A#/WFM.CSV")

Asharp_time = Asharp["in s"]
Asharp_PWM = Asharp["C1 in V"]
Asharp_sine = Asharp["C2 in V"]

# plt.figure(1)
# plt.plot(Asharp_time[1000:8000], Asharp_PWM[1000:8000])
# plt.plot(Asharp_time, Asharp_sine)


# ----- FIT -----

Asharp_guess = [1, 10, 1500, 1]

Asharp_time_range = np.linspace(Asharp_time[0], Asharp_time[len(Asharp_time)-1], len(Asharp_time))

Asharp_para, Asharp_cov = sp.optimize.curve_fit(sine, Asharp_time, Asharp_sine, Asharp_guess, maxfev=100000)

Asharp_range = Asharp_time[len(Asharp_time)-1] - Asharp_time[0]
Asharp_peaks = (Asharp_para[2] / (2*np.pi)) * Asharp_range
Asharp_ind = 10*(int(len(Asharp_time)/Asharp_peaks))

plt.plot(Asharp_time[:Asharp_ind], Asharp_sine[:Asharp_ind])

plt.plot(Asharp_time_range[:Asharp_ind], sine(Asharp_time_range[:Asharp_ind], Asharp_para[0], Asharp_para[1], Asharp_para[2], Asharp_para[3]), lw=3)


# ----- CHI SQUARED VALUE -----

Asharp_sine_fit = sine(Asharp_time_range[:Asharp_ind], Asharp_para[0], Asharp_para[1], Asharp_para[2], Asharp_para[3])
Asharp_residual = Asharp_sine[:Asharp_ind] - Asharp_sine_fit
Asharp_chi = (sum((Asharp_residual**2 *len(Asharp_sine_fit))/Asharp_sine_fit))
print("Asharp chi squared value = %0.6s" % Asharp_chi)


# ----- PARAMETER UNCERTAINTIES -----

Asharp_p1 = Asharp_para[0]
Asharp_e1 = np.sqrt(Asharp_cov[0][0])
print("%0.8s ± %0.8s" % (Asharp_p1, Asharp_e1))
print("Percentage error %0.5s" % ((Asharp_e1/Asharp_p1)*100))
print(" ")

Asharp_p2 = Asharp_para[1]
Asharp_e2 = np.sqrt(Asharp_cov[1][1])
print("%0.8s ± %0.8s" % (Asharp_p2, Asharp_e2))
print("Percentage error %0.5s" % ((Asharp_e2/Asharp_p2)*100))
print(" ")

Asharp_p3 = Asharp_para[2]
Asharp_e3 = np.sqrt(Asharp_cov[2][2])
print("%0.8s ± %0.8s" % (Asharp_p3, Asharp_e3))
print("Percentage error %0.7s" % ((Asharp_e3/Asharp_p3)*100))
print(" ")

Asharp_p4 = Asharp_para[3]
Asharp_e4 = np.sqrt(Asharp_cov[3][3]) 
print("%0.8s ± %0.8s" % (Asharp_p4, Asharp_e4))
print("Percentage error %0.5s" % ((Asharp_e4/Asharp_p4)*100))
print(" ")


Asharp_sum = ((Asharp_e1/Asharp_p1)*100)**2 + ((Asharp_e2/Asharp_p2)*100)**2 + ((Asharp_e3/Asharp_p3)*100)**2 + ((Asharp_e4/Asharp_p4)*100)**2
print("Sum of percentage errors: ", Asharp_sum)

# Asharp_odr_model = odr.Model(sine_odr)
# Asharp_odr_vals = odr.RealData(Asharp_time[:Asharp_ind], Asharp_sine[:Asharp_ind])
# Asharp_odr_vals2 = odr.ODR(Asharp_odr_vals, Asharp_odr_model, beta0=[0, 1, 10, 1400, 1])
# Asharp_odr_out = Asharp_odr_vals2.run()
# Asharp_odr_out.pprint()
# Asharp_odr_bet = Asharp_odr_out.beta
# Asharp_odr_cov = Asharp_odr_out.cov_beta


# plt.plot(Asharp_time_range[:Asharp_ind], sine(Asharp_time_range[:Asharp_ind], Asharp_odr_bet[1], Asharp_odr_bet[2], Asharp_odr_bet[3], Asharp_odr_bet[4]), label='ODR')





#%% -------- B -----------


B = pd.read_csv("Data/B/WFM.CSV")

B_time = B["in s"]
B_PWM = B["C1 in V"]
B_sine = B["C2 in V"]

# plt.figure(1)
# plt.plot(B_time[1000:8000], B_PWM[1000:8000])
# plt.plot(B_time, B_sine)


# ----- FIT -----

B_guess = [1, 10, 1500, 1]

B_time_range = np.linspace(B_time[0], B_time[len(B_time)-1], len(B_time))

B_para, B_cov = sp.optimize.curve_fit(sine, B_time, B_sine, B_guess, maxfev=100000)

B_range = B_time[len(B_time)-1] - B_time[0]
B_peaks = (B_para[2] / (2*np.pi)) * B_range
B_ind = 10*(int(len(B_time)/B_peaks))

plt.plot(B_time[:B_ind], B_sine[:B_ind])

plt.plot(B_time_range[:B_ind], sine(B_time_range[:B_ind], B_para[0], B_para[1], B_para[2], B_para[3]), lw=3)


# ----- CHI SQUARED VALUE -----

B_sine_fit = sine(B_time_range[:B_ind], B_para[0], B_para[1], B_para[2], B_para[3])
B_residual = B_sine[:B_ind] - B_sine_fit
B_chi = (sum((B_residual**2 *len(B_sine_fit))/B_sine_fit))
print("B chi squared value = %0.6s" % B_chi)


# ----- PARAMETER UNCERTAINTIES -----

B_p1 = B_para[0]
B_e1 = np.sqrt(B_cov[0][0])
print("%0.8s ± %0.8s" % (B_p1, B_e1))
print("Percentage error %0.5s" % ((B_e1/B_p1)*100))
print(" ")

B_p2 = B_para[1]
B_e2 = np.sqrt(B_cov[1][1])
print("%0.8s ± %0.8s" % (B_p2, B_e2))
print("Percentage error %0.5s" % ((B_e2/B_p2)*100))
print(" ")

B_p3 = B_para[2]
B_e3 = np.sqrt(B_cov[2][2])
print("%0.8s ± %0.8s" % (B_p3, B_e3))
print("Percentage error %0.7s" % ((B_e3/B_p3)*100))
print(" ")

B_p4 = B_para[3]
B_e4 = np.sqrt(B_cov[3][3]) 
print("%0.8s ± %0.8s" % (B_p4, B_e4))
print("Percentage error %0.5s" % ((B_e4/B_p4)*100))
print(" ")


B_sum = ((B_e1/B_p1)*100) + ((B_e2/B_p2)*100) + ((B_e3/B_p3)*100) + ((B_e4/B_p4)*100)
print("Sum of percentage errors: ", B_sum)


#%% -------- C -----------


C = pd.read_csv("Data/C/WFM.CSV")

C_time = C["in s"]
C_PWM = C["C1 in V"]
C_sine = C["C2 in V"]

# plt.figure(1)
# plt.plot(C_time[1000:8000], C_PWM[1000:8000])
# plt.plot(C_time, C_sine)


# ----- FIT -----

C_guess = [1, 10, 1600, 1]

C_time_range = np.linspace(C_time[0], C_time[len(C_time)-1], len(C_time))

C_para, C_cov = sp.optimize.curve_fit(sine, C_time, C_sine, C_guess, maxfev=100000)

C_range = C_time[len(C_time)-1] - C_time[0]
C_peaks = (C_para[2] / (2*np.pi)) * C_range
C_ind = 10*(int(len(C_time)/C_peaks))

plt.plot(C_time[:C_ind], C_sine[:C_ind])

plt.plot(C_time_range[:C_ind], sine(C_time_range[:C_ind], C_para[0], C_para[1], C_para[2], C_para[3]), lw=3)


# ----- CHI SQUARED VALUE -----

C_sine_fit = sine(C_time_range[:C_ind], C_para[0], C_para[1], C_para[2], C_para[3])
C_residual = C_sine[:C_ind] - C_sine_fit
C_chi = (sum((C_residual**2 *len(C_sine_fit))/C_sine_fit))
print("C chi squared value = %0.6s" % C_chi)


# ----- PARAMETER UNCERTAINTIES -----

C_p1 = C_para[0]
C_e1 = np.sqrt(C_cov[0][0])
print("%0.8s ± %0.8s" % (C_p1, C_e1))
print("Percentage error %0.5s" % ((C_e1/C_p1)*100))
print(" ")

C_p2 = C_para[1]
C_e2 = np.sqrt(C_cov[1][1])
print("%0.8s ± %0.8s" % (C_p2, C_e2))
print("Percentage error %0.5s" % ((C_e2/C_p2)*100))
print(" ")

C_p3 = C_para[2]
C_e3 = np.sqrt(C_cov[2][2])
print("%0.8s ± %0.8s" % (C_p3, C_e3))
print("Percentage error %0.7s" % ((C_e3/C_p3)*100))
print(" ")

C_p4 = C_para[3]
C_e4 = np.sqrt(C_cov[3][3]) 
print("%0.8s ± %0.8s" % (C_p4, C_e4))
print("Percentage error %0.5s" % ((C_e4/C_p4)*100))
print(" ")

C_sum = ((C_e1/C_p1)*100)**2 + ((C_e2/C_p2)*100)**2 + ((C_e3/C_p3)*100)**2 + ((C_e4/C_p4)*100)**2
print("Sum of percentage errors: ", C_sum)


#%% -------- Csharp -----------


Csharp = pd.read_csv("Data/C#/WFM.CSV")

Csharp_time = Csharp["in s"]
Csharp_PWM = Csharp["C1 in V"]
Csharp_sine = Csharp["C2 in V"]

# plt.figure(1)
# plt.plot(Csharp_time[1000:8000], Csharp_PWM[1000:8000])
# plt.plot(Csharp_time, Csharp_sine)


# ----- FIT -----

Csharp_guess = [1, 10, 1700, 1]

Csharp_time_range = np.linspace(Csharp_time[0], Csharp_time[len(Csharp_time)-1], len(Csharp_time))

Csharp_para, Csharp_cov = sp.optimize.curve_fit(sine, Csharp_time, Csharp_sine, Csharp_guess, maxfev=100000)


Csharp_range = Csharp_time[len(Csharp_time)-1] - Csharp_time[0]
Csharp_peaks = (Csharp_para[2] / (2*np.pi)) * Csharp_range
Csharp_ind = 10*(int(len(Csharp_time)/Csharp_peaks))

plt.plot(Csharp_time[:Csharp_ind], Csharp_sine[:Csharp_ind])

plt.plot(Csharp_time_range[:Csharp_ind], sine(Csharp_time_range[:Csharp_ind], Csharp_para[0], Csharp_para[1], Csharp_para[2], Csharp_para[3]), lw=3)


# ----- CHI SQUARED VALUE -----

Csharp_sine_fit = sine(Csharp_time_range[:Csharp_ind], Csharp_para[0], Csharp_para[1], Csharp_para[2], Csharp_para[3])
Csharp_residual = Csharp_sine[:Csharp_ind] - Csharp_sine_fit
Csharp_chi = (sum((Csharp_residual**2 *len(Csharp_sine_fit))/Csharp_sine_fit))
print("Csharp chi squared value = %0.6s" % Csharp_chi)


# ----- PARAMETER UNCERTAINTIES -----

Csharp_p1 = Csharp_para[0]
Csharp_e1 = np.sqrt(Csharp_cov[0][0])
print("%0.8s ± %0.8s" % (Csharp_p1, Csharp_e1))
print("Percentage error %0.5s" % ((Csharp_e1/Csharp_p1)*100))
print(" ")

Csharp_p2 = Csharp_para[1]
Csharp_e2 = np.sqrt(Csharp_cov[1][1])
print("%0.8s ± %0.8s" % (Csharp_p2, Csharp_e2))
print("Percentage error %0.5s" % ((Csharp_e2/Csharp_p2)*100))
print(" ")

Csharp_p3 = Csharp_para[2]
Csharp_e3 = np.sqrt(Csharp_cov[2][2])
print("%0.8s ± %0.8s" % (Csharp_p3, Csharp_e3))
print("Percentage error %0.7s" % ((Csharp_e3/Csharp_p3)*100))
print(" ")

Csharp_p4 = Csharp_para[3]
Csharp_e4 = np.sqrt(Csharp_cov[3][3]) 
print("%0.8s ± %0.8s" % (Csharp_p4, Csharp_e4))
print("Percentage error %0.5s" % ((Csharp_e4/Csharp_p4)*100))
print(" ")

Csharp_sum = ((Csharp_e1/Csharp_p1)*100)**2 + ((Csharp_e2/Csharp_p2)*100)**2 + ((Csharp_e3/Csharp_p3)*100)**2 + ((Csharp_e4/Csharp_p4)*100)**2
print("Sum of percentage errors: ", Csharp_sum)



#%% -------- D -----------


D = pd.read_csv("Data/D/WFM.CSV")

D_time = D["in s"]
D_PWM = D["C1 in V"]
D_sine = D["C2 in V"]

# plt.figure(1)
# plt.plot(D_time[1000:8000], D_PWM[1000:8000])
# plt.plot(D_time, D_sine)


# ----- FIT -----

D_guess = [1, 10, 1800, 1]

D_time_range = np.linspace(D_time[0], D_time[len(D_time)-1], len(D_time))

D_para, D_cov = sp.optimize.curve_fit(sine, D_time, D_sine, D_guess, maxfev=100000)

D_range = D_time[len(D_time)-1] - D_time[0]
D_peaks = (D_para[2] / (2*np.pi)) * D_range
D_ind = 10*(int(len(D_time)/D_peaks))

plt.plot(D_time[:D_ind], D_sine[:D_ind])

plt.plot(D_time_range[:D_ind], sine(D_time_range[:D_ind], D_para[0], D_para[1], D_para[2], D_para[3]), lw=3)


# ----- CHI SQUARED VALUE -----

D_sine_fit = sine(D_time_range[:D_ind], D_para[0], D_para[1], D_para[2], D_para[3])
D_residual = D_sine[:D_ind] - D_sine_fit
D_chi = (sum((D_residual**2 *len(D_sine_fit))/D_sine_fit))
print("D chi squared value = %0.6s" % D_chi)


# ----- PARAMETER UNCERTAINTIES -----

D_p1 = D_para[0]
D_e1 = np.sqrt(D_cov[0][0])
print("%0.8s ± %0.8s" % (D_p1, D_e1))
print("Percentage error %0.5s" % ((D_e1/D_p1)*100))
print(" ")

D_p2 = D_para[1]
D_e2 = np.sqrt(D_cov[1][1])
print("%0.8s ± %0.8s" % (D_p2, D_e2))
print("Percentage error %0.5s" % ((D_e2/D_p2)*100))
print(" ")

D_p3 = D_para[2]
D_e3 = np.sqrt(D_cov[2][2])
print("%0.8s ± %0.8s" % (D_p3, D_e3))
print("Percentage error %0.7s" % ((D_e3/D_p3)*100))
print(" ")

D_p4 = D_para[3]
D_e4 = np.sqrt(D_cov[3][3]) 
print("%0.8s ± %0.8s" % (D_p4, D_e4))
print("Percentage error %0.5s" % ((D_e4/D_p4)*100))
print(" ")


D_sum = ((D_e1/D_p1)*100)**2 + ((D_e2/D_p2)*100)**2 + ((D_e3/D_p3)*100)**2 + ((D_e4/D_p4)*100)**2
print("Sum of percentage errors: ", D_sum)


#%% -------- Dsharp -----------

%matplotlib auto

Dsharp = pd.read_csv("Data/D#/WFM.CSV")

Dsharp_time = Dsharp["in s"]
Dsharp_PWM = Dsharp["C1 in V"]
Dsharp_sine = Dsharp["C2 in V"]

# plt.figure(1)
# plt.plot(Dsharp_time[1000:8000], Dsharp_PWM[1000:8000])
# plt.plot(Dsharp_time, Dsharp_sine)


# ----- FIT -----

Dsharp_guess = [1, 10, 1900, 1]

Dsharp_time_range = np.linspace(Dsharp_time[0], Dsharp_time[len(Dsharp_time)-1], len(Dsharp_time))

Dsharp_para, Dsharp_cov = sp.optimize.curve_fit(sine, Dsharp_time, Dsharp_sine, Dsharp_guess, maxfev=100000)

Dsharp_range = Dsharp_time[len(Dsharp_time)-1] - Dsharp_time[0]
Dsharp_peaks = (Dsharp_para[2] / (2*np.pi)) * Dsharp_range
Dsharp_ind = 10*(int(len(Dsharp_time)/Dsharp_peaks))

plt.plot(Dsharp_time[:Dsharp_ind], Dsharp_sine[:Dsharp_ind])

plt.plot(Dsharp_time_range[:Dsharp_ind], sine(Dsharp_time_range[:Dsharp_ind], Dsharp_para[0], Dsharp_para[1], Dsharp_para[2], Dsharp_para[3]), lw=3)


# ----- CHI SQUARED VALUE -----

Dsharp_sine_fit = sine(Dsharp_time_range[:Dsharp_ind], Dsharp_para[0], Dsharp_para[1], Dsharp_para[2], Dsharp_para[3])
Dsharp_residual = Dsharp_sine[:Dsharp_ind] - Dsharp_sine_fit
Dsharp_chi = (sum(((Dsharp_residual**2)*len(Dsharp_sine_fit))/Dsharp_sine_fit))
print("Dsharp chi squared value = %0.6s" % Dsharp_chi)


# ----- PARAMETER UNCERTAINTIES -----

Dsharp_p1 = Dsharp_para[0]
Dsharp_e1 = np.sqrt(Dsharp_cov[0][0])
print("%0.8s ± %0.8s" % (Dsharp_p1, Dsharp_e1))
print("Percentage error %0.5s" % ((Dsharp_e1/Dsharp_p1)*100))
print(" ")

Dsharp_p2 = Dsharp_para[1]
Dsharp_e2 = np.sqrt(Dsharp_cov[1][1])
print("%0.8s ± %0.8s" % (Dsharp_p2, Dsharp_e2))
print("Percentage error %0.5s" % ((Dsharp_e2/Dsharp_p2)*100))
print(" ")

Dsharp_p3 = Dsharp_para[2]
Dsharp_e3 = np.sqrt(Dsharp_cov[2][2])
print("%0.8s ± %0.8s" % (Dsharp_p3, Dsharp_e3))
print("Percentage error %0.7s" % ((Dsharp_e3/Dsharp_p3)*100))
print(" ")

Dsharp_p4 = Dsharp_para[3]
Dsharp_e4 = np.sqrt(Dsharp_cov[3][3]) 
print("%0.8s ± %0.8s" % (Dsharp_p4, Dsharp_e4))
print("Percentage error %0.5s" % ((Dsharp_e4/Dsharp_p4)*100))
print(" ")

Dsharp_sum = ((Dsharp_e1/Dsharp_p1)*100)**2 + ((Dsharp_e2/Dsharp_p2)*100)**2 + ((Dsharp_e3/Dsharp_p3)*100)**2 + ((Dsharp_e4/Dsharp_p4)*100)**2
print("Sum of percentage errors: ", Dsharp_sum)



#%% -------- E -----------


E = pd.read_csv("Data/E/WFM.CSV")

E_time = E["in s"]
E_PWM = E["C1 in V"]
E_sine = E["C2 in V"]

# plt.figure(1)
# plt.plot(E_time[1000:8000], E_PWM[1000:8000])
# plt.plot(E_time, E_sine)


# ----- FIT -----

E_guess = [1, 10, 2100, 1]

E_time_range = np.linspace(E_time[0], E_time[len(E_time)-1], len(E_time))

E_para, E_cov = sp.optimize.curve_fit(sine, E_time, E_sine, E_guess, maxfev=100000)

E_range = E_time[len(E_time)-1] - E_time[0]
E_peaks = (E_para[2] / (2*np.pi)) * E_range
E_ind = 10*(int(len(E_time)/E_peaks))

plt.plot(E_time[:E_ind], E_sine[:E_ind])

plt.plot(E_time_range[:E_ind], sine(E_time_range[:E_ind], E_para[0], E_para[1], E_para[2], E_para[3]), lw=3)


# ----- CHI SQUARED VALUE -----

E_sine_fit = sine(E_time_range[:E_ind], E_para[0], E_para[1], E_para[2], E_para[3])
E_residual = E_sine[:E_ind] - E_sine_fit
E_chi = (sum((E_residual**2*len(E_sine_fit))/E_sine_fit))
print("E chi squared value = %0.6s" % E_chi)


# ----- PARAMETER UNCERTAINTIES -----

E_p1 = E_para[0]
E_e1 = np.sqrt(E_cov[0][0])
print("%0.8s ± %0.8s" % (E_p1, E_e1))
print("Percentage error %0.5s" % ((E_e1/E_p1)*100))
print(" ")

E_p2 = E_para[1]
E_e2 = np.sqrt(E_cov[1][1])
print("%0.8s ± %0.8s" % (E_p2, E_e2))
print("Percentage error %0.5s" % ((E_e2/E_p2)*100))
print(" ")

E_p3 = E_para[2]
E_e3 = np.sqrt(E_cov[2][2])
print("%0.8s ± %0.8s" % (E_p3, E_e3))
print("Percentage error %0.7s" % ((E_e3/E_p3)*100))
print(" ")

E_p4 = E_para[3]
E_e4 = np.sqrt(E_cov[3][3]) 
print("%0.8s ± %0.8s" % (E_p4, E_e4))
print("Percentage error %0.5s" % ((E_e4/E_p4)*100))
print(" ")

E_sum = ((E_e1/E_p1)*100)**2 + ((E_e2/E_p2)*100)**2 + ((E_e3/E_p3)*100)**2 + ((E_e4/E_p4)*100)**2
print("Sum of percentage errors: ", E_sum)


#%% -------- F -----------

%matplotlib inline

F = pd.read_csv("Data/F/WFM.CSV")

F_time = F["in s"]
F_PWM = F["C1 in V"]
F_sine = F["C2 in V"]

# plt.figure(1)
# plt.plot(F_time[1000:8000], F_PWM[1000:8000])
# plt.plot(F_time, F_sine)


# ----- FIT -----

F_guess = [1, 10, 2200, 1]

F_time_range = np.linspace(F_time[0], F_time[len(F_time)-1], len(F_time))

F_para, F_cov = sp.optimize.curve_fit(sine, F_time, F_sine, F_guess, maxfev=100000)

F_range = F_time[len(F_time)-1] - F_time[0]
F_peaks = (F_para[2] / (2*np.pi)) * F_range
F_ind = 10*(int(len(F_time)/F_peaks))

plt.plot(F_time[:F_ind], F_sine[:F_ind])

plt.plot(F_time_range[:F_ind], sine(F_time_range[:F_ind], F_para[0], F_para[1], F_para[2], F_para[3]), lw=3)


# ----- CHI SQUARED VALUE -----

F_sine_fit = sine(F_time_range[:F_ind], F_para[0], F_para[1], F_para[2], F_para[3])
F_residual = F_sine[:F_ind] - F_sine_fit
F_chi = (sum((F_residual**2*len(F_sine_fit))/F_sine_fit))
print("F chi squared value = %0.6s" % F_chi)


# ----- PARAMETER UNCERTAINTIES -----

F_p1 = F_para[0]
F_e1 = np.sqrt(F_cov[0][0])
print("%0.8s ± %0.8s" % (F_p1, F_e1))
print("Percentage error %0.5s" % ((F_e1/F_p1)*100))
print(" ")

F_p2 = F_para[1]
F_e2 = np.sqrt(F_cov[1][1])
print("%0.8s ± %0.8s" % (F_p2, F_e2))
print("Percentage error %0.5s" % ((F_e2/F_p2)*100))
print(" ")

F_p3 = F_para[2]
F_e3 = np.sqrt(F_cov[2][2])
print("%0.8s ± %0.8s" % (F_p3, F_e3))
print("Percentage error %0.7s" % ((F_e3/F_p3)*100))
print(" ")

F_p4 = F_para[3]
F_e4 = np.sqrt(F_cov[3][3]) 
print("%0.8s ± %0.8s" % (F_p4, F_e4))
print("Percentage error %0.5s" % ((F_e4/F_p4)*100))
print(" ")

F_sum = ((F_e1/F_p1)*100)**2 + ((F_e2/F_p2)*100)**2 + ((F_e3/F_p3)*100)**2 + ((F_e4/F_p4)*100)**2
print("Sum of percentage errors: ", F_sum)



#%% -------- Fsharp -----------

%matplotlib inline

Fsharp = pd.read_csv("Data/F#/WFM.CSV")

Fsharp_time = Fsharp["in s"]
Fsharp_PWM = Fsharp["C1 in V"]
Fsharp_sine = Fsharp["C2 in V"]

# plt.figure(1)
# plt.plot(Fsharp_time[1000:8000], Fsharp_PWM[1000:8000])
# plt.plot(Fsharp_time, Fsharp_sine)


# ----- FIT -----

Fsharp_guess = [1, 10, 2300, 1]

Fsharp_time_range = np.linspace(Fsharp_time[0], Fsharp_time[len(Fsharp_time)-1], len(Fsharp_time))

Fsharp_para, Fsharp_cov = sp.optimize.curve_fit(sine, Fsharp_time, Fsharp_sine, Fsharp_guess, maxfev=100000)

Fsharp_range = Fsharp_time[len(Fsharp_time)-1] - Fsharp_time[0]
Fsharp_peaks = (Fsharp_para[2] / (2*np.pi)) * Fsharp_range
Fsharp_ind = 10*(int(len(Fsharp_time)/Fsharp_peaks))

plt.plot(Fsharp_time[:Fsharp_ind], Fsharp_sine[:Fsharp_ind])

plt.plot(Fsharp_time_range[:Fsharp_ind], sine(Fsharp_time_range[:Fsharp_ind], Fsharp_para[0], Fsharp_para[1], Fsharp_para[2], Fsharp_para[3]), lw=3)


# ----- CHI SQUARED VALUE -----

Fsharp_sine_fit = sine(Fsharp_time_range[:Fsharp_ind], Fsharp_para[0], Fsharp_para[1], Fsharp_para[2], Fsharp_para[3])
Fsharp_residual = Fsharp_sine[:Fsharp_ind] - Fsharp_sine_fit
Fsharp_chi = (sum((Fsharp_residual**2*len(Fsharp_sine_fit))/Fsharp_sine_fit))
print("Fsharp chi squared value = %0.6s" % Fsharp_chi)


# ----- PARAMETER UNCERTAINTIES -----

Fsharp_p1 = Fsharp_para[0]
Fsharp_e1 = np.sqrt(Fsharp_cov[0][0])
print("%0.8s ± %0.8s" % (Fsharp_p1, Fsharp_e1))
print("Percentage error %0.5s" % ((Fsharp_e1/Fsharp_p1)*100))
print(" ")

Fsharp_p2 = Fsharp_para[1]
Fsharp_e2 = np.sqrt(Fsharp_cov[1][1])
print("%0.8s ± %0.8s" % (Fsharp_p2, Fsharp_e2))
print("Percentage error %0.5s" % ((Fsharp_e2/Fsharp_p2)*100))
print(" ")

Fsharp_p3 = Fsharp_para[2]
Fsharp_e3 = np.sqrt(Fsharp_cov[2][2])
print("%0.8s ± %0.8s" % (Fsharp_p3, Fsharp_e3))
print("Percentage error %0.7s" % ((Fsharp_e3/Fsharp_p3)*100))
print(" ")

Fsharp_p4 = Fsharp_para[3]
Fsharp_e4 = np.sqrt(Fsharp_cov[3][3]) 
print("%0.8s ± %0.8s" % (Fsharp_p4, Fsharp_e4))
print("Percentage error %0.5s" % ((Fsharp_e4/Fsharp_p4)*100))
print(" ")

Fsharp_sum = ((Fsharp_e1/Fsharp_p1)*100)**2 + ((Fsharp_e2/Fsharp_p2)*100)**2 + ((Fsharp_e3/Fsharp_p3)*100)**2 + ((Fsharp_e4/Fsharp_p4)*100)**2
print("Sum of percentage errors: ", Fsharp_sum)



#%% -------- G -----------

%matplotlib inline

G = pd.read_csv("Data/G/WFM.CSV")

G_time = G["in s"]
G_PWM = G["C1 in V"]
G_sine = G["C2 in V"]

# plt.figure(1)
# plt.plot(G_time[1000:8000], G_PWM[1000:8000])
# plt.plot(G_time, G_sine)


# ----- FIT -----

G_guess = [1, 10, 2500, 1]

G_time_range = np.linspace(G_time[0], G_time[len(G_time)-1], len(G_time))

G_para, G_cov = sp.optimize.curve_fit(sine, G_time, G_sine, G_guess, maxfev=100000)

G_range = G_time[len(G_time)-1] - G_time[0]
G_peaks = (G_para[2] / (2*np.pi)) * G_range
G_ind = 10*(int(len(G_time)/G_peaks))

plt.plot(G_time[:G_ind], G_sine[:G_ind])

plt.plot(G_time_range[:G_ind], sine(G_time_range[:G_ind], G_para[0], G_para[1], G_para[2], G_para[3]), lw=3)


# ----- CHI SQUARED VALUE -----

G_sine_fit = sine(G_time_range[:G_ind], G_para[0], G_para[1], G_para[2], G_para[3])
G_residual = G_sine[:G_ind] - G_sine_fit
G_chi = (sum((G_residual**2)/(G_sine_fit))) / len(G_sine_fit)
print(len(G_sine_fit))
print("G chi squared value = %0.6s" % G_chi)


# ----- PARAMETER UNCERTAINTIES -----

G_p1 = G_para[0]
G_e1 = np.sqrt(G_cov[0][0])
print("%0.8s ± %0.8s" % (G_p1, G_e1))
print("Percentage error %0.5s" % ((G_e1/G_p1)*100))
print(" ")

G_p2 = G_para[1]
G_e2 = np.sqrt(G_cov[1][1])
print("%0.8s ± %0.8s" % (G_p2, G_e2))
print("Percentage error %0.5s" % ((G_e2/G_p2)*100))
print(" ")

G_p3 = G_para[2]
G_e3 = np.sqrt(G_cov[2][2])
print("%0.8s ± %0.8s" % (G_p3, G_e3))
print("Percentage error %0.7s" % ((G_e3/G_p3)*100))
print(" ")

G_p4 = G_para[3]
G_e4 = np.sqrt(G_cov[3][3]) 
print("%0.8s ± %0.8s" % (G_p4, G_e4))
print("Percentage error %0.5s" % ((G_e4/G_p4)*100))
print(" ")

G_sum = ((G_e1/G_p1)*100)**2 + ((G_e2/G_p2)*100)**2 + ((G_e3/G_p3)*100)**2 + ((G_e4/G_p4)*100)**2
print("Sum of percentage errors: ", G_sum)


#%% -------- Gsharp -----------

%matplotlib auto

Gsharp = pd.read_csv("Data/G#/WFM.CSV")

Gsharp_time = Gsharp["in s"]
Gsharp_PWM = Gsharp["C1 in V"]
Gsharp_sine = Gsharp["C2 in V"]

# plt.figure(1)
# plt.plot(Gsharp_time[1000:8000], Gsharp_PWM[1000:8000])
# plt.plot(Gsharp_time, Gsharp_sine)


# ----- FIT -----

Gsharp_guess = [1, 10, 2600, 1]

Gsharp_time_range = np.linspace(Gsharp_time[0], Gsharp_time[len(Gsharp_time)-1], len(Gsharp_time))

Gsharp_para, Gsharp_cov = sp.optimize.curve_fit(sine, Gsharp_time, Gsharp_sine, Gsharp_guess, maxfev=100000)

Gsharp_range = Gsharp_time[len(Gsharp_time)-1] - Gsharp_time[0]
Gsharp_peaks = (Gsharp_para[2] / (2*np.pi)) * Gsharp_range
Gsharp_ind = 5*(int(len(Gsharp_time)/Gsharp_peaks))

plt.figure("Gsharp")
plt.plot(Gsharp_time[:Gsharp_ind], Gsharp_sine[:Gsharp_ind], color='#0e3b84', lw=3)

plt.plot(Gsharp_time_range[:Gsharp_ind], sine(Gsharp_time_range[:Gsharp_ind], Gsharp_para[0], Gsharp_para[1], Gsharp_para[2], Gsharp_para[3]), lw=3.5, color='#87c707')

plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")


# ----- CHI SQUARED VALUE -----

Gsharp_sine_fit = sine(Gsharp_time_range[:Gsharp_ind], Gsharp_para[0], Gsharp_para[1], Gsharp_para[2], Gsharp_para[3])
Gsharp_residual = Gsharp_sine[:Gsharp_ind] - Gsharp_sine_fit
Gsharp_chi = (sum((Gsharp_residual**2))/(Gsharp_sine_fit)) / len(Gsharp_sine_fit)
print("Gsharp chi squared value = %0.8s" % Gsharp_chi)


# ----- PARAMETER UNCERTAINTIES -----

Gsharp_p1 = Gsharp_para[0]
Gsharp_e1 = np.sqrt(Gsharp_cov[0][0])
print("%0.8s ± %0.8s" % (Gsharp_p1, Gsharp_e1))
print("Percentage error %0.5s" % ((Gsharp_e1/Gsharp_p1)*100))
print(" ")

Gsharp_p2 = Gsharp_para[1]
Gsharp_e2 = np.sqrt(Gsharp_cov[1][1])
print("%0.8s ± %0.8s" % (Gsharp_p2, Gsharp_e2))
print("Percentage error %0.5s" % ((Gsharp_e2/Gsharp_p2)*100))
print(" ")

Gsharp_p3 = Gsharp_para[2]
Gsharp_e3 = np.sqrt(Gsharp_cov[2][2])
print("%0.8s ± %0.8s" % (Gsharp_p3, Gsharp_e3))
print("Percentage error %0.7s" % ((Gsharp_e3/Gsharp_p3)*100))
print(" ")

Gsharp_p4 = Gsharp_para[3]
Gsharp_e4 = np.sqrt(Gsharp_cov[3][3]) 
print("%0.8s ± %0.8s" % (Gsharp_p4, Gsharp_e4))
print("Percentage error %0.5s" % ((Gsharp_e4/Gsharp_p4)*100))
print(" ")

Gsharp_sum = ((Gsharp_e1/Gsharp_p1)*100)**2 + ((Gsharp_e2/Gsharp_p2)*100)**2 + ((Gsharp_e3/Gsharp_p3)*100)**2 + ((Gsharp_e4/Gsharp_p4)*100)**2
print("Sum of percentage errors: ", Gsharp_sum)











#%% ---------- FFT -----------

%matplotlib inline

A_ave = np.mean(A_sine)
A_sine_norm = A_sine - A_ave

h = A_time[3] - A_time[2]
sample_rate = 1/h
duration = A_time[:-1] - A_time[0]
N = sample_rate * duration

A_time
A_sine

fourier_A_sine = np.fft.fft(A_sine)

fourier_N = len(fourier_A_sine)
fourier_n = np.arange(fourier_N)
T = fourier_N / sample_rate

freq = fourier_n/T

plt.stem(freq, fourier_A_time, 'b', markerfmt=" ", basefmt="-b")

plt.xlim(-1000, 10000)

























#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:08:51 2022

@author: laura
"""

# Git test

import numpy as np
import matplotlib.pyplot as plt



#%%

%matplotlib inline

# angle1 = np.arange(0, 2*np.pi, 2.8125*(np.pi/180))
# angle2 = np.arange(0, 2*np.pi, (np.pi/14))

# print(len(angle))

# dc = (np.sin(angle2) + 1)/2


# plt.plot(angle2, dc*100)

# plt.xlabel('Angle [rad]')
# plt.ylabel('Duty Cycle [%]')
# plt.grid()

# print(list(duty*100))



def Duty_Period(n_samples, sine_freq, clock_freq, prescaler):
    
    clock_period = 1 / clock_freq
    sine_period = 1 / sine_freq
    
    print(sine_period)
    
    period = (sine_period / n_samples)          # PWM period fixed by sine freq and sample number
    print(period)                               # Amount of time at each duty cycle

    t = np.arange(0, sine_period, period)       # Time for each sine period
    
    PR2 = (sine_period / (4 * n_samples * clock_period * prescaler)) - 1        # demimal PR2
    
    # plt.plot(t, (sine_period * (np.sin((2 * np.pi * t)/sine_period))), 'x')
    
    CCPCON_b = []
    CCPCON_d = []
    width_vals = []
    
    for i in range(len(t)):
        
        width = (sine_period * (np.sin((2 * np.pi * t[i])/sine_period) + 1)) / (2 * n_samples)
        width_vals.append(width)
        
        CCPCON = width / (clock_period * prescaler)
        CCPCON_d.append(CCPCON)                                         # Decimal CCPCON
        CCPCON_b.append(format(round(CCPCON), 'b'))                     # Binary CCPCON
    
    plt.plot(t, CCPCON_d, 'x-')



    
    return (format(round(PR2), 'b')), CCPCON_b, round(PR2), CCPCON_d , t   # Return binary PR2, CCPCON
                                                                        # and decimal PR2, CCPCON
    



PR2_b, CCPCON_b, PR2_d, CCPCON_d, t = Duty_Period(128, 523.25, 16e6, 1)





# print('PR2: ', PR2_b)
# print(CCPCON)


new_CCPCON = []

for i in range(len(CCPCON_b)):
    # new = CCPCON[i].strip('')
    # new_CCPCON.append(new)
    
    length = len(CCPCON_b[i])
    
    if length < 3:
        new = '0'
    else:
        new = CCPCON_b[i][:length-2]
        new += '00'
    
    back = int(new, 2)
    new_CCPCON.append(back)
    
    # print('MOVLW   ' ,new)
    # print('MOVWF    duty_cycle_upper')	
    # print('')
    # print('CALL     SIGNAL')
    # print('CALL     DELAY')
    # print('')
    
print(new_CCPCON)

plt.plot(t, new_CCPCON)

# print(PR2_d)
# print(CCPCON_d)


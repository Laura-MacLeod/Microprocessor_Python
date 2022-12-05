#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:08:51 2022

@author: laura
"""

# Git test




#%%

angle1 = np.arange(0, 2*np.pi, 2.8125*(np.pi/180))
angle2 = np.arange(0, 2*np.pi, (np.pi/14))

# print(len(angle))

duty = (np.sin(angle2) + 1)/2


plt.plot(angle2, duty*100)

plt.xlabel('Angle [rad]')
plt.ylabel('Duty Cycle [%]')
plt.grid()

print(list(duty*100))







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 02:47:30 2020

@author: dichoski
"""

import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 8
valence = (54.375, 61.876, 66.25, 62.5, 66.875, 66.875, 55.625, 66.25)
arousal = (48.125, 50.625, 60.0, 47.5, 54.275, 55.625, 50.0, 53.75)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, valence, bar_width,
alpha=opacity,
color='b',
label='Валентност')

rects2 = plt.bar(index + bar_width, arousal, bar_width,
alpha=opacity,
color='g',
label='Возбуденост')

plt.xlabel('Модел')
plt.ylabel('Точност')
plt.title('Точност по модел')
plt.xticks(index + bar_width/2, ('1', '2', '3', '4', '5', '6', '7', '8'))
plt.legend()
axes = plt.gca()
axes.set_ylim([40,70])
plt.tight_layout()
plt.show()


#%%
aro=([0.50625, 0.525, 0.6, 0.48125])
val=[0.56875, 0.59375, 0.6625, 0.6]
aro = [x * 100 for x in aro]
val = [x * 100 for x in val]

# create plot
n_groups = 4
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, val, bar_width,
alpha=opacity,
color='b',
label='Валентност')

rects2 = plt.bar(index + bar_width, aro, bar_width,
alpha=opacity,
color='g',
label='Возбуденост')

plt.xlabel('Прозорец')
plt.ylabel('Точност')
plt.title('Точност според должина на прозорец')
plt.xticks(index + bar_width/2, ('1/1', '2/1', '4/2', '6/3'))
plt.legend()

plt.tight_layout()

axes = plt.gca()
# axes.set_xlim([xmin,xmax])
axes.set_ylim([45,70])

plt.show()


#%%
#%%
aro=([0.6, 0.575])
val=[0.6625, 0.575]
aro = [x * 100 for x in aro]
val = [x * 100 for x in val]

# create plot
n_groups = 2
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, val, bar_width,
alpha=opacity,
color='b',
label='Валентност')

rects2 = plt.bar(index + bar_width, aro, bar_width,
alpha=opacity,
color='g',
label='Возбуденост')

plt.xlabel('Број на електроди')
plt.ylabel('Точност')
plt.title('Точност според број на електроди')
plt.xticks(index + bar_width/2, ('10', '32'))
plt.legend()

plt.tight_layout()

axes = plt.gca()
axes.set_ylim([55,68])

plt.show()

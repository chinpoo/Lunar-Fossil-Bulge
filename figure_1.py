# figure 1: temperature and viscosity vs depth for different time

from my_func import heat_conduction,viscosity_temp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import pickle
from math import pi
import matplotlib
#import matplotlib.rcParams.update as font_set

year_to_sec = 365*24*3600
eta0 = 1e22
max_time = eta0/6.5e10/year_to_sec
R0 = 1.74e6
chi = 1.0e-6                           # thermal diffusivity
Ts = 250                               # Surface temperature in K
dT = 1250                              # initial temperature difference between surface and mantle
E = 3.0e5                              # activation energy, J/mol     
epsilon = Ts/dT
gamma = chi*max_time*year_to_sec/R0**2 
r = np.linspace(0.1954,1.0,1000)

out_fig = "figure_1e30_1.eps"
matplotlib.rcParams.update({'font.family': 'sans-serif'})
rc("xtick",labelsize=13)
rc("ytick",labelsize=13)
fig,ax = plt.subplots(1,2,figsize=(10,6),dpi=400)

time = [0.0,1e7,1e8,3e8,1e9,4.3e9]
color = ['k','b','g','orange','r','m']
lns = []
for i,t in enumerate(time):
    T = heat_conduction(r,t/max_time,gamma)
    if t == 0.0:
        T = np.ones(len(r))
    l, = ax[0].plot(Ts+dT*T,r*R0/1000,color=color[i],linewidth=1.5)
    visc = viscosity_temp(T,eta0,E,dT,epsilon)
    ax[1].plot(np.log10(visc),r*R0/1000,color=color[i],linewidth=1.5)
    lns.append(l)

ax[0].legend(lns,["0 Myr","10 Myr","100 Myr","300 Myr","1000 Myr","4300 Myr"],loc=3,fontsize=14)
ax[0].set_xlabel("Temperature (K)",fontsize=14)
ax[0].set_ylabel("Radius (km)",fontsize=14)
ax[0].set_xlim([200,1550])
ax[0].set_ylim([340,1740])
ax[0].set_xticks(np.linspace(250,1500,6))
ax[0].set_yticks(np.linspace(340,1740,8))
ax[1].set_xlabel(r"$log_{10}(\eta)$",fontsize=16)
ax[1].set_ylabel("Radius (km)",fontsize=14)
ax[1].set_xlim([21.5,30.5])
ax[1].set_ylim([340,1740])
ax[1].set_xticks(np.linspace(22,30,9))
ax[1].set_yticks(np.linspace(340,1740,8))

fig.tight_layout()
fig.savefig(out_fig,dpi=400)







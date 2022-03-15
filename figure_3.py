# figure 3: litho thickening, Te vs time for different activation energy

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
epsilon = Ts/dT
gamma = chi*max_time*year_to_sec/R0**2 

matplotlib.rcParams.update({'font.family': 'sans-serif'})
rc("xtick",labelsize=13)
rc("ytick",labelsize=13)
out_fig = "figure_3.eps"
time = np.linspace(0,4.3e9,501)
r = np.linspace(0.1954,1.0,501)
E = [1e5,2e5,3e5,4e5]  # different activation energy
color = ['g','r','k--','b']
labels = ["E=100 kJ/mol","E=200 kJ/mol","E=300 kJ/mol","E=400 kJ/mol"]
Te = np.zeros((len(E),len(time)))
cutoff = 1e26
for j,t in enumerate(time):
	T = heat_conduction(r,t/max_time,gamma)
	if t == 0.0:
		T = np.ones(len(r))
	for i,Ei in enumerate(E):
		visc = viscosity_temp(T,eta0,Ei,dT,epsilon)
		ind = visc < cutoff
		te = R0 - r[ind][-1]*R0
		Te[i,j] = te

fig,ax = plt.subplots(figsize=(8,5),dpi=400)
for i,c in enumerate(color):
	ax.plot(time/1e6,Te[i,:]/1000,c,linewidth=1.5,label=labels[i])
ax.set_xlabel("Time (Myr)",fontsize=14)
ax.set_ylabel("Te (km)",fontsize=14)
ax.set_xlim([0,4300])
ax.set_ylim([0,800])
ax.set_xticks(np.linspace(0,4000,9))
ax.set_yticks(np.linspace(0,800,5))
ax.set_title("Elastic Thickness vs. Time",fontsize=14)
ax.legend(loc="upper left",fontsize=13)

fig.savefig(out_fig,dpi=400)

# 2017-01-24: high-viscosity layer thickness vs. time
#r = np.linspace(0.1954,1.0,1001)
#t_yr = 4e9
#tsq = np.linspace(0,np.sqrt(t_yr),501)
#t = tsq**2/max_time
#visc_cutoff_1 = 1e26
#visc_cutoff_2 = 1e28
#visc = np.zeros(len(t))
#D1 = []  # thickness
#D2 = []  # thickness
#for it in t:
#    T = heat_conduction(r,it,gamma)
#    V = viscosity_temp(T,eta0,E,dT,epsilon)
#    d1 = (1.0 - r[V<visc_cutoff_1][-1])*R0
#    d2 = (1.0 - r[V<visc_cutoff_2][-1])*R0
#    D1.append(d1)
#    D2.append(d2)
#D1 = np.array(D1)
#D2 = np.array(D2)

#fig_6,ax = plt.subplots(1,1,figsize=(8,5),dpi=200)
#ax.plot(t*max_time/1e9,D1/1e3,'blue',Linewidth=2.0,label="cutoff 1e26")
#ax.plot(t*max_time/1e9,D2/1e3,'red',Linewidth=2.0,label="cutoff 1e28")
#ax.legend(loc=0,fontsize=10)
#ax.set_xlim([0,4.0])
#ax.set_xlabel("Time (Gyr)")
#ax.set_ylabel("Thickness (km)")
#fig_5.tight_layout()
#fig_5.savefig("thickness_vs_time.eps",dpi=200)
#plt.show()




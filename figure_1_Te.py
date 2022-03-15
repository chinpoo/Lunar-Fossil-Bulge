# figure 1: temperature and viscosity vs depth for different time

from my_func import viscosity_Te_growth_1,viscosity_Te_growth_smooth
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import pickle
from math import pi
import matplotlib
#import matplotlib.rcParams.update as font_set

R0 = 1.74e6
eta0 = 1e22
tp = 4.4e9
r = np.linspace(0.1954,1.0,2001)

out_fig = "figure_Te_1.png"
matplotlib.rcParams.update({'font.family': 'sans-serif'})
rc("xtick",labelsize=12)
rc("ytick",labelsize=12)
fig,ax = plt.subplots(1,2,figsize=(10,6),dpi=200)

time = [0.0,5e7,2e8,5e8,2e9,4.4e9]
color = ['k','b','g','r','c','m']
lns1 = []
lns2 = []
for i,t in enumerate(time):
    eta1,dd1 = viscosity_Te_growth_1(r,t/1e6,eta0,R0,tp)
    eta2,dd2 = viscosity_Te_growth_smooth(r,t/1e6,eta0,R0,tp,1)
    eta3,dd3 = viscosity_Te_growth_smooth(r,t/1e6,eta0,R0,tp,2)
    l1, = ax[0].plot(np.log10(eta1),r[1:]*R0/1000,color[i],linewidth=1.5)
    l2, = ax[1].plot(np.log10(eta2),r[1:]*R0/1000,color[i],linewidth=1.5)
    l3, = ax[1].plot(np.log10(eta3),r[1:]*R0/1000,color[i]+'--',linewidth=1.5)
    lns1.append(l1)
    lns2.append(l2)
    print(dd1)
    print(dd2)
    print(dd3)

ax[0].legend(lns1,["0 Myr","50 Myr","200 Myr","500 Myr","2000 Myr","4400 Myr"],loc="lower right",fontsize=12)
ax[0].set_title("Gradation Te model")
ax[0].set_xlabel(r"$log_{10}(\eta)$",fontsize=12)
ax[0].set_ylabel("Radius (km)",fontsize=12)
ax[0].set_xlim([21.5,30.5])
ax[0].set_ylim([340,1740])
ax[0].set_xticks(np.linspace(22,30,9))
ax[0].set_yticks(np.linspace(340,1740,8))
ax[1].legend(lns2,["0 Myr","50 Myr","200 Myr","500 Myr","2000 Myr","4400 Myr"],loc="lower right",fontsize=12)
ax[1].set_title("Gradual Te model")
ax[1].set_xlabel(r"$log_{10}(\eta)$",fontsize=12)
ax[1].set_ylabel("Radius (km)",fontsize=12)
ax[1].set_xlim([21.5,30.5])
ax[1].set_ylim([340,1740])
ax[1].set_xticks(np.linspace(22,30,9))
ax[1].set_yticks(np.linspace(340,1740,8))

fig.tight_layout()
fig.savefig(out_fig,dpi=200)







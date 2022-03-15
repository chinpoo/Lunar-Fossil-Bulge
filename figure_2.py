# figure 2: orbital evolution, and bulge size vs time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import pickle
from math import pi
import matplotlib
from my_func import rot_potential_constrained_1,determine_Q
#import matplotlib.rcParams.update as font_set

out_fig = "figure_2.eps"
matplotlib.rcParams.update({'font.family': 'sans-serif'})
rc("xtick",labelsize=13)
rc("ytick",labelsize=13)
fig,ax = plt.subplots(3,1,figsize=(9,15),dpi=400)

pre = "./data/lunar_bulge_Q620_"
case_f = "fluid_all.p"
case0 = ["E300_1.p","E300_2.p"]
case1 = ["E100_1.p","E100_2.p"]
case2 = ["E200_1.p","E200_2.p"]
case3 = ["E400_1.p","E400_2.p"]

color = ['b','r','g']

sol_f = pickle.load(open(pre+case_f,'rb'))
sol_E300 = pickle.load(open(pre+case0[0],'rb'))+pickle.load(open(pre+case0[1],'rb'))
sol_E100 = pickle.load(open(pre+case1[0],'rb'))+pickle.load(open(pre+case1[1],'rb'))
sol_E200 = pickle.load(open(pre+case2[0],'rb'))+pickle.load(open(pre+case2[1],'rb'))
sol_E400 = pickle.load(open(pre+case3[0],'rb'))+pickle.load(open(pre+case3[1],'rb'))

# determine fluid love number k
kf = 0.0
for s in sol_f:
	kf += s["k_end"]
kf /= len(sol_f)
print(kf)

# for E=300kJ/mol, determine 3 cases for figure
for s in sol_E300:
	print("a0={0},k={1}".format(s["a0"],s["k_end"]/kf))
a0_E300 = [10.0,15.0,20.0,5.0,35.0]


# panel (a) ============= from orbit_model_new.py
# model constants
M = 5.972e24
m = 7.348e22
R = 6.371e6
G = 6.67e-11
B = 3*m/M*R**5*(G*(M+m))**0.5
#k2 = 0.97
k2 = 0.3
t = 4.3e9
yr_to_sec = 365*24*3600
t_sec = t*yr_to_sec
C = 13/2*k2*B/R**(13/2)*yr_to_sec
ap = 60.0                      # current Earth-Moon mean distance in Earth radii

# Geological constraints from Williams (2000)
t0_w = np.array([6.2e8,2.45e9])
t_w = t - t0_w
ap_w = ap*np.array([0.965,0.906])
err_w = ap*np.array([0.005,0.029])

dt = 5e6
N = int(t/dt)+1
#N2 = int((t0_w[1]-t0_w[0])/dt)+1
#N3 = int(t0_w[0]/dt)+1

# determine Q for different stages of orbital evolution
Q620 = determine_Q(ap_w[0],ap,t0_w[0],C)
#Q2450 = determine_Q(ap_w[1],ap_w[0],(t0_w[1]-t0_w[0]),C)
Q_model = []
#Q_model_2 = []
for ai in a0_E300:
	q = determine_Q(ai,ap_w[0],t_w[0],C)
	Q_model.append(q)
	#q2 = determine_Q(ai,ap,t,C)
	#Q_model_2.append(q2)
paras = [t_w,ap_w,(C/Q620,-1)]

at_fill = [0]*2
for i,ai in enumerate(a0_E300):
	qi = Q_model[i]
	#qi_2 = Q_model_2[i]
	tt = np.zeros(N)
	at = np.zeros(N)
	#at_2 = np.zeros(N)
	at[0] = ai        # a at t=0
	#at_2[0] = ai
	for j in range(1,N):
		tt[j] = j*dt
		at[j] = rot_potential_constrained_1(tt[j],ai,ap,2,C/qi,paras)
		#at_2[j] = (ai**(13/2)+C/qi_2*tt[j])**(2/13)
	at[1:] = ap/(at[1:]**(1/3))
	tt1 = tt[tt < t_w[0]]
	at1 = at[tt < t_w[0]]
	#ax[0].plot(tt/1e9,at_2,'b--',linewidth=1.0)
	if i < 3:
		ax[0].plot(tt1/1e6,at1,color[i],linewidth=1.5,label="Q={0:.2f}".format(Q_model[i]))
	else:
		at_fill[i-3] = at1

tt2 = tt[tt > t_w[0]]
at2 = at[tt > t_w[0]]
ax[0].plot(tt2/1e6,at2,'k',linewidth=1.5,label="Q={0:.2f}".format(Q620))
#ax[0].fill_between(tt1/1e6,at_fill[0],at_fill[1],color='grey',alpha=0.5)
# observation at 620Ma
ax[0].errorbar(t_w[0]/1e6,ap_w[0],yerr=err_w[0],fmt='o',color='k')
ax[0].set_title("Earth-Moon distance vs time")
ax[0].set_xlabel('Time (Myr)')
ax[0].set_ylabel('Earth-Moon distance (Re)')
ax[0].legend(loc="lower right",fontsize=12)
ax[0].set_xlim([0,4300])
ax[0].set_ylim([0,65])

# panel (b) =======================================
k_fill = [0]*2
for s in sol_f:
	for i,ai in enumerate(a0_E300):
		if s["a0"] == ai:
			if i < 3:
				ax[1].plot(s["time_yr"]/1e6,s["k"]/kf,color[i],linewidth=1.5,label="a0={0:.1f}".format(s["a0"]))
				print(np.max(s["k"])/kf)
			else:
				k_fill[i-3]=s["k"]/kf


#ax[1].fill_between(s["time_yr"]/1e6,k_fill[0],k_fill[1],color='grey',alpha=0.5)
ax[1].fill_between([3500,4300],[15,15],[20,20],color='grey',alpha=0.5)

ax[1].set_title("Bulge size vs time (fluid case)")
ax[1].set_xlabel('Time (Myr)')
ax[1].set_ylabel('Normalized degree-2 shape (k/k0)')
ax[1].legend(loc="upper right",fontsize=12)
ax[1].set_xlim([0,4300])
ax[1].set_ylim([0,80])


# panel (c) =======================================
k_fill = [0]*2
for s in sol_E300:
	for i,ai in enumerate(a0_E300):
		if s["a0"] == ai:
			if i < 3:
				ax[2].plot(s["time_yr"]/1e6,s["k"]/kf,color[i],linewidth=1.5,label="a0={0:.1f}".format(s["a0"]))
				print(np.max(s["k"])/kf)
			else:
				k_fill[i-3]=s["k"]/kf

ax[2].fill_between([3500,4300],[15,15],[20,20],color='grey',alpha=0.5)
#ax[2].fill_between(s["time_yr"]/1e6,k_fill[0],s["time_yr"]/1e6,20,color='grey',alpha=0.5)

ax[2].set_title("Bulge size vs time")
ax[2].set_xlabel('Time (Myr)')
ax[2].set_ylabel('Normalized degree-2 shape (k/k0)')
ax[2].legend(loc="upper right",fontsize=12)
ax[2].set_xlim([0,4300])
ax[2].set_ylim([0,80])




fig.tight_layout()
fig.savefig(out_fig,dpi=400)


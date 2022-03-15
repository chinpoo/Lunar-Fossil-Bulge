import numpy as np
import matplotlib.pyplot as plt

outfile = "orbit_model.eps"

fig,ax = plt.subplots(2,1,figsize=(9,12),dpi=400)

def determine_Q(a0,a1,dt,const):
	Q = const*dt/(a1**(13/2)-a0**(13/2))
	return Q

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
C = 13/2*k2*B/R**(13/2)
ap = 60.0                      # current Earth-Moon mean distance in Earth radii
a0 = np.linspace(5,35,13)

print(C*yr_to_sec)
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
Q620 = determine_Q(ap_w[0],ap,t0_w[0]*yr_to_sec,C)
Q2450 = determine_Q(ap_w[1],ap_w[0],(t0_w[1]-t0_w[0])*yr_to_sec,C)
Q_model = []
Q_model_2 = []
for ai in a0:
	q = determine_Q(ai,ap_w[1],t_w[1]*yr_to_sec,C)
	Q_model.append(q)
	q2 = determine_Q(ai,ap,t*yr_to_sec,C)
	Q_model_2.append(q2)


print(a0);print(Q620);print(Q2450);print(Q_model);print(Q_model_2)

for i,ai in enumerate(a0):
	qi = Q_model[i]
	qi_2 = Q_model_2[i]
	tt = np.zeros(N)
	at = np.zeros(N)
	at_2 = np.zeros(N)
	a_pre = ai
	at[0] = ai        # a at t=0
	at_2[0] = ai
	for j in range(1,N):
		tt[j] = j*dt
		at_2[j] = (ai**(13/2)+C/qi_2*tt[j]*yr_to_sec)**(2/13)
		if tt[j] < t_w[1]:
			Q0 = qi
		else:
			if tt[j] < t_w[0]:
				Q0 = Q2450
			else:
				Q0 = Q620
		at[j] = (a_pre**(13/2)+C/Q0*dt*yr_to_sec)**(2/13)
		a_pre = at[j]
	rot = at**(-3/2)  # rotational rate from at
	rot /= rot[-1]
	tt1 = tt[tt < t_w[1]]
	at1 = at[tt < t_w[1]]
	ax[0].plot(tt/1e9,at_2,'b--',linewidth=1.0)
	ax[0].plot(tt1/1e9,at1,'r',linewidth=1.0)
	ax[1].plot(tt1/1e9,at1,'r--',linewidth=1.5)
# latter stages...
tt2 = tt[np.logical_and(tt >= t_w[1],tt < t_w[0])]
at2 = at[np.logical_and(tt >= t_w[1],tt < t_w[0])]
tt3 = tt[tt > t_w[0]]
at3 = at[tt > t_w[0]]
ax[0].plot(tt2/1e9,at2,'orange',linewidth=1.5)	
ax[0].plot(tt3/1e9,at3,'m',linewidth=1.5)
ax[1].plot(tt2/1e9,at2,'orange',linewidth=1.5)	
ax[1].plot(tt3/1e9,at3,'m',linewidth=1.5)

for i,ti in enumerate(t_w):
	ax[0].errorbar(t_w[i]/1e9,ap_w[i],yerr=err_w[i],fmt='o',color='k')
	ax[1].errorbar(t_w[i]/1e9,ap_w[i],yerr=err_w[i],fmt='o',color='k')

ax[0].set_title("Orbital Evolution Model #1")
ax[0].set_xlabel('Time (Gyr)')
ax[0].set_ylabel('Earth-Moon distance (Re)')
ax[0].set_xlim([0,4.3])
ax[0].set_ylim([0,65])

# my model    
tx = np.linspace(0,t_w[1],401)
b = np.linspace(0.1,0.6,6)
print(b)
for ai in a0:
	for bi in b:
		aa = ai + (ap_w[1]-ai)*(tx/t_w[1])**bi
		ax[1].plot(tx/1e9,aa,'g',linewidth=0.1)
	#	print(ai);print(bi);print(aa[0]);print(aa[-1])

ax[1].set_title("Orbital Evolution Model #2")
ax[1].set_xlabel('Time (Gyr)')
ax[1].set_ylabel('Earth-Moon distance (Re)')
ax[1].set_xlim([0,4.3])
ax[1].set_ylim([0,65])

fig.savefig(outfile,dpi=400)

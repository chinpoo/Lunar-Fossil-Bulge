# functions called by lunar_bulge.py
# import all modules needed
import numpy as np
from math import pi

time_duration = 4.4e9    # in years

def model_readin(file_name):
    keys = ["M","V"]                   # model indicators
    model,ind,tmp = {},[],[]
    fid = open(file_name,'r')  
    for line in fid.readlines():
        tmp.append(line.rstrip('\n'))
    for k in keys:
        ind.append(tmp.index(k))
    ind.append(len(tmp))
    # read model in specific order
    for i,k in enumerate(keys):
        model[k] = [l.split() for l in tmp[ind[i]+2:ind[i+1]]]
        model[k] = np.array(model[k],dtype=np.float32).T    
    fid.close()    
    return model

# interpolate values for given Maxwell time t
def interpolate_t(t,v,i_row):
    n = v.shape[1] - 1      # number of stages in model file
    if t < 0:               # before hydrostatic equilibrium
        return v[i_row,0]
    else:
        for i in range(n):
            if t >= v[0,i]:         # row "0" is Maxwell time
                if t < v[0,i+1]:
                    return v[i_row,i] + (v[i_row,i+1]-v[i_row,i])/(v[0,i+1]-v[0,i])*(t-v[0,i])
                else:
                    continue     

# interpolate initial profile for radial refinement, note we do not cross property interface
def init_property(r,r0,v0):
    n = len(r)-1
    v = np.zeros(n)
    for i in range(n):
        for j in range(len(r0)-1):
            if r[i+1] <= r0[j+1]:
                v[i] = v0[j]
                break
            else:
                continue
    return v
    
# compute gravitational acceleration at each nodal point
def grav_acc(r,rhoc,rhom):
    G = 6.67e-11
    n = len(r)
    c = 4*pi*G/3
    mass = np.zeros(n)
    for l in range(n):
        if l == 0:         # first layer is always core
            mass[l] = rhoc*r[l]**3     
        else:
            mass[l] = mass[l-1] + rhom*(r[l]**3-r[l-1]**3)
    g = c*mass/r**2
    return g

# compute transient temperature profile as a function of time, an analytical solution
def heat_conduction(r,t,gamma,n_sum=1000):
    # r is a numpy array
    T = np.zeros(len(r))
    for n in range(1,n_sum+1):
        ld = n*pi
        c = -(-1)**n*2/ld
        T += c*np.exp(-gamma*ld**2*t)*np.sin(ld*r)/r   # replace r=0 with a small number
    return T
    
# compute viscosity from temperature profile based on Arrhenius equation
def viscosity_temp(T,eta0,E,dT,epsilon,eta_min=3e20,eta_max=1e30):
    R = 8.31
    c = E/R/dT
    eta = eta0*np.exp(c*(1/(T+epsilon)-1/(1+epsilon)))
    eta[eta>eta_max] = eta_max
    eta[eta<eta_min] = eta_min
    return eta

# 03/19/2017: determine viscosity from Te linear growth model
def viscosity_Te_growth_1(r,t,eta0,R0,t1,tp):
    # linear Te growth model used...
    time_Te = np.array([t1/1e6,tp/1e6])               # two step linear growth
    d_Te = np.array([[50,150],[50,150],[50,150]])   # thickness of layers at end of each step
    visc_Te = np.array([1e30,1e27,1e24])            # viscosity of 3 layers
    nl,nc = d_Te.shape
    rate_Te = np.zeros(d_Te.shape)
    for i in range(nc):
        if i==0:
            rate_Te[:,i] = d_Te[:,i]/time_Te[i]
        elif i>0:
            rate_Te[:,i] = (d_Te[:,i]-d_Te[:,i-1])/(time_Te[i]-time_Te[i-1])
    d = np.zeros(nl)
    for i in range(nl):
        for j in range(nc):
            if t <= time_Te[j]:
                if j == 0:
                    d[i] = rate_Te[i,j]*t
                else:
                    d[i] = d_Te[i,j-1] + rate_Te[i,j]*(t-time_Te[j-1])
                break
            else:
                continue
    for i in range(nl):
        if i > 0:
            d[i] += d[i-1]
    r_Te = 1 - d*1e3/R0
    r = r[1:]
    eta = eta0*np.ones(len(r))
    dd = [0]*nl
    for i in range(nl-1,-1,-1):
        ind1 = r>=r_Te[i]
        ind2 = r<r_Te[i]
        eta[ind1] = visc_Te[i]
        # intopolate for one cell
        p1 = np.log10(visc_Te[i])
        p2 = np.log10(eta[ind2][-1])
        r1 = r[ind1][0]
        r2 = r[ind2][-1]
        eta[ind1][0] = 10**(p2+(p1-p2)/(r1-r2)*(r_Te[i]-r2)) 
    # remarks: need to consider when 3 layers in one single cell at very initial time? Maybe not
        dd[i] = d[i]
    return eta

# determine viscosity from Te growth model, but with smoothed 2*Te second high viscous layer beneath
def viscosity_Te_growth_smooth(r,t,eta0,R0,t1,tp,flag,bot_cutoff=1e22):
    # linear Te growth model used...only two layers considered
    time_Te = np.array([t1/1e6,tp/1e6])               # two step linear growth
    d_Te = np.array([[50,150],[100,300]])          # thickness of layers at end of each step
    eta_Te = 1e30 
    nl,nc = d_Te.shape
    rate_Te = np.zeros(d_Te.shape)
    for i in range(nc):
        if i==0:
            if flag == 1:  # this is linear growth of Te, Te = a*t 	    
                rate_Te[:,i] = d_Te[:,i]/time_Te[i]
            elif flag == 2:   # this is sqare root growht of Te, Te = a*sqrt(t)
                rate_Te[:,i] = d_Te[:,i]/np.sqrt(time_Te[i])
        elif i>0:           # at 2nd time segment, always grow linearly
                rate_Te[:,i] = (d_Te[:,i]-d_Te[:,i-1])/(time_Te[i]-time_Te[i-1])
    
    d = np.zeros(nl)
    for i in range(nl):
        for j in range(nc):
            if t <= time_Te[j]:
                if j == 0:
                    if flag == 1:			
                        d[i] = rate_Te[i,j]*t
                    elif flag == 2:
                        d[i] = rate_Te[i,j]*np.sqrt(t)
                else:
                    d[i] = d_Te[i,j-1] + rate_Te[i,j]*(t-time_Te[j-1])
                break
            else:
                continue

    for i in range(nl):
        if i > 0:
            d[i] += d[i-1]
    r_Te = 1 - d*1e3/R0
    r_l,r_u = r[:-1],r[1:]
    n = len(r_l)
    eta = np.zeros(n)
    sign = np.zeros(n)
    # deal with two layers model only....not a good algorithm tho
    p1,p2 = np.log10(bot_cutoff),np.log10(eta_Te)
    if t>0:
        alpha = (p2-p1)/(r_Te[0]-r_Te[1])  # for linear interpolation of power index
    else:
        alpha = None

    ind1 = (r_u <= r_Te[1])
    ind2 = (r_l >= r_Te[0])
    eta[ind1] = np.log10(eta0)
    eta[ind2] = p2
    sign[ind1] = 1
    sign[ind2] = 1
    for i in range(n):
        if sign[i] == 1:
            pass
        elif sign[i] == 0:  # 4 different cases
            if r_l[i] >= r_Te[1]:
                if r_u[i] <= r_Te[0]:
                    rm = 0.5*(r_l[i]+r_u[i])
                    eta[i] = p1+alpha*(rm-r_Te[1])
                else:
                    tmp = p1+alpha*(r_l[i]-r_Te[1])
                    eta[i] = (p2*(r_u[i]-r_Te[0]) + 0.5*(tmp+p2)*(r_Te[0]-r_l[i]))/(r_u[i]-r_l[i])
            else:
                if r_u[i] <= r_Te[0]:
                    tmp = p1+alpha*(r_u[i]-r_Te[1])
                    eta[i] = (p1*(r_Te[1]-r_l[i]) + 0.5*(p1+tmp)*(r_u[i]-r_Te[1]))/(r_u[i]-r_l[i])
                else:
                    eta[i] = (p1*(r_Te[1]-r_l[i]) + 0.5*(p1+p2)*(r_Te[0]-r_Te[1]) + p2*(r_u[i]-r_Te[0]))/(r_u[i]-r_l[i])
            sign[i] = 1
    eta = 10**eta

    return eta

# ad-hoc function of semi-major axis a with respect to time t
def rot_potential(a0,b,t,ap,tp=time_duration):
    if t < 0:
        t = 0
    if b <= 0:
        if t == 0:
            wsq = (ap/a0)**3
        elif t > 0:
            wsq = (ap/ap)**3
    else:    
        a = (ap-a0)*(t/tp)**b + a0
        wsq = (ap/a)**3
    return wsq

# functions for geologically constrained orbital evolution models..

# compute quality factor Q from simple Q formulation
def determine_Q(a0,a1,dt,const):
	Q = const*dt/(a1**(13/2)-a0**(13/2))
	return Q

def rot_potential_constrained(t,a0,ap,flag,x,paras):
	if t < 0:
		t = 0
	tw = paras[0]
	aw = paras[1]
	q = paras[2]
	if t<=tw[1]:
		if flag == 2:
			a = (a0**(13/2)+x*t)**(2/13)
		elif flag == 3:
			a = (aw[1]-a0)*(t/tw[1])**x + a0
	elif t > tw[1] and t <= tw[0]:
		a = (aw[1]**(13/2)+q[1]*(t-tw[1]))**(2/13)
	elif t > tw[0]:
		a = (aw[0]**(13/2)+q[0]*(t-tw[0]))**(2/13)
	wsq = (ap/a)**3
	return wsq

# skip the observation at 2.45Ga because of the unverified result
def rot_potential_constrained_1(t,a0,ap,flag,x,paras):
	if t < 0:
		t = 0
	tw = paras[0][0]
	aw = paras[1][0]
	q = paras[2][0]
	if t<=tw:
		if flag == 2:
			a = (a0**(13/2)+x*t)**(2/13)
		elif flag == 3:
			a = (aw-a0)*(t/tw)**x + a0
	elif t > tw:
		a = (aw**(13/2)+q*(t-tw))**(2/13)
	wsq = (ap/a)**3
	return wsq

# two segments model before 620 Ma, first segment is prescribed with Q and t
def determine_a(a0,t,Q0):
	if t < 0:
		t = 0
	return (a0**(13/2) + Q0*t)**(2/13)

def rot_potential_constrained_2(t,a0,ap,Q,paras):
	if t < 0:
		t = 0
	tw,aw,qw = paras[0],paras[1],paras[2]
	t1,a1,q1 = paras[3],paras[4],paras[5]
	if t<=t1:
		a = determine_a(a0,t,q1)
	elif t > t1 and t < tw:
		a = determine_a(a1,t-t1,Q)
	elif t >= tw:
		a = determine_a(aw,t-tw,qw)
	wsq = (ap/a)**3
	return wsq

# use different dt for different segments of V(t)
def determine_prefactor_dt(prefactors,segments,tp):
    n1 = len(prefactors)
    n2 = len(segments)
    if n1 == n2:
        n = n1
        dt = np.zeros((2,n))
        for i in range(n):
            dt[0,i] = tp*segments[i]
            dt[1,i] = prefactors[i]
        return dt
    else:
        return

# 04/12/2017: NEW!! use different dt for different segments of V(t)
def determine_prefactor_dt_new(prefactors,segments,t_model,tp):
    n1 = len(prefactors)
    n2 = len(segments)
    if n1 == n2:
        n = n1
        t_seg = []
        for i in range(n):
            if tp <= t_model*segments[i]:
                t_seg.append(tp)
                break
            else:
                t_seg.append(t_model*segments[i])
        n = len(t_seg)
        dt = np.zeros((2,n))
        for i in range(n):
            dt[0,i] = t_seg[i]
            dt[1,i] = prefactors[i]
        return dt
    else:
        return

# determine total time steps 
def determine_tot_steps(dt,offset,dt_default,maxwell_time_yr):
    cum_step = np.array([0]*(1+dt.shape[1]))
    step = int(offset/dt_default)         # steps needed before hydrostatic equilibrium
    cum_step[0] = step
    time_c = 0.0                    # current time in yr
    for i in range(dt.shape[1]):
        d_step = int((dt[0,i]-time_c)/(dt[1,i]*maxwell_time_yr)) + 1
        time_c += d_step*maxwell_time_yr*dt[1,i]
        step += d_step
        cum_step[i+1] = step
    return step,cum_step

# the A matrix in isoviscous layer and its eigenvalues
def matrix_a(eta,LL):
    A = np.array([[-2,LL,0,0],
              [-1,1,0,1/eta],
              [12*eta,-6*LL*eta,1,LL],
              [-6*eta,2*(2*LL-1)*eta,-1,-2]])
    return A      

# propagator matrix from v1 to v2
def prop_matrix(A,v1,v2,ld):
    P = np.zeros((4,4),dtype=np.float)
    for i in ld:
        c = np.exp(i*(v2-v1))
        p = np.eye(4,dtype=np.float)
        for j in ld:
            if j != i:
                p = np.dot(p,(A-np.eye(4)*j)/(i-j))
        P = P + c*p       
    # or...
    # P = expm(A*(v2-v1))
    return P

# generalized to N layer model...
def linear_eqn_n(P,DIS,ca,cb):
    a = np.array([[1-ca[1]*P[0,2],0,-(P[0,0]+P[0,2]*(ca[0]+ca[3])),-P[0,1]],
               [-ca[1]*P[1,2],1,-(P[1,0]+P[1,2]*(ca[0]+ca[3])),-P[1,1]],
               [cb[1]+cb[3]-P[2,2]*ca[1],0,cb[0]-(P[2,0]+P[2,2]*(ca[0]+ca[3])),-P[2,1]],
               [-ca[1]*P[3,2],0,-(P[3,0]+P[3,2]*(ca[0]+ca[3])),-P[3,1]]])
               
    b = np.array([[P[0,2]*(ca[2]+ca[4])+DIS[0,0]],
               [P[1,2]*(ca[2]+ca[4])+DIS[1,0]],
               [P[2,2]*(ca[2]+ca[4])-(cb[2]+cb[4]+cb[5])+DIS[2,0]],
               [P[3,2]*(ca[2]+ca[4])+DIS[3,0]]])
               
    return (a,b)
    

# determine prefactors for different segments of V(t) for different orbit model
def determine_prefactor_dt_old(a0,b,ap,prefactors,tp,t_max=time_duration):
    max_range = (ap/a0)**3       # V(t) range
    current_v = 1.0              # V(t) of present day
    n1 = int(np.log2(max_range/current_v))+1   # number of segments of dt
    n2 = len(prefactors)                  
    n = np.min([n1,n2])          # number of segments are min of n1 and n2
    dt_tmp = np.zeros((2,n))         # dt (prefactor) segments
    for i in range(n):
        at = ap/(current_v**(1/3))    # from V(t) determine a(t)
        t = ((at-a0)/(ap-a0))**(1/b)*t_max   # determine t in year
        dt_tmp[0,n-1-i] = t
        dt_tmp[1,n-1-i] = prefactors[n2-1-i]
        current_v *= 2
    # for time_yr < t_max
    if tp < t_max:
        dt = dt_tmp[:,:sum(dt_tmp[0] < tp)+1]
        dt[0,-1] = tp
    else:
        dt = dt_tmp
    return dt


# compute semi-major axis and rotational potential as function of time, from Lambeck(1980).
def rot_potential_old(t1,Q,t0=0.0,n=2000):
    G = 6.67e-11
    m = 7.348e22   # mass of Moon
    M = 5.972e24   # mass of Earth
    Re = 6.4e6      # Earth radius
    k2 = 0.3       # Earth tidal love number
    ca = 3*G*m*Re**5/(G*(M+m))**0.5
    yr_to_sec = 365*24*3600
    coef = 13/2*ca*k2/Q*yr_to_sec
    a1 = 60.0*Re     # present day semi-major axis
    a = np.zeros((2,n+1))
    v = np.zeros((2,n+1))
    dt = (t1-t0)/n
    for i in range(n+1):
        t = dt*i
        a[0,i] = t
        v[0,i] = t
        a0 = a1**6.5 - coef*(t1-t)
        if a0 <= 0:
            print("a0 = "+str(a0))
            print(coef)
            return
        else:
            a[1,i] = a0**(2/13)/Re       # semi-major axis at time t
            v[1,i] = 1.0/a[1,i]**3       # relative amplitude of rot potential
    v[1,:] = v[1,:]/v[1,-1]              # normalized by present day potential
    return v
# -------------------------------------------------------------------

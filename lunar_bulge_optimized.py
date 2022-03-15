
# compute lunar bulge size as effects of conductive cooling and orbital recession of the Moon...

import numpy as np
from math import pi
from numpy.linalg import solve
#from scipy.linalg import expm
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import time as tm
from multiprocessing import Pool
import sys

import my_func as fn

t_start = tm.time()

# ************** Important!! ***************
# 03/22/2017 late night
# 1. Since when a0 is small, it grows exponentially fast. For a=5, it grows to about 20 in only 10 Myrs.
#    Therefore, refinement on time stepping seems very necessary, I will use dt=0.02tau for first ~1e5 yrs, dt=0.2tau till ~1e6 yrs
#    , dt=0.5tau till ~1e7, and so on
# 2. Since Te at very initial time is model dependent, making the first element being assgined with uncertain value that can cause
#    significant variations in the final bulge. I assign the first element, in our used resolution 1km, with 1e30 Pas before Te exceeds the depth of the 1st element
# 3. No further resolution tests before LPSC, but afterwards, they need to be done for the paper
# *****************************************

# global variables...
# *************************************************
l = 2                                  # harmonic degree of forcing
L = l*(l+1)                            # l*(l+1)
lambdas = (l+1,-l,l-1,-l-2)            # eigenvalues of A matrix

model_file = "model_moon.dat"          # input file
#model_file = "model_moon_1e21.dat"          # input file
#model_file = "model_moon_Te.dat"          # input file
data_dir = "./data/"
#dumpfile = "Bulge_Q25_Te_sqrt_1.p"                     # output file for each case
#dumpfile = "Bulge_Q50_fluid.p"                     # output file for each case
case_num = 1
#dumpfile = "Eta1e21_dt_1_Q100_Te_sqrt_{0:d}.p".format(case_num)
#dumpfile = "Eta1e21_dt1x_Q100_Te_sqrt_I.p"
dumpfile = "output_stress_1.p"
#dump_visc = "Visc_Q100_Te_gradation_TEST_1.p"
#dump_time = [1e7,5e7,1e8,2e8,4e8,5e8,1e9,2e9,3e9,4e9]          # time points to check output viscosity

# change the values of a0 and b for different orbital evolution of the Moon...
# orbital model parameters
ap = 60                                # current semi-major axis in Earth radii
#a0 = np.linspace(5,35,13)              # choise of initial Earth-Moon distance
#a0 = np.linspace(5,15,5)              # choise of initial Earth-Moon distance
#a0 = np.linspace(17.5,35,8)              # choise of initial Earth-Moon distance
#a0 = np.linspace(5,28,24)
#a0 = np.linspace(5,12,8)
#a0 = np.linspace(13,20,8)
#a0 = np.linspace(21,28,8)
a0_1 = np.linspace(5,8,4)
a0_2 = np.linspace(9,12,4)
a0_3 = np.linspace(13,16,4)
a0_4 = np.linspace(17,20,4)
a0_5 = np.linspace(21,24,4)
a0_6 = np.linspace(25,28,4)
list_a0 = [a0_1,a0_2,a0_3,a0_4,a0_5,a0_6]
#a0 = list_a0[case_num-1]
a0 = np.array([5,10,15,20])
b = [0.1]
#b = np.linspace(0.05,0.6,12)
#for i in sys.argv[2:]:
#    a0.append(float(i))
#b = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]        # power index in rotation evolution model, charaterizes time scale of orbital recession
print(a0);print(b);print(dumpfile)

# control flags...
time_dependent = 1                     # time-dependent viscosity (from Te growth model or conductive cooling) (1) or time-independent viscosity
use_cooling_model = 0                      # update viscosity from cooling model (1) or from designed Te model (0)
flag_tidal = 1                         # tidal/rotational forcing (1) or surface loading (0)
flag_Te_model = 2                      # apply only when use_cooling_model = 0, graded linear (0) and smoothed linear(1) and sqrt(2) growth model for Te
uniform_init_visc = 1                  # for conductive/Te/fixed-lithosphere model, start calculation with uniform viscosity (1); otherwise, layered (0)
rot_model = 4                          # ad-hoc "a0-b"rotation model (1) or potential read from input file (0) or single Q model with geological constraints from Williams (2000) (2) or a0-b model with geological constraints from Williams(2000) (3) or two segments before 620 Ma with first segment with prescribed Q and t (4)
adaptive_dt = 1                        # dt determined by segments of V(t) (1) or fixed to default dt (0)

# time-stepping parameters...
time_model_yr = 4.4e9                  # total time in years for modeling orbit and lithosphere
time_yr = 0                        # total time of calculation in years
#steps_for_sol = 100                     # output solution for every n time steps
steps_for_sol = 50                    # for 1e21 ref viscosity
steps_for_visc = 100                    # update viscosity (temperature) profile after every n Maxwell times
update_rot_after = 400                 # apply time-changing rotation after n Maxwell times, i.e. after hydrostatic equilibrium forms
#stop_cooling_after = 3.0e8             # stop conductive cooling (lithosphere thicknening) after (yrs)
stop_cooling_after = 1.0e12             # stop conductive cooling (lithosphere thicknening) after (yrs)
prefactor_dt_0 = 1.0                   # default prefactor of time step, dt = prefactor*tau

if adaptive_dt == 1:
	# for when b>0.2, moderate change
	prefactors_1 = [1.0,2.0,5.0,10.0]    # range of dt prefactor for adaptive dt to different segments of V(t), if flag turned on
	segments_1 = [0.25,0.5,0.75,1.0]       # ratio segments in time duration for different prefactors of dt
	# for when b<=0.2, rapid change 
	prefactors_2 = [0.5,1.0,2.0,5.0,10.0]    # range of dt prefactor for adaptive dt to different segments of V(t), if flag turned on
	segments_2 = [0.05,0.25,0.5,0.75,1.0]       # ratio segments in time duration for different prefactors of dt
	# **NEW time stepping with special treatment for very initial time 
	prefactors_new = [0.02,0.2,0.5,1.0,2.0,5.0,10.0]    # range of dt prefactor for adaptive dt to different segments of V(t), if flag turned on
	#prefactors_new = [0.2,2.0,5.0,10.0,20.0,50.0,100.0]    # dt10x, same dt as 1e22 ref viscosity...
	segments_new = [5e-5,5e-4,0.05,0.15,0.3,0.5,1.0]       # ratio segments in time duration for different prefactors of dt

# grid refinement for conductive cooling model
if time_dependent == 1:
	if use_cooling_model == 1:
		r_sep = np.array([0.6,0.8,0.9,0.98])          # refined dr (near surface) determined from resolution tests
		# for when b>0.2
		n_sep_1 = (20,20,20,40,20)                      # number of layers in each division
		# for when b<=0.2
		n_sep_2 = (40,40,40,80,40)                      # number of layers in each division
	else:
		# based on a designed Te model, with 3 top layers grow to 50km in 400Myrs and then 150km at present day
		r_sep = np.array([0.74138,0.91379,0.97126])     # 4 layers, CMB to 450km, to 150 km, to 50km, to surface
		n_sep_1 = (1,100,50,50)                         # resolution, 1/2/3 km for top three layers 
		n_sep_2 = (50,100,50,50)                         # resolution, 1/2/3 km for top three layers 
		init_Te_visc = 1e30                             # initially, first 1km is assigned with fixed viscosity 1e30
		n_visc = 1

# determine temporal and spatial refinement from orbit models
if adaptive_dt == 1:
	if time_dependent == 1:
		if rot_model in [1,3]:
			if np.min(b) > 0.2:
				n_sep,prefactors,segments = n_sep_1,prefactors_1,segments_1
			else:
				n_sep,prefactors,segments = n_sep_2,prefactors_2,segments_2
		if rot_model in [0,2,4]:
			if np.max(a0) <= 40.0:
				#n_sep,prefactors,segments = n_sep_2,prefactors_2,segments_2
				# use new time stepping strategy for now
				n_sep,prefactors,segments = n_sep_2,prefactors_new,segments_new
			else:
				n_sep,prefactors,segments = n_sep_1,prefactors_1,segments_1
	else:
		# use new time stepping even for cases with constant viscosity
		#prefactors,segments = [prefactor_dt_0],[1.0]
		prefactors,segments = prefactors_new,segments_new
	print(prefactors);print(segments)
else:
	if time_dependent == 1:
		n_sep = n_sep_2
	else:
		pass

# constants for orbit models...
G = 6.67e-11
year_to_sec = 365*24*3600              # convesion factor from year to sec
Me = 5.972e24                          # Earth mass
Mm = 7.348e22                          # Moon mass
Re = 6.371e6           
B0 = 3*Mm/Me*Re**5*(G*(Me+Mm))**0.5         # single Q model constant
#k2 = 0.97
k2 = 0.3                              # Earth's deg-2 Love number
C0 = 13/2*k2*B0/Re**(13/2)*year_to_sec
#print(C0)
if rot_model == 4:
	Q1 = 100.0      # give Q for first segment
	t1 = 5e8       # how long first segment lasts
# ------------------------
# Geological constraints from Williams (2000)
if rot_model == 2 or rot_model == 3:
	t0_W = np.array([6.2e8,2.45e9])
	t_W = time_model_yr - t0_W
	ap_W = ap*np.array([0.965,0.906])
	err_W = ap*np.array([0.005,0.029])
	# determine Q values of different stages
	Q620 = fn.determine_Q(ap_W[0],ap,t0_W[0],C0)
	Q2450 = fn.determine_Q(ap_W[1],ap_W[0],(t0_W[1]-t0_W[0]),C0)
	Q = {}
	for ai in a0:
		Q[ai] = fn.determine_Q(ai,ap_W[0],t_W[0],C0)
		#Q[ai] = fn.determine_Q(ai,ap_W[1],t_W[1],C0)   # we skip observation at 2.45Ga
# 2017-03-13: geological constraint for last 620 Ma, two prescribed segments before 620 Ma with first segment Q and t being given
if rot_model == 4:
	t0_W = 6.2e8
	t_W = time_model_yr - t0_W
	ap_W,err_W = ap*0.965,ap*0.005
	Q620 = fn.determine_Q(ap_W,ap,t0_W,C0)
	Q = {}
	A1 = {}
	for ai in a0:
		A1[ai] = fn.determine_a(ai,t1,C0/Q1)
		Q[ai] = fn.determine_Q(A1[ai],ap_W,t_W-t1,C0)

# -------------------------
# conductive cooling parameters
chi = 1.0e-6                           # thermal diffusivity
Ts = 250                               # Surface temperature in K
dT = 1250                              # initial temperature difference between surface and mantle
E = 1.5e5                              # activation energy, J/mol     
epsilon = Ts/dT
#temp_update_shift = 10                 # see program for meaning (unit: Maxwelltime),for cooling model
#if adaptive_dt == 1:
#    temp_update_shift = prefactors_new[0]   # for Te growth model (unit: Maxwelltime)
#else:
#    temp_update_shift = prefactor_dt_0
temp_update_shift = 0
# *************************************************

# Model setup...
# read model from file...
Model = fn.model_readin(model_file)
model_0,V = Model["M"],Model["V"]      # note V is from file, but may be replaced by modeled values

# radius,visc,density,elastic modulus
r0,eta0,rho0,mu0 = model_0[0,:],model_0[1,1:],model_0[2,:],model_0[3,1:]
R0 = r0[-1]                            # outer radius
rhoc,rhom = rho0[0],rho0[1]            # core,mantle density (mantle density constant incompressible)

rsg = 4*pi*G*rhom**2*R0**2/mu0[0]      # non-dim coefficient 1
q0 = 4*pi*G*rhom*R0                    # non-dim coefficient 2

# Maxwell times
tau0 = eta0/mu0                        # Maxwell times of each layer at t=0
#tau_ref = tau0.min()                  # use min Maxwell time as referece, same for viscosity
tau_ref = tau0[0]                      # reference Maxwell time
tau_ref_yr = tau_ref/year_to_sec       # reference Maxwell time in yr
gamma = chi*tau_ref/R0**2              # non-dim coef for conductive cooling

# discretizarion of radial grids if temperature dependent
if time_dependent == 1:
    r_sep = np.sort(np.unique(np.concatenate((r0,r_sep*R0))))
    r1 = np.array([])
    for i in range(len(n_sep)):    # this is problematic if Te model is used (!!)
        r2 = np.linspace(r_sep[i],r_sep[i+1],n_sep[i]+1)
        r1 = np.concatenate((r1,r2))
    r1 = np.sort(np.unique(r1))
else: 
    # use profile from model file    
    r1 = r0
N = len(r1)        # number of solution nodes

#g0 = np.array([9.8]*(N+1))     # Earth...
g0 = fn.grav_acc(r1,rhoc,rhom)         # compute gravitational acceleration

# non-dimensionalization...
r = r1/R0
rb,rs = r[0],r[-1]                     # radii of cmb and surface
v = np.log(r)                          # for propagator matrices
rho = rho0/rhom
d_rho = abs(np.append((rho[:-1]-rho[1:]),rho[-1]))     # density diff at layer interface
drho_b,drho_s = d_rho[0],d_rho[-1]     # note we only consider density interface at surface and CMB
g = g0/q0
gb,gs = g[0],g[-1]                     # gravitational acceleration at cmb and surface
# layer properties at t=-Inf for latter use.
eta1,mu1,tau1 = eta0/eta0[0],mu0/mu0[0],tau0/tau_ref
mu_init = fn.init_property(r1,r0,mu1)
if uniform_init_visc == 0:    	
    eta_init = fn.init_property(r1,r0,eta1)
    Te = 60  #km
    eta_init[r1[:-1] >= R0-Te*1000] = init_Te_visc/eta0[0]
elif uniform_init_visc == 1:
    eta_init = np.ones(N-1)    # visc=1 for first update_rot_after time, to reach hydrostatic equilibrium

# time evolution...
init_time_offset = update_rot_after

def time_evolution(orbit_paras):
    ai = orbit_paras[0]
    bi = orbit_paras[1]
    print("case (a0,b) = ({0:f},{1:f})".format(ai,bi))
    # dt for each time step depends on V(t)...
    # determine dt for segments of V(t) if adaptive_dt == 1, otherwise dt is default value
    if adaptive_dt == 1:
        #prefactor_dt = fn.determine_prefactor_dt(prefactors,segments,time_model_yr)
        # 04/12/2017: run steps till time_yr instead of time_model_yr
        prefactor_dt = fn.determine_prefactor_dt_new(prefactors,segments,time_model_yr,time_yr)
    elif adaptive_dt == 0:
        prefactor_dt = np.zeros((2,1))
        #prefactor_dt[0] = time_model_yr
        prefactor_dt[0] = time_yr
        prefactor_dt[1] = prefactor_dt_0

    # rot models if with geological constraints,passing argument
    if rot_model == 2 or rot_model == 3:
        args = [t_W,ap_W,(C0/Q620,C0/Q2450)]
    if rot_model == 4:
        args = [t_W,ap_W,C0/Q620,t1,A1[ai],C0/Q1]	    
        
    dt = prefactor_dt[1]*tau0[0]/tau_ref    # time step (in Maxwell times)
    # dt_yr = dt*tau_ref_yr                   # time step (in year)
    # determine total time steps from dt segments
    step,dsteps = fn.determine_tot_steps(prefactor_dt,update_rot_after,prefactor_dt_0,tau_ref_yr)
    print(step);print(dsteps);print(prefactor_dt)
    # to save memory for long-term calculations...
    # time-dependent topo, potential at layer interfaces
    N_sol = int(step/steps_for_sol)+1         # total number of solutions
    ur = np.zeros((N,N_sol))                  # time-dependent topo at r1
    phi = np.zeros((N,N_sol))                 # time-dependent potential at r1
    phi_V = np.zeros(N_sol)                   # time dependent external potential at surface
    t = np.zeros(N_sol)                       # time in Maxwell time
    t_yr = np.zeros(N_sol)                       # time in year
    # 04-27-2017: output stress
    srr = np.zeros((N,N_sol))
    srt = np.zeros((N,N_sol))
    visc = np.zeros((N-1,N_sol))
    ut = np.zeros((N,N_sol))                  # time-dependent horizontal displacement
    
    # potential and displacement at every node for previous time step and current time step
    ur_p = np.zeros(N)
    ut_p = np.zeros(N)
    phi_p = np.zeros(N)
    ur_c = np.zeros(N)
    ut_c = np.zeros(N)
    phi_c = np.zeros(N)
    if N > 2:
        # traction components at internal layer interfaces at previous time step and current time step
        trr_p = np.zeros(N-2)
        trt_p = np.zeros(N-2)
        trr_c = np.zeros(N-2)
        trt_c = np.zeros(N-2)
        
    # loop of steps...
    time_c = -init_time_offset         # current elapsed time offset by update_rot_after in unit of Maxwell times
    time_temp_update = 0.0             # record time when temperature field updated    
    for it in range(step):
        # determine dt for current time step
        if it < dsteps[0]:
            dt_c = prefactor_dt_0
        else:
            dt_c = dt[it >= dsteps[:-1]][-1]
        if it%10000 == 0:
            print("time step {0:d}, time {1:f} Myr".format(it,time_c*tau_ref_yr/1e6))
        
#        print("step:{0}, dt={1}".format(it,dt_c))

        # interpolate potential Vn, dVn, rln, for current time
        if rot_model == 0:                            # from model file, linear interpolation                         
            Vn = fn.interpolate_t(time_c*tau_ref_yr/1e6,V,1)     # Vn, interpolated for current time step
        elif rot_model == 1:                          # from orbital recession
            Vn = fn.rot_potential(ai,bi,time_c*tau_ref_yr,ap)
        elif rot_model == 2:                         # Single Q models with geological constraints for the past 2.45 Gyrs 
            #Vn = fn.rot_potential_constrained(time_c*tau_ref_yr,ai,ap,rot_model,C0/Q[ai],args)
            Vn = fn.rot_potential_constrained_1(time_c*tau_ref_yr,ai,ap,rot_model,C0/Q[ai],args)
        elif rot_model == 3:                          # "a-b" models with geological constraints for the past 2.45 Gyrs
            #Vn = fn.rot_potential_constrained(time_c*tau_ref_yr,ai,ap,rot_model,bi,args)
            Vn = fn.rot_potential_constrained_1(time_c*tau_ref_yr,ai,ap,rot_model,bi,args)
        elif rot_model == 4:
            Vn = fn.rot_potential_constrained_2(time_c*tau_ref_yr,ai,ap,C0/Q[ai],args)
	    # ============================

        update_prop_mat = False                 # control flag for update of prop matrix
        
        if it == 0:                             # t=0 is elastic		
            eta_bar = mu_init                   
            Vp = Vn                             # Vp, previous time step
            dVn = Vp                            # dVn, change of Vn in dt
        elif it > 0:                            # t>0 is viscoelastic
            if time_c < 0.0:                    # before update_rot_after, eta is eta_init
                eta = eta_init
            else:                               # update viscosity profile
                if time_dependent == 1:
                    # if viscosity is temperature dependent, update for every steps_for_visc
                    if time_c >= 0 and (time_c == temp_update_shift or time_c-time_temp_update >= steps_for_visc) and time_c*tau_ref_yr <= stop_cooling_after:  # note I shift 10 Maxwell time to compute T, because the series oscillates
                        if use_cooling_model == 1:                # conductive cooling model
                            temp = fn.heat_conduction(r,time_c,gamma)            # update temperature profile
                            T = (temp[:N-1]+temp[1:])/2                         # averaged temperature for each layer
                            eta = fn.viscosity_temp(T,eta0[0],E,dT,epsilon)/eta0[0]    # update viscosity profile
                        elif use_cooling_model == 0: # Te growth model
                            if flag_Te_model == 0:   # graded linear Te growth model				
                                eta = fn.viscosity_Te_growth_1(r,time_c*tau_ref_yr/1e6,eta0[0],R0,t1,time_model_yr)/eta0[0]
                            else:                    # smoothed model dependent on flag
                                eta = fn.viscosity_Te_growth_smooth(r,time_c*tau_ref_yr/1e6,eta0[0],R0,t1,time_model_yr,flag_Te_model)/eta0[0]
                            # assign the topmost element (1km) with 1e30 pas before Te grows deeper than this
                            if eta[-n_visc] < init_Te_visc/eta0[0]+1.0:
                                eta[-n_visc:] = init_Te_visc/eta0[0]
                        #    print("time:{0:.2f}, eta_top:{1:.1e}".format(time_c*tau_ref_yr/1e6,eta[-1]))
                        update_prop_mat = True
                        if time_c > temp_update_shift:
                            time_temp_update = time_c
                    #    print("step {0}, time {1}, diff={2}, dt={3}".format(it,time_c,time_c-time_temp_update,dt_c))
                elif time_dependent == 0:       # fixed viscosity
                    if uniform_init_visc == 1 and time_c == 0:       # if add a lith suddenly, update eta once                
                        eta = fn.init_property(r1,r0,eta1)     
                        update_prop_mat = True

            tau = eta/mu_init
            beta = dt_c/(dt_c+tau)     
            alpha = 1 - beta
            # beta, alpha at cmb and surface
            beta_b,beta_s = beta[0],beta[-1]
            alpha_b,alpha_s = alpha[0],alpha[-1]
            eta_bar = eta/(tau+dt_c)              # for all it>0, the effective viscosity in formulation    
            dVn = Vn - Vp                       # change of rotational potential
                
        # build A and P matrices for current time step 
        if it <= 1 or update_prop_mat:
            #print("eta update at "+str(it))
            P1 = np.array([np.eye(4)]*(N-1))
            P2 = np.array([np.eye(4)]*(N-1))
            for layer in range(N-1):
                A = fn.matrix_a(eta_bar[layer],L)
                P1[layer] = fn.prop_matrix(A,v[layer],v[layer+1],lambdas)
            # also construct P2: P2(rk->rs),P2(rk-1 -> rs),...,P2(rb->rs)        
            for layer in range(N-1):
                if layer == 0:
                    P2[layer] = P1[-1]
                else:
                    P2[layer] = np.dot(P2[layer-1],P1[N-2-layer])
    
        # build a, b coefficients, which relate to solutions at rb,rs
        if it == 0:
            # linear equations coeffs; these coeffs are only associated with values at rb and rs
            ca = np.zeros(5,dtype=np.float32)
            cb = np.zeros(6,dtype=np.float32)
            ca[0] = -rsg*drho_b**2*rb**2/(2*l+1)
            ca[1] = -rsg*drho_b*drho_s*rb**(l+1)/(2*l+1)
            ca[3] = rsg*drho_b*rb*gb
            ca[4] = 0                  # update every time step 
            cb[0] = rsg*drho_s*drho_b*rb**(l+2)/(2*l+1)
            cb[1] = rsg*drho_s**2/(2*l+1)
            cb[3] = -rsg*drho_s*gs   
            cb[4] = 0       # update every time step      
            if flag_tidal == 0:        # if surface loading problem
                Sn = Vn
                cb[5] = -rsg*rs*Sn*gs          # update if area density Sn changes with time
                 
        ca[2] = -rsg*drho_b*rb**(l+1)*dVn       # update each time step
        cb[2] = rsg*drho_s*dVn                  # update each time step
        # discontinuity cumulatives
        DIS = np.zeros((4,1),dtype=np.float32)    # needs to be updated each time step if N > 2
        if N > 2:
            # cc vectors due to discontinuities
            cc =  np.zeros((4,N-2),dtype=np.float32)  
    
        if it > 0:
            ca[4] = -rb*rsg*beta_b*drho_b*(phi_p[0]+rb**l*Vp-gb*ur_p[0])
            cb[4] = rs*rsg*beta_s*drho_s*(phi_p[-1]+rs**l*Vp-gs*ur_p[-1])
            # deal with discontinuities DIS when t>0...
            if N > 2: 
                # update cc and DIS
                for k in range(1,N-1):
                    # note that we already presume mantle density rho[k] is constant
                    cc[2,k-1] = r[k]*(alpha[k-1]-alpha[k])*(trr_p[k-1]+rsg*rho[-1]*(phi_p[k]+r[k]**l*Vp-g[k]*ur_p[k]))
                    cc[3,k-1] = r[k]*(alpha[k-1]-alpha[k])*trt_p[k-1]   
                    DIS += np.dot(P2[N-2-k],cc[:,k-1].reshape((-1,1)))          
            if flag_tidal == 0:        # if surface loading problem
                Sn = Vn - alpha_s*Vp
                cb[5] = -rsg*rs*Sn*gs
                   
        C = fn.linear_eqn_n(P2[-1],DIS,ca,cb)
        
        Y = solve(C[0],C[1]).reshape((4,))    # solution vector
        
        # topo and potential responses at cmb and surface
        d_ur_s = Y[0]
        d_ur_b = Y[2]
        d_ut_s = Y[1]
        d_ut_b = Y[3]
        d_phi_s = (rb**(l+2)*drho_b*Y[2]+rs**l*drho_s*Y[0])/(2*l+1)
        d_phi_b = (rb*drho_b*Y[2]+rb**l*drho_s*Y[0])/(2*l+1)
        
        # important: obtain solutions at internal layer interfaces rm's if N > 2 (viscosity layering)
        if N > 2:
            rY3_b = (ca[0]+ca[3])*Y[2]+ca[1]*Y[0]+(ca[2]+ca[4])
            rY3_s = cb[0]*Y[2]+(cb[1]+cb[3])*Y[0]+(cb[2]+cb[4]) 	    
            X_b = np.array([Y[2],Y[3],rY3_b,0]).reshape((4,1))	    
            Sol_m = np.zeros((N-2,5))
            for k in range(N-2):
                if k == 0:
                    X_m = np.dot(P1[k],X_b)
                elif k > 0:
                    X_m = np.dot(P1[k],(X_m + cc[:,k-1].reshape((-1,1))))
                d_ur_m = X_m[0]
                d_ut_m = X_m[1]
                d_phi_m = (rb**(l+2)/r[k+1]**(l+1)*drho_b*d_ur_b+r[k+1]**l*drho_s*d_ur_s)/(2*l+1)
                d_trr_m = X_m[2]/r[k+1]-rsg*rho[-1]*(d_phi_m+r[k+1]**l*dVn-g[k+1]*d_ur_m)
                d_trt_m = X_m[3]/r[k+1] 
                Sol_m[k,:] = np.array([d_ur_m,d_ut_m,d_phi_m,d_trr_m,d_trt_m])
            X_s = np.dot(P1[-1],X_m+cc[:,-1].reshape((-1,1)))
        #    print(X_s);print(np.array([Y[0],Y[1],rY3_s,0]).reshape((4,1)))	    
             
        # update ur,phi,trr,trt for each time step, but only store solutions for every steps_for_sol
        if it == 0:
            ur_c[0] = d_ur_b
            ur_c[-1] = d_ur_s
            ut_c[0] = d_ut_b
            ut_c[-1] = d_ut_s
            phi_c[0] = d_phi_b
            phi_c[-1] = d_phi_s
            if N > 2:
                ur_c[1:-1] = Sol_m[:,0]
                ut_c[1:-1] = Sol_m[:,1]
                phi_c[1:-1] = Sol_m[:,2]
                trr_c = Sol_m[:,3]
                trt_c = Sol_m[:,4]
        else:
            ur_c[0] = ur_p[0] + d_ur_b
            ur_c[-1] = ur_p[-1] + d_ur_s
            ut_c[0] = ut_p[0] + d_ut_b
            ut_c[-1] = ut_p[-1] + d_ut_s
            phi_c[0] = phi_p[0] + d_phi_b
            phi_c[-1] = phi_p[-1] + d_phi_s
            if N > 2:
                ur_c[1:-1] = ur_p[1:-1] + Sol_m[:,0]
                ut_c[1:-1] = ut_p[1:-1] + Sol_m[:,1]
                phi_c[1:-1] = phi_p[1:-1] + Sol_m[:,2]
                Sol_m[:,3] -= beta[:-1]*rsg*rho[-1]*(phi_p[1:-1]+r[1:-1]**l*Vp-g[1:-1]*ur_p[1:-1])
                trr_c = alpha[:-1]*trr_p + Sol_m[:,3]
                trt_c = alpha[:-1]*trt_p + Sol_m[:,4]
    
        # store solutions...
        if it%steps_for_sol == 0 or it == step - 1:
            if it%steps_for_sol == 0:
                j = int(it/steps_for_sol)
            if it == step - 1:            # also store the last step solution
                j = -1           
            t[j] = time_c
            t_yr[j] = time_c*tau_ref_yr
            phi_V[j] = rs**l*Vn
            ur[:,j] = ur_c
            ut[:,j] = ut_c
            phi[:,j] = phi_c
	    # 04-27-2017 output stress
            srr[0,j] = rsg*rhoc/rhom*(-phi_c[0]-rb**l*Vn+gb*ur_c[0])
            srr[-1,j] = 0
            srt[0,j] = 0
            srt[-1,j] = 0
            if N > 2:
                srr[1:-1,j] = trr_c
                srt[1:-1,j] = trt_c
       #     srr[:,j] -= rsg*rho[-1]*g*ur[:,j]  # from Lagragian to Eulerian
            if it == 0:
                visc[:,j] = eta_init
            else:
                visc[:,j] = eta
            
        time_c += dt_c                   # update current elapsed time    
        # update Vp for next time step
        Vp = Vn
        # record ur_p, phi_p, trr_p, trt_p from current time step
        ur_p = np.copy(ur_c)
        ut_p = np.copy(ut_c)
        phi_p = np.copy(phi_c)
        if N > 2:
            trr_p = np.copy(trr_c)
            trt_p = np.copy(trt_c)
    # end of loop for time
    
    # compute Love numbers at surface
    # (to compare with Zhong 2003, surface load has same density as mantle, H0 is load height)
    V0 = phi_V[-1]
    k1 = (phi[-1,:]+phi_V)/(rs**l*V0)  
    k = phi[-1,:]/(rs**l*V0)             
    if flag_tidal == 0:
        S0 = V0
        H0 = S0/drho_s 
        h = ur[-1,:]/H0         # Zhong 2003 notation
    if flag_tidal == 1:
        h = ur[-1,:]*gs/(rs**l*V0)    # load Love number notation
        f = ut[-1,:]*gs/(rs**l*V0)    # load Love number notation
    # l = Y[1]*gs/(rs**l*V0)
    #output = "k={0:f}, h={1:f}, l={2:f}".format(k,h,l)
    
    print(k[-1])
    print(h[-1])
    print(f[-1])
    t_end = tm.time()
    time_elapsed = (t_end - t_start)/60
    print("time used: "+str(time_elapsed)+" min")
    
    sol = {"time":t,"time_yr":t_yr,"h":h,"l":f,"k":k,"k1":k1,"a0":ai,"b":bi,"pre_dt":prefactor_dt,
           "V":phi_V,"k_end":k[-1],"h_end":h[-1],"l_end":f[-1],
           "step":step,"dsteps":dsteps,"eta_end":eta,"r":r,"Q0":Q620,"Qa":Q[ai],"at":ap/(phi_V**(1/3)),
	   "ur_t":ur,"ut_t":ut,"phi_t":phi,"srr_t":srr,"srt_t":srt,"visc_t":visc}
        
#    outfile = case+"a{0:.1f}_b{1:.1f}_".format(ai,bi)+dumpfile
#    pkl.dump(sol,open(outfile,"wb"))
    return sol

# loop over models
paras = []
t0 = tm.time()
for ind_a,ai in enumerate(a0):
    for ind_b,bi in enumerate(b):
        paras.append((ai,bi))

if __name__ == '__main__':
    with Pool() as p:
        solutions = p.map(time_evolution,paras)
    
outfile = data_dir+dumpfile
pkl.dump(solutions,open(outfile,"wb"))  
print("time used: "+str((tm.time()-t0)/3600)+" hrs")

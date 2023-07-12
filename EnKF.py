import numpy as np
import matplotlib.pyplot as plt

#%%
def lorenzrhs(xyz, *, s=10, r=28, b=2.667): # RHS of the system of Lorenz equations
	x, y, z = xyz
	x_dot = s*(y - x)
	y_dot = r*x - y - x*z
	z_dot = x*y - b*z
	return np.array([x_dot, y_dot, z_dot])

def rk4(func, xinm, dt): # Runge Kutta 4th order for a single timestep
	w1 = 1.0/6.0
	w2 = 1.0/3.0
	w3 = 1.0/3.0
	w4 = 1.0/6.0

	xxm1 = xinm # m stands for mean, one step evolution of mean, for all calls of rk4(,,,,,)
	fp = func(xxm1)
	x1 = dt * fp

	xxm2 = xinm + 0.5 * x1
	fp = func(xxm2)
	x2 = dt * fp

	xxm3 = xinm + 0.5 * x2
	fp = func(xxm3)
	x3 = dt * fp

	xxm4 = xinm + x3
	fp = func(xxm4)
	x4 = dt * fp

	return xinm + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4

def gcorr(J): # general purpose correlation matrix
	return np.identity(J)

def obspat(J): # returns observation operator hh
	return np.identity(J)

def plotaid(k):
	string = ['x', 'y', 'z', 'RMS', 'True RMS']
	axs[k].set_ylabel(string[k], fontsize=14)
	for axis in ['top','bottom','left','right']:
		axs[k].spines[axis].set_linewidth(2)
	axs[k].minorticks_on()
	axs[k].tick_params(labelsize = 14)
	axs[k].tick_params(direction = 'in', right = True, top = True)
	axs[k].tick_params(direction ='in', which = 'minor', length = 3, labelbottom=False)
	axs[k].tick_params(direction ='in', which = 'major', length = 7, labelbottom=False)
	axs[k].grid()

#%%
# Main code starts here
''' Parameters and initial conditions '''

dt = 0.01 # timestep size
ensize = 2 # size of ensemble
num_steps = 800 # total number of timesteps
foonksen = lorenzrhs # the nonlinear/linear model name, have to define a new function if you change model
jdim = 3 # the number of varibles being solved for
time = np.linspace(0, dt * (num_steps + 1), num_steps + 1) # time array
ID = np.identity(jdim) # identity matrix
wt = np.zeros(jdim) # truth
w0 = (1.508870,-1.531271,25.46091)  # Set initial values

#%%
''' Error covariances '''

# define general purpose error covariances of type gaussian
Cor0 = gcorr(jdim) # right now only an identity matrix
d, V = np.linalg.eig(Cor0); D = np.diag(d);  sCor0 = V * np.sqrt(D)

wt = w0 # initiate truth with initial values

stdA = 1.0
pa = stdA**2 * Cor0 # initial guess state / background - error covariance
B = pa # used oi

# initial 'guess' [20, 20, 20]; not the same as initial conditions
wa = np.array([20.0, 20.0, 20.0])#np.zeros(jdim)
waa = np.zeros((ensize,jdim))
wff = np.zeros((ensize,jdim))
stdEni = 2 * stdA
for en in range(0, ensize):
	waa[en,:] = wa + stdEni**2 * np.random.normal(0, np.sqrt(2), size=(jdim))

# %  Set specifics about model error
stdQ = 0 # zero model error
qq = stdQ**2 * Cor0 # model error covariance matrix

# Frequency of observations, every 25 timesteps
tobs  = 25

# Observation matrix = identity matrix
hh = obspat(jdim)

# Construct observation error covariance and observation matrix
stdO = 0.2
rr = stdO**2 * np.identity(jdim)

#%%
''' Start Main Time Loop '''

Va = np.sqrt(pa.trace()) # RMS error initialize
Aa = np.sqrt(np.dot(wt - wa, wt - wa))
WA = np.array(wa) # analysis storage
WA_max = np.array(wa) # analysis storage
WA_min = np.array(wa) # analysis storage
WT = np.array(w0) # truth storage
WO = [] # observation storage
T= [] # time array for observations, smaller in size and more spaced out than the total time array

t = 0
timesteps = 0

for i in range(num_steps):
	t = t + dt
	timesteps = timesteps + 1

	# Evolve true state
	wt = rk4(foonksen, wt, dt)

	# Evolve estimate
	for en in range(0, ensize):
		wff[en,:] = rk4(foonksen, waa[en,:], dt)
	wf = np.mean(wff, axis = 0) # calculate mean of background
	# Evolve covariance
	F = np.ones((ensize,jdim))
	for J in range(0, jdim):
		F[:,J] = wf[J]

	Fdif = F - wff
	pf = np.dot(Fdif.T,Fdif)/ensize

	# % --------------------------------------------------------------------
	# %                Observation/Assimilation
	# % --------------------------------------------------------------------
	if ( timesteps % tobs == 0): # if there is an observation

		# Generate observations by adding random error
		wo = hh.dot(wt) + stdO * np.random.normal(0, np.sqrt(2), size=(jdim)) # works

		# Generate ensemble of observations by adding random error to previous observation
		woo = np.zeros((ensize,jdim))
		for en in range(0, ensize):
			woo[en,:] = wo + stdO * np.random.normal(0, np.sqrt(2), size=(jdim))

		# Compute Kalman Gain
		kk = pf * hh.T * np.linalg.inv( hh * pf * hh.T + rr ) # works

		# Update State and Error Covariance
		for en in range(0, ensize):
			waa[en,:] = wff[en,:] + kk.dot( woo[en,:] - hh.dot(wff[en,:]))
		pa = ( ID - kk * hh ) * pf #* ( ID - kk * hh ).T  +  kk * rr * kk.T # works

		# Append observations and it's corresponding time to plot later
		WO = np.append(WO, wo)
		T = np.append(T, t)

	else: # if there is NO observation, simply update using model
		waa = wff
		wa = np.mean(waa, axis = 0)
		pa = pf

	# True analysis error covariance
	A = np.ones((ensize,jdim))
	for J in range(0, jdim):
		A[:,J] = wa[J]

	err_true = np.sqrt(np.dot(wt - wa, wt - wa)/jdim)

	# Arrays to store analysis, truth and RMS error
# 	print(np.amax(waa, axis = 0))
# 	print(np.amin(waa, axis = 0))
	WA_max = np.append(WA_max, np.amax(waa, axis = 0))
	WA_min = np.append(WA_min, np.amin(waa, axis = 0))
	WA = np.append(WA, wa)
	WT = np.append(WT, wt)
	Va = np.append(Va, np.sqrt(pa.trace()))
	Aa = np.append(Aa, err_true)

# Reshape arrays to obtain x, y and z individually
WT = np.reshape(WT,(-1, jdim))
WA_max = np.reshape(WA_max,(-1, jdim))
WA_min = np.reshape(WA_min,(-1, jdim))
WA = np.reshape(WA,(-1, jdim))
WO = np.reshape(WO,(-1, jdim))

#%%
''' Plot states in time '''

fig, axs = plt.subplots(jdim + 2, figsize=(2.5,7))

for i in range(0,jdim):
	axs[i].plot(T, WO[:,i], '*', color = 'b', mfc = 'w', markersize = 7)
	plotaid(i)
axs[3].plot(time, Va, '-', color = 'k')
plotaid(3)
# axs[3].tick_params(labelbottom=True)
# axs[3].set_xlabel('time', fontsize=14)
axs[3].set_yscale('log')

axs[4].plot(time, Aa, '-', color = 'k')
plotaid(4)
axs[4].tick_params(labelbottom=True)
axs[4].set_xlabel('time', fontsize=14)
axs[4].set_yscale('log')

for i in range(0,jdim):
	axs[i].plot(time, WA[:,i], color = 'r', linewidth = 2)
	axs[i].fill_between(time, WA_max[:,i], WA_min[:,i], color = 'grey')
# 	axs[i].plot(time, WA_max[:,i], color = 'g', linewidth = 2)
# 	axs[i].plot(time, WA_min[:,i], color = 'g', linewidth = 2)
	axs[i].plot(time, WT[:,i], color = 'k')
fig.tight_layout()

# ax = plt.figure(figsize=(5,5)).add_subplot(projection='3d')
# ax.plot(*WA.T, lw=1.5, color='r')
# ax.plot(*WT.T, lw=1.5, color='k')
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

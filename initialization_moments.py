import numpy as np
import matplotlib.pylab as plt
from scipy.stats import lognorm, norm
#np.random.seed(1)
## Hidden states definition
nb_Z = 10          #Number of variables
s_M  = 1           #Prior std mean
s_Z  = 0.5           #Prior std std

mZ = [] #Initialize mean & std of hidden units
sZ = []
for i in range(nb_Z):
    mZ.append(np.random.normal(loc=0,scale=s_M))
    sZ.append(s_Z)

mZ = np.array(mZ)
sZ = np.array(sZ)
s2Z = sZ**2

## MCS verification
nb_mcs = 100000000
Z_s = np.random.normal(loc=mZ,scale=sZ,size=(nb_mcs,nb_Z))


mS_mcs = np.average(np.sum(Z_s,axis=1),axis=0)
s2S_mcs = np.var(np.sum(Z_s,axis=1),axis=0)

print('mS  (MCS):', mS_mcs)
print('s2S (MCS):', s2S_mcs)

mS_ana = 0
s2S_ana = nb_Z*(s_Z**2)

print('mS  (ana):', mS_ana)
print('s2S (ana):', s2S_ana)

mS2_mcs = np.average(np.sum(Z_s**2,axis=1),axis=0)
s2S2_mcs = np.var(np.sum(Z_s**2,axis=1),axis=0)

print('mS2  (MCS):', mS2_mcs)
print('s2S2 (MCS):', s2S2_mcs)

mS2_ana = nb_Z*(s_M**2+s_Z**2)
s2S2_ana = nb_Z*(2*s_Z**4+4*s_M**2*s_Z**2)

print('mS2  (ana):', mS2_ana)
print('s2S2 (ana):', s2S2_ana)
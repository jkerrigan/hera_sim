import numpy as np
from hera_sim.SimEnviron import SimEnviron
import pylab as pl
from itertools import product
import h5py
from hera_qm import xrfi
import pyuvdata
import sys
# generate visibility data
# w/ standard IDR1 time/freq samples

save = False
fname = 'SimVis_v5.h5'
realizations = 2
data_array = []
flag_array = [] #[Nbls,Nt,Nf]
plot = True
files_ = sys.argv[1:]
print files_

def time_avg(data):
    t_avg = np.mean(data,0)
    return data - t_avg

def normscale(data):
    mu = np.mean(data)
    data -= mu
    data /= data.max()
    return data

def meansubtime(data):
    times,freqs = np.shape(data)
    for i in range(freqs):
        data[:,i] -= np.mean(data[:,i])
    data /= np.var(data)
    return data

def meansubfreq(data):
    times,freqs= np.shape(data)
    for i in range(times):
        data[i,:] -= np.mean(data[i,:])
    data /= np.var(data)
    return data

def timediff(data):
    tdiff = np.copy(data)
    times,freqs= np.shape(data)
    for i in range(times-1):
        tdiff[i,:] = data[i+1,:] - data[i,:]
    tdiff /= np.nanmax(np.abs(tdiff)[~np.isinf(np.abs(tdiff))])
    return tdiff

def freqdiff(data):
    fdiff = np.copy(data)
    times,freqs= np.shape(data)
    for i in range(freqs-1):
        fdiff[:,i] = data[:,i+1] - data[:,i]
    fdiff /= np.nanmax(np.abs(fdiff)[~np.isinf(np.abs(fdiff))])
    return fdiff


def preprocess(data):
   # data array size should be (batch,time,freq)
   data_a = np.copy(np.array(data))
   data_b = np.copy(np.array(data))
   t_num,f_num = np.shape(data)
   data_a = freqdiff(data_a)   
   data_b = timediff(data_b)
   return data_a,data_b #reshape(batch,t_num,f_num,2)

def preprocess3(data):
   # data array size should be (batch,time,freq)                                                                                               
   data_a = np.copy(data)
   batch,t_num,f_num = np.shape(data)
   # initialize output array                                                                                                                   
   data_out = np.zeros((t_num,f_num,2))
#   for b in range(batch):
   data_ = np.copy(data)
   data_ -= np.nanmean(data_)
   data_ /= np.nanmax(np.abs(data_))
   data_out[:,:,0] = np.log10(np.abs(data_))
   data_out[:,:,1] = np.angle(data_)
   return np.nan_to_num(data_out)

def preprocess2(data):
    data = np.copy(data)
    data -= np.nanmean(data)
    data /= np.nanmax(np.abs(data))
    return np.nan_to_num(np.log10(np.abs(data)))

def clog(data):
    c_data = np.log(np.abs(data))+1j*np.angle(data)
    return np.abs(c_data)

#uv = pyuvdata.miriad.Miriad()
#uv.read_miriad('zen.2457555.40356.xx.HH.uvcT')

ct = 0
while ct < realizations:
    print(ct)
    bsl_len = np.random.normal(loc=50.,scale=75.)
    if bsl_len < 40.:
        continue
    hera = SimEnviron()
    hera.gen_HERA_vis(61,1024,bl_len_ns=bsl_len,add_rfi='../hera_sim/rfi_sims/rfi2.yml')
    #AT.plot_vis(rfi=True)
    data_array.append(hera.return_vis(rfi=True)[:60,:])#np.nan_to_num(uv.get_data(ap)))
    #flag_array.append(AT.return_flags(rfi=rfi)[:60,:])#xrfi.xrfi(np.abs(uv.get_data(ap))))
    flag_array.append(hera.return_flags(rfi=True)[:60,:])
    gain_flucts = np.random.normal(loc=1.,scale=np.pi/8,size=(60,1024))
    if plot:
        pl.subplot(311)
        pl.imshow(np.log10(np.abs(data_array[ct]*gain_flucts)),aspect='auto')
        pl.colorbar()
        pl.subplot(312)
        pl.imshow(np.angle(data_array[ct]*gain_flucts),aspect='auto')
        pl.colorbar()
        pl.subplot(313)
        #pl.imshow(flag_array[ct],aspect='auto')
        pl.colorbar()
        pl.show()
    ct+=1

#    del(uv)
#    pre_data_t,pre_data_f = preprocess(np.log10(np.abs(data_array[0])))
    #print np.shape(pre_data)
#    out2 = (pre_data_t)
#    out1 = (pre_data_f)
#    pl.subplot(311)
#    pl.imshow(np.log10(np.abs(data_array[0])),aspect='auto')
#    pl.colorbar()
#    pl.subplot(312)
#    pl.imshow(np.abs(out1),aspect='auto')
#    pl.colorbar()
#    pl.subplot(313)
#    pl.imshow(np.abs(out2),aspect='auto')
#    pl.colorbar()
#    pl.show()
#    del(AT)
print 'Total of Training set: ',ct
    
#data_array = np.array(data_array)
#flag_array = np.array(flag_array)
print np.shape(data_array),np.shape(flag_array)
if save:
    h5f = h5py.File(fname, 'w')
    h5f.create_dataset('data', data=data_array)
    h5f.create_dataset('flag', data=flag_array)
    h5f.close()






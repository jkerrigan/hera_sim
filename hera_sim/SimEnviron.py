import numpy as np
import pylab as pl
import hera_qm as hq
import time
from hera_sim import foregrounds,noise,sigchain,rfi
import aipy as a
from scipy import signal
from scipy.ndimage import gaussian_filter
from FRBSim import gen_simulated_frb
import yaml
from rfi_alt import RFI_Sim

class SimEnviron:

    def __init__(self):
        pholder = None

    def gen_HERA_vis(self,tsamples,fsamples,bl_len_ns=400.,add_rfi=False,inject_frb=False):
        #### Convert time samples to appropriate LST hours where 60 time samples = 10 min
        #### !!!! LST is in rads not hrs
        sph = 60./.1667
        lst_add = tsamples/sph
        fqs = np.linspace(.1,.2,fsamples,endpoint=False)
        lsts = np.linspace(np.pi/2.,np.pi/2. + lst_add,tsamples)
        times = lsts / (2.*np.pi) * a.const.sidereal_day

        #### FOREGROUNDS ####
        # Diffuse
        Tsky_mdl = noise.HERA_Tsky_mdl['xx']
        vis_fg_diffuse = foregrounds.diffuse_foreground(Tsky_mdl, lsts, fqs, bl_len_ns)
        # Point Sources
        vis_fg_pntsrc = foregrounds.pntsrc_foreground(lsts, fqs, bl_len_ns, nsrcs=1000)
        # FRBs
        vis_fg_frb = np.asarray(gen_simulated_frb(NFREQ=1024,NTIME=61,width=1.,sim=True,delta_t=10.,freq=(200,100),FREQ_REF=150.,fluence=(60.,600.),scintillate=True,dm=(300., 1800.))[0].T,dtype=np.complex128)
        vis_fg_frb *= np.exp(1j*(lsts[30]+.1*np.pi*np.random.randn(61,1024)))
        # Combined
        if inject_frb:
            vis_fg = vis_fg_diffuse + vis_fg_pntsrc + vis_fg_frb
        else:
            vis_fg = vis_fg_diffuse + vis_fg_pntsrc
        #### Noise ####
        tsky = noise.resample_Tsky(fqs,lsts,Tsky_mdl=noise.HERA_Tsky_mdl['xx'])
        t_rx = 150.
        nos_jy = noise.sky_noise_jy(tsky + t_rx, fqs, lsts)
        # Add Noise
        vis_fg_nos = vis_fg + nos_jy

        #### RFI ####
        if add_rfi:
            g = sigchain.gen_gains(fqs, [1,2,3])
            with open(add_rfi,'r') as infile:
                RFIdict = yaml.load(infile)
            rfi = RFI_Sim(RFIdict)
            rfi.applyRFI()
            self.rfi_true = rfi.getFlags()
            vis_fg_nos_rfi = np.copy(vis_fg_nos) + rfi.getRFI()
            vis_total_rfi = sigchain.apply_gains(vis_fg_nos_rfi, g, (1,2))
            self.data_rfi = vis_total_rfi
        else:
            g = sigchain.gen_gains(fqs, [1,2,3])
            vis_total_norfi = sigchain.apply_gains(vis_fg_nos, g, (1,2))
            self.data = vis_total_norfi

    def time_RFIalg(self,alg,pholder):
        if alg == 'xrfi':
            t0 = time.time()
            hq.xrfi.xrfi(np.abs(self.data),Kt=pholder,Kf=pholder)
            return time.time()-t0
        if alg == 'detrend_medminfilt':
            t0 = time.time()
            hq.xrfi.detrend_medminfilt(np.abs(self.data),Kt=pholder,Kf=pholder)
            return time.time()-t0
    
    def run_RFIalg(self,alg,*args):
        if alg == 'xrfi':
            try:
                self.flags_rfi = np.array(hq.xrfi.xrfi(np.abs(self.data_rfi),*args[0]))
            except:
                pass
            self.flags = np.array(hq.xrfi.xrfi(np.abs(self.data),*args[0]))
        elif alg == 'xrfi_simple':
            try:
                self.flags_rfi = np.array(hq.xrfi.xrfi_simple(np.abs(self.data_rfi),*args[0]))
            except:
                pass
            self.flags = np.array(hq.xrfi.xrfi_simple(np.abs(self.data),*args[0]))
        elif alg == 'watershed':
            try:
                self.flags_rfi = np.array(hq.xrfi.watershed_flag(np.abs(self.data_rfi)))
            except:
                pass
            self.flags = np.array(hq.xrfi.watershed_flag(np.abs(self.data)))


    def count_RFI(self,ret=False):
        try:
            empty_ct = np.sum(self.flags)/(1.*np.size(self.flags))
        except:
            pass
        try:
            rfi_ct = np.sum((self.rfi_true.astype(int)+self.flags_rfi.astype(int)) == 2)/(1.*np.sum(self.rfi_true))            
            false_pos = np.sum((self.rfi_true.astype(int)-self.flags_rfi.astype(int)) == -1)/(1.*np.size(self.rfi_true))

            false_neg = np.sum((self.rfi_true.astype(int)-self.flags_rfi.astype(int)) == 1)/(1.*np.size(self.rfi_true))
            if ret:
                return rfi_ct,false_pos,false_neg
            else:
                print 'No RFI Baseline: '+str(100.*empty_ct)+'% ('+str(np.sum(self.flags))+')'
                print 'RFI Correctly Found:  '+str(100.*rfi_ct)+' %'
                print 'False Positive Rate:  '+str(100.*false_pos)+' %'
                print 'False Negative Rate:  '+str(100.*false_neg)+' %'
                
        except:
            pass

    def plot_flags(self,rfi=False):
        if rfi:
            ### Lets plot correct flags as +1 and incorrect as -1, no flags should be 0
            _flags = np.zeros_like(self.flags_rfi).astype(int)
            _flags[self.rfi_true==0] = 1
            _flags[(self.rfi_true==1)==(self.flags_rfi==0)] = -1
            pl.imshow(_flags,aspect='auto')
            pl.show()
        else:
            pl.imshow(self.flags,aspect='auto')
            pl.show()

    def return_vis(self,rfi=False):
        if rfi:
            return self.data_rfi
        else:
            return self.data

    def return_flags(self,rfi=False):
        if rfi:
            return self.rfi_true
        else:
            return np.zeros_like(self.rfi_true)
        
    def plot_vis(self,rfi=False):
        if rfi:
            pl.imshow(np.log10(np.abs(self.data_rfi)),aspect='auto',cmap='jet')
            pl.colorbar()
            pl.show()
            pl.plot(np.log10(np.abs(self.data_rfi[30,:])))
            pl.show()
        else:
            pl.imshow(np.log10(np.abs(self.data)),aspect='auto',cmap='jet')
            pl.show()


    def logfit(self,x,y):
        ### Use a MSE loss function to iterate over a parameter space
        ### form of m*x*log(n*x) + c
        x = np.array(x)
        y = np.array(y)
        m = np.logspace(-3,4,100)
        n = np.logspace(-3,4,100)
        c = np.linspace(np.mean(y),2.*np.mean(y),100)
        sum_res = np.zeros((100,100,100))
        ct_m = 0
        for i in m:
            ct_n = 0
            for j in n:
                ct_c = 0
                for k in c:
                    sum_res[ct_m,ct_n,ct_c] = np.sum(np.abs(i*x*np.log10(j*x)+k - y))
        
        min_m = m[np.where(np.min(sum_res)==sum_res)[0][0]]
        min_n = n[np.where(np.min(sum_res)==sum_res)[1][0]]
        min_c = c[np.where(np.min(sum_res)==sum_res)[2][0]]
        print min_m,min_n,min_c
        return min_m*x*np.log10(min_n*x)+min_c

    def logloss(self):
        eps = 1e-15
        clipped = np.clip(self.flags_rfi, eps, 1-eps)
        ifone = -1.*np.log(clipped[self.rfi_true==1])
        ifzero = -1.*np.log(1-clipped[self.rfi_true==0])
        return np.sum(ifone)+np.sum(ifzero)

    def logloss2(self):
        eps = 1e-15
        clipped = np.clip(self.flags_rfi, eps, 1-eps)
        ifone = -1.*np.log(clipped[self.rfi_true==1])
        ifzero = -1.*np.log(1-clipped[self.rfi_true==0])
        corr, fp, fn = self.count_RFI(ret=True)
        return (np.sum(ifone)+np.sum(ifzero))*np.exp((fp+fn)/corr)

    def MSE(self):
        return np.sum(np.abs(self.rfi_true.flatten()-self.flags_rfi.flatten()))

    def MinFPFN(self):
        corr, fp, fn = self.count_RFI(ret=True)
        return (fp+fn)/corr


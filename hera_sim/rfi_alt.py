import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import scipy.ndimage
import scipy

class RFI_Sim():
    
    def __init__(self,RFIdict,telescope='HERA'):
        self.RFIdict = RFIdict
        if telescope == 'HERA':
            self.chans = 1024
            self.times = 61
        self.baseline_data = np.zeros((self.times,self.chans),dtype=np.complex64)

    def scatterRFI(self):
        #Collect parameters From RFIdict
        max_times = self.RFIdict.get('scatter').get('max_times')
        max_freq = self.RFIdict.get('scatter').get('max_freqs')
        mu = self.RFIdict.get('scatter').get('mu')
        sigma = self.RFIdict.get('scatter').get('sigma')
        points = self.RFIdict.get('scatter').get('points')
        delta_times = self.RFIdict.get('scatter').get('delta_times')
        period = self.RFIdict.get('scatter').get('period')
        #Defines range of times and frequencies. Defines phase
        times = np.random.randint(0,max_times,size=points)
        freq = np.random.randint(0,max_freq,size=points)
        #Apply a rough bleed to all scatter

        #Loops for each point to be adjusted
        for i in range(points):
            scatter_canvas_phs = np.zeros((self.times,self.chans))
            scatter_canvas_amp = np.zeros((self.times,self.chans))
            #sinusoidal_noise = (np.cos((((2. * np.pi)/(period)) * delta_times) * i ))
            amplitude = np.random.normal(loc=mu,scale=sigma)
            phase = np.random.normal(loc=0, scale=np.pi)
            scat_signal = amplitude * np.exp(1j*phase)
            rnd_bleed = np.random.rand()
            scatter_canvas_amp[times[i],freq[i]] = amplitude 
            scatter_canvas_phs[times[i],freq[i]] = phase
            scatter_canvas_amp = scipy.ndimage.gaussian_filter(scatter_canvas_amp,sigma=rnd_bleed)
            scatter_canvas_phs = scipy.ndimage.gaussian_filter(scatter_canvas_phs,sigma=rnd_bleed)
            self.baseline_data += scatter_canvas_amp*np.exp(1j*scatter_canvas_phs)


    def narrowbandRFI(self):
        #From RFIdict
        raw_packages = self.RFIdict.get('narrowband').get('raw_packages')
        delta_times = self.RFIdict.get('narrowband').get('delta_times')
        max_times = self.times#self.RFIdict.get('narrowband').get('max_times')
        f = np.linspace(.1,.2,self.chans) #freqs in GHz
        #Defines times
        times = range(max_times)
        for package in raw_packages:
            channel = np.argmin(np.abs(f - package.get('frequency')))
            #From RFIdict, raw_packages
            mu = package.get('mu')
            sigma = package.get('sigma')
            period = package.get('period')

            #Periodic component
            sinusoidal_noise = (np.cos((((2. * np.pi)/(period)) * delta_times) * np.asarray(times, dtype = float)))
            amplitude_canvas = np.zeros((self.times,self.chans))
            amplitude_canvas[times,channel] = sinusoidal_noise * np.random.normal(loc=mu, scale=sigma, size=self.times)

            phase_canvas = np.zeros((self.times,self.chans))
            phase_canvas[times,channel] = np.random.normal(loc=0, scale=np.pi, size=self.times)

            if(package.has_key('bleed')):
                # Apply channel bleed (aka Gaussian blur)
                bleed = package.get('bleed')
                phase_gaus = scipy.ndimage.gaussian_filter(phase_canvas, sigma=bleed)
                amplitude_gaus = scipy.ndimage.gaussian_filter(amplitude_canvas, sigma=bleed)
                complex_conjugate = amplitude_gaus * np.exp(1j*phase_gaus)
            else:
                complex_conjugate = amplitude_canvas * np.exp(1j*phase_canvas)

            #Increases signal along entire frequency channel
            #baseline_data += complex_conjugate
            self.baseline_data += complex_conjugate

    def burstRFI(self):
        #From RFIdict
        if self.RFIdict.get('burst').get('raw_packages')[0].has_key('random'):
            random_bursts = True
            raw_packages = range(np.random.randint(10))
        else:
            raw_packages = self.RFIdict.get('burst').get('raw_packages')
            random_bursts = False
        max_times = self.RFIdict.get('burst').get('max_times')
        delta_times = self.RFIdict.get('burst').get('delta_times')
        f = np.linspace(.1,.2,1024)
        #iterate through all raw_packages
        for package in raw_packages:
            #Takes the frequency of package and sets the frequency channel
            #From RFIdict, for current package
            if random_bursts:
                channel = np.argmin(np.abs(f - np.random.uniform(.1,.2)))
                mu = np.abs(np.random.normal(loc=1000.,scale=10000.))
                sigma = np.abs(np.random.normal(loc=1000.,scale=1000.))
                start_time = np.random.randint(0,self.times)
                duration = np.abs(int(np.random.normal(loc=10,scale=10)))
            else:
                channel = np.argmin(np.abs(f - package.get('frequency')))
                mu = package.get('mu')
                sigma = package.get('sigma')
                period = package.get('period')
                start_time = package.get('start_time')
                duration = package.get('duration')
            #Half the duration or 1 to avoid  division by zero
            max_taper = (duration/2) if (duration >= 2) else (1)
            #Randomizes the taper length without exceeding the full length of max taper
            taper_length = np.random.randint(0,max_taper)
            #Center frequency that tapers step up towards
            center_strength = int(mu+sigma)
            amplitude_canvas = np.zeros((self.times,self.chans))
            phase_canvas = np.zeros((self.times,self.chans))
            #Generates signal for time duration on specific frequency
            rnd_bleed = np.random.uniform(.0,.5)
            #for i in range(duration):
                #Position amplitude is being applied to
                #time_coordinate = i + start_time + 1
                #Phase of signal
            phase = np.random.normal(loc=0, scale=np.pi,size=duration)
                #Amplitude of signal
            amplitude = np.random.normal(loc=mu, scale=sigma,size=duration)
                #Combination of amplitude (real number) and phase (imaginary number)
                #Stops loop when time coordinate exceeds max time integrations
                #if(time_coordinate >= max_times):
                #    break
                #Applies starting taper
                #elif(i < taper_length):
                    #Value that amplitude is multiplied by to produces taper in linear symmetrical pattern
                #    taper_multiplier =  (center_strength * ((i+1) / float(taper_length+1)))
                #    amplitude_canvas[time_coordinate,channel] += amplitude*taper_multiplier
                #    phase_canvas[time_coordinate,channel] += phase*taper_multiplier
                    #Increases specific point in baseline to combined signal
                    #self.baseline_data += complex_canvas
                #Applies ending taper
                #elif(duration - i <= taper_length):
                    #Value that intensity is multiplied by to produces taper in linear symmetrical pattern
                #    taper_multiplier = (center_strength * ((duration - i+1) / float(taper_length+1)))
                    #Increases specific point in baseline to combined signal
                #    amplitude_canvas[time_coordinate,channel] += amplitude*taper_multiplier
                #    phase_canvas[time_coordinate,channel] += phase*taper_multiplier
                    #complex_canvas[time_coordinate,channel] *= taper_multiplier
                    #self.baseline_data += complex_canvas
                #Applies center strength signal
                #else:
            # This is a quick fix
            try:
                amplitude_canvas[start_time:start_time+duration,channel] += amplitude*scipy.signal.gaussian(duration,1.)
                phase_canvas[start_time:start_time+duration,channel] += phase*scipy.signal.gaussian(duration,1.)
                  #Increases specific point in baseline to combined signal
            except:
                pass
                    #self.baseline_data += complex_canvas
            amplitude_canvas = scipy.ndimage.gaussian_filter(amplitude_canvas,sigma=rnd_bleed)
            phase_canvas = scipy.ndimage.gaussian_filter(phase_canvas,sigma=rnd_bleed)
            self.baseline_data += amplitude_canvas*np.exp(1j*phase_canvas)

    def applyRFI(self):
        if self.RFIdict.has_key('narrowband'):
            self.narrowbandRFI()
        if self.RFIdict.has_key('scatter'):
            self.scatterRFI()
        if self.RFIdict.has_key('burst'):
            self.burstRFI()
                
    def getRFI(self):
        return self.baseline_data

    def getFlags(self):
        return np.where(np.abs(self.baseline_data) != 0., np.ones_like(self.baseline_data).astype(np.float32),np.zeros_like(self.baseline_data).astype(np.float32))

from scipy.interpolate import interp1d
import numpy as np


try:  # PySM >= 3.2.1
    import pysm3.units as u
except ImportError:
    import pysm.units as u

import mapsims.noise_v2

# An example simple survey with external noise curves and hitsmaps.
class surveyFromExternalData:
    def __init__(self,nbands,fwhms,noise_files,hitsmap_filenames,ivar_map_filenames=None,white_noises=None):
        """[summary]
        
        [description]
        
        Arguments:
            nbands {[int]} -- the number of bands to be passed
            fwhms {[float]} -- a list of the fwhms for each band
            noise_files {[type]} -- a list of the noise levels for each band
        
            hitsmap_filenames {[type]} -- [description] (default: {None})
        Keyword Arguments:
            ivar_map_filenames {[type]} -- If you have external ivar_maps pass the filenames as a list here (default: {None})
            white_noises {[type]} -- a list of the white noise levels for each band. Note this is not necessary for basic functionality
                                    However if not passed some functions of the noise class will not work. E.g. rescaling the white noise level
                                    of generating an ivar_map.
        """
        self.nbands = nbands
        self.fwhms = np.array(fwhms)
        self.white_noises = white_noises
        self.hitsmap_filenames = hitsmap_filenames
        self.ivar_map_filenames = ivar_map_filenames
        self.noise_files = noise_files
        if (len(fwhms)!=nbands): raise AssertionError("Each band needs a fwhm")
        if(len(noise_files)!=nbands): raise AssertionError("Each band needs a noisefile")
        if(len(hitsmap_filenames)!=nbands): raise AssertionError("Each band needs a hitsmap_filename")
        if white_noises is not None:
            self.white_noises = np.array(white_noises)
            if(len(white_noises)!=nbands): raise AssertionError("Each band needs a whitenoise level")
        if ivar_map_filenames is not None:
            if(len(ivar_map_filenames)!=nbands): raise AssertionError("Each band needs a ivar_map_filenames")
   
    def get_beams(self):
        return self.fwhms
    
    def get_hitsmap_filenames(self):
        return self.hitsmap_filenames
    
    def get_ivar_map_filenames(self):
        return self.ivar_map_filenames

    def get_white_noise(self,f_sky, units='arcmin2',*args,**kwargs):
        if self.white_noises is None:
            return None
        A = self.white_noises**2
        if units == 'arcmin2':
            A *= (60*180/np.pi)**2
        elif units != 'sr':
            raise ValueError("Unknown units '%s'." % units)
        return A
    
    def get_noise_curves(self, f_sky, ell_max, delta_ell, deconv_beam=True,
                         full_covar=False, rolloff_ell=None):
        ell = np.arange(2, ell_max, delta_ell)
        T_out = np.zeros([self.nbands,self.nbands,ell.shape[0]])
        P_out = np.zeros([self.nbands,self.nbands,ell.shape[0]])
        if np.shape(self.noise_files)==2:
            for i in range(self.nbands):
                for j in range(self.nbands):
                    noise_ells,noise_T,noise_P = np.loadtxt(self.noise_files[i][j],unpack=True)
                    T_out[i,j] = interp1d(noise_ells,noise_T,bounds_error=False,fill_value=0)(ell)
                    P_out[i,j] = interp1d(noise_ells,noise_P,bounds_error=False,fill_value=0)(ell)
            
        else:
            for i in range(self.nbands):
                noise_ells,noise_T,noise_P = np.loadtxt(self.noise_files[i],unpack=True)
                T_out[i,i] = interp1d(noise_ells,noise_T,bounds_error=False,fill_value=0)(ell)
                P_out[i,i] = interp1d(noise_ells,noise_P,bounds_error=False,fill_value=0)(ell)
        return (ell, T_out, P_out)


baseDir =mapsims.__path__[0]+'/tests/data/'

# Create your survey pass the noise files, hitsmaps_filenames and ivar maps (if this is desired)
survey = surveyFromExternalData(1,fwhms=[1.],noise_files=[baseDir+'/example_extern_noise_curves.txt'],hitsmap_filenames = [baseDir+'/example_hitsmaps_nside_16.fits'])

# Create a channel. Set the noise_band_index to the corresponding index of the noise_file and hitsmpa_filenames lists. 
ch_0 = mapsims.channel_utils.Channel(tag='AA',telescope='AA'
                                        ,band='150',tube='PA1',beam=1.5*u.arcmin,
                                        center_frequency=150*u.GHz,noise_band_index=0,
                                       )


noiseSim_test = mapsims.noise_v2.ExternalNoiseSimulator(nside= 16,channels_list=[ch_0],survey=survey)

# Simulate a tube
noiseSim_test.simulate('PA1')

# File: norread_channel_extraction
# Author: Rick Morrison
# Copyright 2015 SocialEyes Corporation
'''
OpenCV/Python/Numpy code for retinal image enhancement
The conversion to c/c++ code will be relatively straight-forward.

There are 5 example code sets that are similar to that which will
be implemented for the tablet based analysis. These examples are:

* example1_color_enhance - Improve display of initial image
* example2_nored -         Zero out image red channel and convert to gray scale
* example3_tone -          Improve contrast and gamma of nored image
* example4_sharpen -       Apply unsharp mask to nored image
* example5_deconvolve -    Apply deconvolution

Each example can be passed either a image data array or read an image data file.

The first several functions are the example code. This is followed by
a class that implements the functionality and finishing with the main program.

The examples either rely on the retinal_image_processing class or when
they are suffixed with explicit they stand separate from the class.

'''

# Libraries used in analysis
import numpy as np
import cv2

#-------------------------------------------------------------------------------------------------------

def example3_tone (imgChannel=None):     # Adjust contrast and gamma
    if (imgChannel==None):  imgChannel = cv2.imread ('nored_output.png') 
    if (imgChannel==None):  print 'Image file data not found and read; Abort'; return None, None, None
    imgChannel = imgChannel[:,:,0]
    rip = retinal_image_processing()
    rip.set_image_channel(imgChannel)
    # Create curves
    hist, histcum, nbins = rip.extract_channel_histogram (imgChannel)
    histcum = histcum / histcum.max()
    mn = histcum[histcum<0.01].size
    mx = histcum[histcum<0.99].size
    tcurve = hist.copy()
    tcurve[0:mn-1] = 0
    for i in range(mn,mx):       tcurve[i] = (nbins-1)*(i-mn)/(mx-mn)
    tcurve[mx:nbins-1] = (nbins-1)
    tcurve0 = hist.copy()
    for i in range(nbins):   tcurve0[i] = i

    tmp = rip.apply_channel_tone_curve(tcurve0, imgChannel)  # No corrections
    imgChannel0 = rip.apply_channel_gamma_correction(1.0, tmp)
    
    tmp = rip.apply_channel_tone_curve (tcurve, imgChannel)
    imgChannel1 = rip.apply_channel_gamma_correction(0.8, tmp) 
 
    tmp = rip.apply_channel_tone_curve (tcurve, imgChannel)      
    imgChannel2 = rip.apply_channel_gamma_correction(1.2, tmp) 
        
    return imgChannel0, imgChannel1, imgChannel2

def display (name, imgset):   
    cv2.imshow(name, imgset)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
#-------------------------------------------------------------------------------------------------------
class retinal_image_processing ():
    
    def __init__(self):
        self.img        = None
        self.imgChannel = None
        self.filename   = None
        self.tone_curve = None
        self.gamma_value= None
    
    def read_image(self, filename):     # Read image using opencv
        self.img = cv2.imread (filename)      
        return self.img
    
    def write_image(self, filename, img=None):    # Write image data using opencv
        if (img!=None): return cv2.imwrite(filename, img)
        else: return cv2.imwrite(filename, self.img)
        
    def write_image_channel(self, filename, imgChannel=None):
        if (imgChannel!=None): return cv2.imwrite(filename, imgChannel)
        else: return cv2.imwrite(filename, self.imgChannel)
        
    def set_image(self, img):           # Send in image data
        self.img = img.copy()
        return
    
    def set_image_channel(self,imgChan):
        self.imgChannel = imgChan.copy()
        return
    
    def unsharp_mask_image_channel(self, imgChannel):
        if (imgChannel!=None):       self.imgChannel = imgChannel.copy()
        if (self.imgChannel==None):  return None
        return self.img_channel
    
    def extract_red(self, img):    
        if (img!=None):       self.img = img.copy()
        if (self.img==None):  return None          # red channel to imgChannel
        return self.extract_rgb_channel(rgbChan1=2)
    
    def extract_green(self, img):            # green channel to imgChannel
        if (img!=None):       self.img = img.copy()
        if (self.img==None):  return None
        return self.extract_rgb_channel(rgbChan1=1)
        
    def extract_blue(self, img):             # blue channel to imgChannel
        if (img!=None):       self.img = img.copy()
        if (self.img==None):  return None
        return self.extract_rgb_channel(rgbChan1=0)
    
    def extract_nored(self, img=None):           # green and blue channel to imgChannel
        if (img!=None):       self.img = img.copy()
        if (self.img==None):  return None
        return self.extract_rgb_channel(img, rgbChan1=23)
         
    def extract_rgb_channel(self, img=None, rgbChan1=2):    # conversion engine
        if (img !=None):      self.img = img.copy()
        if (self.img==None):  return None
        # Note: cv2 images are stored as bgr format
        if (rgbChan1==0 | rgbChan1==1 or rgbChan1==2):
            self.imgChannel = (self.img[:,:,rgbChan1]).copy() 
        elif (rgbChan1==23):
            tmp = self.img.copy()
            tmp[:,:,2] = 0
            self.imgChannel = cv2.cvtColor (tmp, cv2.COLOR_BGR2GRAY)
       
        return self.imgChannel
    
    def extract_L_channel(self, img=None):  # L channel from L*a*b color space
        return      # Later
    
    def extract_channel_histogram(self, imgChannel=None):
        if (imgChannel !=None):      self.imgChannel = imgChannel.copy()
        if (self.imgChannel==None):  return None
        nbins = 100
        if self.imgChannel.dtype == np.uint8:  nbins = 256
        if self.imgChannel.dtype == np.uint16: nbins = pow(2,16)
        #hist, bins = np.histogram(self.imgChannel.flatten(), nbins, [0,nbins])
        hist = cv2.calcHist ([self.imgChannel], [0], None, [nbins], [0,nbins])
        histcum = hist.cumsum()
        return hist, histcum, nbins     
    
    def apply_channel_tone_curve (self, curve=None, imgChannel=None):
        if (imgChannel!=None):      self.imgChannel = imgChannel
        if (self.imgChannel==None): return None        
        if (curve!=None): self.tone_curve = curve.copy()
        if (self.tone_curve==None): return None
        tmp = self.tone_curve[self.imgChannel]
        self.imgChannel = tmp.astype(self.imgChannel.dtype)
        return self.imgChannel
        
    def apply_channel_gamma_correction (self, gamma_value=1.0, imgChannel=None):
        if (imgChannel!=None):      self.imgChannel = imgChannel
        if (self.imgChannel==None): return None  
        self.gamma_value = gamma_value
        if (self.imgChannel.dtype == 'uint8'):      imgMax = pow(2,8)
        elif (self.imgChannel.dtype == 'uint16'):   imgMax = pow(2,16)
        tmp = imgMax * pow (self.imgChannel.astype('double') / imgMax , gamma_value)
        self.imgChannel = tmp.astype(self.imgChannel.dtype)
        return self.imgChannel
    
    def apply_channel_sharpen (self, value, imgChannel=None):    # Unsharp mask using gaussian blur subtracted from original
        if (imgChannel!=None):      self.imgChannel = imgChannel
        if (self.imgChannel==None): return None               
        tmp = cv2.GaussianBlur(self.imgChannel, (0,0), value)
        self.imgChannel = cv2.addWeighted(self.imgChannel, 1.5, tmp, -0.5, 0)
        return self.imgChannel

    def blur_edge(self, img, d=31):   # Ised by deconvolution
        h, w  = img.shape[:2]
        img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
        img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
        y, x = np.indices((h, w))
        dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
        w = np.minimum(np.float32(dist)/d, 1.0)
        return img*w + img_blur*(1-w)
    
    def defocus_kernel(self, d, sz=65):
        kern = np.zeros((sz, sz), np.uint8)
        cv2.circle(kern, (sz, sz), d, 255, -1, cv2.CV_AA, shift=1)
        kern = np.float32(kern) / 255.0
        return kern

    def apply_channel_deconvolution (self, img, psfsize=10, snrVal=8):    # Based on deconvolution.py in python samples of opencv
        
        img = img.astype('double')/255.0
        img = self.blur_edge(img)
        IMG = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
    
        if (psfsize==0): return img
        
        defocus = True
    
        ang = 0
        d = psfsize
        snr = snrVal
        noise = 10**(-0.1*snr)

        if defocus:
            psf = self.defocus_kernel(d)
        else:
            psf = self.motion_kernel(ang, d)

        psf /= psf.sum()
        psf_pad = np.zeros_like(img)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf
        PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
        PSF2 = (PSF**2).sum(-1)
        iPSF = PSF / (PSF2 + noise)[...,np.newaxis]
        RES = cv2.mulSpectrums(IMG, iPSF, 0)
        res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        res = np.roll(res, -kh//2, 0)
        res = np.roll(res, -kw//2, 1)

        return res
    
    def apply_channel_CLAHE (self):
        return  # Need to upgrade cv2 version
    
             
#-------------------------------------------------------------------------------------------------------
# Examples of how to use the class and methods
import sys

if __name__ == "__main__":
    
    #-----------------------------------------------------------------------
    # The following are the four examples to be implemented on the tablet
    # example 3 - Gamma correction
    res21, res22, res23 = example3_tone ()  
    im3 = np.hstack((res21, res22, res23 )) #, res13, cv2.cvtColor(res14, cv2.COLOR_GRAY2BGR) ))
    display ('Example 3 gamma correction images', im3)
    cv2.imwrite('example3.png', im3)
    cv2.imwrite('tone_output_1.png', res22)
    cv2.imwrite('tone_output_2.png', res23)
        
    sys.exit(0)
    
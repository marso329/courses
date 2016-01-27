from scipy.io.wavfile import *
from scipy.signal import *
from scipy.fftpack import *
import scipy
import matplotlib.pyplot as plt
import numpy as np

#open file
rate,data=read("signal-marso329.wav")

#given information
f_s=400000

#compute fft
fft_x = np.fft.fft(data)
fft_x_abs=np.abs(fft_x)
n = len(fft_x)
freq = np.fft.fftfreq(n, 1.0/f_s)
#shift it for doublesided spectrum
fft_x_shifted = np.fft.fftshift(fft_x_abs)
freq_shifted = np.fft.fftshift(freq)
#plot it
plt.figure(1)
plt.plot(freq_shifted, np.abs(fft_x_shifted))
plt.xlabel("Frequency (Hz)")
scale_plot=1000

#we sample with 400k and have 7.6M samples so we got 19.5s of data 
t_axis=np.linspace(0,19.5,num=7800000)
t_axis_scale=t_axis[1::scale_plot]

#Filter everything between 20k Hz and 70k Hz (first part)
plt.figure(2)
B,A=butter(10,[0.15,0.225])
data_first=lfilter(B,A,data)
data_first_plot=data_first[1::scale_plot]
plt.plot(t_axis_scale,data_first_plot,"-",linewidth=0.1)
plt.xlabel("First data")
#this plot only shows white noise

#Filter everything between 80k Hz and 110k Hz (second part)
plt.figure(3)
B,A=butter(10,[0.425,0.525])
data_second=lfilter(B,A,data)
data_second_plot=data_second[1::scale_plot]
plt.plot(t_axis_scale,data_second_plot,"-",linewidth=0.1)
plt.xlabel("Second data")
#strong signal but looks like noise

#Filter everything between 120k Hz and 150k Hz (third part)
plt.figure(4)
B,A=butter(10,[0.6,0.725])
data_third=lfilter(B,A,data)
data_third_plot=data_third[1::scale_plot]
plt.plot(t_axis_scale,data_third_plot,"-",linewidth=0.1)
plt.xlabel("Third data")
#strongest signal and it looks like it contains some data and we like data

#lets assume that the carrier freq is 133Hz

plt.figure(5)

convolve_data=fftconvolve(data_second,data_second,mode="same")
convolve_data_plot=convolve_data[1::scale_plot]
plt.plot(t_axis_scale,convolve_data_plot,"-",linewidth=0.3)

#the convolution is largest with tau=4, sidetops at +-1.67 . 1.67*400000=668 000 samples




plt.show()



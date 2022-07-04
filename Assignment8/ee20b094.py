'''
Title    : Assignment-8
Author   : Niyas Mon P(EE20B094) 
Date     : 11-May-2022
'''

from pylab import *
from numpy import *
from random import *


x=linspace(0,2*pi,128)
y=sin(5*x)
Y=fft.fft(y)
figure()
subplot(2,1,1)
plot(abs(Y),lw=2)
grid(True)
subplot(2,1,2)
plot(unwrap(angle(Y)),lw=2)
grid(True)
show()

# plotting the spectrum of sin(5t)
x=linspace(0,2*pi,129);x=x[:-1]
y=sin(5*x)
Y=fft.fftshift(fft.fft(y))/128.0
w=linspace(-64,63,128)
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2,markersize = 4)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin(5t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2,markersize = 2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2,markersize = 5)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$k$",size=16)
grid(True)
savefig("1.png")
show()


# AM Modulation | (1 + 0.1cos(t))cos(10t)
t=linspace(0,2*pi,129);t=t[:-1]
y=(1+0.1*cos(t))*cos(10*t)
Y=fft.fftshift(fft.fft(y))/128.0
w=linspace(-64,63,128)
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2,markersize = 4)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2,markersize = 4)
xlim([-15,15])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("2.png")
show()

# plotting after changing ranges and number of points
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=(1+0.1*cos(t))*cos(10*t)
Y=fft.fftshift(fft.fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2,markersize = 4)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2,markersize = 1)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2,markersize = 4)
xlim([-15,15])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("3.png")
show()


# Spectrum of sin^3(t)
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=sin(t)**3
Y=fft.fftshift(fft.fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2,markersize = 4)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin^3(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2,markersize = 0.5)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2,markersize = 4.5)
xlim([-15,15])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("4.png")
show()

# Spectrum of cos^3(t)
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=cos(t)**3
Y=fft.fftshift(fft.fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\cos^3(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2,markersize = 0.5)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2,markersize = 4.5)
xlim([-15,15])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("5.png")
show()

    
# Spectrum of cos(20t + 5cos(t))
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=cos(20*t + 5*cos(t))
Y=fft.fftshift(fft.fft(y))/512.0
w=linspace(-80,80,513);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2,markersize = 4)
xlim([-40,40])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $cos(20t+5cos(t))$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2,markersize = 1)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2,markersize = 4)
xlim([-40,40])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("6.png")
show()

# function to plot Gaussian spectrum
def spec_gaussian(time_range):
    t=linspace(-time_range*pi,time_range*pi,2049);t=t[:-1]
    y= exp(-(t**2)/2)
    Y=fft.fftshift(fft.fft(fft.ifftshift(y)))/2048.0
    Y = Y * sqrt(2*pi)/max(Y)
    w=linspace(-100,100,2049);w=w[:-1]
    figure()
    subplot(2,1,1)
    plot(w,abs(Y),lw=2)
    xlim([-10,10])
    ylabel(r"$|Y|$",size=16)
    title(r"Spectrum of Gaussian $\exp(-t^2/2)$")
    grid(True)
    subplot(2,1,2)
    plot(w,angle(Y),'ro',lw=2,markersize=1)
    ii=where(abs(Y)>1e-3)
    plot(w[ii],angle(Y[ii]),'go',lw=2,markersize = 5)
    xlim([-10,10])
    plt.ylim([-2, 2])
    ylabel(r"Phase of $Y$",size=16)
    xlabel(r"$\omega$",size=16)
    grid(True)
    show()

# plotting Gaussian spectrum for different time ranges
for i in range(8,17,2):
    spec_gaussian(i)
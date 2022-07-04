'''
Title : Assignment 8
Author:Niyas Mon P
Date :13-05-2022
'''

from pylab import *

# working out exapmles given in question
t = linspace(-pi,pi,65);t = t[:-1]
dt = t[1] - t[0]; fmax = 1/dt
y = sin(sqrt(2)*t)
y[0] = 0 # the sample corresponding to -tmax should be set zero
y = fftshift(y)
Y = fftshift(fft(y))/64
w=linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
xlabel(r"$w$",size=16)
title(r"spectrum of $\sin\left(\sqrt{2}t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("1.png")
show()

t1=linspace(-pi,pi,65);t1=t1[:-1]
t2=linspace(-3*pi,-pi,65);t2=t2[:-1]
t3=linspace(pi,3*pi,65);t3=t3[:-1]
# y=sin(sqrt(2)*t)
figure(2)
plot(t1,sin(sqrt(2)*t1),'b',lw=2)
plot(t2,sin(sqrt(2)*t2),'r',lw=2)
plot(t3,sin(sqrt(2)*t3),'r',lw=2)
ylabel(r"$y$",size=16)
xlabel(r"$t$",size=16)
title(r"$\sin\left(\sqrt{2}t\right)$")
grid(True)
savefig("2.png")
show()

t1=linspace(-pi,pi,65);t1=t1[:-1]
t2=linspace(-3*pi,-pi,65);t2=t2[:-1]
t3=linspace(pi,3*pi,65);t3=t3[:-1]
y=sin(sqrt(2)*t1)
figure(3)
plot(t1,y,'bo',lw=2)
plot(t2,y,'ro',lw=2)
plot(t3,y,'ro',lw=2)
ylabel(r"$y$",size=16)
xlabel(r"$t$",size=16)
title(r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$ ")
grid(True)
savefig("3.png")
show()

t=linspace(-pi,pi,65);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
y=t
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/64.0
w=linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
figure()
semilogx(abs(w),20*log10(abs(Y)),lw=2)
xlim([1,10])
ylim([-20,0])
xticks([1,2,5,10],["1","2","5","10"],size=16)
ylabel(r"$|Y|$ (dB)",size=16)
title(r"Spectrum of a digital ramp")
xlabel(r"$\omega$",size=16)
grid(True)
savefig("4.png")
show()


t1=linspace(-pi,pi,65);t1=t1[:-1]
t2=linspace(-3*pi,-pi,65);t2=t2[:-1]
t3=linspace(pi,3*pi,65);t3=t3[:-1]
n=arange(64)
wnd=fftshift(0.54+0.46*cos(2*pi*n/63))
y=sin(sqrt(2)*t1)*wnd
figure(3)
plot(t1,y,'bo',lw=2)
plot(t2,y,'ro',lw=2)
plot(t3,y,'ro',lw=2)
ylabel(r"$y$",size=16)
xlabel(r"$t$",size=16)
title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
grid(True)
savefig("5.png")
show()

t=linspace(-pi,pi,65);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(64)
wnd=fftshift(0.54+0.46*cos(2*pi*n/63))
y=sin(sqrt(2)*t)*wnd
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/64.0
w=linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
figure(4)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-8,8])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-8,8])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("6.png")

t=linspace(-4*pi,4*pi,257);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(256)
wnd=fftshift(0.54+0.46*cos(2*pi*n/256))
y=sin(sqrt(2)*t)
# y=sin(1.25*t)
y=y*wnd
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/256.0
w=linspace(-pi*fmax,pi*fmax,257);w=w[:-1]
figure(5)
subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("7.png")
show()


# function for windowing
def window(n):
  return fftshift(0.54 + 0.46 * cos((2 * pi * n)/(len(n) - 1)))

# Defining function to plot FFT of a given function
# This function will return w and tranform Y 
def Plot_FFT(lim,N,fn,Title,windowing=False,xlimit=10):
    t=linspace(-lim,lim,N+1);t=t[:-1]
    dt=t[1]-t[0]
    fmax=1/dt
    y = fn(t)

    # If we want windowing then this block will run
    if (windowing):  
        k=arange(N)
        y = y*window(k)

    y[0]=0 
    y=fftshift(y) 
    Y=fftshift(fft(y))/float(N) 
    w=linspace(-pi*fmax,pi*fmax,N+1)[:-1]

    figure()
    subplot(2,1,1)
    plot(w,abs(Y),lw=2)
    xlim([-xlimit,xlimit])
    ylabel(r"$|Y|$",size=16)
    title(Title)
    grid(True)

    subplot(2,1,2)
    angle(Y)[where(abs(Y)<3e-3)] = 0
    plot(w,angle(Y),'ro')
    plot()
    xlim([-xlimit,xlimit])
    ylabel(r"Phase of $Y$",size=16)
    xlabel(r"$\omega$",size=16)
    grid(True)
    show()

    return w,Y
  
######### Question 2 ##############
# defining cos^3(w0.t) function
def cos_3(t,w0=0.86):
    return (cos(w0*t))**3
# plotting the FFT of cos^3(w0.t)
w,Y = Plot_FFT(4*pi,64*4,cos_3,r"Spectrum of $cos^3(w_0t)$",windowing=False,xlimit= 4)
w,Y = Plot_FFT(4*pi,64*4,cos_3,r"Spectrum of $cos^3(w_0t)$ * $\omega$(t)",xlimit= 4,windowing=True)


######### Question 3 ##############
## Function to estimate w0 and delta
def estimate_w_delta(w, Y, phase):
    ii = where(abs(Y) > 0.2)

    w_avg = sum((Y[ii]**2) * abs(w[ii]))/sum(Y[ii]**2)

    delta_avg = mean(abs(phase[ii]))

    print("Estimated w_0:", w_avg.round(6))
    print("Estimated delta:", delta_avg.round(6))

#FFT of cos(wt+delta) * w(t) to estimate w, delta
def cos_delta(t,w0=1.4,delta=0.5):
    return cos(w0*t + delta)

w, Y = Plot_FFT(pi,128,cos_delta,xlimit= 10,windowing=True, Title = r"Spectrum of $cos(w_0t + \delta)$ without noise")
print("\nEstimations for cos(1.4t+0.5):")
estimate_w_delta(w, Y, angle(Y))

######### Question 4 ##############
# Now with Gaussian noise added 
def cos_delta_noise(t,w0=1.4,delta=0.5):
    return cos(w0*t + delta) + 0.1*randn(len(t))

w, Y = Plot_FFT(pi,128,cos_delta_noise,xlimit= 10,windowing=True, Title = r"Spectrum of $cos(w_0t + \delta)$ with noise")
print("\nEstimations for cos(1.4t+0.5)+n(t):")
estimate_w_delta(w, Y, angle(Y))


######### Question 5 ##############

def chirp(t):
    return cos(16 * t * (1.5 + t/(2 * pi)))
t1 = linspace(-pi, pi, 1000)[:-1]
t2 = linspace(-3*pi, -pi, 1000)[:-1]
t3 = linspace(pi, 3*pi, 1000)[:-1]
y = chirp(t1)
figure(2)
title("Plot of chirped signal ($cos(16t(1.5+t/2\\pi))$)")
plot(t1, y, color = 'green')
plot(t2, y, color = 'red')
plot(t3, y, color = 'red')
xlabel("$t$ (in s)", size = 13)
ylabel("$cos(16t(1.5+t/2\\pi))$", size = 13)
grid(True)
show()



y2 = chirp(t1) * window(arange(len(t1)))
figure(3)
title("Plot of chirped signal ($cos(16t(1.5+t/2\\pi))$)")
plot(t1, y2, color = 'green')
plot(t2, y2, color = 'red')
plot(t3, y2, color = 'red')
xlabel("$t$ (in s)", size = 13)
ylabel("$cos(16t(1.5+t/2\\pi))$", size = 13)
grid(True)
show()

w,Y = Plot_FFT(pi,1024,chirp,xlimit= 70,windowing=False, Title = r"Spectrum of chirped signal without windowing")

w,Y = Plot_FFT(pi,1024,chirp,xlimit= 70,windowing=True, Title = r"Spectrum of chirped signal with windowing")

########## Question 6 ##############
## Defining time and frequency variables
T = linspace(-pi, pi, 1025)[:-1]
tt = reshape(T, (16, 64))
magnitudes = [] #list to store magnitudes and phases for different time ranges
phases = []
w = linspace(-512, 512, 65)[:-1]

# computing DFTs of different ranges of time
for t in tt:
    y = chirp(t)
    y[0] = 0
    y = fftshift(y)
    Y = fftshift(fft(y))/64

    magnitudes.append(abs(Y))
    phases.append(angle(Y))

magnitudes = array(magnitudes)
phases = array(phases)

Y = linspace(-pi, pi, 17)[:-1]
w, Y = meshgrid(w, Y)

# 3d plot of Frequency response vs frequency and time
fig = plt.figure(figsize=plt.figaspect(0.25))
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf=ax.plot_surface(w,Y,magnitudes, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.set_title("Surface plot of Magnitude Response vs Frequency and Time")
ax.set_xlabel("$\\omega$ (in rad/s)") 
ax.set_ylabel("$t$ (in s)")
ax.set_zlabel("|Y|")


ax = fig.add_subplot(1,2,2, projection = '3d')
surf = ax.plot_surface(w, Y, phases, cmap = cm.coolwarm_r)
fig.colorbar(surf, shrink = 0.5)
ax.set_title("Surface plot of Phase Response vs Frequency and Time")
ax.set_xlabel("$\\omega$ (in rad/s)") 
ax.set_ylabel("$t$ (in s)")
ax.set_zlabel("Phase of Y (in rad)")
show()







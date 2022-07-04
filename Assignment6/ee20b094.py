'''
Assignment 6 : The laplace Transform
Author       : Niyas Mon P
Roll. No.    : EE20B094
'''


# importing necessary libraries
import numpy as np
import scipy.signal as sp
from matplotlib import pyplot as plt

# function to plot impulse response of the given system
def x_t(ang_freq,sigma):
  # frequency response of a spring is being calculated
  num = np.poly1d([1,sigma])
  den1 = np.poly1d([1,2*sigma,sigma**2 + ang_freq**2])
  den2 = np.poly1d([1,0,2.25])
  den = den1 * den2

  # Coefficients are converted into LTI system type for getting impulse response
  X_S = sp.lti(num,den)
  t,x = sp.impulse(X_S,None,np.linspace(0,50,1001))
  # plotting the impulse response
  plt.plot(t,x)
  plt.xlabel("t$\\longrightarrow$")
  plt.ylabel('x(t)$\\longrightarrow$')
  plt.show()

# plotting the x(t) function for different decay parametres 
plt.figure(1)
plt.title("Forced Damping Oscillator with decay=0.5")
x_t(1.5,0.5)  #decay parametre = 0.5

plt.figure(2)
plt.title("Forced Damping Oscillator with decay=0.05")
x_t(1.5,0.05) #decay parametre = 0.05



# defining LTI system whose transfer function is given by
#  H(s) = X(s)/F(s) = 1/(s^2 + 2.25)
#'input' represent the input function to the LTI system.
# w0 represent the input frequncy
# sigma represent the decay paramtre of input signal
def LTI(input_fn,w0,sigma):
  Num = np.poly1d([1]) 
  Den = np.poly1d([1,0,2.25])
  H_S = sp.lti(Num,Den)
  t = np.linspace(0,100,1001)
  u = input_fn(t,w0,sigma)
  t,x,svec = sp.lsim(H_S,u,t)
  plt.plot(t,x) #plotting the output signal
  plt.xlabel('time$\\longrightarrow$')
  plt.ylabel('x(t)$\\longrightarrow$')
  plt.show()

def fn(t,w0,sigma): #defining input f(t)
  return np.cos(w0*t)*np.exp(-sigma*t)

# getting out for f(t) with different ang.frequencies as per question
w = np.arange(1.4,1.6,0.05) 
for ww in w:
  plt.figure()
  plt.title("Forced Damped Oscillator with frequency=" + str(ww))
  LTI(fn,ww,0.05)


# SOLVING COUPLED SYSTEM OF DIFFERENTIAL EQUATION
# function to plot impulse response
def inverse_LT(Num,Den,out_fn_label): 
  F_S = sp.lti(Num,Den)
  t,x = sp.impulse(F_S,None,np.linspace(0,20,1001))
  plt.plot(t,x,label = out_fn_label)

plt.figure(2)
# getting x(t) from X(s)
Num1 = np.poly1d([1,0,2])
Den1 = np.poly1d([1,0,3,0])
inverse_LT(Num1,Den1,"x(t)")

# getting y(t) from Y(s)
Num2 = np.poly1d([2])
Den2 = np.poly1d([1,0,3,0]) 
inverse_LT(Num2,Den2,"y(t)")

plt.title("Coupled spring problem")
plt.xlabel('time$\\longrightarrow$')
plt.ylabel('output$\\longrightarrow$')
plt.legend()
plt.show()


# SOLVING LCR LOW PASS FILTER

L=1e-6
C=1e-6
R=100

# Transfer function for the LCR system
def LCR_TransferFunction(L,C,R): 
  Num = np.poly1d([1])
  Den = np.poly1d([L*C,R*C,1])
  H_S = sp.lti(Num,Den)
  w,S,phi = H_S.bode()
  plt.subplot(2,1,1)
  plt.semilogx(w,S)
  plt.xlabel('w$\\longrightarrow$')
  plt.ylabel('Magnitude(dB)')
  plt.subplot(2,1,2)
  plt.xlabel('w$\\longrightarrow$')
  plt.ylabel('Phase')
  plt.semilogx(w,phi)
# plotting the magnitude and phase of LCR LTI system
LCR_TransferFunction(L,C,R)
plt.show()



# defining input function vi(t) = cos(10^3*t)u(t) − cos(10^6*t)u(t)
def v_i(t,w0):
  return np.cos(w0[0]*t) - np.cos(w0[1]*t)

# defining function to get output v_o(t) for the above LCR system for the given input
def v_o(input,w0,time):
  Num = np.poly1d([1])
  Den = np.poly1d([L*C,R*C,1])
  H_S = sp.lti(Num,Den)
  t,h = sp.impulse(H_S,None,np.linspace(0,10,1001))
  u = input(time,w0)
  t,x,svec = sp.lsim(H_S,u,time)
  plt.xlabel('time$\\longrightarrow$')
  plt.ylabel('v_o(t)$\\longrightarrow$')
  plt.plot(t,x)
  plt.show()

# plotting output for t < 30 us
plt.title('output for t < 30 us')
v_o(v_i,[1e3,1e6],time = np.linspace(0,30e-6,101))
# plotting output for t < 10 ms
plt.title('output for t < 10 ms')
v_o(v_i,[1e3,1e6],time = np.linspace(0,10e-3,100001))

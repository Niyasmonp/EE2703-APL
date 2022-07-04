# importing nesessary libraries
from __future__ import division
from sympy import *
import pylab as p
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as sp


s = symbols('s')


# defining lowpass filter
def lowpass(R1,R2,C1,C2,G,Vi):
  A = Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0],[0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
  b=Matrix([0,0,0,-Vi/R1])
  V=A.inv()*b
  return (A,b,V)


# potting the transfer function of low-pass filter
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
Vo=V[3]
ww=p.logspace(0,8,801)
ss=1j*ww
hf=lambdify(s,Vo,"numpy")
v=hf(ss)
p.loglog(ww,abs(v),lw=2)
p.title("Magnitude response of the low-pass filter")
p.xlabel("w$\\longrightarrow$")
p.ylabel("|H(w)|$\\longrightarrow$")
p.grid(True)
p.show()


# plotting the step response of the low-pass filter
step_response=lowpass(10000,10000,1e-9,1e-9,1.586,1/s)[2][3] #here the input is 1/s
n,d = fraction(step_response)
n = n.as_poly(s)
d = d.as_poly(s)
num = np.array(n.all_coeffs(),dtype=float)
den = np.array(d.all_coeffs(),dtype=float)
print(type(num))
step_resp_S = sp.lti(num,den)
t,y = sp.impulse(step_resp_S,None,np.linspace(0,0.001,1001))
plt.plot(t,y)
plt.title("Step response of the low-pass filter")
plt.xlabel("t$\\longrightarrow$")
plt.ylabel('x(t)$\\longrightarrow$')
plt.show()


# function to plot the output of low-pass filter along with input
def v_out_lowpass(func,t):
    H=lowpass(10000,10000,1e-9,1e-9,1.586,1)[2][3]
    n,d = fraction(H)
    n = n.as_poly(s)
    d = d.as_poly(s)
    num = np.array(n.all_coeffs(),dtype=float)
    den = np.array(d.all_coeffs(),dtype=float)
    H_S = sp.lti(num,den)
    v_ii = func(t)
    plt.plot(t,v_ii,label="V_in")
    t,y,svec = sp.lsim(H_S,v_ii,t)
    plt.plot(t,y,label = "V_out")
    plt.xlabel("t$\\longrightarrow$")
    plt.ylabel('v(t)$\\longrightarrow$')
    plt.legend()
    plt.show()


def v_i(t):  # input = [sin(2*10e3*pi*t) + cos(2*10e6*pi*t)].u(t)
  return p.sin(2*1000*p.pi*t) + p.cos(2*1000000*p.pi*t)

t = np.linspace(0,0.001,1000001)

plt.title("output for Vi(t) = [sin(2.pi.$10^3$t) + cos(2.pi.$10^6$t)].u(t)")
v_out_lowpass(v_i,t)  #plotting the output along with input



# defining highpass filter
def highpass(R1,R3,C1,C2,G,Vi):
  A = Matrix([[s*C1+s*C2+1/R1,-s*C2,0,-1/R1],[-s*C2,s*C2+1/R3,0,0],[0,0,G,-1],[0,-G,G,1]])
  b=Matrix([s*C1*Vi,0,0,0])
  V=A.inv()*b
  return (A,b,V)

# potting the transfer function of high-pass filter
A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1)
Vo=V[0]
ww=p.logspace(0,8,801)
ss=1j*ww
hf=lambdify(s,Vo,"numpy")
v=hf(ss)
p.loglog(ww,abs(v),lw=2)
p.title("Magnitude response of the high-pass filter")
p.xlabel("w$\\longrightarrow$")
p.ylabel("|H(w)|$\\longrightarrow$")
p.grid(True)
p.show()

# plotting the step response of the high pass filter
step_response=highpass(10000,10000,1e-9,1e-9,1.586,1/s)[2][3] #here the input is 1/s
n,d = fraction(step_response)
n = n.as_poly(s)
d = d.as_poly(s)
num = np.array(n.all_coeffs(),dtype=float)
den = np.array(d.all_coeffs(),dtype=float)
print(type(num))
step_resp_S = sp.lti(num,den)
t,y = sp.impulse(step_resp_S,None,np.linspace(0,0.001,1001))
plt.plot(t,y)
plt.title("Step  response of the high-pass filter")
plt.xlabel("t$\\longrightarrow$")
plt.ylabel('x(t)$\\longrightarrow$')
plt.show()


# function to plot the output of high-pass filter along with input
def v_out_highpass(func,t):
    H=highpass(10000,10000,1e-9,1e-9,1.586,1)[2][3]
    n,d = fraction(H)
    n = n.as_poly(s)
    d = d.as_poly(s)
    num = np.array(n.all_coeffs(),dtype=float)
    den = np.array(d.all_coeffs(),dtype=float)
    H_S = sp.lti(num,den)
    v_ii = func(t)
    plt.plot(t,v_ii,label="V_in")
    t,y,svec = sp.lsim(H_S,v_ii,t)
    plt.plot(t,y,label="V_out")
    plt.xlabel("t$\\longrightarrow$")
    plt.ylabel('v(t)$\\longrightarrow$')
    plt.legend()
    plt.show()


 # input = e^(-500t).cos(2.10^3.pi.t)].u(t)
def v_in1(t): 
  return p.exp(-500*t)*p.cos(2*p.pi*1000*t)
t = np.linspace(0,0.01,1000000)
plt.title("High pass output for Vi(t) = exp(-500t)cos(2.pi.$10^3$.t)")
v_out_highpass(v_in1,t)  #plotting the output along with input


# input = [e^(-5000t).cos(2.10^6.pi.t)].u(t)
def v_in2(t):  
  return p.exp(-5000*t)*p.cos(2*p.pi*1000000*t)
t = np.linspace(0,0.001,1000000)
plt.title("High pass output for Vi(t) = exp(-5000t)cos(2.pi.$10^6$.t)")
v_out_highpass(v_in2,t)  #plotting the output along with input


# input = [e^(-500t).(sin(2.10^3.pi.t)+cos(2.10^6.pi.t))].u(t)
# comparing both high pass and low pass filter output for the given input
def v_in3(t):  
  return p.exp(-500*t)*(p.sin(2*1000*p.pi*t) + p.cos(2*1000000*p.pi*t))
t = np.linspace(0,0.01,1000000)
plt.title("High pass output for Vi(t) = exp(-500t)[sin(2.pi.$10^3$.t) + cos(2.pi.$10^6$.t)]")
v_out_highpass(v_in3,t)  #plotting the output along with input for high pass filter
plt.title("Low pass output for Vi(t) = exp(-500t)[sin(2.pi.$10^3$.t) + cos(2.pi.$10^6$.t)]")
v_out_lowpass(v_in3,t) #plotting the output along with input for low pass filter
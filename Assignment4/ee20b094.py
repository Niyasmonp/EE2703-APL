import numpy as np
import matplotlib.pyplot as plt 
from pylab import *
from scipy.integrate import quad

# defining required functions: exp(x)and cos(cos(x))
def exp_fn(x):
  x = np.array(x)
  y = np.exp(x)
  return y

def coscos_fn(x):
  x = np.array(x)
  y = np.cos(np.cos(x))
  return y

# plotting the functions 
# as well as periodic functions from -pi to 2pi in semilog scale
a=(linspace(-2*pi,4*pi,500))
figure(1)
semilogy(a,exp_fn(a),label = "$e^x$")
semilogy(a,exp_fn(a%(2*pi)),label = r"function generated from forier series")
xlabel(r'$x$',size = 10)
ylabel(r'$e^x$', size =10)
title("Q1:Plot of $e^x$")
grid(True)
legend()
show()

figure(2)
plot(a,coscos_fn(a),label="cos(cos(x))")
plot(a,coscos_fn(a%(2*pi)),label="function generated from forier series")
xlabel(r'$x$',size = 10)
ylabel(r'$cos(cos(x))$', size =10)
title("Q1:Plot of cos(cos(x)) function")
grid(True)
legend(loc= "upper right")
show()



#obtaining the first 51  Forier coefficients of both functions

# function that outputs required no of forier coefficients 
def FS_coeffs(fn,limit_lower,limit_upper,n):
  coeffs = np.zeros(n)
  def fn_cos_k(x,k,fn):
    return fn(x)*np.cos(k*x)
  def fn_sin_k(x,k,fn):
    return fn(x)*np.sin(k*x)

  coeffs[0] = quad(fn,0,2*pi)[0] / (2*pi)

  for i in range(1,n):
    if i%2==1:
      coeffs[i] = quad(fn_cos_k,limit_lower,limit_upper, args = (i//2 +1,fn))[0] /pi
    else:
      coeffs[i] = quad(fn_sin_k,limit_lower,limit_upper, args = (i//2,fn))[0] /pi
  return coeffs


# storing first 51 forier coeff of exp(x) and cos(cos(x))
coeff_coscos = FS_coeffs(coscos_fn,0,2*pi,51)
coeff_exp = FS_coeffs(exp_fn,0,2*pi,51)


# plot of FS_coeff of exp(x) in semilog scale
figure(3)
semilogy(abs(coeff_exp),'ro')
ylabel(r'$e^x$ ($log$)',size = 10)
xlabel(r'$n$ th coefficient', size =10)
title("Plot of coefficients of $e^x$ in semilog scale")
grid(True)
legend()
show()

# plot of FS_coeff of exp(x) in log-log scale
figure(4)
loglog(abs(coeff_exp),'ro')
ylabel(r'$e^x$ ($log$)',size = 10)
xlabel(r'$n$ th coefficient ($log$)', size =10)
title("Plot of coefficients of $e^x$ in log-log scale")
grid(True)
legend()
show()

# plot of FS_coeff of cos(cos((x)) in semilog scale
figure(5)
semilogy(abs(coeff_coscos),'ro')
ylabel(r'$cos(cos(x))$ ($log$)',size = 10)
xlabel(r'$n$ th coefficient', size =10)
title("Plot of coefficients of cos(cos(x)) in semilog scale")
grid(True)
legend()
show()

# plot of FS_coeff of cos(cos((x)) in log-log scale
figure(6)
loglog(abs(coeff_coscos),'ro')
ylabel(r'$cos(cos(x))$ ($log$)',size = 10)
xlabel(r'$n$ th coefficient ($log$)', size =10)
title("Plot of coefficients of cos(cos(x)) in log-log scale")
grid(True)
legend()
show()

# finding FS coeffs using least squares approach

# function which return Forier coeff of given function using lstsq method
def least_square_estimation_of_coeffiecients(fn):
  x = linspace(0,2*pi,401)
  x=x[:-1]
  A = zeros((400,51))
  f_x = fn(x)
  A[:,0] = 1
  for k in range(1,26):
    A[:,2*k-1] = np.cos(k*x)
    A[:,2*k] = np.sin(k*x)

  return lstsq (A,f_x,rcond = None)[0]

# storing the forier coeffs of exp(x) and cos(cos(x)) found using lstsq method
lstsq_coeff_exp = least_square_estimation_of_coeffiecients(exp_fn)
lstsq_coeff_coscos = least_square_estimation_of_coeffiecients(coscos_fn)

# plotting the estimated forier coeffients along with true coefficients
figure(3)
semilogy(abs(coeff_exp),'ro',markersize = 6,label = 'true coefficients')
semilogy(abs(lstsq_coeff_exp),'go',markersize = 4,label='lstsq estimated coefficients')
ylabel(r'$e^x$ ($log$)',size = 10)
xlabel(r'$n$ th coefficient', size =10)
title("Plot of coefficients of $e^x$ in semilog scale")
grid(True)
legend()
show()

figure(4)
loglog(abs(coeff_exp),'ro',markersize = 6,label = 'true coefficients')
loglog(abs(lstsq_coeff_exp),'go',markersize = 4,label='lstsq estimated coefficients')
ylabel(r'$e^x$ ($log$)',size = 10)
xlabel(r'$n$ th coefficient ($log$)', size =10)
title("Plot of coefficients of $e^x$ in log-log scale")
grid(True)
legend()
show()

figure(5)
semilogy(abs(coeff_coscos),'ro',markersize = 6,label = 'true coefficients')
semilogy(abs(lstsq_coeff_coscos),'go',markersize = 4,label='lstsq estimated coefficients')
ylabel(r'$cos(cos(x))$ ($log$)',size = 10)
xlabel(r'$n$ th coefficient', size =10)
title("Plot of coefficients of cos(cos(x)) in semilog scale")
grid(True)
legend()
show()

figure(6)
loglog(abs(coeff_coscos),'ro',markersize = 6,label = 'true coefficients')
loglog(abs(lstsq_coeff_coscos),'go',markersize = 4,label='lstsq estimated coefficients')
ylabel(r'$cos(cos(x))$ ($log$)',size = 10)
xlabel(r'$n$ th coefficient ($log$)', size =10)
title("Plot of coefficients of cos(cos(x)) in log-log scale")
grid(True)
legend()
show()


# finding max deviation of estimated forier coeffs from true values
error_exp = np.abs(coeff_exp - lstsq_coeff_exp)
error_coscos = np.abs(coeff_coscos - lstsq_coeff_coscos)

error_exp_max = np.amax(error_exp)
error_coscos_max = np.amax(error_coscos)

print("The largest deviation of least square coefficient of $e^x$ from true value is : ",error_exp_max )
print("The largest deviation of least square coefficient of cos(cos(x)) from true value is : ",error_coscos_max )

# generating the original functions from the estimated coefficients
def estimated_fn(coeff,x):
  A = zeros((len(x),51))
  A[:,0] = 1
  for k in range(1,26):
    A[:,2*k-1] = np.cos(k*x)
    A[:,2*k] = np.sin(k*x)
  return np.matmul(A,coeff)

a1 =(linspace(0,2*pi,500))
estimated_exp = estimated_fn(lstsq_coeff_exp,a1)
estimated_coscos = estimated_fn(lstsq_coeff_coscos,a1)

# plotting the estimated functions along with actual functions
figure(1)
semilogy(a1,exp_fn(a1),label = r"$e^x$")
semilogy(a1,estimated_exp,'go',markersize = 3,label = r"estimated $e^x$")
xlabel(r'$x$',size = 10)
ylabel(r'$e^x$', size =10)
title("Plot of exp(x) function")
grid(True)
legend()
show()


figure(2)
plot(a1,coscos_fn(a1),label="cos(cos(x))")
plot(a1,estimated_coscos,'go',markersize = 3,label = r"estimated $cos(cos(x))$")
xlabel(r'$x$',size = 10)
ylabel(r'$cos(cos(x))$', size =10)
title("Plot of cos(cos(x)) function")
grid(True)
legend(loc= "upper right")
show()




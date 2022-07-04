from ast import arg
from glob import glob
from sys import argv
from pylab import * 
import numpy as np
import scipy.linalg as sp
import mpl_toolkits.mplot3d.axes3d as p3

# storing default values
Nx = 25      #size along x
Ny = 25      #size along y
radius = 8   #radius of central lead
Niter = 1500 #number of iterations to perform


#taking user inputs for Nx,Ny,radius&Niter
if len(argv) == 2:
    Nx = int(argv[1])
    Ny = 25
    radius = 8
    Niter = 1500

if len(argv) == 3:
    Nx = int(argv[1])
    Ny = int(argv[2])
    radius = 8
    Niter = 1500

if len(argv) == 4:
    Nx = int(argv[1])
    Ny = int(argv[2])
    radius = int(argv[3])
    Niter = 1500

if len(argv) == 5:
    Nx = int(argv[1])
    Ny = int(argv[2])
    radius = int(argv[3])
    Niter = int(argv[4])

radius = radius/(Nx-1)  #maping radius to cm

#initialising potential array
phi = np.zeros((Ny,Nx),dtype= float )
x = np.linspace(-0.5,0.5,Nx)  # x coordinates 
y = np.linspace(-0.5,0.5,Ny)  # y coordinates
X,Y = np.meshgrid(x,y)
ii = np.where(np.square(X) + np.square(Y) <= (radius*1.001)**2)
phi[ii] = 1.0  #set the potential of sentral lead as 1.0

# plotting the initial potential
figure(1)
title("Initial Potentials")
xlabel("X $\longrightarrow$")
ylabel("Y $\longrightarrow$")
contourf(X, -Y, phi)
plot(x[ii[0]], y[ii[1]], 'ro')
colorbar()
grid()
show()


errors = np.zeros(Niter) #declaring array to store errors in each iteration with the previous one

for j in range(Niter):
    oldphi = phi.copy() 
    #updating potential as a average of neighbouring values
    phi[1:-1, 1:-1] = 0.25 * (oldphi[1:-1, 0:-2] + oldphi[1:-1, 2:] + oldphi[0:-2, 1:-1] + oldphi[2:, 1:-1])

    #assigning boundary conditions
    phi[:, 0] = phi[:, 1]       
    phi[:,-1] = phi[:,-2]      
    phi[0, :] = phi[1, :] 
    phi[-1, :] = 0

    #making central potential as 1.0
    phi[ii] = 1.0   

    #storing error
    errors[j] = np.max(np.abs(phi - oldphi))


# Plotting the errors in semilog  scale
figure(2)
title("Plot of errors in semilog scale")
xlabel("Number of iterations")
ylabel("Errors")
semilogy(np.array(range(Niter)), errors, label= 'errors in every iteration')
semilogy(np.array(range(Niter))[::50], errors[::50],'ro',label = ' error in every 50th iteraion')
legend()
grid()
show()

# Plotting the errors in loglog scale
figure(3)
title("Plot of errors in log-log scale")
xlabel("Number of iterations")
ylabel("Errors")
loglog(np.array(range(Niter)), errors, label= 'errors in every iteration')
loglog(np.array(range(Niter))[::50], errors[::50],'ro',label = ' error in every 50th iteraion')
legend()
grid()
show()


# Fitting the errors
# function to fit the errors as an exponential curve using lstsq
def fit_log_error(errors, start_index):
    n_iter = len(errors)
    log_errors = np.log(errors)
    M = np.c_[np.ones((n_iter,1)),np.array(range(start_index,n_iter+start_index))]
    return sp.lstsq(M, log_errors)[0]

log_A1, B1 = fit_log_error(errors,0)  #fitting error for all values
log_A2, B2 = fit_log_error(errors[500:], 500) #fitting errors after 500th ieration
K = np.array(range(Niter))

print("Error fitted into the form of y=A*e^(Bx)")
print(f"using entire errors, we estimate log A = {round(log_A1,5)} and B = {round(B1,5)}")
print(f"using errors after 500th iteration, we estimate log A = {round(log_A2,5)} and B = {round(B2,5)}")


# Plotting of the estimated errors in semilog scale
figure(4)
title("Plot of actual errors vs fitted errors in semilog scale")
xlabel("Number of iterations")
ylabel("Errors")
semilogy(K, errors)
estimated_values1 = np.exp(log_A1 + B1 * (K))[::100]
estimated_values2 = np.exp(log_A2 + B2 * (K))[::100]
semilogy(K[::100], estimated_values1, 'ro')
semilogy(K[::100], estimated_values2, 'go', markersize = 4)
legend(["errors","fit1","fit2"])
grid()
show()

# Plotting of the estimated errors in log-log scale
figure(5)
title("Plot of actual errors vs fitted errors in loglog scale")
xlabel("Number of iterations")
ylabel("Errors")
loglog(K, errors)
loglog(K[::100], estimated_values1, 'ro')
loglog(K[::100], estimated_values2, 'go', markersize = 4)
legend([" Actual errors", "fit1", "fit2"])
grid()
show()


# cumulative error calculation
N_iterations = np.arange(1, 1501, 50)
Error_iteration = -(log_A1/B1) * np.exp(B1 * (N_iterations + 0.5))
Error_iteration


# Plotting Cumulative error in loglog scale against number of iterations
figure(6)
title("Plot of Cumulative error in loglog scale")
xlabel("Number of iterations")
ylabel("Maximum error in computation")
loglog(N_iterations, np.abs(Error_iteration), 'ro', markersize = 3)
grid()
show()


# plots of Final Potentials
fig1 = figure(7)
ax = p3.Axes3D(fig1)
title('3-D Surface Plot of Potentials')
surf = ax.plot_surface(Y, X, phi, rstride=1, cstride=1, cmap=plt.cm.jet)
show()


## Plotting 2-D Contour Plot of Final Potential values
figure(8)
title("2-D Contour Plot of Potentials")
xlabel("X")
ylabel("Y")
plot(x[ii[0]], y[ii[1]],'ro')
contourf(X[::-1],-Y, phi, cmap='magma')
colorbar()
grid()
show()


# Finding current density distribution
Jx, Jy = 1/2 * (phi[1:-1, 0:-2] - phi[1:-1, 2:]), 1/2 * (phi[:-2, 1:-1] - phi[2:, 1:-1])


# Plotting of current density
figure(9)
title("Vector plot of current flow")
quiver(X[1:-1, 1:-1], -Y[1:-1, 1:-1], -Jx[:, ::-1], -Jy, scale=6)
plot(x[ii[0]], y[ii[1]],'ro')
grid()
show()

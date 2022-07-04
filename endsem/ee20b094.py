'''
Title : End Semester Examination-2022
Author: Niyas Mon P (EE20B094)
Date  : 13-05-2022
'''

# importing necessary libraries
from numpy import *
from pylab import *
from math import sqrt


# defining constants
N = 100
l = 0.5
Im = 1.0
c = 2.9979e8
mu0 = 4e-7*pi
a = 0.01
lamda = l*4.0
f = c/lamda
k = 2*pi/lamda
dz = l/N

# ============ QUESTION 1  =============

# creating z matrix ie, position of all currents
z = array([i*dz for i in range(-N,N+1,1)])
# creating current vector corresponding to the position of z
I = zeros((2*N+1,),dtype = float)
I[N] = Im 

# u is unknown current points
u = concatenate((z[1:N],z[N+1:-1]))
# J is unknown current vector 
J = zeros((2*N-2,1),dtype = float)

# printing the z and u vectors
print("The z vector is :")
print("------------------")
print(z)
print()
print("The u vector is :")
print("------------------")
print(u)
print("====================")


# ============ QUESTION 2  ===============

# defining function for computing matrix M
def compute_M(radius,no_unknown_currents):
  M = zeros((no_unknown_currents,no_unknown_currents),dtype = float)
  fill_diagonal(M,1)
  M = M / (2*pi*radius)
  return M


# ============ QUESTION 3  ===============

# defining function for computing Rz and Ru
# as specified in question
def compute_Rz_Ru(N,r):   # N : no of sections in each half length , r: radius
  Rz = zeros((2*N+1,2*N+1),dtype = float)
  Ru = zeros((2*N-2,2*N-2),dtype = float)
  for i in range(0,2*N+1):
    for j in range(0,2*N+1):
      Rz[i][j] = sqrt(r**2 + (z[i]-z[j])**2)
  for i in range(0,2*N-2):
    for j in range(0,2*N-2):
      Ru[i][j] = sqrt(r**2 + (u[i]-u[j])**2)
  return Rz,Ru

# computing Rz and Ru
Rz ,Ru = compute_Rz_Ru(N,a)
# computing the matrix P
# P is the matrix of vector potential contributed by unknown currents
P = exp(-1j*k*Ru)/Ru *1e-7 * dz
# computing the matrix PB
# PB is contribution to vector potential due to current I[N]
R_iN = array([sqrt(a**2 + u[i]**2) for i in range(0,2*N-2)])
PB = exp(-1j*k*R_iN)/R_iN *1e-7 * dz

print("Rz :")
print("-----------")
print(Rz)

print("Ru :")
print("-----------")
print(Ru)
print("====================")


print("The P matrix is :  (after multiplying by 10^8)")
print((P*1e8).round(2)) 
print("The PB matrix is :  (after multiplying by 10^8)")
print((PB*1e8).round(2)) 
print("====================")


# ============ QUESTION 4  =============

# creation of matrix Q and QB as in question 4
Q = -P * (a/mu0) * ((-1j*k/Ru)-1/(Ru*Ru))
QB = PB * (a/mu0) * ((-1j*k/R_iN)-1/(R_iN*R_iN))

print("Q matrix:")
print(Q.round(2))
print("QB matrix:")
print(QB.round(2))
print("====================")


# ============ QUESTION 5  =============

# finding the final solution 
M = compute_M(a,2*N-2)
# finding the J vector  
J = matmul(inv(M-Q),QB) * Im
J = abs(real(J))

# filling I vector with the values in the J vector
I[1:N] = J[0:N-1]
I[N+1:-1] = J[N-1:]
# printing J and I vectors
print("The J vector is :")
print("------------------")
print(J.round(3))
print()
print("The I vector is :")
print("------------------")
print(I.round(3))
print("====================")


# finding the mean square error between calculated cureents and assumption
I_standard_assumption = Im*sin(k*(l-abs(z)))
print(f"The mean square error is :{(1/len(I)*sum(abs(I-I_standard_assumption)**2)).round(5)}")


# plotting the actual value of currents with the aprroximation
figure(0)
title(f"Calculated current VS approximated current for N = {N}")
plot(z,Im*sin(k*(l-abs(z))),label = r'aproximated sin function')
plot(z,I,label = r'actual value computed')
xlabel("z$\longrightarrow$")
ylabel("I$\longrightarrow$")
legend()
show()


## =============== THE END  =================== ##

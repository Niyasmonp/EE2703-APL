'''
Applied Programming Lab(EE2703)
Name : Niyas Mon P
Week3:Fitting Data to Models
'''


from pylab import *
import scipy.special as sp
import numpy as np

#defining the true values of A and B in A*J(t)+B*t
A_TRUE = 1.05   
B_TRUE = -0.105

#(question 2) Extracting time column and data columns
data_points= loadtxt("fitting.dat")
t= data_points [:,:1]  #time
data = data_points[:,1:10]  #data

# (q3) making plots with noise in it
sigma = logspace(-1,-3,9) 
sigma = around(sigma,3) #rounding as per given in sample plots
figure(0)
for i in range(len(data[0])):  #plotting different noisy data in data
  plot(t,data[:,i:i+1],label='$\sigma_{} = {}$'.format(i, sigma[i]))
  title(r'plot of data {} column with noise'.format(i),size= 15,)


# q4 : plotting of true values
# function to create g(t;A,B) = AJ(t) + Bt 
def g(t,A,B):
  return A*sp.jn(2,t) + B*t

y_true = g(t,A_TRUE,B_TRUE)

plot(t,y_true,label="True value",color='black')
xlabel('$t$',size=10)
ylabel('$f(t)+n$',size=10)
title(r'Q4: Data to fitted to theory',size=15)
legend()
show()

# q5:Making errorbar plot of first data column
figure(1)
errorbar(t[::5], data[:,:1][::5], 0.1, fmt='ro', label='Error Bar')
plot(t, y_true, label='f(t)', color='black')
xlabel('$t\longrightarrow$')
ylabel('$f(t)\longrightarrow$')
title(f'Q5: Data points for $\sigma$ = {sigma[0]} along with exact function',size = 15)
legend()
grid()
show()

# q6
J = sp.jn(2,t)
M= c_[J, t]
AB = [A_TRUE,B_TRUE] #vector with true value of A and B
x = np.matmul(M,AB)
if x.all() == g(t,A_TRUE,B_TRUE).all():
  print("vector M*AB and g(t,A0,B0) are equal")
else:
  print("vector M*AB and g(t,A0,B0) are not equal")

# q7
A_values = arange(0,2.1,0.1)
B_values = append(arange(-0.2,0,0.01),0)
#making epsilon matrix
E_matrix = zeros((len(A_values),len(B_values)))

data1= []  #storing the data column 1  
for w in range(len(data[:,:1])):
  data1.append(data[w][0])

#making eplison matrix for the data in column 1
for p in range(len(A_values)):
  for q in range(len(B_values)):
    E_matrix[p][q] =  mean(square(data[:,:1] - g(t,A_values[p],B_values[q])))


# q8: plotting epsilon matrix
figure(2) 
contour_plot=contour(A_values,B_values,E_matrix,levels=20)
xlabel('A values')
ylabel('B values')
title("Q8:Contour plot of $\epsilon_{ij}$")
clabel(contour_plot,fontsize=8, inline=True)
plot(A_TRUE,B_TRUE,'r*',markersize=10)
grid()
annotate("Exact Location", (1.05, -0.105), xytext=(-10,10), textcoords="offset points")
show()

# q9: obtaining the best estimate for A and B using lstsq function
p , resid , rank , sig = lstsq(M,data1,rcond=None)
print(f"Estimated value of A for the data1 is {p[0]} & B = {p[1]}")

# q10: estimating the best values for A and B in each data with different noise
# and estimating the errors with true values and plotting them

#function to return mean square error
def error_AB(AB_estimated):
  return square(AB_estimated[0]-A_TRUE) , square(AB_estimated[1]-B_TRUE)

errorA = zeros((9,1));errorB = zeros((9,1)) #for storing the errors in A and B

for f in range(9):
  data_x =[] #just for storing the datas for plotting
  for g in range(len(data[:,:1])):
    data_x.append(data[g][f])

  pp , resid , rank , sig = lstsq(M,data_x,rcond=None)
  errorA[f],errorB[f] = error_AB(pp)

figure(3)
plot(sigma,errorA,'ro--',label='Aerr',linewidth=1)
plot(sigma,errorB,'go--',label='Berr',linewidth=1)
title("Q10 : variation of error with noise",size=10)
xlabel(r"Noise standard deviation$\longrightarrow$",size=12.5)
ylabel(r"MS error$\longrightarrow$",size= 12.5)
legend()
grid()
show()

# q11:Plot of Aerr & Berr and sigma using loglog
figure(4)
loglog(sigma, errorA, 'ro', label="Aerr")
errorbar(sigma, errorA, std(errorA), fmt='ro')
loglog(sigma, errorB, 'go', label="Berr")
errorbar(sigma, errorB, std(errorB), fmt='go')
title("Q11 : Variation of error with noise")
xlabel("$\sigma_{n}\longrightarrow$")
ylabel("$MS error\longrightarrow$")
legend()
grid()
show()

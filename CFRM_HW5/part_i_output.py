import numpy as np
import pandas as pd

#Definitions
sigma1 = 0.38
sigma2 = 0.12
sigma11 = np.power(sigma1,2)
sigma22 = np.power(sigma2,2)
rho = 0.7
sigma12 = rho*sigma1*sigma2
r0= 0.02

print('Case 2: r0 = 0.02, rho=0.7')
#vectors and etc.
#Givens and calculations
mu = np.array([[0.20],[0.13]])
mu_t = np.transpose(mu)
Sigma = np.array([[sigma11, sigma12],[sigma12, sigma22]])
Sigma_inverse = np.linalg.inv(Sigma)
one_v = np.array([[1.0],[1.0]])
one_v_t = np.array([[1.0,1.0]])
mu_e = mu - r0*one_v
mu_e_t = np.array([[0.1,0.03]])

#calculate tangency portfolio
top = np.dot(Sigma_inverse, mu_e)
B = np.dot(one_v_t, Sigma_inverse)
B = np.dot(B, mu)
A = np.dot(one_v_t, Sigma_inverse)
A = r0*np.dot(A, one_v)
bottom = B-A
w_t = top/bottom
print('')
print('Tangency Portfolio is (Transpose): ', np.transpose(w_t))

#calculate the weights for each investor
gamma_values = np.array([0.2,1.25,0.5762031250000001])
weights = {'investor':[],'risk_free':[], 'Asset_1':[], 'Asset_2':[]}
for i in range(len(gamma_values)):
    W_1 = (gamma_values[i])*np.dot(Sigma_inverse, mu_e)
    Asset_1 = W_1[0][0]
    Asset_2 = W_1[1][0]
    risk_free = 1 - Asset_1 - Asset_2
    weights['investor'].append(i+1)
    weights['risk_free'].append(risk_free)
    weights['Asset_1'].append(Asset_1)
    weights['Asset_2'].append(Asset_2)
weights = pd.DataFrame.from_dict(weights)
weights = weights.set_index('investor')
weights['sum'] = weights.sum(axis=1)
print('')
print(weights)

#Equations to Satisfy equilibrium
#Equation 1 showing that all wealth is invested in the market as the weight on risk free asset is in zero net supply
wealth = np.array([10.0,3.0,4.0])
M = wealth.sum()
gamma_values = np.array([0.2,1.25,0.5762031250000001])
summation = 0.0
for i in range(len(gamma_values)):
    W_1 = (gamma_values[i])*np.dot(Sigma_inverse, mu_e)
    value = wealth[i]*W_1
    summation += value 
market = M*w_t
print('')
print('Equation 1 show all wealth invested in the market:')
print('Transpose Investor side: ', np.transpose(summation))
print('Transpose Market side: ', np.transpose(market))

#Equation 2 Showing that summation of all wealth invested in risk free asset sums to zero
summation = 0.0
for i in range(len(weights)):
    value = wealth[i]*weights['risk_free'][i+1]
    summation += value
print('')
print('Equation 2 show summation of all investors wealth invested in risk free asset sum to zero:')
print('Wealth invested in risk-free assets:', summation)

#Equation 3 showint that demand is equal to supply in the market
summation = 0.0
for i in range(len(weights)):
    value = (wealth[i] - wealth[i]*weights['risk_free'][i+1])*w_t 
    summation += value
print('')
print('Equation 3 show that the demand and supply are equal in the market:')
print('Sum of Demand from all investors (Transpose):', np.transpose(summation))
print('Sum of Supply from market (Transpose)', np.transpose(market))

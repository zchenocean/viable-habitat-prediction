import numpy as np
import seawater as sw

kb_inv = 1./(8.6e-5) #K/eV
R_gasconst = 8.3144621 # J/mol/K
T0_Kelvin = 273.15
Tref = 15. # °C
Tref_K = 288.15 #Tref + T0_Kelvin
XiO2 = 0.209 # Mean atmospheric O2 mixing ratio
V = 32e-6 # partial molar volume of O2 (m3/mol)
Patm = 1. # Atm pressure
db2Pa = 1e4 # convert pressure: decibar to Pascal
t68 = 1.00024 # temp conv
tc9 = 298.15
A0 = 5.80871
A1 = 3.20291
A2 = 4.17887
A3 = 5.10006
A4 = -9.86643e-2
A5 = 3.80369
B0 = -7.01577e-3
B1 = -7.70028e-3
B2 = -1.13864e-2
B3 = -9.51519e-3
C0 = -2.75915e-7
a0 = 999.842594
a1 = 6.793952e-2
a2 = -9.095290e-3
a3 = 1.001685e-4
a4 = -1.120083e-6
a5 = 6.536332e-9
b0 = 8.24493e-1
b1 = -4.0899e-3
b2 = 7.6438e-5
b3 = -8.2467e-7
b4 = 5.3875e-9
c0 = -5.72466e-3
c1 = 1.0227e-4
c2 = -1.6546e-6
d0 = 4.8314e-4
h0 = 3.239908
h1 = 1.43713e-3
h2 = 1.16092e-4
h3 = -5.77905e-7
k0 = 8.50935e-5
k1 = -6.12293e-6
k2 = 5.2787e-8
e0 = 19652.21
e1 = 148.4206
e2 = -2.327105
e3 = 1.360477e-2
e4 = -5.155288e-5
j0 = 1.91075e-4
i0 = 2.2838e-3
i1 = -1.0981e-5
i2 = -1.6078e-6
m0 = -9.9348e-7
m1 = 2.0816e-8
m2 = 9.1697e-10
f0 = 54.6746
f1 = -0.603459
f2 = 1.09987e-2
f3 = -6.1670e-5
g0 = 7.944e-2
g1 = 1.6483e-2
g2 = -5.3009e-4
#
def dPhidO2(O2,T,S,depth,Ac,Eo,dEodT):
    """compute partial derivative of Phi to O2 """
    P = sw.pres(depth, lat=0.)  # seawater pressure [db] !! Warning - z*0 neglects gravity differences w/ latitude
    return 1000.0*Ac*Patm*XiO2*(-0.1*depth/(S*(S**0.5*(T*t68*(T*g2*t68 + g1) + g0) + T*t68*(T*t68*(T*f3*t68 + f2) + f1) + f0) + T*t68*(T*t68*(T*t68*(T*e4*t68 + e3) + e2) + e1) + 0.1*depth*(S*(S**0.5*j0 + T*t68*(T*i2*t68 + i1) + i0) + T*t68*(T*t68*(T*h3*t68 + h2) + h1) + 0.1*depth*(S*(T*t68*(T*m2*t68 + m1) + m0) + T*t68*(T*k2*t68 + k1) + k0) + h0) + e0) + 1)*np.exp(kb_inv*(Eo + dEodT*(T - Tref))*(1.0/(T + T0_Kelvin) - 1.0/Tref_K))*np.exp(P*V*db2Pa/(R_gasconst*(T + T0_Kelvin)))*np.exp(-A0 - A1*np.log((-T + tc9)/(T + T0_Kelvin)) - A2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 - A3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0 - A4*np.log((-T + tc9)/(T + T0_Kelvin))**4.0 - A5*np.log((-T + tc9)/(T + T0_Kelvin))**5.0 - C0*S**2.0 - S*(B0 + B1*np.log((-T + tc9)/(T + T0_Kelvin)) + B2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 + B3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0))/(S**2*d0 + S*(T*t68*(T*t68*(T*t68*(T*b4*t68 + b3) + b2) + b1) + b0) + S**1.5*(T*t68*(T*c2*t68 + c1) + c0) + T*t68*(T*t68*(T*t68*(T*t68*(T*a5*t68 + a4) + a3) + a2) + a1) + a0)
#
def dPhidT(O2,T,S,depth,Ac,Eo,dEodT):
    """compute partial derivative of Phi to T """
    P = sw.pres(depth, lat=0.)  # seawater pressure [db] !! Warning - z*0 neglects gravity differences w/ latitude
    return -1000.0*Ac*O2*P*Patm*V*XiO2*db2Pa*(-0.1*depth/(S*(S**0.5*(T*t68*(T*g2*t68 + g1) + g0) + T*t68*(T*t68*(T*f3*t68 + f2) + f1) + f0) + T*t68*(T*t68*(T*t68*(T*e4*t68 + e3) + e2) + e1) + 0.1*depth*(S*(S**0.5*j0 + T*t68*(T*i2*t68 + i1) + i0) + T*t68*(T*t68*(T*h3*t68 + h2) + h1) + 0.1*depth*(S*(T*t68*(T*m2*t68 + m1) + m0) + T*t68*(T*k2*t68 + k1) + k0) + h0) + e0) + 1)*np.exp(kb_inv*(Eo + dEodT*(T - Tref))*(1.0/(T + T0_Kelvin) - 1.0/Tref_K))*np.exp(P*V*db2Pa/(R_gasconst*(T + T0_Kelvin)))*np.exp(-A0 - A1*np.log((-T + tc9)/(T + T0_Kelvin)) - A2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 - A3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0 - A4*np.log((-T + tc9)/(T + T0_Kelvin))**4.0 - A5*np.log((-T + tc9)/(T + T0_Kelvin))**5.0 - C0*S**2.0 - S*(B0 + B1*np.log((-T + tc9)/(T + T0_Kelvin)) + B2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 + B3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0))/(R_gasconst*(T + T0_Kelvin)**2*(S**2*d0 + S*(T*t68*(T*t68*(T*t68*(T*b4*t68 + b3) + b2) + b1) + b0) + S**1.5*(T*t68*(T*c2*t68 + c1) + c0) + T*t68*(T*t68*(T*t68*(T*t68*(T*a5*t68 + a4) + a3) + a2) + a1) + a0)) - 100.0*Ac*O2*Patm*XiO2*depth*(-S*(S**0.5*(T*g2*t68**2 + t68*(T*g2*t68 + g1)) + T*t68*(T*f3*t68**2 + t68*(T*f3*t68 + f2)) + t68*(T*t68*(T*f3*t68 + f2) + f1)) - T*t68*(T*t68*(T*e4*t68**2 + t68*(T*e4*t68 + e3)) + t68*(T*t68*(T*e4*t68 + e3) + e2)) - 0.1*depth*(S*(T*i2*t68**2 + t68*(T*i2*t68 + i1)) + T*t68*(T*h3*t68**2 + t68*(T*h3*t68 + h2)) + 0.1*depth*(S*(T*m2*t68**2 + t68*(T*m2*t68 + m1)) + T*k2*t68**2 + t68*(T*k2*t68 + k1)) + t68*(T*t68*(T*h3*t68 + h2) + h1)) - t68*(T*t68*(T*t68*(T*e4*t68 + e3) + e2) + e1))*np.exp(kb_inv*(Eo + dEodT*(T - Tref))*(1.0/(T + T0_Kelvin) - 1.0/Tref_K))*np.exp(P*V*db2Pa/(R_gasconst*(T + T0_Kelvin)))*np.exp(-A0 - A1*np.log((-T + tc9)/(T + T0_Kelvin)) - A2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 - A3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0 - A4*np.log((-T + tc9)/(T + T0_Kelvin))**4.0 - A5*np.log((-T + tc9)/(T + T0_Kelvin))**5.0 - C0*S**2.0 - S*(B0 + B1*np.log((-T + tc9)/(T + T0_Kelvin)) + B2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 + B3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0))/((S*(S**0.5*(T*t68*(T*g2*t68 + g1) + g0) + T*t68*(T*t68*(T*f3*t68 + f2) + f1) + f0) + T*t68*(T*t68*(T*t68*(T*e4*t68 + e3) + e2) + e1) + 0.1*depth*(S*(S**0.5*j0 + T*t68*(T*i2*t68 + i1) + i0) + T*t68*(T*t68*(T*h3*t68 + h2) + h1) + 0.1*depth*(S*(T*t68*(T*m2*t68 + m1) + m0) + T*t68*(T*k2*t68 + k1) + k0) + h0) + e0)**2*(S**2*d0 + S*(T*t68*(T*t68*(T*t68*(T*b4*t68 + b3) + b2) + b1) + b0) + S**1.5*(T*t68*(T*c2*t68 + c1) + c0) + T*t68*(T*t68*(T*t68*(T*t68*(T*a5*t68 + a4) + a3) + a2) + a1) + a0)) + 1000.0*Ac*O2*Patm*XiO2*(-0.1*depth/(S*(S**0.5*(T*t68*(T*g2*t68 + g1) + g0) + T*t68*(T*t68*(T*f3*t68 + f2) + f1) + f0) + T*t68*(T*t68*(T*t68*(T*e4*t68 + e3) + e2) + e1) + 0.1*depth*(S*(S**0.5*j0 + T*t68*(T*i2*t68 + i1) + i0) + T*t68*(T*t68*(T*h3*t68 + h2) + h1) + 0.1*depth*(S*(T*t68*(T*m2*t68 + m1) + m0) + T*t68*(T*k2*t68 + k1) + k0) + h0) + e0) + 1)*(dEodT*kb_inv*(1.0/(T + T0_Kelvin) - 1.0/Tref_K) - 1.0*kb_inv*(Eo + dEodT*(T - Tref))/(T + T0_Kelvin)**2)*np.exp(kb_inv*(Eo + dEodT*(T - Tref))*(1.0/(T + T0_Kelvin) - 1.0/Tref_K))*np.exp(P*V*db2Pa/(R_gasconst*(T + T0_Kelvin)))*np.exp(-A0 - A1*np.log((-T + tc9)/(T + T0_Kelvin)) - A2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 - A3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0 - A4*np.log((-T + tc9)/(T + T0_Kelvin))**4.0 - A5*np.log((-T + tc9)/(T + T0_Kelvin))**5.0 - C0*S**2.0 - S*(B0 + B1*np.log((-T + tc9)/(T + T0_Kelvin)) + B2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 + B3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0))/(S**2*d0 + S*(T*t68*(T*t68*(T*t68*(T*b4*t68 + b3) + b2) + b1) + b0) + S**1.5*(T*t68*(T*c2*t68 + c1) + c0) + T*t68*(T*t68*(T*t68*(T*t68*(T*a5*t68 + a4) + a3) + a2) + a1) + a0) + 1000.0*Ac*O2*Patm*XiO2*(-0.1*depth/(S*(S**0.5*(T*t68*(T*g2*t68 + g1) + g0) + T*t68*(T*t68*(T*f3*t68 + f2) + f1) + f0) + T*t68*(T*t68*(T*t68*(T*e4*t68 + e3) + e2) + e1) + 0.1*depth*(S*(S**0.5*j0 + T*t68*(T*i2*t68 + i1) + i0) + T*t68*(T*t68*(T*h3*t68 + h2) + h1) + 0.1*depth*(S*(T*t68*(T*m2*t68 + m1) + m0) + T*t68*(T*k2*t68 + k1) + k0) + h0) + e0) + 1)*(-S*(T*t68*(T*t68*(T*b4*t68**2 + t68*(T*b4*t68 + b3)) + t68*(T*t68*(T*b4*t68 + b3) + b2)) + t68*(T*t68*(T*t68*(T*b4*t68 + b3) + b2) + b1)) - S**1.5*(T*c2*t68**2 + t68*(T*c2*t68 + c1)) - T*t68*(T*t68*(T*t68*(T*a5*t68**2 + t68*(T*a5*t68 + a4)) + t68*(T*t68*(T*a5*t68 + a4) + a3)) + t68*(T*t68*(T*t68*(T*a5*t68 + a4) + a3) + a2)) - t68*(T*t68*(T*t68*(T*t68*(T*a5*t68 + a4) + a3) + a2) + a1))*np.exp(kb_inv*(Eo + dEodT*(T - Tref))*(1.0/(T + T0_Kelvin) - 1.0/Tref_K))*np.exp(P*V*db2Pa/(R_gasconst*(T + T0_Kelvin)))*np.exp(-A0 - A1*np.log((-T + tc9)/(T + T0_Kelvin)) - A2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 - A3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0 - A4*np.log((-T + tc9)/(T + T0_Kelvin))**4.0 - A5*np.log((-T + tc9)/(T + T0_Kelvin))**5.0 - C0*S**2.0 - S*(B0 + B1*np.log((-T + tc9)/(T + T0_Kelvin)) + B2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 + B3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0))/(S**2*d0 + S*(T*t68*(T*t68*(T*t68*(T*b4*t68 + b3) + b2) + b1) + b0) + S**1.5*(T*t68*(T*c2*t68 + c1) + c0) + T*t68*(T*t68*(T*t68*(T*t68*(T*a5*t68 + a4) + a3) + a2) + a1) + a0)**2 + 1000.0*Ac*O2*Patm*XiO2*(-0.1*depth/(S*(S**0.5*(T*t68*(T*g2*t68 + g1) + g0) + T*t68*(T*t68*(T*f3*t68 + f2) + f1) + f0) + T*t68*(T*t68*(T*t68*(T*e4*t68 + e3) + e2) + e1) + 0.1*depth*(S*(S**0.5*j0 + T*t68*(T*i2*t68 + i1) + i0) + T*t68*(T*t68*(T*h3*t68 + h2) + h1) + 0.1*depth*(S*(T*t68*(T*m2*t68 + m1) + m0) + T*t68*(T*k2*t68 + k1) + k0) + h0) + e0) + 1)*(-A1*(T + T0_Kelvin)*(-(-T + tc9)/(T + T0_Kelvin)**2 - 1/(T + T0_Kelvin))/(-T + tc9) - 2.0*A2*(T + T0_Kelvin)*(-(-T + tc9)/(T + T0_Kelvin)**2 - 1/(T + T0_Kelvin))*np.log((-T + tc9)/(T + T0_Kelvin))**1.0/(-T + tc9) - 3.0*A3*(T + T0_Kelvin)*(-(-T + tc9)/(T + T0_Kelvin)**2 - 1/(T + T0_Kelvin))*np.log((-T + tc9)/(T + T0_Kelvin))**2.0/(-T + tc9) - 4.0*A4*(T + T0_Kelvin)*(-(-T + tc9)/(T + T0_Kelvin)**2 - 1/(T + T0_Kelvin))*np.log((-T + tc9)/(T + T0_Kelvin))**3.0/(-T + tc9) - 5.0*A5*(T + T0_Kelvin)*(-(-T + tc9)/(T + T0_Kelvin)**2 - 1/(T + T0_Kelvin))*np.log((-T + tc9)/(T + T0_Kelvin))**4.0/(-T + tc9) - S*(B1*(T + T0_Kelvin)*(-(-T + tc9)/(T + T0_Kelvin)**2 - 1/(T + T0_Kelvin))/(-T + tc9) + 2.0*B2*(T + T0_Kelvin)*(-(-T + tc9)/(T + T0_Kelvin)**2 - 1/(T + T0_Kelvin))*np.log((-T + tc9)/(T + T0_Kelvin))**1.0/(-T + tc9) + 3.0*B3*(T + T0_Kelvin)*(-(-T + tc9)/(T + T0_Kelvin)**2 - 1/(T + T0_Kelvin))*np.log((-T + tc9)/(T + T0_Kelvin))**2.0/(-T + tc9)))*np.exp(kb_inv*(Eo + dEodT*(T - Tref))*(1.0/(T + T0_Kelvin) - 1.0/Tref_K))*np.exp(P*V*db2Pa/(R_gasconst*(T + T0_Kelvin)))*np.exp(-A0 - A1*np.log((-T + tc9)/(T + T0_Kelvin)) - A2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 - A3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0 - A4*np.log((-T + tc9)/(T + T0_Kelvin))**4.0 - A5*np.log((-T + tc9)/(T + T0_Kelvin))**5.0 - C0*S**2.0 - S*(B0 + B1*np.log((-T + tc9)/(T + T0_Kelvin)) + B2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 + B3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0))/(S**2*d0 + S*(T*t68*(T*t68*(T*t68*(T*b4*t68 + b3) + b2) + b1) + b0) + S**1.5*(T*t68*(T*c2*t68 + c1) + c0) + T*t68*(T*t68*(T*t68*(T*t68*(T*a5*t68 + a4) + a3) + a2) + a1) + a0)
#
def dPhidS(O2,T,S,depth,Ac,Eo,dEodT):
    """compute partial derivative of Phi to S """
    P = sw.pres(depth, lat=0.)  # seawater pressure [db] !! Warning - z*0 neglects gravity differences w/ latitude
    return -100.0*Ac*O2*Patm*XiO2*depth*(-1.5*S**0.5*(T*t68*(T*g2*t68 + g1) + g0) - T*t68*(T*t68*(T*f3*t68 + f2) + f1) - 0.1*depth*(1.5*S**0.5*j0 + T*t68*(T*i2*t68 + i1) + 0.1*depth*(T*t68*(T*m2*t68 + m1) + m0) + i0) - f0)*np.exp(kb_inv*(Eo + dEodT*(T - Tref))*(1.0/(T + T0_Kelvin) - 1.0/Tref_K))*np.exp(P*V*db2Pa/(R_gasconst*(T + T0_Kelvin)))*np.exp(-A0 - A1*np.log((-T + tc9)/(T + T0_Kelvin)) - A2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 - A3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0 - A4*np.log((-T + tc9)/(T + T0_Kelvin))**4.0 - A5*np.log((-T + tc9)/(T + T0_Kelvin))**5.0 - C0*S**2.0 - S*(B0 + B1*np.log((-T + tc9)/(T + T0_Kelvin)) + B2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 + B3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0))/((S*(S**0.5*(T*t68*(T*g2*t68 + g1) + g0) + T*t68*(T*t68*(T*f3*t68 + f2) + f1) + f0) + T*t68*(T*t68*(T*t68*(T*e4*t68 + e3) + e2) + e1) + 0.1*depth*(S*(S**0.5*j0 + T*t68*(T*i2*t68 + i1) + i0) + T*t68*(T*t68*(T*h3*t68 + h2) + h1) + 0.1*depth*(S*(T*t68*(T*m2*t68 + m1) + m0) + T*t68*(T*k2*t68 + k1) + k0) + h0) + e0)**2*(S**2*d0 + S*(T*t68*(T*t68*(T*t68*(T*b4*t68 + b3) + b2) + b1) + b0) + S**1.5*(T*t68*(T*c2*t68 + c1) + c0) + T*t68*(T*t68*(T*t68*(T*t68*(T*a5*t68 + a4) + a3) + a2) + a1) + a0)) + 1000.0*Ac*O2*Patm*XiO2*(-0.1*depth/(S*(S**0.5*(T*t68*(T*g2*t68 + g1) + g0) + T*t68*(T*t68*(T*f3*t68 + f2) + f1) + f0) + T*t68*(T*t68*(T*t68*(T*e4*t68 + e3) + e2) + e1) + 0.1*depth*(S*(S**0.5*j0 + T*t68*(T*i2*t68 + i1) + i0) + T*t68*(T*t68*(T*h3*t68 + h2) + h1) + 0.1*depth*(S*(T*t68*(T*m2*t68 + m1) + m0) + T*t68*(T*k2*t68 + k1) + k0) + h0) + e0) + 1)*(-1.5*S**0.5*(T*t68*(T*c2*t68 + c1) + c0) - 2*S*d0 - T*t68*(T*t68*(T*t68*(T*b4*t68 + b3) + b2) + b1) - b0)*np.exp(kb_inv*(Eo + dEodT*(T - Tref))*(1.0/(T + T0_Kelvin) - 1.0/Tref_K))*np.exp(P*V*db2Pa/(R_gasconst*(T + T0_Kelvin)))*np.exp(-A0 - A1*np.log((-T + tc9)/(T + T0_Kelvin)) - A2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 - A3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0 - A4*np.log((-T + tc9)/(T + T0_Kelvin))**4.0 - A5*np.log((-T + tc9)/(T + T0_Kelvin))**5.0 - C0*S**2.0 - S*(B0 + B1*np.log((-T + tc9)/(T + T0_Kelvin)) + B2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 + B3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0))/(S**2*d0 + S*(T*t68*(T*t68*(T*t68*(T*b4*t68 + b3) + b2) + b1) + b0) + S**1.5*(T*t68*(T*c2*t68 + c1) + c0) + T*t68*(T*t68*(T*t68*(T*t68*(T*a5*t68 + a4) + a3) + a2) + a1) + a0)**2 + 1000.0*Ac*O2*Patm*XiO2*(-0.1*depth/(S*(S**0.5*(T*t68*(T*g2*t68 + g1) + g0) + T*t68*(T*t68*(T*f3*t68 + f2) + f1) + f0) + T*t68*(T*t68*(T*t68*(T*e4*t68 + e3) + e2) + e1) + 0.1*depth*(S*(S**0.5*j0 + T*t68*(T*i2*t68 + i1) + i0) + T*t68*(T*t68*(T*h3*t68 + h2) + h1) + 0.1*depth*(S*(T*t68*(T*m2*t68 + m1) + m0) + T*t68*(T*k2*t68 + k1) + k0) + h0) + e0) + 1)*(-B0 - B1*np.log((-T + tc9)/(T + T0_Kelvin)) - B2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 - B3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0 - 2.0*C0*S**1.0)*np.exp(kb_inv*(Eo + dEodT*(T - Tref))*(1.0/(T + T0_Kelvin) - 1.0/Tref_K))*np.exp(P*V*db2Pa/(R_gasconst*(T + T0_Kelvin)))*np.exp(-A0 - A1*np.log((-T + tc9)/(T + T0_Kelvin)) - A2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 - A3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0 - A4*np.log((-T + tc9)/(T + T0_Kelvin))**4.0 - A5*np.log((-T + tc9)/(T + T0_Kelvin))**5.0 - C0*S**2.0 - S*(B0 + B1*np.log((-T + tc9)/(T + T0_Kelvin)) + B2*np.log((-T + tc9)/(T + T0_Kelvin))**2.0 + B3*np.log((-T + tc9)/(T + T0_Kelvin))**3.0))/(S**2*d0 + S*(T*t68*(T*t68*(T*t68*(T*b4*t68 + b3) + b2) + b1) + b0) + S**1.5*(T*t68*(T*c2*t68 + c1) + c0) + T*t68*(T*t68*(T*t68*(T*t68*(T*a5*t68 + a4) + a3) + a2) + a1) + a0) 

#
#def Phi(pO2, T, Ac, Eo, dEodT):
#    """compute the metabolic index"""
#    return Ac * pO2 * np.exp(kb_inv * (Eo + dEodT * (T - Tref)) * (1./(T + T0_Kelvin) - 1./Tref_K))


#def compute_pO2(O2, T, S, depth):
    # Solubility with pressure effect

#Phi = Ac * (O2 / (1e-3 * (np.exp(A0 + A1*(np.log((tc9-T)/(T0_Kelvin+T))) + A2*(np.log((tc9-T)/(T0_Kelvin+T)))**2. + A3*(np.log((tc9-T)/(T0_Kelvin+T)))**3. + A4*(np.log((tc9-T)/(T0_Kelvin+T)))**4. + A5*(np.log((tc9-T) /(T0_Kelvin+T)))**5. + S*(B0 + B1*(np.log((tc9-T)/(T0_Kelvin+T))) + B2*(np.log((tc9-T)/(T0_Kelvin+T)))**2. + B3*(np.log((tc9-T) /(T0_Kelvin+T)))**3.) + C0 * S**2.)) * ((a0+(a1+(a2+(a3+(a4+a5*T*t68)*T*t68)*T*t68)*T*t68)*T*t68+(b0+(b1+(b2+(b3+b4*T*t68)*T*t68)*T*t68)*T*t68)*S+(c0+(c1+c2*T*t68)*T*t68)*S*S**0.5 + d0*S**2) /(1-depth/10./((e0+(e1+(e2+(e3+e4*T*t68)*T*t68)*T*t68)*T*t68 + (f0+(f1+(f2+f3*T*t68)*T*t68)*T*t68+(g0+(g1+g2*T*t68)*T*t68)*S**0.5)*S) + (h0+(h1+(h2+h3*T*t68)*T*t68)*T*t68 + (i0+(i1+i2*T*t68)*T*t68 + j0*S**0.5)*S + (k0+(k1+k2*T*t68)*T*t68 + (m0+(m1+m2*T*t68)*T*t68)*S)*depth/10.)*depth/10.)))/(Patm*XiO2)))*np.exp(V*P*db2Pa/(R_gasconst*(T+To_Kelvin)))*np.exp(kb_inv*(Eo+dEodT*(T-Tref))*(1./(T+T0_Kelvin)-1./Tref_K))



#def dens(S, T, depth):
#(a0+(a1+(a2+(a3+(a4+a5*T*1.00024)*T*1.00024)*T*1.00024)*T*1.00024)*T*1.00024 + (b0+(b1+(b2+(b3+b4*T*1.00024)*T*1.00024)*T*1.00024)*T*1.00024)*S + (c0+(c1+c2*T*1.00024)*T*1.00024)*S*S**0.5 + d0*S**2) / (1 - depth/10./ ((e0+(e1+(e2+(e3+e4*T*1.00024)*T*1.00024)*T*1.00024)*T*1.00024 + (f0+(f1+(f2+f3*T*1.00024)*T*1.00024)*T*1.00024+(g0+(g1+g2*T*1.00024)*T*1.00024)*S**0.5)*S) + (h0+(h1+(h2+h3*T*1.00024)*T*1.00024)*T*1.00024 + (i0+(i1+i2*T*1.00024)*T*1.00024 + j0*S**0.5)*S + (k0+(k1+k2*T*1.00024)*T*1.00024 + (m0+(m1+m2*T*1.00024)*T*1.00024)*S)*depth/10.)*depth/10.))
#


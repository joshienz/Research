# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:21:51 2021

@author: NICHOLSJA18
"""

from scipy.special import factorial
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy import linalg
from scipy.integrate import quad
from scipy import special
from numba import jit
from sympy import *

PI = np.pi

# allows quad to call a function and see it as a cfunc 
#-- this speeds up numericla integration
from numba import cfunc, carray, jit
from numba.types import int32, float64, CPointer
from scipy import LowLevelCallable
def jit_quad_function(quad_function):
    jitted_function = jit(quad_function, nopython=True, cache=True)
   
    @cfunc(float64(int32, CPointer(float64)))
    def wrapped(len_u, u):
        #result = jitted_function(len_u, values)
        values = carray(u, len_u, dtype=float64)
        result = jitted_function(values[0], values[1:]) #values[1],values[2],values[3])
        return result
    
    return LowLevelCallable(wrapped.ctypes)




#returns the appropriate j function for specified l,k,r
@jit
def j_funct(l,params):
    r,k=params
    z = k*r
    if(l==0):
        return np.sin(z)
    elif(l==1):
        return (1/z)*np.sin(z)-np.cos(z)
    elif(l==2):
        return ((3/z**2)-1)*np.sin(z)-(3/z)*np.cos(z)
    elif(l==3):
        return ((15/z**3)-(6/z))*np.sin(z)+((-15/z**2)+1)*np.cos(z)
    elif(l==4):
        return((105/z**4)-(45/z**2)+1)*np.sin(z)+((-105/z**3)+(10/z))*np.cos(z)
    elif(l==5):
        return ((945/z**5)-(420/z**3)+(15/z))*np.sin(z)-((945/z**4)-(105/z**2)+1)*np.cos(z)
    elif(l==6):
        return ((10395/z**6)-(4725/z**4)+(210/z**2)-1)*np.sin(z)-((10395/z**5)-(1260/z**3)+(21/z))*np.cos(z)
    # elif(l==7):
    else:
        return
    
#returns the appropriate h function for specified l,k,r
@jit
def h_funct(l,params):
    r,k=params
    z = k*r
    i=1j
    x=i*z
    if(l==0):
        return np.exp(x)
    elif(l==1):
        return np.exp(x)*(1-x)/z
    elif(l==2):
        return -(np.exp(x)*(-3+3*x+z**2)/z**2)
    elif(l==3):
        return np.exp(x)*(15-15*x-6*z**2+i*z**3)/z**3
    elif(l==4):
        return np.exp(x)*(105-105*x-45*z**2+10*i*z**3+z**4)/z**4
    elif(l==5):
        return np.exp(x)*(945-945*x-420*z**2+105*i*z**3+15*z**4-i*z**5)/z**5
    elif(l==6):
        return (-np.exp(x)*(-10395+10395*x+4725*z**2-1260*i*z**3-210*z**4+21*i*z**5+z**6)/z**6)
    # elif(l==7):
    else:
        return


# # calculate integral for basis funtion of M00 matrix  
@jit_quad_function
def M00_integrand(r,params):  # use *params with @jit
    b,k,l=params
    param=(r,k)
    V = -1.0*np.exp(-r) 
    
    J = j_funct(l,param)
    integ = J*V*J/k
    
    return integ
# 

# # calculate integral for basis funtion over the MQ0 vector at i=0
@jit_quad_function
# =============================================================================
def MQ0_0_integrand(r,params):  # use *params with @jit
    b,k, c,l = params
    param=(r,k)
    V = -1.0*np.exp(-r) 
    
    J = j_funct(l,param)
    H= h_funct(l,param)
    G=(1-np.exp(-b*r))**(2*l+1)

    
    integ = G*H*V*J/k

    if c == 0:
        return integ.real
    else:
        return integ.imag 

#

# calculate integral for basis funtion over the MQ0 at i>0 vector 
@jit_quad_function
def MQ0_i_integrand(r,params):  # use *params with @jit
    i,b,k,a,l = params
    param=(r,k)
    V = -1.0*np.exp(-r) 
    
    J = j_funct(l,param)
    Q = r**i*np.exp(-a*r)
    integ = Q*V*J/np.sqrt(k)

    return integ


# =============================================================================
# calculate integral for the MQQ i=j=0 
@jit_quad_function
def MQQ_00_integrand(r,params):  # use *params with @jit
    b,k,c,l = params
    z=k*r
    param=(r,k)
    H= h_funct(l,param)
    if l==0:
        H_dot=1j*k*np.exp(1j*z)
        G_dub_dot=-b**2*np.exp(-b*r)
    else:
        H_dot = k*((l*h_funct(l-1,param)-(l+1)*h_funct(l+1,param))/(2*l+1))+H/r
        G_dub_dot=((2*l+1)*(2*l)*b**2*np.exp(-2*b*r)*(1-np.exp(-b*r))**(2*l-1)-
               (2*l+1)*b**2*np.exp(-b*r)*(1-np.exp(-b*r))**(2*l))

    V = -1.0*np.exp(-r) 
    G=(1-np.exp(-b*r))**(2*l+1)
    G_dot= ((2*l+1)*b*np.exp(-b*r)*(1-np.exp(-b*r))**(2*l))
   
    integ = (G*H/(k))*((-1/2)*G_dub_dot*H-G_dot*H_dot+V*H*G)

    if c == 0:
        return integ.real
    else:
        return integ.imag


# calculate integral for basis funtion over the MQQ i>0,j=0 
@jit_quad_function
def MQQ_i0_integrand(r,params):  # use *params with @jit
    i, b, k, a,c,l= params
    z=k*r
    param=(r,k)
    H= h_funct(l,param)
    if l==0:
        H_dot=1j*k*np.exp(1j*z)
        G_dub_dot=-b**2*np.exp(-b*r)
    else:
        H_dot = k*((l*h_funct(l-1,param)-(l+1)*h_funct(l+1,param))/(2*l+1))+H/r
        G_dub_dot=((2*l+1)*(2*l)*b**2*np.exp(-2*b*r)*(1-np.exp(-b*r))**(2*l-1)-
               (2*l+1)*b**2*np.exp(-b*r)*(1-np.exp(-b*r))**(2*l))

    V = -1.0*np.exp(-r) 
    G=(1-np.exp(-b*r))**(2*l+1)
    G_dot= ((2*l+1)*b*np.exp(-b*r)*(1-np.exp(-b*r))**(2*l))
    
    Q=r**i*np.exp(-a*r)
    
    integ = (Q/np.sqrt(k))*((-1/2)*G_dub_dot*H-G_dot*H_dot+V*H*G)

    if c == 0:
        return integ.real
    else:
        return integ.imag
# =============================================================================
# # calculate integral for basis funtion over the MQQ matrix at i>0,j>0

@jit_quad_function
def MQQ_ij_integrand(r,params):  # use *params with @jit
    i, j, b, k, a,l= params
    V = -1.0*np.exp(-r) 
    Qi=r**i*np.exp(-a*r)
    Qj=r**j*np.exp(-a*r)
    
    #using -1/2(d^2/dr^2+v-k^2/2+l(l+1)/2*r^2)
    integ = Qj*((l*(l+1)/(2*r**2))*Qi-(k**2/2)*Qi+V*Qi+
                (1/2)*(-np.exp(-a*r)*(i-1)*i*r**(i-2)+
                       2*a*np.exp(-a*r)*i*r**(i-1)-
                       a**2*np.exp(-a*r)*r**i))

    return integ

#
    
def main(): 
    for nbasis in range (1,2):
        '''user input'''
        realVal = -1.7449393#0.000419228 #.0195485 #-1.7449393  #Known value for these integrands
        B=1.0
        k=.1496663
        alpha=2.5
        l=0
        debug = True
        
        ''' END input'''
        
        M_mat =np.zeros((nbasis+1,nbasis+1),dtype=np.complex) 
        s_vec=np.zeros(nbasis+1,dtype=np.complex)
        s_T=np.zeros(nbasis+1,dtype=np.complex)
            
        for row in range (0, nbasis):
            for col in range (0, row+1):
    #                 
    #          # Writes in MQQ matrix where i and j > 0
    
               args_MQQ_ij =(row+1, col+1, B, k, alpha,l) 
               res0_MQQ_ij = quad(MQQ_ij_integrand, 0.0, np.inf, args_MQQ_ij, limit = 2000,epsrel=1e-13,epsabs=1e-13)                
    
               M_mat[row+1,col+1]= res0_MQQ_ij[0]+0j
               M_mat[col+1,row+1]= M_mat[row+1,col+1]
    
        # Add in the MQQ at i=j=0 elem real and imag parts
    
        args_MQQ_00=(B,k,0,l)    
        res0_MQQ_00_real = quad(MQQ_00_integrand, 0.0, np.inf, args_MQQ_00, limit = 2000,epsrel=1e-13,epsabs=1e-13) 
        args_MQQ_00=(B,k,1,l)    
        res0_MQQ_00_imag = quad(MQQ_00_integrand, 0.0, np.inf, args_MQQ_00, limit = 2000,epsrel=1e-13,epsabs=1e-13) 
        M_mat[0,0] =res0_MQQ_00_real[0] + 1j*res0_MQQ_00_imag[0]
    
# =============================================================================
#       First node in MQ0 vector
        args_MQ0_0= (B,k,0,l)
        res0_MQ0_0_real=  quad(MQ0_0_integrand, 0.0, np.inf, args_MQ0_0, limit = 5000,epsrel=1e-11,epsabs=1e-11)  
        args_MQ0_0= (B,k,1,l)
        res0_MQ0_0_imag=  quad(MQ0_0_integrand, 0.0, np.inf, args_MQ0_0, limit = 5000,epsrel=1e-11,epsabs=1e-11)  
        s_vec[0] = res0_MQ0_0_real[0] + 1j*res0_MQ0_0_imag[0]
        s_T[0] = res0_MQ0_0_real[0] + 1j*res0_MQ0_0_imag[0]
#     
# =============================================================================
        #diagonally symeytric so only need one half 
        for col in range (1,nbasis+1):
            #inserting Edge elements to MQQ matrix
            args_MQQ_i0 = (col, B, k, alpha,0,l)
            res0_MQQ_i0_real =  quad(MQQ_i0_integrand, 0.0, np.inf, args_MQQ_i0, limit = 2000,epsrel=1e-11,epsabs=1e-11) 
            args_MQQ_i0 = (col, B, k, alpha,1,l)
            res0_MQQ_i0_imag =  quad(MQQ_i0_integrand, 0.0, np.inf, args_MQQ_i0, limit = 5000,epsrel=1e-11,epsabs=1e-11) 
            M_mat[0,col]=res0_MQQ_i0_real[0] + 1j*res0_MQQ_i0_imag[0]
            #insert identical diagonal
            M_mat[col,0]= M_mat[0,col]  
            #MQ0 vector 
            args_MQ0_i=(col,B,k,alpha,l)
            res0_MQ0_i = quad(MQ0_i_integrand, 0.0, np.inf, args_MQ0_i, limit = 2000,epsrel=1e-13,epsabs=1e-13) 
            s_vec[col]=res0_MQ0_i[0]+0j
            s_T[col]=res0_MQ0_i[0]+0j



        args_M00 =(B,k,l)
        res0_M00 = quad(M00_integrand, 0.0, np.inf, args_M00, limit = 2000,epsrel=1e-13,epsabs=1e-13)  
        M00 = res0_M00[0]+0j
        M_inv=np.linalg.inv(M_mat)
        lmbda =(M00-s_vec@M_inv@s_vec)
        if debug:
            print(M00)
            print(s_vec)
            print(M_mat)
            print(lmbda)
       
        lmbda_est = lmbda.imag/lmbda.real
        print(nbasis)
        print("lmbda Estimate: %.7f"%(lmbda_est))
        lmbdaRatio = lmbda_est/realVal
        print("Ratio to real value: %.4f"%(lmbdaRatio))

        #(Fmat1_vals, Fmat1_vecs) = linalg.eig(M_inv)
        #print(Fmat1_vals)
        if nbasis==1:
            simple_lmbda = (-2.0)*(M00-((s_vec[0])*(s_vec[0])/(M_mat[0,0])))
            #test =   -1.7449393  #  -2j*1.7449393/(1.0+1j*1.7449393)
            print("simple")
            print(simple_lmbda)
            """
            correct_T =-test/(PI*(1-1j*test))
            print( correct_T)
            con_T = np.conjugate(correct_T)
            print(con_T)
            con_K = -1j*PI*con_T/(1j+PI*con_T)
            print(con_K)
            print(" ")
            #print(-2*jVj)
            """
    # =============================================================================
    #     
    #             #    print(norm[row],Ham[row,col])
    #         for col in range (1, nbasis+1):
    #             args_GH =(order[col-1],  norm[col-1], c, scale,a,k,0)
    #             res0_GH_real = quad(GH_integrand, 0.0, np.inf, args_GH, limit = 2000,epsrel=1e-13,epsabs=1e-13)  
    #             args_GH =(order[col-1],  norm[col-1], c, scale,a,k,1)
    #             res0_GH_imag = quad(GH_integrand, 0.0, np.inf, args_GH, limit = 2000,epsrel=1e-13,epsabs=1e-13)              
    #             args_Hj =(order[col-1],  norm[col-1], c, scale,a,k)
    #             res0_Hj = quad(Hj_integrand, 0.0, np.inf, args_Hj, limit = 2000,epsrel=1e-13,epsabs=1e-13)             
    # 
    #             M_mat[0,col] = res0_GH_real[0] + 1j*res0_GH_imag[0]
    #             M_mat[col,0] = M_mat[0,col]
    #             
    #             s_vec[col]=res0_Hj[0]+0j
    #     
    #         args_Gj =(order[col-1],  norm[col-1], c, scale,a,k,0)
    #         res0_Gj_real = quad(Gj_integrand, 0.0, np.inf, args_Gj, limit = 2000,epsrel=1e-13,epsabs=1e-13) 
    #         args_Gj =(order[col-1],  norm[col-1], c, scale,a,k,1)
    #         res0_Gj_imag = quad(Gj_integrand, 0.0, np.inf, args_Gj, limit = 2000,epsrel=1e-13,epsabs=1e-13) 
    #         
    #         s_vec[0] = res0_Gj_real[0]+1j*res0_Gj_imag[0]
    # 
    #         args_jVj =(k,1)
    #         res0_jVj = quad(jVj_integrand, 0.0, np.inf, args_jVj, limit = 2000,epsrel=1e-13,epsabs=1e-13) 
    #         
    #         jVj = res0_jVj[0]+0j
    #         
    #         M_inv = np.linalg.inv(M_mat)
    #         s_T= s_vec.getH()
    #         lmbda = -(2.0/k)*(jVj+s_T@M_inv@s_vec)
    # =============================================================================
        
     
     
        

   
    
if __name__ == '__main__':
    main()
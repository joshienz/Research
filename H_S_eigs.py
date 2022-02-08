# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:21:51 2021
updated 7/9/2021

This program uses basis functions to get the phase shift of a 

@author: NICHOLSJA18
"""

from scipy.special import sph_harm
import numpy as np
import scipy
import math
from scipy.integrate import nquad
from numba import jit, prange

PI = np.pi

# allows nnquad to call a function and see it as a cfunc 
#-- this speeds up numericla integration
from numba import cfunc, carray
import numba as nb
from numba.types import int32, float64, CPointer
from scipy import LowLevelCallable
#:::::::::::::::::::::::::::::::::numerical integration functions for potential
#...........................................................jit_nquad3_function
# This function wraps the integration in C code, which runs faster.
def jit_nquad3_function(quad_function): 
# Last modified June 9, 2020 by mcf, bjc
    jitted_function = jit(quad_function, nopython=True)
    @cfunc(float64(int32, CPointer(float64)))
    def wrapped(len_u, u):
        values = carray(u, len_u, dtype=float64)
        result = jitted_function(values) 
        return result
    return LowLevelCallable(wrapped.ctypes)

import matplotlib.pyplot as plt

# Adjust for screen resolution
if plt.get_backend() == 'Qt5Agg':
    import sys
    from matplotlib.backends.qt_compat import QtWidgets
    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = 0.8*qApp.desktop().physicalDpiX()


from scipy.special import spherical_jn, spherical_yn

# Adjust for screen resolution
if plt.get_backend() == 'Qt5Agg':
    import sys
    from matplotlib.backends.qt_compat import QtWidgets
    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = 0.8*qApp.desktop().physicalDpiX()




from numba import njit
import ctypes
from numba.extending import get_cython_function_address

# =============================================================================
# create numba compatible spherical Bessel and spherical Neumann functions
#   (these functions take a double argument and return a double)
# =============================================================================
scale_thresh = .1

pi_val = 3.14159265358979323846
_dble = ctypes.c_double
functype = ctypes.CFUNCTYPE(_dble, _dble, _dble)

addr_jv = get_cython_function_address("scipy.special.cython_special",
                                      '__pyx_fuse_1jv')
jv_float64_fn = functype(addr_jv)

addr_yv = get_cython_function_address("scipy.special.cython_special",
                                      '__pyx_fuse_1yv')
yv_float64_fn = functype(addr_yv)

# spherical Bessel function  (assumes z>=0 always)
@jit
def sph_jn_numba(n,z):
    if z < 1e-100:    
        if n==0.0:
            return 1.0
        else:
            return 0.0
    else: 
        return (pi_val/(2*z))**0.5 * jv_float64_fn(n+0.5, z)
 
# spherical Neumann function  (assumes z>=0 always)
@jit
def sph_yn_numba(n,z):
    if z < 1e-305:
        return np.inf
    else: 
        return -(pi_val/(2*z))**0.5 * yv_float64_fn(n+0.5, z)

# =============================================================================
# end spherical Bessel and spherical Neumann functions
# =============================================================================



#==============================================================================
#==============================================================================
# A list of each Real Spherical Harmonics for each symmetry. Each RSH has its 
    # own list of the form: l,m. 
basisTypes = [
    [[1,0],[0,0]]
    #[[3,0],[2,0],[1,0],[0,0]],
    #[[3,0],[2,0],[1,0],[0,0]]#,
    #[[3,1],[2,1],[1,1]],
    #[[3,1],[2,1],[1,1]]#,
    #[[2,-2]],
    #[[2,-2]]
    ]

# These are the exponential constants that can be used. They only need to be 
    # changed if you change the basis set. Each row corresponds to an l value.
    # ex: if a symmetry has a p orbital (l=1), it will include the second list
    
alpha_standards = [ [5,2.5,1.25,0.625,0.3125,0.15625,0.078125],#,0.0390625,0.01953125,0.009765625,0.0048828125,0.0024414063,0.001220703,0.0006103516],
                   [5,2.5,1.25,0.625,0.3125,0.15625,0.078125],#,0.0740740740,0.02646913556],#,2.5,1.25,0.625,0.3125,0.1625,0.08125],#3.3333333,1.111111111,0.37037037,0.12345679,0.0411522634],#,0.0137174211,0.00457247],
                   [5,2.5,1.25,0.625,0.3125,0.15625,0.078125],#0.6666667,0.22222222222,0.07407407,0.024691358,0.0082304527],
                   [5,2.5,1.25,0.625,0.3125,0.15625,0.078125],#,0.1666666666],#,0.055555556]
                   [3,1.5]
                   ]


# the number of terms for each basisfunction, accessed by the coordinates:
    # [l][m+l]. Extra space is filled by zeroes because numba will not
    # accept multidimensional arrays of non-uniform length. The zeros should 
    # never be accessed, and even if they were, they would not create problems.
numberOfTerms = np.array([
        # s orbitals, l=0, m = 0
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        # p orbitals, l=1, m= -1,0,1
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        # d orbitals, l=2, m= -2...2
        [1, 1, 3, 1, 2, 0, 0, 0, 0],
        # f orbitals, l=3, m= -3...3
        [2, 1, 3, 3, 3, 2, 2, 0, 0],
        # g orbitals, l=4, m= -4...4
        [2, 2, 3, 3, 6, 3, 4, 2, 3]
        ], dtype=np.int32)
        
        
        
'''*************************************************************************'''
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::Other Globals
# Last modified July 6, 2020 by bjc

# This is used to keep ints from being floored. int(float) = floor(float), so 
    # we use int(float + int_c).
int_c = 0.000001
# orientation matrices: a coefficient, xpower, ypower, zpower for every term 
    # associated with that orientation of that orbital. These are then put in
    # the orbitals_matrix. Extra space is filled with zeroslists because 
    # numba will not accept multidimensional arrays of non-uniform length.
zeroslist1 = [0,0,0,0]
s_orbital_0 = [[1,0,0,0],zeroslist1,zeroslist1,
               zeroslist1,zeroslist1,zeroslist1]

p_orbital_0 = [[1,0,0,1],zeroslist1,zeroslist1,
               zeroslist1,zeroslist1,zeroslist1]
p_orbital_1 = [[1,1,0,0],zeroslist1,zeroslist1,
              zeroslist1,zeroslist1,zeroslist1]
p_orbital_n1 = [[1,0,1,0],zeroslist1,zeroslist1,
                zeroslist1,zeroslist1,zeroslist1]

d_orbital_0 = [[-1,2,0,0],[-1,0,2,0],[2,0,0,2],
               zeroslist1,zeroslist1,zeroslist1]
d_orbital_1 = [[1,1,0,1],zeroslist1,zeroslist1,
               zeroslist1,zeroslist1,zeroslist1]
d_orbital_n1 = [[1,0,1,1],zeroslist1,zeroslist1,
               zeroslist1,zeroslist1,zeroslist1]
d_orbital_2 = [[1,2,0,0],[-1,0,2,0],zeroslist1,
               zeroslist1,zeroslist1,zeroslist1]
d_orbital_n2 = [[1,1,1,0],zeroslist1,zeroslist1,
                zeroslist1,zeroslist1,zeroslist1]

f_orbital_0 = [[-3,2,0,1],[-3,0,2,1],[2,0,0,3],
               zeroslist1,zeroslist1,zeroslist1]
f_orbital_1 = [[4,1,0,2],[-1,3,0,0],[-1,1,2,0],
               zeroslist1,zeroslist1,zeroslist1]
f_orbital_n1 = [[4,0,1,2],[-1,2,1,0],[-1,0,3,0],
                zeroslist1,zeroslist1,zeroslist1]
f_orbital_2 = [[1,2,0,1],[-1,0,2,1],zeroslist1,
               zeroslist1,zeroslist1,zeroslist1]
f_orbital_n2 = [[1,1,1,1],zeroslist1,zeroslist1,
                zeroslist1,zeroslist1,zeroslist1]
f_orbital_3 = [[1,3,0,0],[-3,1,2,0],zeroslist1,
               zeroslist1,zeroslist1,zeroslist1]
f_orbital_n3 = [[3,2,1,0],[-1,0,3,0],zeroslist1,
                zeroslist1,zeroslist1,zeroslist1]

g_orbital_0 = [[1.5,4,0,0],[1.5,0,4,0],[-12,2,0,2],
               [-12,0,2,2],[3,2,2,0],[1,0,0,4]]
g_orbital_1 = [[-7.5,3,0,1],[-7.5,1,2,1],[10,1,0,3],
               zeroslist1,zeroslist1,zeroslist1]
g_orbital_n1 = [[-7.5,2,1,1],[-7.5,0,3,1],[10,0,1,3],
                zeroslist1,zeroslist1,zeroslist1]
g_orbital_2 = [[-7.5,4,0,0],[7.5,0,4,0],[45,2,0,2],
               [-45,0,2,2],zeroslist1,zeroslist1]
g_orbital_n2 = [[-7.5,3,1,0],[-7.5,1,3,0],[45,1,1,2],
                zeroslist1,zeroslist1,zeroslist1]
g_orbital_3 = [[1,3,0,1],[-3,1,2,1],zeroslist1,
               zeroslist1,zeroslist1,zeroslist1]
g_orbital_n3 = [[3,2,1,1],[-1,0,3,1],zeroslist1,
                zeroslist1,zeroslist1,zeroslist1]
g_orbital_4 = [[1,4,0,0],[-6,2,2,0],[1,0,4,0],
               zeroslist1,zeroslist1,zeroslist1]
g_orbital_n4 = [[1,3,1,0],[-1,1,3,0],zeroslist1,
                zeroslist1,zeroslist1,zeroslist1]

zeroslist2 = [zeroslist1,zeroslist1,zeroslist1,
              zeroslist1,zeroslist1,zeroslist1]

# This array gives the information about each term for each basisfunction. The 
    # coordinates are given by the [l][m+l] of the basisfunction.
orbitals_matrix = np.array([ 
            # s orbital, l=0,m=0
            [s_orbital_0, zeroslist2, zeroslist2, zeroslist2,
             zeroslist2, zeroslist2, zeroslist2, zeroslist2, zeroslist2],
            # p orbitals, l=1, m= -1,0,1
            [p_orbital_n1, p_orbital_0, p_orbital_1, zeroslist2,
             zeroslist2, zeroslist2, zeroslist2, zeroslist2, zeroslist2],
            # d orbitals, l=2, m= -2...2
            [d_orbital_n2, d_orbital_n1, d_orbital_0, d_orbital_1, 
             d_orbital_2, zeroslist2, zeroslist2, zeroslist2, zeroslist2],
            # f orbitals, l=3, m= -3...3
            [f_orbital_n3, f_orbital_n2, f_orbital_n1, f_orbital_0, 
             f_orbital_1, f_orbital_2, f_orbital_3, zeroslist2, zeroslist2],
            # g orbitals, l=4, m= -4...4
            [g_orbital_n4, g_orbital_n3, g_orbital_n2, g_orbital_n1, 
             g_orbital_0, g_orbital_1, g_orbital_2, g_orbital_3, 
             g_orbital_4]
          ]) 


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::matrix constructors
@jit
def construct_T(numSymmetry,norm,sym,scaleFactor): 
# Last modified July 9, 2020 by bjc
    z_atom1 = 1.0
    z_atom2 = -1.0
    # The number of basis functions used in this symmetry.
    numbasis = num_Basis_Functions[numSymmetry]
    # The basisFunctionSet for this symemtry
    SymSet = Symmetries[numSymmetry]
    
    # Creating the T matrix to be filled later.
    Tmat = np.zeros((numbasis,numbasis), dtype=np.float64)
    
    # We need every non-duplicate combination of basis functions 
        # in this symmetry. Each basis function refers to a list 
        # of l, m, alpha.
    for basisfunction1 in range(0, numbasis):
        lma1 = SymSet[basisfunction1]
        l1 = np.int32(lma1[0] + int_c)
        m1 = np.int32(lma1[1] + int_c)
        ml1 = m1 + l1
        # combo_sign determines whether the terms are added or subtracted. This
            # is contingent on the orbital type, and symmetry.
        combo_sign1 = sym * (-1)**ml1
        
        # If the alpha is less than the scale threshold, we multiply it
            # by the scaleFactor.
        alpha1 = lma1[2]
        if alpha1 < scale_thresh:
            alpha1 *= scaleFactor
            
        # This is the array of terms associated with the orbital in this 
            # basisfunction.
        orbital_1 = orbitals_matrix[l1][ml1]
    
        # Same stuff for the other basis function.
        for basisfunction2 in range(0, basisfunction1+1):
            lma2 = SymSet[basisfunction2]
            l2 = np.int32(lma2[0] + int_c)
            m2 = np.int32(lma2[1] + int_c)
            ml2 = m2 + l2
            combo_sign2 = sym * (-1)**ml2
            
            alpha2 = lma2[2]
            if alpha2 < scale_thresh:
                alpha2 *= scaleFactor
                
            orbital_2 = orbitals_matrix[l2][ml2]
            
            # Calculating the elements of the kinetic energy matirx.
            
            # The number of terms for each basis function is determined by
                # the orbital type it is composed of. The number of terms is
                # found in the numberOfTerms array and can be accessed with the 
                # l and m+l of the orbital as coordinates.
            for i in range(0, numberOfTerms[l1][ml1]):
                # Each term for a basis function is also determined by the 
                    # orbital it is composed of. The information for each 
                    # term is found in the orbitals matrix. Each term is given 
                    # as a list of 4 numbers: coefficient of the term, xpower,
                    # ypower, zpower.
                term_b1 = orbital_1[i]
                c1 = term_b1[0]
                xpow1 = term_b1[1]
                ypow1 = term_b1[2]
                zpow1 = term_b1[3]
               
                # Same stuff as above, but for the other basisfunction.
                for j in range(0, numberOfTerms[l2][ml2]):
                    term_b2 = orbital_2[j]
                    c2 = term_b2[0]
                    xpow2 = term_b2[1]
                    ypow2 = term_b2[2]
                    zpow2 = term_b2[3]
                    
                    # Here is the calculation for the elements of the T matrix.
                        # Each element is a combination of four terms: t1 - t4.
                        # It is also normalized.
                    t1 = (kinetic(
                            alpha1, xpow1, ypow1, zpow1, 
                            z_atom1, 
                            alpha2, xpow2, ypow2, zpow2,
                            z_atom1))
                    t2 = (kinetic(
                            alpha1, xpow1, ypow1, zpow1,
                            z_atom1,
                            alpha2, xpow2, ypow2, zpow2,
                            z_atom2) )
                    t3 = (kinetic(
                            alpha1, xpow1, ypow1, zpow1, 
                            z_atom2, 
                            alpha2, xpow2, ypow2, zpow2,
                            z_atom1))
                    t4 = (kinetic(
                            alpha1, xpow1, ypow1, zpow1,
                            z_atom2,
                            alpha2, xpow2, ypow2, zpow2,
                            z_atom2) )
                    
                    Tmat[basisfunction1,basisfunction2] += (
                            norm[basisfunction1] * norm[basisfunction2] *
                           c1*c2 * (t1 + combo_sign2*t2 + 
                                  combo_sign1*t3 
                             + combo_sign1*combo_sign2*t4)  )
            
            # Tmat is symmetric, so row,col = col,row.
            Tmat[basisfunction2,basisfunction1] = (
                    Tmat[basisfunction1,basisfunction2])
                    
    return Tmat
#constsruct_T

@jit
def kinetic(alpha1, l1, m1, n1, za, alpha2, l2, m2, n2, zb):
    """
    This function uses a pre-integrated formula to calculate elements for the 
    T matrix. It is called by construct_T.

    Parameters
    ----------
    alpha1, alpha2 : The alphas for the two basis functions.
    l1,l2 : The power on x for each basis function.
    m1,m2 : The power on y for each basis function.
    n1,n2 : The power on z for each basis function.
    za,zb : The z coordinates of the two atoms

    Returns
    -------
    term0+term1+term2: The result of the integration.

    """
# Last modified Jun 12, 2020 by bjc
    
    term0 = alpha2*(2*(l2+m2+n2)+3)*overlap(alpha1, l1, m1, n1, za,
                                            alpha2, l2, m2, n2, zb)

    term1 = -2*(alpha2**2.0)*(overlap(alpha1, l1, m1, n1, za,
                                      alpha2, l2+2, m2, n2, zb)
                              +overlap(alpha1, l1, m1, n1, za,
                                       alpha2, l2, m2+2, n2, zb)
                              +overlap(alpha1, l1, m1, n1, za,
                                       alpha2, l2, m2, n2+2, zb))
    
    term2 = -0.5*(l2*(l2-1)*overlap(alpha1, l1, m1, n1, za,
                            alpha2, l2-2, m2, n2, zb) +
            m2*(m2-1)*overlap(alpha1, l1, m1, n1, za,
                            alpha2, l2, m2-2, n2, zb)+
            n2*(n2-1)*overlap(alpha1, l1, m1, n1, za, 
                            alpha2, l2, m2, n2-2, zb))
    return term0+term1+term2
#kinetic

@jit
def overlap(alpha1, l1, m1, n1, za, alpha2, l2, m2, n2, zb):
    """
    This method finds the total overlap by calling overlap_1D to find 
        the overlap in each dimension separately.

    Parameters
    ----------
    alpha1, alpha2 : The alphas for the two basis functions.
    l1,l2 : The power on x for each basis function.
    m1,m2 : The power on y for each basis function.
    n1,n2 : The power on z for each basis function.
    za,zb : The z coordinates of the two atoms

    Returns
    -------
    pre*wx*wy*wz: The result of the integration.

    """
# Last modified July 7, 2020 by bjc
    
    # Computes z coordinate of the center of the product of the 
        # gaussians.
    gamma = alpha1 + alpha2
    zp = (alpha1*za + alpha2*zb)/gamma
    
    # Calculating a prefactor coefficient.
    zdiff = za-zb
    rab2 = (zdiff)*(zdiff) # (xdiff)*(xdiff) + (ydiff)*(ydiff) + (zdiff)*(zdiff)
    pre = ((PI/gamma)**1.5)*np.exp(-alpha1*alpha2*rab2/gamma)
                              
    # Calculating the overlap in each dimension.
    wx = overlap_1D(l1, l2, 0.0,   0.0,   gamma)
    wy = overlap_1D(m1, m2, 0.0,   0.0,   gamma)
    wz = overlap_1D(n1, n2, zp-za, zp-zb, gamma)
    
    return pre*wx*wy*wz
#overlap

@jit    
def overlap_1D(pow1, pow2, PA, PB, gamma):
    """
    This function computes the overlap for one dimension.

    Parameters
    ----------
    pow1, pow2 : The powers on the relevant dimension of each basis 
                function. Ex: for the x dimension, pow1 = the power on x
                for the first basis function, pow2 = the power on x for  
                the second basis function.
    PA,PB : The distance from the center of the product in the 
             relevant dimension
    gamma : The sum of the alphas from the two basis functions.

    Returns
    -------
    sum: The sum of the overlap.

    """
    
#Last modified July 2020 bjc
    sum = 0.0
    i = 0
    while (i < 1+math.floor(0.5*(pow1+pow2))):
        sum += ( binomial_prefactor(2*i, pow1, pow2, PA, PB)*fact2(2*i-1)/
                ((2.0*gamma)**i)  )
        i += 1
    return sum
#overlap_1D

#Table of factorials
fact_LOOKUP_TABLE = np.array([
     1, 1, 2, 6, 24, 120, 720, 5040, 40320,
     362880, 3628800, 39916800, 479001600,
     6227020800, 87178291200, 1307674368000,
     20922789888000, 355687428096000, 6402373705728000,
     121645100408832000, 2432902008176640000], dtype='float64')


@jit#(cache=True)
def fact(m):
    """
    This function computes single factorials. Since this function is 
        used so often, it is faster to look up a value from a list than 
        to keep doing the same calculations over and over.

    Parameters
    ----------
    m : The integer we want the factorial of.

    Returns
    -------
    y or fact_LOOKUP_TABLE[n]: The factorial of the given int.

    """
# Last modified June 16, 2020 by bjc
    
    n = int(m + int_c)
    # The table contains factorials up to 20!. If n is greater, we calculate 
         # it starting at 20! so that we don't do redundant calculations.
    if n > 20:
        y = fact_LOOKUP_TABLE[20]
        for i in prange(21,n+1):
            y *= i
        return y
    return fact_LOOKUP_TABLE[n]
#fact

#Table of double factorials
fact2_LOOKUP_TABLE = np.array([
     1, 1, 2, 3, 8, 15, 48, 105, 384, 945,
     3840, 10395, 46080, 135135, 645120,
     2027025, 10321920, 34459425, 185794560,
     654729075, 3715891200, 13749310575,
     81749606400, 316234143225, 1961990553600,
     7905853580625, 51011754393600, 1], dtype='float64')
#mcf  Note: 1 added at end of list so that -1!! = 1 

@jit#(cache=True)
def fact2(m):
    """
    This function computes single factorials. Since this function is 
        used so often, it is faster to look up a value from a list than 
        to keep doing the same calculations over and over.

    Parameters
    ----------
    m : The integer we want the factorial of.

    Returns
    -------
    y or fact_LOOKUP_TABLE[n]: The factorial of the given int.

    """
# Last modified June 16, 2020 by bjc
     
    #if m == -1: return 1.0  # implemented in list
    n = int(m + int_c)
    # The table contains factorials -1!! - 26!!. If n is greater, we calculate 
        # it starting at 26!! so that we don't do redundant calculations.
    if n > 26:
        if n & 1:  # n odd
            y = fact2_LOOKUP_TABLE[25]
            for i in prange(27,n+1,2):
                y *= i
            return y
        else:   # n even
            y = fact2_LOOKUP_TABLE[26]
            for i in prange(28,n+1,2):
                y *= i
            return y
    return fact2_LOOKUP_TABLE[n]
#fact2

#...........................................................binomial_prefactors
@jit
def binomial_prefactor(s, ia, ib, xpa, xpb): 
# Last modified ???
    sum = 0.0
    for t in range (0, s+1):
        if (s-ia <= t) and (t <= ib):
            sum += ( binomial(ia,s-t)*binomial(ib,t)*
                    (xpa**(ia-s+t))*(xpb**(ib-t)) )
    return sum

#......................................................................binomial
@jit
def binomial(a, b):
    """
    This function calculates and returns the binomial coefficient for a and b.
    """
    return fact(a)/(fact(b)*fact(a-b))

#...................................................................construct_S
# This function creates the Overlap and norm matrices. 
# Parameters -> numSymmetry: An integer index that refers to the 
#                   current symmetry.
#               sym: 1 or -1. This determines whether the symmetry is 
#                   bonding or antibonding.
#               scaleFactor: This is the amount all alphas below the
#                   scale threshold are scaled by.
# Returns -> norm: A matrix of list of normalization constants.
#            Smat: The overlap matrix.
@jit
def construct_S(numSymmetry,sym,scaleFactor): 
# last modified July 9, 2020 by bjc
    z_atom1 = 1.0
    z_atom2 = -1.0
    # The number of basis functions used in this symmetry.
    numbasis = num_Basis_Functions[numSymmetry]
    # The basisFunctionSet for this symemtry
    SymSet = Symmetries[numSymmetry]
    
    # Creating matrices to be filled later.
    norm = np.zeros((numbasis), dtype=np.float64)
    Smat = np.zeros((numbasis,numbasis), dtype=np.float64)

     # We need every non-duplicate combination of basis functions 
        # in this symmetry. Each basis function refers to a list 
        # of l, m, alpha.
    for basisfunction1 in range(0,numbasis):
        lma1 = SymSet[basisfunction1]
        l1 = np.int32(lma1[0] + int_c)
        ml1 = l1 + np.int32(lma1[1] + int_c)
        # combo_sign determines whether the terms are added or subtracted. This
            # is contingent on the orbital type, and symmetry.
        combo_sign1 = sym * (-1)**ml1
        
        # If the alpha is less than the scale threshold, we multiply it
           # by the scaleFactor.
        alpha1 = lma1[2]
        if alpha1 < scale_thresh:
            alpha1 *= scaleFactor
    
        # This is the array of terms associated with the orbital in this 
            # basisfunction.
        orbital_1 = orbitals_matrix[l1][ml1]
        
        # Same stuff for the other basis function.
        for basisfunction2 in range(0,basisfunction1+1):
            lma2 = SymSet[basisfunction2]
            l2 = np.int32(lma2[0] + int_c)
            ml2 = l2 + np.int32(lma2[1] + int_c)
            combo_sign2 = sym * (-1)**ml2
            
            alpha2 = lma2[2]
            if alpha2 < scale_thresh:
                alpha2 *= scaleFactor
                
            orbital_2 = orbitals_matrix[l2][ml2]
            
            # Calculating the elements of the overlap matrix.
            
            # The number of terms for each basis function is determined by
                # the orbital type it is composed of. The number of terms is
                # found in the numberOfTerms array and can be accessed with the 
                # l and m+l of the orbital as coordinates.
            for i in range(0, numberOfTerms[l1][ml1]):
                # Each term for a basis function is also determined by the 
                    # orbital it is composed of. The information for each 
                    # term is found in the orbitals matrix. Each term is given 
                    # as a list of 4 numbers: coefficient of the term, xpower,
                    # ypower, zpower.
                term_b1 = orbital_1[i]
                c1 = term_b1[0]
                xpow1 = term_b1[1]
                ypow1 = term_b1[2]
                zpow1 = term_b1[3]
                
                # Same stuff for the other basis function.
                for j in range(0, numberOfTerms[l2][ml2]):
                    term_b2 = orbital_2[j]
                    c2 = term_b2[0]
                    xpow2 = term_b2[1]
                    ypow2 = term_b2[2]
                    zpow2 = term_b2[3]
                    
                    # Calculating each element of the S matrix. Each element is
                        # a combination of four terms: t1 - t4.
                    t1 = (overlap(
                            alpha1, xpow1, ypow1, zpow1, 
                            z_atom1, 
                            alpha2, xpow2, ypow2, zpow2,
                            z_atom1))
                    t2 = (overlap(
                            alpha1, xpow1, ypow1, zpow1,
                            z_atom1,
                            alpha2, xpow2, ypow2, zpow2,
                            z_atom2) )
                    t3 = (overlap(
                            alpha1, xpow1, ypow1, zpow1, 
                            z_atom2, 
                            alpha2, xpow2, ypow2, zpow2,
                            z_atom1))
                    t4 = (overlap(
                            alpha1, xpow1, ypow1, zpow1,
                            z_atom2,
                            alpha2, xpow2, ypow2, zpow2,
                            z_atom2) )
                    
                    Smat[basisfunction1,basisfunction2] += (
                            c1 * c2 * 
                            (t1 + combo_sign2*t2 + combo_sign1*t3 + 
                            combo_sign1*combo_sign2*t4) )
                    
            # This creates the norm matrix.
            if basisfunction1 == basisfunction2:
                norm[basisfunction1] = 1.0/math.sqrt(
                        Smat[basisfunction1,basisfunction2])
    
    for row in range (0, numbasis):
        for col in range (0, row+1):
            # This normalizes the S matrix.
            Smat[row,col] = Smat[row,col]* norm[row] * norm[col]
            # The S matrix is symmetric, so row,col = col,row.
            Smat[col,row] = Smat[row,col]          

    return norm,Smat    

#==============================================================================
#==============================================================================
#Trig tables used in the trig_table function to calculate exact answers of
    #simple trigonometric integrals
TRIG_TABLE_1 = np.array ([
    np.array([0.0,  1j*PI, 0.0, 3*1j*PI/4, 0.0, 5*1j*PI/8,  0.0, 35*1j*PI/64], dtype=np.complex128),
    np.array([PI,  0.0, PI/4,  0.0, PI/8,   0.0, 5*PI/64,   0.0],dtype=np.complex128),
    np.array([0.0,  1j*PI/4, 0.0, 1j*PI/8, 0.0, 5*1j*PI/64,  0.0, 7*1j*PI/128], dtype=np.complex128), 
    np.array([3*PI/4, 0.0, PI/8,  0.0, 3*PI/64, 0.0, 3*PI/128,  0.0],dtype=np.complex128),
    np.array([0.0,  1j*PI/8, 0.0, 3*1j*PI/64, 0.0, 3*1j*PI/128,  0.0, 7*1j*PI/512], dtype=np.complex128), 
    np.array([5*PI/8,  0.0, 5*PI/64,   0.0, 3*PI/128, 0.0, 5*PI/512, 0.0],dtype=np.complex128),
    np.array([0.0,  5*1j*PI/64, 0.0, 3*1j*PI/128, 0.0, 5*1j*PI/512,  0.0, 5*1j*PI/1024], dtype=np.complex128), 
    np.array([35*PI/64,  0.0, 7*PI/128,   0.0, 7*PI/512, 0.0, 5*PI/1024, 0.0],dtype=np.complex128)
    ])
TRIG_TABLE_2 = np.array([
    np.array([0.0,  0.0, -PI/2,   0.0, -PI/2, 0.0, -15*PI/32, 0.0], dtype=np.complex128),
    np.array([0.0,  1j*PI/2, 0.0, 1j*PI/4, 0.0, 5*1j*PI/32,  0.0, 7*1j*PI/64], dtype=np.complex128),
    np.array([PI/2,  0.0, 0.0,  0.0, -PI/32, 0.0, -PI/32, 0.0],dtype=np.complex128),
    np.array([0.0,  1j*PI/4, 0.0, 3*1j*PI/32, 0.0, 3*1j*PI/64,  0.0, 7*1j*PI/256], dtype=np.complex128),
    np.array([PI/2, 0.0, PI/32,  0.0, 0.0, 0.0, -PI/256,  0.0],dtype=np.complex128),
    np.array([0.0,  5*1j*PI/32, 0.0, 3*1j*PI/64, 0.0, 5*1j*PI/256,  0.0, 5*1j*PI/512], dtype=np.complex128),
    np.array([15*PI/32,  0.0, PI/32,   0.0, PI/256, 0.0, 0.0, 0.0],dtype=np.complex128),
    np.array([0.0,  7*1j*PI/64, 0.0, 7*1j*PI/256, 0.0, 5*1j*PI/512,  0.0, 35*1j*PI/8192], dtype=np.complex128)
       ])
TRIG_TABLE_3 = np.array ([
    np.array([0.0, 0.0, 0.0, -1j*PI/4, 0.0, -5*1j*PI/16,  0.0, -21*1j*PI/64], dtype=np.complex128),
    np.array([0.0,  0.0, -PI/4,  0.0, -3*PI/16,   0.0, -9*PI/64,   0.0],dtype=np.complex128),
    np.array([0.0,  1j*PI/4, 0.0, 1j*PI/16, 0.0, 1j*PI/64,  0.0, 0.0], dtype=np.complex128), 
    np.array([PI/4, 0.0, -PI/16,  0.0, -3*PI/64, 0.0, -PI/32,  0.0],dtype=np.complex128),
    np.array([0.0,  3*1j*PI/16, 0.0, 3*1j*PI/64, 0.0, 1j*PI/64,  0.0, 3*1j*PI/512], dtype=np.complex128), 
    np.array([5*PI/16,  0.0, -PI/64, 0.0, -PI/64, 0.0, -5*PI/512, 0.0],dtype=np.complex128),
    np.array([0.0,  9*1j*PI/64, 0.0, 1j*PI/32, 0.0, 5*1j*PI/512,  0.0, 15*1j*PI/4096], dtype=np.complex128), 
    np.array([21*PI/64, 0.0, 0.0, 0.0, -3*PI/512, 0.0, -15*PI/4096, 0.0],dtype=np.complex128)
    ])
TRIG_TABLE_4 = np.array([
    np.array([0.0,  0.0, 0.0,   0.0, PI/8, 0.0, 3*PI/16, 0.0], dtype=np.complex128),
    np.array([0.0,  0.0, 0.0, -1j*PI/8, 0.0, -1j*PI/8,  0.0, -7*1j*PI/64], dtype=np.complex128),
    np.array([0.0,  0.0, -PI/8,  0.0, -PI/16, 0.0, -PI/32, 0.0],dtype=np.complex128),
    np.array([0.0,  1j*PI/8, 0.0, 0.0, 0.0, -1j*PI/64,  0.0, -1j*PI/64], dtype=np.complex128),
    np.array([PI/8, 0.0, -PI/16,  0.0, -PI/32, 0.0, -PI/64,  0.0],dtype=np.complex128),
    np.array([0.0,  1j*PI/8, 0.0, 1j*PI/64, 0.0, 0.0,  0.0, -5*1j*PI/2048], dtype=np.complex128),
    np.array([3*PI/16,  0.0, -PI/32,   0.0, -PI/64, 0.0, -15*PI/2048, 0.0],dtype=np.complex128),
    np.array([0.0,  7*1j*PI/64, 0.0, 1j*PI/64, 0.0, 5*1j*PI/2048,  0.0, 0.0], dtype=np.complex128)
       ])

@jit
def trig_table(x_pow,y_pow,m,c):
    """
    Used to calucalte exact integrals of phi parts in integrands
    updated 7/13/2021 - NicholsJA
    Parameters
    ----------
    x_pow : power of the x part of basis function
    y_pow : power of the y part of basis function
    m : Quantum number 
    c : 0 or 1 determine if the functions returns real or imag part

    Returns
    -------
    Exact value of simple integrand, either real or imaginary

    """
    TRIG_TABLE_0 = np.array([
    [2*PI,  0.0, PI,   0.0, 3*PI/4,   0.0, 5*PI/8,   0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [PI,  0.0, PI/4,  0.0, PI/8,   0.0, 5*PI/64,   0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    
    [3*PI/4, 0.0, PI/8,  0.0, 3*PI/64, 0.0, 3*PI/128,  0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5*PI/8,  0.0, 5*PI/64,   0.0, 3*PI/128, 0.0, 5*PI/512, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [35*PI/64,  0.0, 7*PI/128,   0.0, 7*PI/512, 0.0, 5*PI/1024, 0.0]
       ],dtype=np.float64)
    if c==0:
        if m==0:
            return TRIG_TABLE_0[x_pow,y_pow].real
        
        elif m==1:
            return TRIG_TABLE_1[x_pow,y_pow].real    
        elif m==2:
            return TRIG_TABLE_2[x_pow,y_pow].real
        elif m==3:
            return TRIG_TABLE_3[x_pow,y_pow].real
        elif m==4: 
            return TRIG_TABLE_4[x_pow,y_pow].real
    else:
        if m==0:
            return TRIG_TABLE_0[x_pow,y_pow].imag
        
        elif m==1:
            return TRIG_TABLE_1[x_pow,y_pow].imag    
        elif m==2:
            return TRIG_TABLE_2[x_pow,y_pow].imag
        elif m==3:
            return TRIG_TABLE_3[x_pow,y_pow].imag
        elif m==4: 
            return TRIG_TABLE_4[x_pow,y_pow].imag
#trig_table


@jit_nquad3_function
def MQQ_ij_integrand(args):
    """
    integrand for MQQ matrix when i>0 and j >0

    Parameters
    ----------
    params : params : contains( rho,z,b,k,c,ol,ol_prime,om,z_atom1,z_atom2)
    z : Argument integrating over from 0 to inf
    rho : second argument integrating over from 0 to inf
    c : 0 or 1 used to control the return of real or imaginary part
    sym: 1 or -1 based on symmetry type
    ol1 : quantum number of Hankle function
    om1 : quantum number of Hankle function
    alph1: alpha of the first basis function
    ol2: quantum number of second basis function
    om2: quantum number of second basis function
    alph2: alpha of the second basis function


    Returns
    -------
    integ : the triple integral from -inf to inf dz,0 to 2pi dphi, 0 to inf dp
    It is multiplied by 2 because we are only integrating 
        from 0 to inf instead of -inf to inf. 
    It has a factor of 2Pi as well because the phi part of the integral can 
        be ignored and 1 integrated over 0 to 2Pi is 2Pi.
    Cylindrical coords so it is mult. by rho.
    It also multiplies by the sum of all the basis functions

    """
    #set coords of atoms
    z_atom1 = 1.0    
    z_atom2 = -1.0
    # Unpacking args
    z,rho,c,sym, ol1, om1, alp1, ol2, om2, alp2=args
    #nothing is dependant on m in the integrand but the trig tables call an m
        #so set it to 0
    m=0
    # l and m+l will be used as indices to look up information, so they must 
        # be converted to ints.
    l1 = int(ol1 + int_c)
    ml1 = l1 + int(om1 + int_c)
    l2 = int(ol2 + int_c)
    ml2 = l2 + int(om2 + int_c)
    # combo_sign determines whether the terms are added or subtracted. This
            # is contingent on the orbital type, and symmetry.
    combo_sign1 = sym * ((-1)**ml1)
    combo_sign2 = sym * ((-1)**ml2)
    
    #all the conversions from cylindrical to spherical for an easier integrand
    rhosq = rho*rho
    za_diff = z-z_atom1
    zb_diff = z-z_atom2
    rsq = rhosq+z*z
    r=rsq**0.5
    rasq = rhosq + za_diff*za_diff
    rbsq = rhosq + zb_diff*zb_diff
    #part of integrand with basis functions and phi 
    Vless_integrand = 0.0
    #potential
    V =-2.35*(np.exp(-rbsq)+np.exp(-rasq))

    # The terms associated with the atomic orbital of each basisfunction.
    orbital_1 = orbitals_matrix[l1][ml1]
    orbital_2 = orbitals_matrix[l2][ml2]
    # The integrand is V times the sum of the products of all the basisfunction
        # terms for every i and j.
    for i in range (0, numberOfTerms[l1][ml1]):
        term_b1 = orbital_1[i]
        coeff_b1 = term_b1[0]
        xpow1 = term_b1[1]
        ypow1 = term_b1[2]
        zpow1 = term_b1[3]
        
        for j in range (0, numberOfTerms[l2][ml2]):
            term_b2 = orbital_2[j]
            xpow2 = term_b2[1]
            ypow2 = term_b2[2]
            zpow2 = term_b2[3]
            
            # We pull out the phi dimension (which is an integral of only
                # sines and cosines) and look up its integrated value.
            Rho = rho**(xpow1 + xpow2 + ypow1 + ypow2)
            phi = trig_table(int(xpow1 + xpow2),int(ypow1 + ypow2),m,0)
            
            Vless_integrand += coeff_b1 * term_b2[0] * Rho * phi * (
                np.exp(-rasq*(alp1 + alp2)) *  
                    za_diff**(zpow1 + zpow2)
                + combo_sign1 * np.exp(-rbsq*alp1 - rasq*alp2) * 
                    za_diff**zpow2 * zb_diff**zpow1 
                + combo_sign2 * np.exp(-rasq*alp1 - rbsq*alp2) * 
                    za_diff**zpow1 * zb_diff**zpow2
                + combo_sign1*combo_sign2 * np.exp(-rbsq*(alp1 + alp2)) * 
                    zb_diff**(zpow1 + zpow2) 
                )
               
    integ=  2*Vless_integrand * V * rho

    return integ


# go to top to change basis functions/types
# hit control+shift+o to see outline


def main():
    """
    Code to find resonance energy from MQQIJ matrix with the H and S eigen vals
    """
    
    B=1.0   #constant used in cutoff function
    k=.447   
    lmax=6 #maximum L, cannot exceed 6
    odd = 0 #make 1 if lmax is odd
    m=0
    
    
    step = 2
    sym = -1 #symmetry var make it negative for odd L
    mat_sz = int(lmax/step) #var used to properly build matricies based on number of Ls being used
    debug = True
    """ import from Ben's code"""
            
    # atom coordinates
    z_atom1 = 1.0
    z_atom2 = -1.0
    

    scale_list = np.arange(0.05,0.3,0.01)
    for scale in scale_list:
        nbasis = 0
        #print(scale)
        # this builds list of basis functions including alphas from standards list
        numberOfBasisFunctions = []
        SymmetriesList = []
        basisFunctionSet = []
        # The counter keeps track of how many basisfunctions are in each symmetry.
        counter = 0
        
        # Each symmetry contains one or more atomic orbitals that the 
            # basisfunctions are created from. Each orbital was given by the user 
            # in the basisTypes list above and has a unique l and m.
        for orbital in basisTypes[0]:
            ol = orbital[0]
            om = orbital[1]
            # Each atomic orbital is used to create multiple basisfunctions, each 
                # with a different alpha value. These alphas were predetermined
                # when the basis set was chosen, and are accessed from the 
                # alpha_standards list by the index l of the atomic orbital.
            for alph in alpha_standards[ol]:
                
                #if alph <= scale_thresh:
                   # alph = alph*scale
                #print(alph)
                basisFunctionSet.append([ol, om, alph])
                counter += 1
                nbasis+=1
            
                
                
        # Once a set of basisfunctions for a symmetry has been created, it is 
            # appended to SymmetriesList. The program will be able to access nearly
            # all of the basisfunction information from this list with a single,
            # unique index for each symemtry.
        SymmetriesList.append(basisFunctionSet)
        # We also need to know the number of basisfunctions in each symmetry. This 
            # has been calculated by the counter in the above loops. We need this 
            # because we cannot perform a direct element loop inside numba, so we 
            # need a way to determine a range to iterate over. Index N of 
            # numberOfBasisFunctions corresponds to index N of SymmetriesList.
        numberOfBasisFunctions.append(counter)
        # Numba does not accept multidimensional arrays of non-uniform length. However, 
            # the symemtries are not required to have the same number of basisfunctions,
            # which means the 2d basisFunctionSet arrays inside the SymmetriesList
            # array may not always have the same length. To remedy this, we determine 
            # the largest number of basisFunctions used, and add extra dummy arrays 
            # to any symmetries that have fewer. This way all the arrays inside 
            # SymmetriesList will have the same length and numba will accept it. These
            # dummy arrays will never be used in calculations because the iteration 
            # range is determined by the numberOfBasisFunctions list.
        # This determines the largest number of basisfunctions used.
        most = max(numberOfBasisFunctions)
        # This loops over all the symemtries and if any symmetries have fewer 
            # basisfunctions, it adds extra dummy arrays until all have the same size.
        for index, element in enumerate(numberOfBasisFunctions):
            if element != most:
                for i in range(0, most-element):
                    SymmetriesList[index].append([0,1,0]) # These are all zeros so even
                        # if this part gets called (which it shouldn't), there will be
                        # no effect.
                        
        # Because numba cannot access global lists, we convert these to arrays.
        # numba wants this to have type int64.
        global num_Basis_Functions 
        num_Basis_Functions = np.array(numberOfBasisFunctions, dtype=np.int64)
        global Symmetries 
        Symmetries = np.array(SymmetriesList)
        SymSet = Symmetries[0]
        arg_val = []
        # this loop creates the arguments for MQQ_ij matrix elms.
        for basisfunction1 in range(0, nbasis):
            # Each basisfunction is given by a row of [l, m, alpha] in SymSet.
            lma1 = SymSet[basisfunction1] 
            l1 = lma1[0]
            m1 = lma1[1]
            alpha1 = lma1[2]
            if alpha1 < scale_thresh:
                alpha1 *= scale
            
            for basisfunction2 in range(0, basisfunction1 +1):
                lma2 = SymSet[basisfunction2] 
                l2 = lma2[0]
                m2 = lma2[1]
                alpha2 = lma2[2]
                if alpha2 < scale_thresh:
                    alpha2 *= scale
                # arg_val is a list of lists. Each list contains:
                    # a 0 to return only the real part of the integral
                    # An int 1/-1 for bonding/antibonding,
                    # The l, m, alpha for the first basisfunction.
                    # The l, m, alpha for the second basisfunction.
                arguments = [0,sym, l1, m1, alpha1, l2, m2, alpha2]
                arg_val.append(list(arguments))
            
        #print(arg_val)
        #print("Sym Mat",Symmetries)
        N,S = construct_S(0,sym,scale)
        T = construct_T(0,N,sym,scale)
        #print("eigs",S_eig)
        #print("S ",S)
        #print("T ",T)
        #print("N ",N)
        
        H =np.zeros((nbasis,nbasis),dtype=np.float64) 
        counter = 0
        for row in range (0, nbasis):
            for col in range (0, row+1):
        #                 
        #          # Writes in MQQ matrix where i and j > 0
        
               res0_MQQ_ij = nquad(MQQ_ij_integrand, [(0.0, np.inf),(0.0,np.inf)], arg_val[counter])                
               #print(res0_MQQ_ij[0])
               H[row,col]= (T[row,col]+N[row]*N[col]*(res0_MQQ_ij[0]))
               H[col,row]= H[row,col]
               counter= counter+1
        
        (Eig_vals, Eig_vecs) = scipy.linalg.eig(H,S)
        #Eig_vals = Eig_vals.sort()
        for eig in Eig_vals:
            print(scale,eig.real)
     
            

   
    
if __name__ == '__main__':
    main()
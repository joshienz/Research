# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 2021
Last updated 2/7/2022 by NICHOLSJA18

Implements the Complex Kohn Variational method to find a phase shift and 
    scattering amplitude of a non-spherical potential.
    The potential as a double well and is cylindrically symmetric 
    Mimicking two atoms at +1 and -1 on the z- axis these are the centers
    of the potential wells and basis functions used to approx the orbital shapes.
    The integrands use a spherical coordinate system
@author: NICHOLSJA18
"""

from scipy.special import sph_harm, lpmn
import numpy as np
import math
from scipy.integrate import nquad
from numba import jit, prange

from multiprocessing import Pool
import time



PI = np.pi

# allows nnquad to call a function and see it as a cfunc 
#-- this speeds up numericla integration
from numba import cfunc, carray
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

import ctypes
from numba.extending import get_cython_function_address

# =============================================================================
# create numba compatible spherical Bessel and spherical Neumann functions
#   (these functions take a double argument and return a double)
# =============================================================================

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
    [[1,1]]#[1,0],[0,0]]
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
                   [5,2.5,1.25,0.625,0.3125,0.15625,0.078125,0.0390625,0.01953125,0.009765625],#,2.5,1.25,0.625,0.3125,0.1625,0.08125],#3.3333333,1.111111111,0.37037037,0.12345679,0.0411522634],#,0.0137174211,0.00457247],
                   [5,2.5,1.25,0.625,0.3125,0.15625,0.078125],#0.6666667,0.22222222222,0.07407407,0.024691358,0.0082304527],
                   [5,2.5,1.25,0.625,0.3125,0.15625,0.078125],#,0.1666666666],#,0.055555556]
                   [3,1.5]
                   ]


alpha_standards = [[0.8*i for i in j] for j in alpha_standards]


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
def construct_T(numSymmetry,norm,sym,z_atom1,z_atom2):
    """
    This function creates the Kinetic Energy matrix. 
    
    Parameters
    ----------
    numSymmetry : An integer index that refers to the current symmetry.
    norm : A list of normalization constants.
    sym : 1 or -1. This determines whether the symmetry is 
        bonding or antibonding.
    z_atom1 : Position of atom 1 along z-axis
    z_atom2 : Position of atom 2 along z-axis
        ## We only use two in this code

    Returns
    -------
    Tmat: The kinetic energy matrix.

    """
# Last modified July 9, 2020 by bjc
    
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
        if n & 1:  # n lmin
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
def construct_S(numSymmetry,sym,z_atom1,z_atom2):
    """
    This function creates the Overlap and norm matrices. 

    Parameters
    ----------
    numSymmetry : An integer index that refers to the current symmetry.
    norm : A list of normalization constants.
    sym : 1 or -1. This determines whether the symmetry is 
        bonding or antibonding.
    z_atom1 : Position of atom 1 along z-axis
    z_atom2 : Position of atom 2 along z-axis
        ## We only use two in this code

    Returns
    -------
    norm: A matrix of list of normalization constants.
    Smat: The overlap matrix.

    """
    
# last modified July 9, 2020 by bjc
   
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
                norm[basisfunction1] = 1.0/np.sqrt(
                        Smat[basisfunction1,basisfunction2])
    
    for row in range (0, numbasis):
        for col in range (0, row+1):
            # This normalizes the S matrix.
            Smat[row,col] = Smat[row,col]* norm[row] * norm[col]
            # The S matrix is symmetric, so row,col = col,row.
            Smat[col,row] = Smat[row,col]          

    return norm,Smat    
#construct_S

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


@jit
def legendre_poly(l,m,tht):
    """
    The associative legendre polynomial for specified l and m.
    Uses recursion past l=4
    and includes the Normalization constant from spherical harmonic

    Parameters
    ----------
    l : Quantum number l
    m : Quantum number m
    theta : coordinate used in legendre polynomials for sin and cos

    Returns
    -------
    N - normalization constant
    N*Appropriate legendre polynomial

    """
    #Normalization cponstant which is included and part of 
        #the spherical harmonic being integrated
    N = ((-1)**(m)*np.sqrt(((2*l+1)/(4*PI))*(fact(l-m)/fact(l+m))))
    #conversion to cos(theta) and sin(theta)
    sin = np.sin(tht)
    cos = np.cos(tht)
    #getting squares and cubes to simplify table
    cospow2 = cos*cos
    sinpow2 = sin*sin
    cospow3 = cos*cospow2
    sinpow3 = sin*sinpow2
    cospow4 = cos*cospow3
    sinpow4 = sinpow3*sin
    
    #table of hard-coded in legendre terms up to l=4, m=4 
        #(does not include -m's)
    LEGENDRE_TABLE = np.array([
    [1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [cos,  -sin, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0],
    [.5*(3*cospow2-1),  -3*cos*sin, 3*sinpow2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [.5*(5*cospow3-3*cos), -1.5*(5*cospow2-1)*sin,
         15*cos*sinpow2, -15*sinpow3, 0.0, 0.0, 0.0, 0.0],
    [0.125*(35*cospow4-30*cospow2+3), -2.5*(7*cospow3-3*cos)*sin, 
         7.5*(7*cospow2-1)*sinpow2, -105*cos*sinpow3, 105*sinpow4, 0.0, 0.0, 0.0]
       ])
    
    #if the term is in the table return N* table value
    if l<5:
        
        return N*LEGENDRE_TABLE[l,m]
  
    else:
        #Otherwise do a recursive loop where the recursive formula is
        #(l-m+1)Pl+1=(2l+1)*x*Pl-(l+m)Pl-1
        #and x = cos(theta)
        L = 4
        P_l = LEGENDRE_TABLE[L,m]
        P_l_old = LEGENDRE_TABLE[L-1,m]
        while L<l:
            
            P_l_new = ((2*L+1)*cos*P_l
             -(L+m)*P_l_old)/(L-m+1)
            P_l_old = P_l
            P_l = P_l_new
            
            L=L+1
            
        return N*P_l_new
#legendre_poly
    
     
@jit_nquad3_function        # use *params with @jit
def M00_integrand(params):
    """
    Integrand for the M00 matrix

    Parameters
    ----------
    params : contains( rho,z,k,ol,ol_prime,om,z_atom1,z_atom2)
    r : Argument integrating over from 0 to inf
    tht : second argument integrating over from 0 to PI/2
    k : Momentum constant
    ol : L of the Bessel fnct on the right
    ol_prime : L of the bessel fnct on the left
    om : m of both bessel functions
    z_atom1, z_atom2 : z-coords of the two atoms
    
    Returns
    -------
    integ : the triple integral from 0 to inf dr,0 to pi dtht, 0 to 2PI dphi
    It is multiplied by 2 because we are only integrating 
        from 0 to inf instead of 0 to PI/2. 
    It has a factor of 2Pi as well because the phi part of the integral can 
        be ignored and 1 integrated over 0 to 2Pi is 2Pi.
    Spherical coords so it is mult. by r^2 sin(tht).
    And because we use the spherical bessel functions we get cancellations 
    resulting in only the potential term left with a Jl1*V*Jl2 with their 
    corresponding legendre polies and divide by the sqrt of k for each J funct

    """
    #unpack params
    r,tht,k,ol,ol_prime,om,z_atom1,z_atom2=params
    #numba causes all params to be floats so convert to ints
    l = int(ol+int_c)
    m = int(om+int_c)
    l_prime=int(ol_prime+int_c)
    
    #all the conversions from cylindrical to spherical for an easier integrand
    x = r*np.sin(tht)
    z=r*np.cos(tht)
    rsq=r*r
    za_diff = z-z_atom1
    zb_diff = z-z_atom2
    rasq = x*x + za_diff*za_diff
    rbsq = x*x + zb_diff*zb_diff
    #Potential 
    #Use the width to narrow the well, and edit the 4.5 to change the attraction
        #width = 0.96
        #V =  -(4.5/2)*(np.exp(-width*rbsq)+np.exp(-width*rasq))
    # V = -(1/2)*(np.exp(-rbsq)+np.exp(-rasq))
    #V = -(2.35)*(np.exp(-rbsq)+np.exp(-rasq))
    #potential
    wwidth= .5
    V =-(3.2)*(np.exp(-wwidth*rbsq)+np.exp(-wwidth*rasq))

    #Spherical bessel functions 
    z__ = k*r
    J = sph_jn_numba(l,z__)
    J_prime = sph_jn_numba(l_prime,z__)
    #get associative legendre polynomials
    Legendre = legendre_poly(l,m,tht)
    Legendre_prime = legendre_poly(l_prime,m,tht )
    #calculate the integrand
    integ = 4*PI*rsq*np.sin(tht)*J_prime*V*J*Legendre*Legendre_prime/k
    return integ
#M00_integrand

# # calculate integral for basis funtion over the MQ0 vector at i=0
@jit_nquad3_function        # use *params with @jit
def MQ0_0_integrand(params):
    """
    Integrand for the MQ0 matrix when i = 0

    Parameters
    ----------
    params : params : contains( rho,z,b,k,c,ol,ol_prime,om,z_atom1,z_atom2)
    r : Argument integrating over from 0 to inf
    tht : second argument integrating over from 0 to inf
    b : constant to control Cutoff function (G)
    k : Momentum constant
    c : 0 or 1 used to control the return of real or imaginary part
    ol : L of the Bessel fnct
    ol_prime : L o fthe Hankle fnct
    om : m of either function
    z_atom1, z_atom2 : z-coords of the two atoms

    Returns
    -------
    integ : the triple integral from 0 to inf dr,0 to pi dtht, 0 to 2PI dphi
    It is multiplied by 2 because we are only integrating 
        from 0 to inf instead of 0 to PI/2. 
    It has a factor of 2Pi as well because the phi part of the integral can 
        be ignored and 1 integrated over 0 to 2Pi is 2Pi.
    Spherical coords so it is mult. by r^2 sin(tht).
    And because we use the spherical bessel functions we get cancellations 
    resulting in only the potential term left with a G*Hl1*V*Jl2 with their 
    corresponding legendre polies and divide by the sqrt of k for each J or H funct

    """
    #unpack params
    r,tht,b,k, c,ol,ol_prime,om,z_atom1,z_atom2 = params
    #numba causes all params to be floats so convert to ints
    l = int(ol+int_c)
    m = int(om+int_c)
    l_prime=int(ol_prime+int_c)
    
    #all the conversions from cylindrical to spherical for an easier integrand
    x = r*np.sin(tht)

    z=r*np.cos(tht)
    rsq=r*r
    za_diff = z-z_atom1
    zb_diff = z-z_atom2
    rasq = x*x + za_diff*za_diff
    rbsq = x*x + zb_diff*zb_diff
    #Potential 
    #Use the width to narrow the well, and edit the 4.5 to change the attraction
        #width = 0.96
        #V =  -(4.5/2)*(np.exp(-width*rbsq)+np.exp(-width*rasq))
    #V = -(1/2)*(np.exp(-rbsq)+np.exp(-rasq))
    #potential
    wwidth= .5
    V =-(3.2)*(np.exp(-wwidth*rbsq)+np.exp(-wwidth*rasq))

    #V = -(2.35)*(np.exp(-rbsq)+np.exp(-rasq))
    #Spherical bessel functions 
    z__=k*r
    J = sph_jn_numba(l,z__)
    #spherical Hankle function
    H= sph_yn_numba(l_prime,z__)+1j*sph_jn_numba(l_prime,z__)
    # cutoff function
    G=(1-np.exp(-b*r))**(2*l_prime+1)
    #asociative legendre terms
    Legendre = legendre_poly(l,m,tht )
    Legendre_prime = legendre_poly(l_prime,m,tht )

    
    integ = 4*PI*G*H*V*J*Legendre*rsq*np.sin(tht)*Legendre_prime/k


    if c == 0:
      
        return integ.real
    else:
        return integ.imag

#MQ0_0_integrand


@jit_nquad3_function
def MQ0_i_integrand(args):
    """
    integrand for MQ0 matrix when i>0 

    Parameters
    ----------
    params : params : contains( rho,z,b,k,c,ol,ol_prime,om,z_atom1,z_atom2)
    r : Argument integrating over from 0 to inf
    tht : second argument integrating over from 0 to inf
    b : constant to control Cutoff function (G)
    k : Momentum constant
    c : 0 or 1 used to control the return of real or imaginary part
    ol : quantum number of Hankle function
    om : quantum number of Hankle function
    sym: 1 or -1 based on symmetry type
    l1: quantum number of basis function
    m1: quantum number of basis function
    z_atom1, z_atom2 : z-coords of the two atoms

    Returns
    -------
    integ : the triple integral from 0 to inf dr,0 to pi dtht, 0 to 2PI dphi
    It is multiplied by 2 because we are only integrating 
        from 0 to inf instead of 0 to PI/2. 
    It has a factor of 2Pi as well because the phi part of the integral can 
        be ignored and 1 integrated over 0 to 2Pi is 2Pi.
    Spherical coords so it is mult. by r^2 sin(tht).
    And because we use the spherical bessel functions we get cancellations 
    resulting in the potential mult. by J an dthe basis functions 
    and the corresponding legendre polies for each J function
    and divide by the sqrt of k for each J funct

    """
    #unpack params
    r,tht,ol,om,b,k,alpha,c,sym,l1,m1,z_atom1,z_atom2 = args
    #numba causes all params to be floats so convert to ints
    l= int(ol + int_c)
    m = int(om + int_c)
    l1 = int(l1+int_c)
    m1 = int(m1+int_c)
    #sign as to how to add the basis functions can be 1 or -1
    combo_sign1 = sym * ((-1)**(m1+l1))
    #all the conversions from cylindrical to spherical for an easier integrand
    x = r*np.sin(tht)

    z=r*np.cos(tht)
    rsq=r*r
    za_diff = z-z_atom1
    zb_diff = z-z_atom2
    rasq = x*x + za_diff*za_diff
    rbsq = x*x + zb_diff*zb_diff
    #Potential 
    #Use the width to narrow the well, and edit the 4.5 to change the attraction
        #width = 0.96
        #V =  -(4.5/2)*(np.exp(-width*rbsq)+np.exp(-width*rasq))
    #potential
    wwidth= .5
    V =-(3.2)*(np.exp(-wwidth*rbsq)+np.exp(-wwidth*rasq))

    #V =-(2.35)*(np.exp(-rbsq)+np.exp(-rasq)) #V = -(1/2)*(np.exp(-rbsq)+np.exp(-rasq))
    z__=k*r
    J = sph_jn_numba(l,z__)
    # Legendre Variables
    Legendre = legendre_poly(l, m, tht)
    # The terms associated with the atomic orbital of each basisfunction.
    orbital_1 = orbitals_matrix[l1][m1+l1]
    Q = 0.0
    
    #Q is the sum of all the basis functions in i and j>0
    for i in range (0, numberOfTerms[l1][m1+l1]):
        term_b1 = orbital_1[i]

        coeff_b1 = term_b1[0]
        x_pow = term_b1[1]
        y_pow = term_b1[2]
        z_pow = term_b1[3]
        #phi part of teh integral from trig tables
        phi_re = trig_table(int(x_pow), int(y_pow), m,0)
        phi_im = trig_table(int(x_pow), int(y_pow), m,1)
        Q += (coeff_b1*(phi_re+phi_im*1j)*
        (np.exp(-alpha*rasq)*za_diff**z_pow+
         combo_sign1*np.exp(-alpha*rbsq)*zb_diff**z_pow))
        
    #integrand with J function and the potential and the basis fncts
    integ= 2*Q*V*J*rsq*np.sin(tht)*Legendre/np.sqrt(k)


    if c == 0:
        return integ.real
    else:
        return integ.imag
#MQ0_i_integrand


@jit_nquad3_function
def MQQ_00_integrand(params):
    """
    integrand for MQQ matrix when i and j = 0
    and l == l_prime
    Parameters
    ----------
    params : params : contains( rho,z,b,k,c,ol,ol_prime,om,z_atom1,z_atom2)
    r : Argument integrating over from 0 to inf
    tht : second argument integrating over from 0 to inf
    b : constant to control Cutoff function (G)
    k : Momentum constant
    c : 0 or 1 used to control the return of real or imaginary part
    ol : L of hankle fnct on right
    ol_prime : L of hankle fnct on the left
    om : m of hankle functions 
    z_atom1, z_atom2 : z-coords of the two atoms

    Returns
    -------
    integ : the triple integral from 0 to inf dr,0 to pi dtht, 0 to 2PI dphi
    It is multiplied by 2 because we are only integrating 
        from 0 to inf instead of 0 to PI/2. 
    It has a factor of 2Pi as well because the phi part of the integral can 
        be ignored and 1 integrated over 0 to 2Pi is 2Pi.
    Spherical coords so it is mult. by r^2 sin(tht).
    And because we use the spherical hankle functions we get cancellations 
    resulting in several left-over terms that come out of the second derivative of G*H 
    and the potential along with the corresponding legendre polies for each H
    and divide by the sqrt of k for each J or H funct

    """
    #unpack params
    r,tht,b,k,c,ol,ol_prime,om,z_atom1,z_atom2 = params
    #numba causes all params to be floats so convert to ints
    l = int(ol+int_c)
    m = int(om+int_c)
    l_prime=int(ol_prime+int_c)
    #all the conversions from cylindrical to spherical for an easier integrand
    z=r*np.cos(tht)
    x = r*np.sin(tht)

    rsq = r*r
    za_diff = z-z_atom1
    zb_diff = z-z_atom2
    rasq = x*x + za_diff*za_diff
    rbsq = x*x + zb_diff*zb_diff
    z__=k*r
    #spherical hankle functions
    H= sph_yn_numba(l,z__)+1j*sph_jn_numba(l,z__)

    H_prime= sph_yn_numba(l_prime,z__)+1j*sph_jn_numba(l_prime,z__)

    Legendre = legendre_poly(l,m,tht)
    Legendre_prime = legendre_poly(l_prime,m,tht )
    
    #derivatives of hankle and cutoff functions
        #at l=0 derivs are harcoded in
    if l==0:
        H_dot=-(np.exp(1j*k*r)/(k*r)**2)+(1j*np.exp(1j*k*r)/(k*r))
        G_dub_dot=-b**2*np.exp(-b*r)
        
    else:
        H_dot = (l*(sph_yn_numba(l-1,z__)+1j*sph_jn_numba(l-1,z__))-(l+1)*(sph_yn_numba(l+1,z__)+1j*sph_jn_numba(l+1,z__)))/(2*l+1)
        
        G_dub_dot=((2*l+1)*(2*l)*b**2*np.exp(-2*b*r)*(1-np.exp(-b*r))**(2*l-1)-
               (2*l+1)*b**2*np.exp(-b*r)*(1-np.exp(-b*r))**(2*l))
    #Potential 
    #Use the width to narrow the well, and edit the 4.5 to change the attraction
        #width = 0.96
        #V =  -(4.5/2)*(np.exp(-width*rbsq)+np.exp(-width*rasq))
    #potential
    wwidth= .5
    V =-(3.2)*(np.exp(-wwidth*rbsq)+np.exp(-wwidth*rasq))

    #V =-(2.35)*(np.exp(-rbsq)+np.exp(-rasq))#V = -(1/2)*(np.exp(-rbsq)+np.exp(-rasq))
    #V =  -(1/2)*(np.exp(-rbsq)+np.exp(-rasq))
    G=(1-np.exp(-b*r))**(2*l+1)
    G_prime = (1-np.exp(-b*r))**(2*l_prime+1)
    G_dot= ((2*l+1)*b*np.exp(-b*r)*(1-np.exp(-b*r))**(2*l))
    #integrand contains potentail and terms from second deriv of G*H after cancellation
    integ = 4*PI*Legendre_prime*(G_prime*H_prime)*((-1/2)*G_dub_dot*H-G_dot*(k*H_dot+H/r)+V*H*G)*Legendre*rsq*np.sin(tht)/k
    
    if c == 0:
        return integ.real
    else:
        return integ.imag
#MQQ_00_integrand

@jit_nquad3_function
def MQQ_00_integrand_2(params):
    """
    integrand for MQQ matrix when i and j = 0
    but l!= l_prime

    Parameters
    ----------
    params : params : contains( rho,z,b,k,c,ol,ol_prime,om,z_atom1,z_atom2)
    r : Argument integrating over from 0 to inf
    tht : second argument integrating over from 0 to inf
    b : constant to control Cutoff function (G)
    k : Momentum constant
    c : 0 or 1 used to control the return of real or imaginary part
    ol : L of hankle fnct on right
    ol_prime : L of hankle fnct on the left
    om : m of hankle functions 
    z_atom1, z_atom2 : z-coords of the two atoms

    Returns
    -------
    integ : the triple integral from 0 to inf dr,0 to pi dtht, 0 to 2PI dphi
    It is multiplied by 2 because we are only integrating 
        from 0 to inf instead of 0 to PI/2. 
    It has a factor of 2Pi as well because the phi part of the integral can 
        be ignored and 1 integrated over 0 to 2Pi is 2Pi.
    Spherical coords so it is mult. by r^2 sin(tht).
    Includes just the potential mulitplied by the two hankle functions

    """
    #unpack params
    r,tht,b,k,c,ol,ol_prime,om,z_atom1,z_atom2 = params
    #numba causes all params to be floats so convert to ints
    l = int(ol+int_c)
    m = int(om+int_c)
    l_prime=int(ol_prime+int_c)
    #all the conversions from cylindrical to spherical for an easier integrand
    z=r*np.cos(tht)
    x = r*np.sin(tht)

    rsq = r*r
    za_diff = z-z_atom1
    zb_diff = z-z_atom2
    rasq = x*x + za_diff*za_diff
    rbsq = x*x + zb_diff*zb_diff
    z__=k*r
    #spherical hankle functions
    H= sph_yn_numba(l,z__)+1j*sph_jn_numba(l,z__)

    H_prime= sph_yn_numba(l_prime,z__)+1j*sph_jn_numba(l_prime,z__)

    Legendre = legendre_poly(l,m,tht)
    Legendre_prime = legendre_poly(l_prime,m,tht )

    #Potential 
    #Use the width to narrow the well, and edit the 4.5 to change the attraction
        #width = 0.96
        #V =  -(4.5/2)*(np.exp(-width*rbsq)+np.exp(-width*rasq))
    #V =-(2.35)*(np.exp(-rbsq)+np.exp(-rasq))#V = -(1/2)*(np.exp(-rbsq)+np.exp(-rasq))
    
    #potential
    wwidth= .5
    V =-(3.2)*(np.exp(-wwidth*rbsq)+np.exp(-wwidth*rasq))
    #cutoff fncts
    G=(1-np.exp(-b*r))**(2*l+1)
    G_prime = (1-np.exp(-b*r))**(2*l_prime+1)
    #integrand contains potentail and terms from second deriv of G*H after cancellation
    integ = 4*PI*Legendre_prime*(G_prime*H_prime)*(V*H*G)*Legendre*rsq*np.sin(tht)/k
    
    if c == 0:
        return integ.real
    else:
        return integ.imag

@jit_nquad3_function
def MQQ_i0_integrand(args):
    """
    integrand for MQQ matrix when i>0 and j = 0

    Parameters
    ----------
    params : params : contains( rho,z,b,k,c,ol,ol_prime,om,z_atom1,z_atom2)
    r : Argument integrating over from 0 to inf
    tht : second argument integrating over from 0 to inf
    b : constant to control Cutoff function (G)
    k : Momentum constant
    c : 0 or 1 used to control the return of real or imaginary part
    ol : quantum number of Hankle function
    om : quantum number of Hankle function
    sym: 1 or -1 based on symmetry type
    l1: quantum number of basis function
    m1: quantum number of basis function
    z_atom1, z_atom2 : z-coords of the two atoms

    Returns
    -------
    integ : the triple integral from 0 to inf dr,0 to pi dtht, 0 to 2PI dphi
    It is multiplied by 2 because we are only integrating 
        from 0 to inf instead of 0 to PI/2. 
    It has a factor of 2Pi as well because the phi part of the integral can 
        be ignored and 1 integrated over 0 to 2Pi is 2Pi.
    Spherical coords so it is mult. by r^2 sin(tht).
    And because we use the spherical hankle functions we get cancellations 
    resulting in several left-over terms that come out of the second derivative of G*H 
    and the potential along with the corresponding legendre polies for each H
    and divide by the sqrt of k for each J or H funct

    """
    #unpack params
    r,tht,ol,om,b,k,alpha,c,sym,l1,m1,z_atom1,z_atom2 = args
    #numba causes all params to be floats so convert to ints
    l= int(ol+int_c)
    m = int(om+int_c)
    l1 = int(l1+int_c)
    m1 = int(m1+int_c)

    #sign as to how to add the basis functions can be 1 or -1
    combo_sign1 = sym * ((-1)**(m1+l1))
    
    #all the conversions from cylindrical to spherical for an easier integrand
    x = r*np.sin(tht)

    z=r*np.cos(tht)
    rsq=r*r
    za_diff = z-z_atom1
    zb_diff = z-z_atom2
    rasq = x*x + za_diff*za_diff
    rbsq = x*x + zb_diff*zb_diff
    #hankle function
    z__=k*r
    H= (sph_yn_numba(l,z__)+1j*sph_jn_numba(l,z__))
    #derivatives of hankle and cutoff functions
        #at l=0 derivs are harcoded in
    if l==0:
        H_dot=-(np.exp(1j*k*r)/(k*r)**2)+(1j*np.exp(1j*k*r)/(k*r))
        G_dub_dot=-b**2*np.exp(-b*r)
        
    else:
        H_dot = (l*(sph_yn_numba(l-1,z__)+1j*sph_jn_numba(l-1,z__))-(l+1)*(sph_yn_numba(l+1,z__)+1j*sph_jn_numba(l+1,z__)))/(2*l+1)
        
        G_dub_dot=((2*l+1)*(2*l)*b**2*np.exp(-2*b*r)*(1-np.exp(-b*r))**(2*l-1)-
               (2*l+1)*b**2*np.exp(-b*r)*(1-np.exp(-b*r))**(2*l))
    #Potential 
    #Use the width to narrow the well, and edit the 4.5 to change the attraction
        #width = 0.96
        #V =  -(4.5/2)*(np.exp(-width*rbsq)+np.exp(-width*rasq))
    #V =-(2.35)*(np.exp(-rbsq)+np.exp(-rasq)) #V = -(1/2)*(np.exp(-rbsq)+np.exp(-rasq)) 
    #potential
    wwidth= .5
    V =-(3.2)*(np.exp(-wwidth*rbsq)+np.exp(-wwidth*rasq))

    G=(1-np.exp(-b*r))**(2*l+1)
    G_dot= ((2*l+1)*b*np.exp(-b*r)*(1-np.exp(-b*r))**(2*l))

    # Legendre Variables
    Legendre = legendre_poly(l,m,tht )
    # The terms associated with the atomic orbital of each basisfunction.
    orbital_1 = orbitals_matrix[l1][m1+l1]
    Q = 0.0
    # The integrand is V times the sum of the products of all the basisfunction
        # terms for every i and j.
    for i in range (0, numberOfTerms[l1][m1+l1]):
        term_b1 = orbital_1[i]
        coeff_b1 = term_b1[0]
        x_pow = term_b1[1]
        y_pow = term_b1[2]
        z_pow = term_b1[3]
        phi_re = trig_table(int(x_pow), int(y_pow), m,0)
        phi_im = trig_table(int(x_pow), int(y_pow), m,1)

        Q += (coeff_b1*(phi_re+phi_im*1j)*
        (np.exp(-alpha*rasq)*za_diff**z_pow+
         combo_sign1*np.exp(-alpha*rbsq)*zb_diff**z_pow))
    #integrand contains potentail and terms from second deriv of G*H after cancellation
    integ = 2*(Q/np.sqrt(k))*((-1/2)*G_dub_dot*H-G_dot*(k*H_dot+H/r)+V*H*G)*rsq*np.sin(tht)*Legendre

    if c == 0:
        return integ.real
    else:
        return integ.imag
#MQ0_i_integrand

@jit_nquad3_function
def MQQ_ij_integrand(args):
    """
    integrand for MQQ matrix when i>0 and j >0

    Parameters
    ----------
    params : params : contains( rho,z,b,k,c,ol,ol_prime,om,z_atom1,z_atom2)
    r : Argument integrating over from 0 to inf
    tht : second argument integrating over from 0 to inf
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
    integ : the triple integral from 0 to inf dr,0 to pi dtht, 0 to 2PI dphi
    It is multiplied by 2 because we are only integrating 
        from 0 to inf instead of 0 to PI/2. 
    It has a factor of 2Pi as well because the phi part of the integral can 
        be ignored and 1 integrated over 0 to 2Pi is 2Pi.
    Spherical coords so it is mult. by r^2 sin(tht).
    It also multiplies by the sum of all the basis functions

    """
    #set coords of atoms
    z_atom1 = 1.0
    z_atom2 = -1.0
    # Unpacking args
    r,tht,c,sym, ol1, om1, alp1, ol2, om2, alp2=args
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
    x = r*np.sin(tht)
    z=r*np.cos(tht)
    rsq=r*r
    za_diff = z-z_atom1
    zb_diff = z-z_atom2
    rasq = x*x + za_diff*za_diff
    rbsq = x*x + zb_diff*zb_diff
    #part of integrand with basis functions and phi 
    Vless_integrand = 0.0
    #Potential 
    #Use the width to narrow the well, and edit the 4.5 to change the attraction
        #width = 0.96
        #V =  -(4.5/2)*(np.exp(-width*rbsq)+np.exp(-width*rasq))
    #V = -(2.35)*(np.exp(-rbsq)+np.exp(-rasq))#-(1/2)*(np.exp(-rbsq)+np.exp(-rasq))
    #potential
    wwidth= .5
    V =-(3.2)*(np.exp(-wwidth*rbsq)+np.exp(-wwidth*rasq))

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
            phi = trig_table(int(xpow1 + xpow2),int(ypow1 + ypow2),m,0)
            
            Vless_integrand += coeff_b1 * term_b2[0] * phi * (
                np.exp(-rasq*(alp1 + alp2)) *  
                    za_diff**(zpow1 + zpow2)
                + combo_sign1 * np.exp(-rbsq*alp1 - rasq*alp2) * 
                    za_diff**zpow2 * zb_diff**zpow1 
                + combo_sign2 * np.exp(-rasq*alp1 - rbsq*alp2) * 
                    za_diff**zpow1 * zb_diff**zpow2
                + combo_sign1*combo_sign2 * np.exp(-rbsq*(alp1 + alp2)) * 
                    zb_diff**(zpow1 + zpow2) 
                )
               
    integ=  2*Vless_integrand * V * rsq*np.sin(tht)

    return integ



def main_fxn(k):  
    """
    Implements the Complex Kohn Variational method to find a phase shift and 
    scattering amplitude of a non-spherical potential.
    The potential as a double well and is cylindrically symmetric 
    Mimicking two atoms at +1 and -1 on the z- axis these are the centers
    of the potential wells and basis functions used to approx the orbital shapes.
    The integrands use a spherical coordinate system

    """
    B=1   #constant used in cutoff function
    lmax= 8#maximum L to loop over
    lmin = 2 #make 1 if lmax is lmin
    m=1    #Quantum number m based on symmetry
    scale =0    #used to multiply in finding scattering amplitude
    if m ==0:
        scale = 1
    else:
        scale = 2
    step = 2 #how to step between the ls
    nbasis = 0  #number of basis functions
    sym = 1 #symmetry var make it negative for lmin L
    mat_sz = int((lmax-lmin)/step)+1 #var used to properly build matricies based on number of Ls being used
    debug = False #print out matricies 
            
    # atom coordinates
    z_atom1 = 1.0
    z_atom2 = -1.0
    

    
    
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
        
        for basisfunction2 in range(0, basisfunction1 +1):
            lma2 = SymSet[basisfunction2] 
            l2 = lma2[0]
            m2 = lma2[1]
            alpha2 = lma2[2]
            
            # arg_val is a list of lists. Each list contains:
                # a 0 to return only the real part of the integral
                # An int 1/-1 for bonding/antibonding,
                # The l, m, alpha for the first basisfunction.
                # The l, m, alpha for the second basisfunction.
            arguments = [0,sym, l1, m1, alpha1, l2, m2, alpha2]
            arg_val.append(list(arguments))
    #print(arg_val)
    #print("Sym Mat",Symmetries)
    N,S = construct_S(0,sym,1.0,-1.0)
    T = construct_T(0,N,sym,1.0,-1.0)
    #S_eig = np.linalg.eigvals(S)
    #print("eigs",S_eig)
    #print("S ",S)
    #print("T ",T)
    #print("N ",N)
    #creating the MQQ and S matricies 
    M_mat =np.zeros((nbasis+(mat_sz),nbasis+(mat_sz)),dtype=np.complex128) 
    S_mat =np.zeros((mat_sz+nbasis,(mat_sz)),dtype=np.complex128)
    # Writes in MQQ matrix where i and j > 0    
    counter = 0
    for row in range (0, nbasis):
        for col in range (0, row+1):
    
    
           res0_MQQ_ij = nquad(MQQ_ij_integrand, [(0.0, np.inf),(0.0,PI/2)], arg_val[counter])                
           #print(res0_MQQ_ij[0])
           #must include kinetic and overlp and potential pieces of matrix elements
           M_mat[row+(mat_sz),col+(mat_sz)]= (T[row,col]+N[row]*N[col]*(res0_MQQ_ij[0]-(k**2/2)*S[row,col]))+0j
           M_mat[col+(mat_sz),row+(mat_sz)]= M_mat[row+(mat_sz),col+(mat_sz)]
           counter= counter+1
    
 
    #create the T matrix and M00
    big_T = np.zeros((mat_sz,mat_sz),dtype=np.complex128)
    M00 = np.zeros((mat_sz,mat_sz),dtype=np.complex128)
    
    #Big loop which calls everything and finds all the different pieces 
        #of the matricies to get a phase shift
    for l in range(lmin,lmax+1,step):
        #arg vector for s matrix
        arg_vec = []
        arg_vec_im = []
        #this is the argument list created for MQQ_i0 and S matrix
        for basisfunction1 in range(0,nbasis):
                # Each basisfunction is given by a row of [l, m, alpha] in SymSet.
                lma1 = SymSet[basisfunction1] 
                l1 = lma1[0]
                m1 = lma1[1]
                alpha1 = lma1[2]
                # sarg_vec is a list of lists. Each list contains:
                    # an l and m for symmetry type
                    # a B for the cutoff fnct
                    # a k constant
                    # a c to return the real or imag part of the integral
                    # An int 1/-1 for bonding/antibonding,
                    # The l, m, alpha for the basisfunction.
                    # coords of atoms
                #need to include all four vectors to create all the pieces of the MQQ_i0 Matrix
                arguments=[l,m,B,k,alpha1,0,sym,l1,m1,z_atom1,z_atom2]
                arg_vec.append(list(arguments)) 
                arguments=[l,m,B,k,alpha1,1,sym,l1,m1,z_atom1,z_atom2]
                arg_vec_im.append(list(arguments))

        for l_prime in range(lmin,l+1,step):
            if l== l_prime:
                # MQQ at i=j=0 elem real and imag parts
                args_MQQ_00=(B,k,0,l,l_prime,m,z_atom1,z_atom2)    
                res0_MQQ_00_real = nquad(MQQ_00_integrand, [(0.0, np.inf),(0.0,PI/2)], args_MQQ_00) 
                args_MQQ_00=(B,k,1,l,l_prime,m,z_atom1,z_atom2)    
                res0_MQQ_00_imag = nquad(MQQ_00_integrand, [(0.0, np.inf),(0.0,PI/2)], args_MQQ_00) 
                M_mat[int((l-lmin)/step),int((l_prime-lmin)/step)] =res0_MQQ_00_real[0] + 1j*res0_MQQ_00_imag[0]
                args_MQQ_00=(B,k,0,l_prime,l,m,z_atom1,z_atom2)    
                res0_MQQ_00_real = nquad(MQQ_00_integrand, [(0.0, np.inf),(0.0,PI/2)], args_MQQ_00) 
                args_MQQ_00=(B,k,1,l_prime,l,m,z_atom1,z_atom2)    
                res0_MQQ_00_imag = nquad(MQQ_00_integrand, [(0.0, np.inf),(0.0,PI/2)], args_MQQ_00) 
                M_mat[int((l_prime-lmin)/step),int((l-lmin)/step)] =res0_MQQ_00_real[0] + 1j*res0_MQQ_00_imag[0]
            elif l!=l_prime:#faster integrand because it excludes the kinetic part 
                # MQQ at i=j=0 elem real and imag parts
                args_MQQ_00=(B,k,0,l,l_prime,m,z_atom1,z_atom2)    
                res0_MQQ_00_real = nquad(MQQ_00_integrand_2, [(0.0, np.inf),(0.0,PI/2)], args_MQQ_00) 
                args_MQQ_00=(B,k,1,l,l_prime,m,z_atom1,z_atom2)    
                res0_MQQ_00_imag = nquad(MQQ_00_integrand_2, [(0.0, np.inf),(0.0,PI/2)], args_MQQ_00) 
                M_mat[int((l-lmin)/step),int((l_prime-lmin)/step)] =res0_MQQ_00_real[0] + 1j*res0_MQQ_00_imag[0]
                args_MQQ_00=(B,k,0,l_prime,l,m,z_atom1,z_atom2)    
                res0_MQQ_00_real = nquad(MQQ_00_integrand_2, [(0.0, np.inf),(0.0,PI/2)], args_MQQ_00) 
                args_MQQ_00=(B,k,1,l_prime,l,m,z_atom1,z_atom2)    
                res0_MQQ_00_imag = nquad(MQQ_00_integrand_2, [(0.0, np.inf),(0.0,PI/2)], args_MQQ_00) 
                M_mat[int((l_prime-lmin)/step),int((l-lmin)/step)] =res0_MQQ_00_real[0] + 1j*res0_MQQ_00_imag[0]
    
        #     
        # =============================================================================
            #build M00 matrix 
            args_M00 =(k,l,l_prime,m,z_atom1,z_atom2)
            res0_M00 = nquad(M00_integrand, [(0.0, np.inf),(0.0,PI/2)], args_M00)  
            M00[int((l-lmin)/step),int((l_prime-lmin)/step)] = res0_M00[0]+0j
            M00[int((l_prime-lmin)/step),int((l-lmin)/step)] =M00[int((l-lmin)/step),int((l_prime-lmin)/step)]
            
        #MQQ_i0 is diagonally symeytric so loop over half of matrix elms
        for col in range ((mat_sz),nbasis+(mat_sz)):
            #inserting Edge elements to MQQ matrix
            res0_MQQ_i0_real =  nquad(MQQ_i0_integrand, [(0.0, np.inf),(0.0,PI/2)], arg_vec[col-(mat_sz)]) 
            res0_MQQ_i0_imag =  nquad(MQQ_i0_integrand, [(0.0, np.inf),(0.0,PI/2)], arg_vec_im[col-(mat_sz)])
            #print("i0, l,lp",l,l_prime,res0_MQQ_i0_real[0] + 1j*res0_MQQ_i0_imag[0])
            M_mat[int((l-lmin)/step),col]=(res0_MQQ_i0_real[0] + 1j*res0_MQQ_i0_imag[0])*N[col-(mat_sz)]
            M_mat[col,int((l-lmin)/step)]=M_mat[int((l-lmin)/step),col]
            
            # Building MQ0 matrix
        for L in range(lmin,lmax+1,step):
            args_MQ0_0= (B,k,0,L,l,m,z_atom1,z_atom2)
            res0_MQ0_0_real=  nquad(MQ0_0_integrand, [(0.0, np.inf),(0.0,PI/2)], args_MQ0_0)  
            args_MQ0_0= (B,k,1,L,l,m,z_atom1,z_atom2)
            res0_MQ0_0_imag=  nquad(MQ0_0_integrand, [(0.0, np.inf),(0.0,PI/2)], args_MQ0_0)  
            S_mat[int((l-lmin)/step),int((L-lmin)/step)] = res0_MQ0_0_real[0] + 1j*res0_MQ0_0_imag[0]
            
        for col in range ((mat_sz),nbasis+(mat_sz)):
            #MQ0 where i>0 
            res0_MQ0_i = nquad(MQ0_i_integrand, [(0.0, np.inf),(0.0,PI/2)], arg_vec[col-(mat_sz)]) 
            #print(l,sarg_vec[col-(1+mat_sz)],res0_MQ0_i[0])
            S_mat[col,int((l-lmin)/step)]=(res0_MQ0_i[0])*N[col-(mat_sz)]
            

    
    #need transpose of S matrix
    S_T =np.transpose(S_mat)
    M_inv= np.linalg.inv(M_mat)
    #debugging
    if debug:
        print("M00 ",M00)
        print("s matrix ",S_mat)
        print("s matrix T",S_T)
        print("M_mat ",M_mat)
        print("M_inv ",M_inv)
        print("I",M_inv@M_mat)
    #make the T matrix based on complex kohn variational method
    big_T=(M00-S_T@M_inv@S_mat)
    #print("Big_t",big_T)
    lmbda_ah = np.linalg.eigvals(big_T)
    lmbda_est = lmbda_ah.imag/lmbda_ah.real
    
    
    # file.write("{}".format(k))
    # for val in lmbda_est:
    #     file.write("\t{}".format(val))
    
    # file.write("\n")
    
    
    #find scattering amplitudes
    theta_in =0
    theta_out = [theta_in,theta_in+PI/4,theta_in+PI/2]
    phi = 0
    pho = PI
    F = []
    #find f for each theta out
    for tht in theta_out:
        sum_T = 0
        sum_2 = 0
        for lp in range(lmin,lmax+1,step):
            
            for l in range(lmin,lmax+1,step):
                #useing equations from papers on CKV to find scattering amp
                sum_T = (sum_T+(1j)**(l-lp)*(2/PI)*big_T[int((lp-lmin)/step),int((l-lmin)/step)]*
                         sph_harm(m,lp,phi,theta_in)*sph_harm(m,l,phi,tht))
                #use a second sum to get the other theta out value where you must
                # rotate phi becasue theta is between 0 and PI
                sum_2 = (sum_2+(1j)**(l-lp)*(2/PI)*big_T[int((lp-lmin)/step),int((l-lmin)/step)]*
                         sph_harm(m,lp,pho,theta_in)*sph_harm(m,l,phi,tht))
                                        
            
        F.append(-2*scale*PI**2*k*sum_T)
        F.append(-2*scale*PI**2*k*sum_2)
        
        #print("F: ",F)
        """
        range from 0.05 to 0.25 ish center is 0.15 = k
        """
    
    return k, lmbda_est


# go to top to change basis functions/types
# hit control+shift+o to see outline
def main():
    """
    Implements the Complex Kohn Variational method to find a phase shift and 
    scattering amplitude of a non-spherical potential.
    
    Uses multiprocessing to run the program in parallel

    """
    
    num_processes = 25  # number of processes to use for the computations
    with Pool(num_processes) as pool:
        results = pool.map(main_fxn, np.arange(.01,.2,0.0001))
        #results2 = pool.map(main_fxn, np.arange(.0500,.3,0.0001),1.1)
    #print(results)
    
    with open('text.txt','w') as file:
        for res in results:
            k, lmbda_est = res
            file.write("{}".format(k))
            #eigenSum = 0
            for val in lmbda_est:
                #eigenSum += math.atan(val)
                
               file.write("\t{}".format(math.atan(val)))
            #file.write("\t{}".format(math.atan(lmbda_est)))
            file.write("\n")
            """
    with open('text2.txt','w') as file:
        for res in results2:
            k, lmbda_est = res
            file.write("{}".format(k))
            eigenSum = 0
            for val in lmbda_est:
                eigenSum += math.atan(val)
               #file.write("\t{}".format(math.atan(val)))
            file.write("\t{}".format(eigenSum))
            file.write("\n")
            file.close()
            """
    
#maybe make a new file with atans and sum, read them in as np.arrays
#maybe graph it with python and combine two moved glitches

    
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Elapsed time: {:.2f} seconds'.format(end_time-start_time))

"""
This file contains the functions necessary to encode a message with Reed
Solomon error-correcting code, and to decode an RS codeword into its original
message, with error checking and correction.

The primitive polynomial, n and k have to be known to build the generator
tables required for the encoding and decoding.

"""

#cython: language_level=3, boundscheck=False, wraparound=False
from GaloisField import * 
import itertools, math
from cpython.array cimport array, clone
import numpy as np
cimport numpy as np

cdef array array_int_template = array('i')

###############################################################################

# Find a RS generator polynomial given the lower and upper powers of alpha,
# we usually use 1 until n-k.
# g(x) = (x-alpha^l) (x-alpha^(l+1)) ... (x-alpha^(l+2t-1))

# The result is a polynomial of x where each element is a list
# representing a polynomial of alpha.
# To simplify this polynomial, we need exptable which is in another function.
# lower = l, upper = l+2t-1
cdef list RS_generator_polynomial(int lower, int upper):
    cdef:
        list t, poly, gx, gx_list = []
        int i
    
    # [[1], [-1,0]] is interpreted as 
    # ( 1 ) * (x^1) + (-1 * \alpha^1 + 0 * \alpha^0) * (x^0)
    # or simplified, (x - \alpha)
    for i in range(lower, upper+1):
        t = [[1], [-1]+[0]*i]
        gx_list.append(t)
        
    gx = gx_list[0]
    for poly in gx_list[1:]:
        gx = GF_polynomial_product(gx, poly, modulo=2)
    return gx


# Mainly to fill the exponential and log tables used for encoding and decoding.
# Veys of exptable range from 0 to (2^n)-1.
# Its values are the integer forms of binary polynomials as result of
# exponentiating alpha to the power indicated by its key, then take the modulo
# of its division with given primitive polynomial.
# This ensures all values always have the same length, which is one less than
# the primitive polynomial.
# On the other hand, logtable is the inverse of exptable.
# The keys of logtable are the values in exptable while its values are the
# keys in exptable.
cdef tuple RS_generate_tables(int n, list primitive_poly):
    
    cdef:
        array[int] exptable, logtable, rem_array
        dict int2binstr_dict = {}
        int polydegree, order, bits_per_symbol, exponent, remainder_int, i
        list x_coefs, remainder, t, logkeys
        np.ndarray[int, ndim=2] multiplication_table
         
    polydegree = len(primitive_poly)-1
    order = 2**(polydegree)-1
    bits_per_symbol = math.ceil(math.log2(n))
         
    exptable = clone(array_int_template, order, True)
    logtable = clone(array_int_template, order+1, True)
    multiplication_table = np.zeros((order+1,order+1), dtype=np.int32)
         
    # log(0) = -infinity
    # x^(-infinity) = 0 so we don't put in exptable
    logtable.data.as_ints[0] = -1
    
    # x**0 = 1, log(1) = 0
    exptable.data.as_ints[0] = 1
    logtable.data.as_ints[1] = 0
    
    
    # coefficients of x, starting from x^degree to x^0
    # we initialize it to 1
    x_coefs = [0]*(polydegree-1) + [1]
    
    int2binstr_dict[0] = '0'*bits_per_symbol
    
    # start from alpha^1
    exponent = 1
    while exponent < order:
        
        # multiply the previous result by x
        x_coefs = x_coefs + [0]
        
        # then find the remainder of its division by primitive polynomial
        remainder = GF_polynomial_div_remainder(x_coefs, primitive_poly, 
                                          returnlen=polydegree, modulo=2)
        rem_array = array('i', remainder)
        remainder_int = bin2int(rem_array)
        exptable[exponent] = remainder_int
        logtable[remainder_int] = exponent
        int2binstr_dict[exponent] = int2binstr(exponent, 
                                               returnlen=bits_per_symbol)
        exponent += 1
    int2binstr_dict[exponent] = int2binstr(exponent, 
                                           returnlen=bits_per_symbol)
    
    # first row is all zeros, skip
    for i in range(1,order+1):
        for j in range(order+1):
            multiplication_table[i,j] = GF_product(i, j, n, exptable,logtable)
            
    return exptable, logtable, int2binstr_dict, multiplication_table


# Calls the functions to create the generator polynomial and relevant tables.
# Also converts the calculated generator polynomial into simpler form
# by use of the lookup table.
cpdef tuple RS_generator(int n, int k, list primitive_poly):

    cdef:
        tuple gf_tables
        dict int2binstr_dict
        int[:,:] multiplication_table
        list generator_poly_alpha, generator_poly_int, x_coef
        int i, j, t, alpha_degree, polylen
        array[int] gpoly
        int[:] exptable, logtable

    gf_tables = RS_generate_tables(n, primitive_poly)
    exptable, logtable, int2binstr_dict, multiplication_table = gf_tables
    
    generator_poly_alpha = RS_generator_polynomial(1,n-k)
    
    polylen = len(generator_poly_alpha)
    generator_poly_int = [0]*polylen
    
    for i in range(polylen):
        x_coef = generator_poly_alpha[i]
        alpha_degree = len(x_coef)-1
        t = 0
        
        for j in range(alpha_degree+1):
            if x_coef[j] != 0:
                t ^= exptable[(alpha_degree-j) % n]
        
        generator_poly_int[i] = t
    
    gpoly = clone(array_int_template, polylen, False)
    for i in range(polylen):
        gpoly.data.as_ints[i] = generator_poly_int[i]
    return gf_tables, gpoly


###############################################################################
# A few functions for Reed-Solomon decoder


# calculating distance of Berlekamp-Massey algorithm at kth iteration
cdef int BM_delta(int[:] syndromes, int[:] cx,
                  int[:,:] multiplication_table, int k):
    cdef int i, d = 0, lencx = len(cx)
    
    # No bound checking as we rely on the fact that syndrome array is always
    # bigger than cx array. Length of syndrome array is 2t, while cx at most
    # has a degree of t.
    for i in range(lencx):
            d ^= (multiplication_table[syndromes[k-lencx+i]][cx[i]])
    return d

# Berlekamp-Massey algorithm, finding the error locator polynomial of 
# a codeword given syndromes.
cdef array[int] Berlekamp_Massey(int[:] syndromes, int n, int k, 
                                 int[:] exptable, int[:] logtable,
                                 int[:,:] multiplication_table):
    
    cdef:
        array[int] cx, px, tx, subtractby
        int N, double_t, L, l, dm, delta, dm_inv, dx_dm_inv
    
    N = n-k
    double_t = n-k
    
    # c(x) and p(x) is initialized to [1]
    #cx, px = [1], [1]
    cx = clone(array_int_template, 1, False)
    cx.data.as_ints[0] = 1
    px = clone(array_int_template, 1, False)
    px.data.as_ints[0] = 1
    L, l, dm = 0, 1, 1
    
    # k=0 is used for initial state so we start from k=1
    for k in range(1,N+1):
        delta = BM_delta(syndromes, cx, multiplication_table, k)
        
        if delta == 0:
            l += 1
        
        else:
            if 2*L >= k:
                
                # cx = cx - d * dm^{-1} * px * x^{l}
                dm_inv = GF_inverse(dm, n, exptable,logtable)
                dx_dm_inv = (multiplication_table[delta][dm_inv])

                subtractby = clone(array_int_template, len(px)+l, True)
                for i in range(len(px)):
                    subtractby.data.as_ints[i] = (multiplication_table
                                                  [dx_dm_inv]
                                                  [px.data.as_ints[i]])
                
                cx = GF2_poly_add(cx, subtractby)
                l += 1
                
            # update L,px,dm and reset l after finding cx for this iteration
            else:
                tx = cx
                
                dm_inv = GF_inverse(dm, n, exptable,logtable)
                dx_dm_inv = (multiplication_table[delta][dm_inv])

                subtractby = clone(array_int_template, len(px)+l, True)
                for i in range(len(px)):
                    subtractby.data.as_ints[i] = (multiplication_table
                                                  [dx_dm_inv]
                                                  [px.data.as_ints[i]])
                
                cx = GF2_poly_add(cx, subtractby)
                
                px = tx
                dm = delta
                L = k-L
                l = 1
        
    # \Lambda(x) is the c(x) after final iteration
    return cx

# Forney's algorithm, finding the error polynomial e(x)
# given a known error locator polynomial.
# As r(x) = c(x)+e(x), we subtract r(x) by e(x) to get the correct c(x).
cdef array[int] Forney(int[:] Lambda_poly, int[:] Syndromes, int n, int k,
                       int[:] exptable, int[:] logtable,
                       int[:,:] multiplication_table):
    
    cdef:
        array[int] Synd_poly, Syndrome_x_Lambda_poly, \
             x_exp_2t, Omega_poly, Lambda_poly_ddx, Lx_roots, \
             error_locs, roots_Omega_eval, roots_Lambda_ddx_eval, \
             error_poly, result
        int i, ome, lam, expnt, error, lenerr
    
    # reverse the syndrome list since the syndrome written as polynomial
    # starts with S1 * x^0 + S2 * x^1 + ...
    Synd_poly = clone(array_int_template, n-k, False)
    for i in range(n-k):
        Synd_poly.data.as_ints[i] = Syndromes[n-k-1-i]
    Syndrome_x_Lambda_poly = GF2_poly_product(Lambda_poly, Synd_poly,
                                             multiplication_table)
    
    # divide by x^{2t} and take the remainder
    #x_exp_2t = [1] + [0]*(n-k)
    x_exp_2t = clone(array_int_template, n-k+1, True)
    x_exp_2t.data.as_ints[0] = 1
    
    Omega_poly = GF2_remainder_monic_divisor(Syndrome_x_Lambda_poly, 
                                             x_exp_2t, multiplication_table)
    
    Lambda_poly_ddx = GF2_polynomial_derivative(Lambda_poly)
    
    # roots of L(x) are all exp where L(\alpha^{exp}) evaluates to 0
    Lx_roots = GF2_poly_eval(Lambda_poly, n, k, array('i', range(0, n)),
                             exptable, multiplication_table,
                             rootsonly=True)
    
    lenerr = len(Lx_roots)
    
    # error locations are given by the exponent of inverse of L(x) roots
    #error_locs = [(order-i)%order for i in Lx_roots]
    error_locs = clone(array_int_template, lenerr, False)
    for i in range(lenerr):
        error_locs.data.as_ints[i] = (-Lx_roots.data.as_ints[i]) % n

    roots_Omega_eval = GF2_poly_eval(Omega_poly, n, k, Lx_roots,
                                     exptable,multiplication_table)
    
    roots_Lambda_ddx_eval = GF2_poly_eval(Lambda_poly_ddx, n, k, Lx_roots,
                                          exptable,multiplication_table)
    
    # 0 is no error
    error_poly = clone(array_int_template, n, True)
    
    for i in range(lenerr):
        ome = roots_Omega_eval.data.as_ints[i]
        lam = roots_Lambda_ddx_eval.data.as_ints[i]
        
        # unreachable? roots cannot be zero
        if ome==0 or lam==0:
            error = 0
        else:
            expnt = (logtable[ome] - logtable[lam]) % n
            error = exptable[expnt]
        
        error_poly.data.as_ints[error_locs.data.as_ints[i]] = error

    # reverse the list as the error locator puts lowest exponent of x
    # at the start
    result = clone(array_int_template, n, False)
    for i in range(n):
        result.data.as_ints[i] = error_poly.data.as_ints[n-1-i]
    
    return result


###############################################################################
# ENCODER AND DECODER

# Encoding function for Reed-Solomon code. In systematic encoding, the 
# resulting codeword is the input message (m) of length k concatenated with
# n-k parity bits.
# Input and output are in binary strings (0s and 1s).

cpdef str RS_encode(str m, int n, int k, 
              tuple gf_tables, int[:] gx, 
              systematic=True):
    cdef:
        dict int2binstr_dict
        int[:,:] multiplication_table
        array[int] mx, tx, rx
        int s, double_t, lenm, i
        str codeword, rx_binstr
    
    int2binstr_dict, multiplication_table = gf_tables
    # symbol size, bits per symbol
    s = math.ceil(math.log2(n))
    # n-k = 2t
    double_t = n-k
    lenm = len(m)
    
    if lenm < k*s:
        m += '0'*(k*s-lenm)
    
    # split codeword into equal length s then
    # turn it into polynomial form
    mx = binstr2int_eqlen(m, s)
    
    # r(x) = (m(x) * x^{2t}) modulo g(x)
    # c(x) = m(x) * x^{2t} - r(x)
    
    # We multiply message polynomial by x^(2t), then divide
    # by generator polynomial, take the remainder and concatenate.
    tx = clone(array_int_template, n, True)
    for i in range(k):
        tx.data.as_ints[i] = mx.data.as_ints[i]
    rx = GF2_remainder_monic_divisor(tx, gx,
                                     multiplication_table, returnlen=n-k)
    
    rx_binstr = ''.join([int2binstr_dict[i] for i in rx])
    
    codeword = m + rx_binstr
    return codeword
    
    # TODO non-systematic encoding


# Decoding function of Reed-Solomon code. In systematic encoding, we chop off
# the last n-k bits from received message (recv) of length n to return the
# decoded message of length k.
# Input and output are in binary strings (0s and 1s).

cpdef str RS_decode(str recv, int n, int k, 
              tuple gf_tables, 
              systematic=True):
    
    cdef:
        int[:] exptable, logtable
        dict int2binstr_dict
        int[:,:] multiplication_table
        array[int] correctx, errorx, Lx, recvx, Syn
        int i, s, sumsyn, lenrecv
    
    exptable, logtable, int2binstr_dict, multiplication_table = gf_tables
    
    # s represents the number of bits per symbol.
    s = math.ceil(math.log2(n))
    
    recvx = binstr2int_eqlen(recv, s)
    
    # Calculate RS syndromes by evaluating the codeword polynomial.
    # Generator poly assumed to be (x-alpha)*(x-alpha^2)*...*(x-alpha^(n-k)),
    # so we evaluate cx starting from alpha^1 for S_1 to alpha^(n-k)
    # for the last entry.
    Syn = GF2_poly_eval(recvx, n, k, array('i', range(1,n-k+1)),
                        exptable, multiplication_table)
    
    sumsyn = 0
    for i in range(n-k):
        sumsyn += Syn.data.as_ints[i]
    lenrecv = len(recvx)
    
    if sumsyn != 0:
        # find error locator polynomial with Berlekamp-Massey
        # then use Forney to calculate the error polynomial
        Lx = Berlekamp_Massey(Syn, n, k, 
                              exptable, logtable, multiplication_table)
        errorx = Forney(Lx, Syn, n, k,
                    exptable, logtable, multiplication_table)

        # c(x) + e(x) = r(x)
        # c(x) = r(x) - e(x) ---> r(x) ^ e(x)
        # length not checked for speed, ensure they are same before this point
        correctx = clone(array_int_template, lenrecv, True)
        for i in range(lenrecv):
            correctx.data.as_ints[i] = (recvx.data.as_ints[i] ^
                                        errorx.data.as_ints[i])
        #correctx = [recvx[i] ^ errorx[i] for i in range(lenrecv)]

        # we only need k symbols out of n, so we cut the tail after k
        # if using systematic encoding
        
        decoded = ''.join([int2binstr_dict[i] for i in correctx[:k]])
        
        return decoded
    else:
        return recv[:k*s]
    
    # TODO non-systematic decoding

###############################################################################


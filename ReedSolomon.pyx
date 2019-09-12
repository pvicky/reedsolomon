#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from . import GaloisField as GF
from . import auxiliary as aux
import math
from cpython.array cimport array, clone, zero
import numpy as np
cimport numpy as np

cdef array array_int_template = array('i')
cdef array array_char_template = array('b')

"""
This file contains the functions necessary to encode a message with Reed
Solomon error-correcting code, and to decode an RS codeword into its original
message, with error checking and correction.

The primitive polynomial, n and k have to be known to build the generator
tables required for the encoding and decoding.

"""

###############################################################################

# Find a RS generator polynomial given the lower and upper powers of alpha,
# we usually use 1 until n-k.
# g(x) = (x-alpha^l) (x-alpha^(l+1)) ... (x-alpha^(l+2t-1))

# The result is a polynomial of x where each element is a list
# representing a polynomial of alpha.
# To simplify this polynomial, we need exptable which is in another function.
# lower = l, upper = l+2t-1
cdef list RS_generator_polynomial(int lower, int upper, int q):
    cdef:
        list t, poly, gx, gx_list = []
        array[int] t_ar
        int i
    
    # [[0,-1], [1]] is interpreted as 
    # {(0 * a^0) + (-1 * a^1)} * (x^0) +  {1 * a^0} * (x^1)
    # or simplified written as (x - a)
    for i in range(lower, upper+1):
        t_ar = array('i', [0]*i+[-1])
        t = [t_ar, array('i', [1])]
        gx_list.append(t)
    
    gx = gx_list[0]
    for poly in gx_list[1:]:
        gx = GF.nested_polynomial_product(gx, poly, modulo=q)
    
    return gx


# Mainly to fill the exponential and log tables used for encoding and decoding.
# Keys of exptable are in range [0, (2^n)-2].
# Its values are the integer forms of binary polynomials as result of
# exponentiating alpha to the power indicated by its key, then take the modulo
# of its division with given primitive polynomial.
# This ensures all values always have the same length, which is one less than
# the primitive polynomial.
# On the other hand, logtable is the inverse of exptable.
# The keys of logtable are the values in exptable while its values are the
# keys in exptable.
# Here, the number of bits per symbol is determined by the length of the
# primitive polynomial.
cdef tuple RS_bin_tables(list primitive_poly):
    
    cdef:
        array[int] exptable, logtable, x_coefs, primpoly_ar, \
                   remainder_descending, remainder
        list int2bin_list
        int polydegree, order, bits_per_symbol, exponent, remainder_int, i, j
        np.ndarray[int, ndim=2] multiplication_table
         
    polydegree = len(primitive_poly)-1
    bits_per_symbol = polydegree
    order = (1 << polydegree) - 1
    
    # move to array because division does not accept list
    primpoly_ar = clone(array_int_template, polydegree+1, True)
    for i in range(polydegree+1):
        primpoly_ar.data.as_ints[i] = primitive_poly[i]
    
    exptable = clone(array_int_template, order, True)
    logtable = clone(array_int_template, order+1, True)
    multiplication_table = np.zeros((order+1,order+1), dtype=np.int32)
         
    # log(0) = -infinity
    # x^(-infinity) = 0 so we don't put in exptable
    logtable.data.as_ints[0] = -1
    
    # x**0 = 1, log(1) = 0
    exptable.data.as_ints[0] = 1
    logtable.data.as_ints[1] = 0
    
    
    # coefficients of x, starting from x^0 as first element
    # we initialize it to 1
    x_coefs = clone(array_int_template, order, True)
    x_coefs.data.as_ints[0] = 1
    
    int2bin_list = [None]*(order+1)
    int2bin_list[0] = array('b', [0]*bits_per_symbol)
    
    # start from alpha^1
    exponent = 1
    while exponent < order:
        
        # multiply the previous result by x
        x_coefs.data.as_ints[exponent-1] = 0
        x_coefs.data.as_ints[exponent] = 1
        
        # then find the remainder of its division by primitive polynomial
        remainder = GF.GF_polynomial_div_remainder(x_coefs, primpoly_ar, 
                                          returnlen=polydegree, modulo=2)
        
        # reverse the array because we want the poly notation of exptable
        # to have the highest power of x at the start
        remainder_descending = array('i', remainder[::-1])
        remainder_int = aux.bin2int(remainder_descending, bits_per_symbol)
        exptable[exponent] = remainder_int
        logtable[remainder_int] = exponent
        int2bin_list[exponent] = aux.int2bin(exponent, 
                                             returnlen=bits_per_symbol)
        exponent += 1
        
    int2bin_list[exponent] = aux.int2bin(exponent, returnlen=bits_per_symbol)
    
    # first row is all zeros, skip
    for i in range(1,order+1):
        for j in range(order+1):
            multiplication_table[i,j] = GF.GF_product(i, j, order, 
                                                      exptable,logtable)
            
    return exptable, logtable, multiplication_table, int2bin_list


# Calls the functions to create the generator polynomial and relevant tables.
# Also converts the calculated generator polynomial into simpler form
# by use of the lookup table.
cpdef tuple RS_bin_generator(int n, int k, list primitive_poly):

    cdef:
        tuple gf_tables
        list int2bin_list
        int[:,::1] multiplication_table
        list generator_poly_alpha, generator_poly_int
        int i, j, t, alpha_degree, polylen, real_n
        array[int] gpoly, x_coef
        int[::1] exptable, logtable

    gf_tables = RS_bin_tables(primitive_poly)
    exptable, logtable, multiplication_table, int2bin_list = gf_tables
    
    generator_poly_alpha = RS_generator_polynomial(1,n-k, 2)
    
    polylen = len(generator_poly_alpha)
    generator_poly_int = [0]*polylen
    
    real_n = (1 << (len(primitive_poly)-1)) - 1
    for i in range(polylen):
        x_coef = generator_poly_alpha[i]
        alpha_degree = len(x_coef)-1
        t = 0
        
        for j in range(alpha_degree+1):
            if x_coef[j] != 0:
                t ^= exptable[j % real_n]
        
        generator_poly_int[i] = t
    
    gpoly = clone(array_int_template, polylen, False)
    for i in range(polylen):
        gpoly.data.as_ints[i] = generator_poly_int[i]
    return gf_tables, gpoly


###############################################################################
# A few functions for Reed-Solomon decoder


# calculating distance of Berlekamp-Massey algorithm at kth iteration
cdef int BM_delta_bin(int[::1] syndromes, int[::1] cx, int lencx,
                      int[:,::1] multiplication_table, int k):
    cdef int i, t, d = 0
    
    # No bound checking as we rely on the fact that syndrome array is always
    # bigger than cx array. Length of syndrome array is 2t, while cx at most
    # has a degree of t.
    for i in range(lencx):
        t = syndromes[k-1-i]
        d ^= (multiplication_table[t, cx[i]])
    return d

# Berlekamp-Massey algorithm, finding the error locator polynomial of 
# a codeword given syndromes.
cdef array[int] Berlekamp_Massey_bin(int[::1] syndromes, int n, int k, 
                                     int[::1] exptable, int[::1] logtable,
                                     int[:,::1] multiplication_table):
    
    cdef:
        array[int] cx, px, tx, subtractby
        int N, L, l, dm, delta, dm_inv, dx_dm_inv, \
            lencx, lenpx, i, subtractlen
    
    # N = 2t
    N = n-k
    
    # c(x) and p(x) is initialized to [1]
    # c(x) has to be filled with zeros because the result is used in other
    # function which does not know its length. Even if this function calculates
    # result correctly it may cause error when finding the error locator.
    cx = clone(array_int_template, N, True)
    cx.data.as_ints[0] = 1
    px = clone(array_int_template, N, False)
    px.data.as_ints[0] = 1
    
    tx = clone(array_int_template, N, False)
    subtractby = clone(array_int_template, N, True)
    
    lenpx, lencx = 1, 1
    L, l, dm = 0, 1, 1
    
    # k=0 is used for initial state so we start from k=1
    # this k is different to k given in argument
    for k in range(1,N+1):
        delta = BM_delta_bin(syndromes, cx, lencx, multiplication_table, k)
        
        if delta == 0:
            l += 1
        
        else:
            # cx = cx - d * dm^{-1} * px * x^{l}
            dm_inv = exptable[(n - logtable[dm]) % n]
            dx_dm_inv = (multiplication_table[delta][dm_inv])
            
            zero(subtractby)
            for i in range(lenpx):
                subtractby.data.as_ints[l + i] = (multiplication_table
                                            [dx_dm_inv, px.data.as_ints[i]])
            subtractlen = lenpx + l
            
            if 2*L >= k:

                GF.GF2_poly_add(cx, lencx, subtractby, subtractlen)
                l += 1
                
            # update L,px,dm and reset l after finding cx for this iteration
            else:
                # tx = cx
                for i in range(lencx):
                    tx.data.as_ints[i] = cx.data.as_ints[i]
                
                GF.GF2_poly_add(cx, lencx, subtractby, subtractlen)
                
                # px = tx
                # We make tx points to the old px so we don't lose the pointer
                # to px. Otherwise, px then points to tx and we lose track of
                # pointer to the initial px forever.
                px, tx = tx, px
                
                # The assignment of p(x) only happens when L is incremented.
                # Length of c(x) increases only in iterations where L changes.
                lenpx, lencx = lencx, subtractlen
                
                dm = delta
                L = k-L
                l = 1
                
    # \Lambda(x) is the c(x) after final iteration
    tx = clone(array_int_template, lencx, False)
    for i in range(lencx):
        tx.data.as_ints[i] = cx.data.as_ints[i]
    return tx


# Forney's algorithm, finding the error polynomial e(x)
# given a known error locator polynomial.
# As r(x) = c(x)+e(x), we subtract r(x) by e(x) to get the correct c(x).
cdef array[int] Forney_bin(int[::1] Lambda_poly, int[::1] Syndromes, int n, int k,
                           int[::1] exptable, int[::1] logtable,
                           int[:,::1] multiplication_table):
    
    cdef:
        array[int] Syndrome_x_Lambda_poly, x_exp_2t, Omega_poly, \
             Lambda_poly_ddx, Lx_roots, error_locs, roots_Omega_eval, \
             roots_Lambda_ddx_eval, error_poly, buffer_n
        int i, double_t = n-k, ome, lam, expnt, error, lenerr
    
    buffer_n = clone(array_int_template, n, True)
    for i in range(n):
        buffer_n.data.as_ints[i] = i
    
    # divide syndrome*Lx by x^{2t} and take the remainder
    x_exp_2t = clone(array_int_template, double_t+1, True)
    x_exp_2t.data.as_ints[double_t] = 1
    
    Syndrome_x_Lambda_poly = GF.GF2_poly_product(Lambda_poly, Syndromes,
                                             multiplication_table)
    
    Omega_poly = GF.GF2_remainder_monic_divisor(Syndrome_x_Lambda_poly,
                                                x_exp_2t,
                                                multiplication_table)
    
    Lambda_poly_ddx = aux.GF2_formal_derivative(Lambda_poly)
    
    # roots of L(x) are all exp where L(\alpha^{exp}) evaluates to 0
    # evaluate L(x) for all alpha^i, i=0,1,...,n
    Lx_roots = GF.GF2_poly_eval(Lambda_poly, n, buffer_n,
                             exptable, multiplication_table,
                             rootsonly=True)
    
    # evaluate roots of Lambda(x) on Omega(x)
    roots_Omega_eval = GF.GF2_poly_eval(Omega_poly, n, Lx_roots,
                                     exptable,multiplication_table)
    
    # evaluate roots of Lambda(x) on Lambda'(x)
    roots_Lambda_ddx_eval = GF.GF2_poly_eval(Lambda_poly_ddx, n, Lx_roots,
                                          exptable,multiplication_table)
    
    lenerr = len(Lx_roots)
    # error locations are given by the exponent of inverse of Lambda(x) roots
    # error_locs = [(order-i)%order for i in Lx_roots]
    error_locs = clone(array_int_template, lenerr, True)
    for i in range(lenerr):
        error_locs.data.as_ints[i] = (n - Lx_roots.data.as_ints[i]) % n
    
    # 0 is no error
    zero(buffer_n)
    
    for i in range(lenerr):
        ome = roots_Omega_eval.data.as_ints[i]
        lam = roots_Lambda_ddx_eval.data.as_ints[i]
        
        # unreachable? roots cannot be zero
        if ome==0 or lam==0:
            error = 0
        else:
            # error coefficient = - Omega(x) / Lambda'(x)
            # we can discard negative since we use XOR for binary
            expnt = (logtable[ome] - logtable[lam] +n) % n
            error = exptable[expnt]
        
        buffer_n.data.as_ints[error_locs.data.as_ints[i]] = error
    
    return buffer_n


###############################################################################
# ENCODER AND DECODER

# Encoding function for Reed-Solomon code. In systematic encoding, the 
# resulting codeword is the input message (m) of length k concatenated with
# n-k parity bits.
# Input and output are in binary strings (0s and 1s).

cpdef array[char] RS_bin_encode(char[::1] msg, 
                                int n, int k, int bits_per_symbol,
                                int[:,::1] multiplication_table, 
                                list int2bin_list,
                                int[::1] gx, 
                                systematic=True, ascending_poly=True):
    
    cdef:
        array[int] mx, tempx, rx
        int double_t, lenmsg, i, j, counter, real_n, real_k
        array[char] codeword, tarray
    
    # n-k = 2t
    double_t = n-k
    # length of message, in bits
    lenmsg = len(msg)
    
    real_n = (1 << bits_per_symbol)-1
    real_k = real_n - (n-k)
    
    # split codeword into equal length bits_per_symbol then
    # turn it into polynomial form
    mx = aux.binarray2intarray(msg, real_k, bits_per_symbol)
    
    # r(x) = (m(x) * x^{2t}) modulo g(x)
    # c(x) = m(x) * x^{2t} - r(x)
    
    # We multiply message polynomial by x^(2t), then divide
    # by generator polynomial, take the remainder and concatenate.
    # During calculations, the array of tempx must contain polynomial
    # coefficients in ascending order, so put mx in reverse when
    # ascending_poly = False.
    tempx = clone(array_int_template, n, True)
    if ascending_poly:
        for i in range(k):
            tempx.data.as_ints[double_t + i] = mx.data.as_ints[i]
    else:
        for i in range(k):
            tempx.data.as_ints[double_t + k - i - 1] = mx.data.as_ints[i]
            
    rx = GF.GF2_remainder_monic_divisor(tempx, gx,
                                     multiplication_table, returnlen=double_t)
    
    codeword = clone(array_char_template, bits_per_symbol*n, True)
    
    counter = 0
    if ascending_poly:
        # copy parity bits to buffer
        # length of parity bits: bits per symbol * 2t
        for i in range(double_t):
            tarray = int2bin_list[rx.data.as_ints[i]]
            for j in range(bits_per_symbol):
                codeword.data.as_schars[counter] = tarray.data.as_schars[j]
                counter += 1
        
        # append original message to the end
        for i in range(lenmsg):
            codeword.data.as_schars[i+counter] = msg[i]
    
    else:
        # original message at the start
        for i in range(lenmsg):
            codeword.data.as_schars[i] = msg[i]
        
        # parity bits at the end
        counter = lenmsg
        for i in range(double_t):
            tarray = int2bin_list[rx.data.as_ints[double_t - i - 1]]
            for j in range(bits_per_symbol):
                codeword.data.as_schars[counter] = tarray.data.as_schars[j]
                counter += 1
        
    
    return codeword
    
    # TODO non-systematic encoding


# Decoding function of Reed-Solomon code. In systematic encoding, we chop off
# the last n-k bits from received message (recv) of length n to return the
# decoded message of length k.
# Input and output are in binary strings (0s and 1s).

cpdef array[char] RS_bin_decode(char[::1] recv, 
                                int n, int k, int bits_per_symbol,
                                int[::1] exptable, int[::1] logtable, 
                                int[:,::1] multiplication_table, 
                                list int2bin_list,
                                int[::1] gx, 
                                systematic=True, trim_parity=True,
                                ascending_poly=True):
    
    cdef:
        array[int] recvx, tempx, remainder, Syn, Lx, cx, ex, buffer_2t
        array[char] decoded, tarray
        int real_n, real_k, i, j, counter, double_t, sumremainder, \
            lenrecv, lenrecvx, copy_start, copy_len
    
    double_t = n-k
    real_n = (1 << bits_per_symbol) - 1
    real_k = real_n - (n-k)
    
    # We only need k symbols out of n, so we cut the first n-k symbols
    # with systematic encoding and copy start from the end of parity bits 
    # (no matter what value ascending_poly takes, because first element always
    # represent lowest coefficient in internal calculations).
    # copy_len indicates how many symbols to be copied, not bits
    if trim_parity:
        decoded = clone(array_char_template, bits_per_symbol*k, True)
        copy_start = double_t
        copy_len = k
    # just correct the codeword, but keep parity bits intact
    else:
        decoded = clone(array_char_template, bits_per_symbol*n, True)
        copy_start = 0
        copy_len = n
    
    recvx = aux.binarray2intarray(recv, real_n, bits_per_symbol)
    # lenrecv is the real received length, that is the number of received bits
    # in recv divided by the number of bits per symbol
    # in other words, the number of received symbols
    # lenrecvx is full length of symbols, contains 2^(bits per symbol) elements
    # so lenrecvx >= lenrecv
    lenrecv = len(recv)/bits_per_symbol
    lenrecvx = len(recvx)
    
    # During calculations, the array of tempx must contain polynomial
    # coefficients in ascending order, so put recvx in reverse when
    # ascending_poly = False.
    tempx = clone(array_int_template, lenrecvx, True)
    if ascending_poly:
        for i in range(lenrecv):
            tempx.data.as_ints[i] = recvx.data.as_ints[i]
    else:
        for i in range(lenrecv):
            tempx.data.as_ints[i] = recvx.data.as_ints[lenrecv - i - 1]
    
    remainder = GF.GF2_remainder_monic_divisor(tempx, gx,
                                            multiplication_table)
    sumremainder = 0
    for i in range(double_t):
        sumremainder += remainder.data.as_ints[i]
    
    # if sumremainder != 0, there are errors
    if sumremainder:
        cx = clone(array_int_template, lenrecvx, True)
        
        # Calculate RS syndromes by evaluating the codeword polynomial.
        # Generator poly assumed to be (x-alpha)(x-alpha^2)*...*(x-alpha^(n-k))
        # so we evaluate rx starting from alpha^1 for S_1 to alpha^(n-k)
        # for the last entry.
        buffer_2t = clone(array_int_template, double_t, True)
        for i in range(double_t):
            buffer_2t.data.as_ints[i] = i+1
        Syn = GF.GF2_poly_eval(tempx, real_n, buffer_2t,
                            exptable, multiplication_table)
        
        # find error locator polynomial with Berlekamp-Massey
        # then use Forney to calculate the error polynomial
        Lx = Berlekamp_Massey_bin(Syn, real_n, real_k, 
                                  exptable, logtable, multiplication_table)
        ex = Forney_bin(Lx, Syn, real_n, real_k,
                        exptable, logtable, multiplication_table)
        
        # c(x) = r(x) - e(x)
        # length not checked for speed, ensure they are same before this point
        for i in range(lenrecvx):
            cx.data.as_ints[i] = (tempx.data.as_ints[i] ^
                                        ex.data.as_ints[i])
        
        # Copying of cx to result array starts here.
        counter = 0
        if ascending_poly:
            for i in range(copy_start, copy_start+copy_len):
                tarray = int2bin_list[cx.data.as_ints[i]]
                for j in range(bits_per_symbol):
                    decoded.data.as_schars[counter] = tarray.data.as_schars[j]
                    counter += 1
        else:
            for i in range(copy_start+copy_len - 1, copy_start - 1, -1):
                tarray = int2bin_list[cx.data.as_ints[i]]
                for j in range(bits_per_symbol):
                    decoded.data.as_schars[counter] = tarray.data.as_schars[j]
                    counter += 1
        
        return decoded
    
    # if there are no errors
    else:
        if ascending_poly:
            counter = copy_start * bits_per_symbol
            for i in range(0, copy_len * bits_per_symbol):
                decoded.data.as_schars[i] = recv[counter]
                counter += 1
        else:
            for i in range(0, copy_len * bits_per_symbol):
                decoded.data.as_schars[i] = recv[i]
            
        return decoded
    
    # TODO non-systematic decoding


###############################################################################

# Generate tables for non-binary Galois field.
# q is a prime integer
cpdef tuple RS_generate_tables(q):
    
    cdef:
        array[int] exptable, logtable, rem_array
        int order, n, exponent, i, j, generator_int, tempint
        np.ndarray[int, ndim=2] multiplication_table
    
    order = q-1
         
    exptable = clone(array_int_template, order, True)
    logtable = clone(array_int_template, order+1, True)
    multiplication_table = np.zeros((order+1,order+1), dtype=np.int32)
         
    # log(0) = -infinity
    # x^(-infinity) = 0 so we don't put in exptable
    logtable.data.as_ints[0] = -1
    
    # x**0 = 1, log(1) = 0
    exptable.data.as_ints[0] = 1
    logtable.data.as_ints[1] = 0
    
    generator_int = 0
    # find a positive integer < q that is valid generator
    # in total there are phi(q-1), but we settle for the first valid integer
    for i in range(1,q):
        tempint = i
        ctr = 1
        while tempint != 1:
            tempint = (tempint*i) % q
            ctr += 1
        if ctr == q-1:
            generator_int = i
            break
    
    # if we can't find a generator, exit
    if generator_int == 0:
        return
    
    tempint = generator_int
    
    exponent = 1
    while exponent < order:
        
        exptable[exponent] = tempint
        logtable[tempint] = exponent
        tempint = (tempint*generator_int) % q
        exponent += 1
    
    # first row is all zeros, skip
    for i in range(1,order+1):
        for j in range(order+1):
            multiplication_table[i,j] = GF.GF_product(i, j, order, 
                                                      exptable,logtable)
            
    return exptable, logtable, multiplication_table


cpdef tuple RS_generator(int n, int k, int q):

    cdef:
        tuple gf_tables
        list generator_poly_alpha, generator_poly_int
        int i, j, t, alpha_degree, polylen, real_n
        array[int] gpoly, x_coef
        int[::1] exptable, logtable
        int[:,::1] multiplication_table

    gf_tables = RS_generate_tables(q)
    exptable, logtable, multiplication_table = gf_tables
    
    generator_poly_alpha = RS_generator_polynomial(1,n-k, q)
    
    polylen = len(generator_poly_alpha)
    generator_poly_int = [0]*polylen
    
    real_n = q-1
    for i in range(polylen):
        x_coef = generator_poly_alpha[i]
        alpha_degree = len(x_coef)-1
        t = 0
        
        for j in range(alpha_degree+1):
            if x_coef[j] != 0:
                t += (x_coef[j] * exptable[j % real_n])
        
        generator_poly_int[i] = t % q
    
    gpoly = clone(array_int_template, polylen, False)
    for i in range(polylen):
        gpoly.data.as_ints[i] = generator_poly_int[i]
    return gf_tables, gpoly


# Reed-Solomon encoding for non-binary fields.
# The input msg is an array of integers of length k.
cpdef array[int] RS_encode(int[::1] msg, 
                            int n, int k, int q,
                            int[::1] gx, 
                            systematic=True, ascending_poly=True):
    
    cdef:
        array[int] tempx, rx, parity, codeword
        int double_t, lenmsg, i, counter, real_n, real_k
    
    # n-k = 2t
    double_t = n-k
    # length of message
    lenmsg = len(msg)
    
    real_n = q
    real_k = real_n - (n-k)
    
    # r(x) = (m(x) * x^{2t}) modulo g(x)
    # c(x) = m(x) * x^{2t} - r(x)
    
    # We multiply message polynomial by x^(2t), then divide
    # by generator polynomial, take the remainder and concatenate.
    # During calculations, the array of tempx must contain polynomial
    # coefficients in ascending order.
    tempx = clone(array_int_template, n, True)
    if ascending_poly:
        for i in range(k):
            tempx.data.as_ints[double_t + i] = msg[i]
    else:
        for i in range(k):
            tempx.data.as_ints[double_t + k - i - 1] = msg[i]
    
    rx = GF.GF_polynomial_div_remainder(tempx, gx, 
                                        returnlen=double_t, modulo=q)
    
    # because parity is = 0 - remainder
    parity = clone(array_int_template, double_t, True)
    for i in range(double_t):
        parity.data.as_ints[i] = (q - rx.data.as_ints[i]) % q
    
    codeword = clone(array_int_template, n, True)
    
    counter = 0
    if ascending_poly:
        # copy parity bits to buffer
        # length of parity bits: bits per symbol * 2t
        for i in range(double_t):
            codeword.data.as_ints[counter] = parity.data.as_ints[i]
            counter += 1
        
        # append original message to the end
        for i in range(lenmsg):
            codeword.data.as_ints[i+counter] = msg[i]
    
    else:
        # original message at the start
        for i in range(lenmsg):
            codeword.data.as_ints[i] = msg[i]
        
        # parity bits at the end
        # start from last element of parity
        counter = lenmsg
        for i in range(double_t):
            codeword.data.as_ints[counter] = parity.data.as_ints[double_t-i-1]
            counter += 1
    
    return codeword


cdef int BM_delta(int[::1] syndromes, int[::1] cx, int lencx,
                  int[:,::1] multiplication_table, int k,
                  modulo=0):
    cdef int i, t, d = 0
    
    for i in range(lencx):
        t = syndromes[k-1-i]
        d += (multiplication_table[t, cx[i]])
    if modulo != 0:
        d = d%modulo
    return d


cdef array[int] Berlekamp_Massey(int[::1] syndromes, int n, int k, int q,
                                 int[::1] exptable, int[::1] logtable,
                                 int[:,::1] multiplication_table):
    
    cdef:
        array[int] cx, px, tx, subtractby
        int N, L, l, dm, delta, dm_inv, dx_dm_inv, \
            lencx, lenpx, i, subtractlen
    
    N = n-k
    
    cx = clone(array_int_template, N, True)
    cx.data.as_ints[0] = 1
    px = clone(array_int_template, N, True)
    px.data.as_ints[0] = 1
    
    tx = clone(array_int_template, N, True)
    subtractby = clone(array_int_template, N, True)
    
    lenpx, lencx = 1, 1
    L, l, dm = 0, 1, 1
    
    for k in range(1,N+1):
        delta = BM_delta(syndromes, cx, lencx, multiplication_table, k,
                         modulo=q)
        
        if delta == 0:
            l += 1
        
        else:
            dm_inv = exptable[((q-1) - logtable[dm]) % (q-1)]
            dx_dm_inv = (multiplication_table[delta][dm_inv])
            
            zero(subtractby)
            for i in range(lenpx):
                subtractby.data.as_ints[l + i] = -(multiplication_table
                                            [dx_dm_inv, px.data.as_ints[i]])
            subtractlen = lenpx + l
            
            if 2*L >= k:

                GF.GF_poly_add(cx, lencx, subtractby, subtractlen, modulo=q)
                l += 1
            
            else:
                for i in range(lencx):
                    tx.data.as_ints[i] = cx.data.as_ints[i]
                
                GF.GF_poly_add(cx, lencx, subtractby, subtractlen, modulo=q)
                
                px, tx = tx, px
                
                lenpx, lencx = lencx, subtractlen
                
                dm = delta
                L = k-L
                l = 1
    
    tx = clone(array_int_template, lencx, False)
    for i in range(lencx):
        tx.data.as_ints[i] = cx.data.as_ints[i]
    return tx


cdef array[int] Forney(int[::1] Lambda_poly, int[::1] Syndromes, 
                       int n, int k, int q,
                       int[::1] exptable, int[::1] logtable,
                       int[:,::1] multiplication_table):
    
    cdef:
        array[int] Syndrome_x_Lambda_poly, x_exp_2t, Omega_poly, \
             Lambda_poly_ddx, Lx_roots, error_locs, roots_Omega_eval, \
             roots_Lambda_ddx_eval, error_poly, buffer_n
        int i, double_t = n-k, ome, lam, expnt, error, lenerr
    
    buffer_n = clone(array_int_template, n, True)
    for i in range(n):
        buffer_n.data.as_ints[i] = i
    
    # divide syndrome*Lx by x^{2t} and take the remainder
    x_exp_2t = clone(array_int_template, double_t+1, True)
    x_exp_2t.data.as_ints[double_t] = 1
    
    Syndrome_x_Lambda_poly = GF.polynomial_product(Lambda_poly, Syndromes)
    
    Omega_poly = GF.GF_polynomial_div_remainder(Syndrome_x_Lambda_poly,
                                                x_exp_2t, modulo=q)
    
    # Lambda'(x) is formal derivative of Lambda(x)
    Lambda_poly_ddx = aux.formal_derivative(Lambda_poly, q)
    
    # roots of L(x) are all exp where L(\alpha^{exp}) evaluates to 0
    # evaluate L(x) for all alpha^i, i=0,1,...,n
    Lx_roots = GF.GF_poly_eval(Lambda_poly, n, buffer_n,
                               exptable, multiplication_table,
                               modulo=q,
                               rootsonly=True)
    
    # evaluate roots of Lambda(x) on Omega(x)
    roots_Omega_eval = GF.GF_poly_eval(Omega_poly, n, Lx_roots,
                                     exptable,multiplication_table, modulo=q)
    
    # evaluate roots of Lambda(x) on Lambda'(x)
    roots_Lambda_ddx_eval = GF.GF_poly_eval(Lambda_poly_ddx, n, Lx_roots,
                                          exptable,multiplication_table, modulo=q)
    
    lenerr = len(Lx_roots)
    # error locations are given by the exponent of inverse of Lambda(x) roots
    # error_locs = [(order-i)%order for i in Lx_roots]
    error_locs = clone(array_int_template, lenerr, True)
    for i in range(lenerr):
        error_locs.data.as_ints[i] = ((q-1) - Lx_roots.data.as_ints[i]) % (q-1)
    
    # 0 is no error
    zero(buffer_n)
    
    for i in range(lenerr):
        ome = roots_Omega_eval.data.as_ints[i]
        lam = roots_Lambda_ddx_eval.data.as_ints[i]
        
        # unreachable? roots cannot be zero
        if ome==0 or lam==0:
            error = 0
        else:
            # error coefficient = - Omega(x) / Lambda'(x)
            expnt = (logtable[ome] - logtable[lam] + (q-1)) % (q-1)
            error = - exptable[expnt]
        
        buffer_n.data.as_ints[error_locs.data.as_ints[i]] = error
    
    return buffer_n


cpdef array[int] RS_decode(int[::1] recv, 
                           int n, int k, int q,
                           int[::1] exptable, int[::1] logtable, 
                           int[:,::1] multiplication_table,
                           int[::1] gx, 
                           systematic=True, trim_parity=True,
                           ascending_poly=True):
    
    cdef:
        array[int] recvx, tempx, remainder, Syn, Lx, cx, ex, buffer_2t, \
                   decoded, tarray
        int real_n, real_k, i, j, counter, double_t, sumremainder, \
            lenrecv, lenrecvx, copy_start, copy_len
    
    double_t = n-k
    real_n = q
    real_k = real_n - (n-k)
    
    # We only need k symbols out of n, so we cut the first n-k symbols
    # with systematic encoding and copy start from the end of parity bits 
    # (no matter what value ascending_poly takes, because first element always
    # represent lowest coefficient in internal calculations).
    # copy_len indicates how many symbols to be copied, not bits
    if trim_parity:
        decoded = clone(array_int_template, k, True)
        copy_start = double_t
        copy_len = k
    # just correct the codeword, but keep parity bits intact
    else:
        decoded = clone(array_int_template, n, True)
        copy_start = 0
        copy_len = n
    
    lenrecv = len(recv)
    
    tempx = clone(array_int_template, real_n, True)
    if ascending_poly:
        for i in range(lenrecv):
            tempx.data.as_ints[i] = recv[i]
    else:
        for i in range(lenrecv):
            tempx.data.as_ints[i] = recv[lenrecv - i - 1]
    
    remainder = GF.GF_polynomial_div_remainder(tempx, gx,
                                               returnlen=double_t, 
                                               modulo=q)
    
    sumremainder = 0
    for i in range(double_t):
        sumremainder += remainder.data.as_ints[i]
    
    # if sumremainder != 0, there are errors
    if sumremainder:
        cx = clone(array_int_template, lenrecv, True)
        
        # Calculate RS syndromes by evaluating the codeword polynomial.
        # Generator poly assumed to be (x-alpha)(x-alpha^2)*...*(x-alpha^(n-k))
        # so we evaluate rx starting from alpha^1 for S_1 to alpha^(n-k)
        # for the last entry.
        buffer_2t = clone(array_int_template, double_t, True)
        for i in range(double_t):
            buffer_2t.data.as_ints[i] = i+1
        
        Syn = GF.GF_poly_eval(tempx, real_n, buffer_2t,
                            exptable, multiplication_table, modulo=q)
        
        # find error locator polynomial with Berlekamp-Massey
        # then use Forney to calculate the error polynomial
        Lx = Berlekamp_Massey(Syn, real_n, real_k, q, 
                              exptable, logtable, multiplication_table)
        
        ex = Forney(Lx, Syn, real_n, real_k, q,
                    exptable, logtable, multiplication_table)
        
        # c(x) = r(x) - e(x)
        # length not checked for speed, ensure they are same before this point
        for i in range(lenrecv):
            cx.data.as_ints[i] = (((tempx.data.as_ints[i] -
                                      ex.data.as_ints[i]) % q) + q ) % q
        
        counter = 0
        if ascending_poly:
            for i in range(copy_start, copy_start+copy_len):
                decoded.data.as_ints[counter] = cx.data.as_ints[i]
                counter += 1
        else:
            for i in range(copy_start+copy_len - 1, copy_start - 1, -1):
                decoded.data.as_ints[counter] = cx.data.as_ints[i]
                counter += 1
                
        return decoded
    
    # if there are no errors
    else:
        if ascending_poly:
            counter = copy_start
            for i in range(0, copy_len):
                decoded.data.as_ints[i] = recv[counter]
                counter += 1
        else:
            for i in range(0, copy_len):
                decoded.data.as_ints[i] = recv[i]
            
        return decoded
    

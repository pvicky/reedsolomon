#cython: language_level=3, boundscheck=False, wraparound=False
import math
from auxiliary import *
from cpython.array cimport array, clone
import numpy as np
cimport numpy as np

cdef array array_int_template = array('i')

"""

a polynomial p(x) in the GF(2^n) can be written as:

p(x) = \alpha^{e_1} * x^{n-1} + \alpha^{e_2} * x^{n-2} + ... 
       + \alpha^{e_n-1} * x + \alpha^{e_n}

e is a placeholder for exponent, with range as given by [0, (2^n)-2].
To represent a 0, we use \alpha^{-\infty}

"""


# Multiply a and b, where both are integers form of some powers of alpha.
# Note that a and b themselves are not the exponents of an alpha.
# Returns an integer which represents \alpha^{log a + log b}.
# If either a or b is 0, it means multiply by 0, or to be precise 
# multiplying with alpha with an exponent of negative infinity.
cpdef int GF_product(int a, int b, 
                     int[:] exptable, int[:] logtable):
    
    # multiplied by x^{-\infty}
    if a==0 or b==0:
        return 0
    
    cdef:
        int order
    
    order = len(exptable)
    
    if a < order+1 and b < order+1:
        return exptable[(logtable[a] + logtable[b]) % order]
    else:
        return -1


# Given a binary polynomial (in integer), find its inverse in the
# exponential table -> x * x^{-1} = 1
# Returns the integer form of a binary polynomial
cpdef int GF_inverse(int x, 
                     int[:] exptable, int[:] logtable):
    cdef:
        int order
    
    order = len(exptable)
    
    # inverse of 0 (x^{-\infty}) is not defined
    if x == 0:
        return -1
    
    return exptable[(order - logtable[x]) % order]


# Multiply poly_a and poly_b, where both are the integer form of two
# polynomials in Galois Field.
# Returns an list which represents the resulting polynomial.
cpdef array[int] GF2_poly_product(int[:] poly_a, int[:] poly_b, 
                                  int[:,:] multiplication_table):
    
    # similar to convolution but look up table whenever multiplication occurs
    cdef:
        int i, j, total, support_end, lena, lenb
        array[int] result
    
    lena = len(poly_a)
    lenb = len(poly_b)
    support_end = lena+ lenb - 2
    
    # result is initialized to [0] * (support_end+1)
    result = clone(array_int_template, support_end+1, True)
    
    for i in range(0, support_end+1):
        total = 0
        for j in range(0, i+1):
            if j < lena and i-j < lenb:
                total ^= multiplication_table[poly_a[j]][poly_b[i-j]]
                
        result.data.as_ints[i] = total
    return result


# The input arguments dividend and divisor are polynomials where
# each element is integer representation of a binary number that itself
# represent a polynomial of x.
# For example, 11 represent 1011 meaning x^3 + x + 1
# If you want the polynomials represented as some power as alpha for each
# power of x, then take the logtable of each element.
cpdef list GF2_div_remainder(list dividend, list divisor, tuple gf_tables,
                             int returnlen=0):
    
    cdef:
        list exptable, logtable, remainder, subtractby
        int order, lendiv, i, j, divisor_lead_x_exponent, quot
    
    if len(dividend) < len(divisor):
        return dividend
    
    exptable, logtable = gf_tables
    order = len(exptable)
    lendiv = len(divisor)
    
    remainder = dividend[:]
    divisor_lead_x_exponent = logtable[divisor[0]]
    
    
    i = 0
    # trim leading zeros before first iteration
    while len(remainder)>i and remainder[i] == 0:
        i += 1
    
    while len(remainder) >= i+lendiv:
        # since we trim leading zeros first this will hold true if remainder
        # is all zeros, so we can avoid doing a sum to prevent subtraction
        # with None
        if remainder[i] == 0:
            break
        
        # find how much the divisor should be multiplied by, in terms of
        # power of alpha
        quot = (logtable[remainder[i]] - divisor_lead_x_exponent) % order
    
        # multiply divisor by quotient
        subtractby = [exptable[(logtable[x]+quot) % order] if x!=0 else 0 
                        for x in divisor]
        
        # XOR with the remainder
        # the result will reduce the degree of remainder by at least one
        for j in range(lendiv):
            remainder[i+j] = remainder[i+j] ^ subtractby[j]
        
        # remove leading zero of the remainder
        while len(remainder)>i and remainder[i] == 0:
            i += 1
    
    # result from the loop above should have length of at most len(generator)-1
    # so we can add 0s in front until the desired length
    # in cases where remainder's length is more than the generator's
    # (usually when codeword is all 0), we take the tail with length as needed
    if returnlen:
        if returnlen > len(remainder):
            return [0]*(returnlen-len(remainder)) + remainder
        else:
            return remainder[len(remainder) - returnlen:]
    else:
        return remainder


# Functionally a faster version of GF2_div_remainder, but limited to only
# monic polynomial as divisor.
cpdef array[int] GF2_remainder_monic_divisor(int[:] dividend,
                                             int[:] divisor, 
                                             int[:,:] multiplication_table, 
                                             int returnlen=0):
    
    cdef:
        int lendiv = len(divisor), lenrem = len(dividend), \
            lead_remainder, i, j, lendiff
        array[int] remainder, temparray
        
    
    if lenrem < lendiv:
        if returnlen > lenrem:
            remainder = clone(array_int_template, returnlen, True)
            lendiff = returnlen-lenrem
            for i in range(lenrem):
                remainder.data.as_ints[lendiff+i] = dividend[i]
                # equivalent to [0]*(returnlen-lenrem) + dividend
                return remainder
        else:
            return dividend[lenrem - returnlen:]
    
    remainder = clone(array_int_template, lenrem, False)
    for i in range(lenrem):
        remainder.data.as_ints[i] = dividend[i]
    
    i = 0
    # trim leading zeros before first iteration
    while lenrem > i and remainder.data.as_ints[i] == 0:
        i += 1
    
    while lenrem >= i+lendiv:
        lead_remainder = remainder.data.as_ints[i]

        # XOR with the remainder
        # the result will reduce the degree of remainder by at least one
        # i.e. lead_remainder will be 0 after the loop, and may be followed
        # by some zeros
        for j in range(lendiv):
            remainder.data.as_ints[i+j] = (remainder.data.as_ints[i+j] ^
                            multiplication_table[divisor[j]][lead_remainder])
        
        while lenrem > i and remainder.data.as_ints[i] == 0:
            i += 1
    
    # result from the loop above should have length of at most len(generator)-1
    # so we can add 0s in front until the desired length
    # in cases where remainder's length is more than the generator's
    # (usually when codeword is all 0), we take the tail with length as needed
    if returnlen:
        if returnlen > lenrem:
            temparray = clone(array_int_template, returnlen, True)
            lendiff = returnlen-lenrem
            for i in range(lenrem):
                temparray.data.as_ints[lendiff+i] = remainder.data.as_ints[i]
            # equivalent to [0]*(returnlen-lenrem) + remainder
            return temparray
        else:
            return remainder[lenrem - returnlen:]
    else:
        return remainder


# Evaluate the input polynomial px for each element in alphaexps.
# px is a polynomial where each element is the integer representation of some
# binary polynomial for a power of x (so we look up the entry in logtable to 
# know what power of alpha it is).
# On the other hand, each element of alphaexps is a power of alpha for which 
# we wish to evaluate, i.e. substitute x with \alpha^{exp}.
cpdef array[int] GF2_poly_eval(int[:] px, int n, int k, int[:] alphaexps, 
                         int[:] exptable, int[:,:] multiplication_table,
                         rootsonly=False):
    cdef:
        array[int] results, temp
        int degree=len(px)-1, order, reps=len(alphaexps), \
            i, j, r_i, expnt, numroots = 0
    
    order=len(exptable)
    
    results = clone(array_int_template, reps, True)
    
    for i in range(reps):
        r_i = 0
        
        # expnt is what x is substituted with
        expnt = alphaexps[i]
        for j in range(degree+1):
            
            # r_i + \alpha^{log(px[j])} * x^{degree-j}
            r_i ^= (multiplication_table[px[j]]
                        [exptable[expnt*(degree-j)%order]])
        results.data.as_ints[i] = r_i
        if r_i == 0:
            numroots += 1
    
    if rootsonly:
        temp = clone(array_int_template, numroots, False)
        j = 0
        for i in range(reps):
            if results.data.as_ints[i] == 0:
                temp.data.as_ints[j] = alphaexps[i]
                j += 1
        return temp
    else:
        return results


cpdef array[int] GF2_poly_add(int[:] poly_a, int[:] poly_b):
    cdef:
        int lena = len(poly_a), lenb = len(poly_b)
        int i, maxlen = max(lena, lenb), lendiff
        array[int] result = clone(array_int_template, maxlen, True)
    
    if lena > lenb:
        lendiff = maxlen - lenb
        for i in range(lendiff):
            result.data.as_ints[i] = poly_a[i]
        for i in range(lenb):
            result.data.as_ints[lendiff+i] = poly_a[lendiff+i] ^ poly_b[i]

    else:
        lendiff = maxlen - lena
        for i in range(lendiff):
            result.data.as_ints[i] = poly_b[i]
        for i in range(lena):
            result.data.as_ints[lendiff+i] = poly_a[i] ^ poly_b[lendiff+i]

    return result
    

###############################################################################

cdef list polynomial_product(list poly1, list poly2):
    cdef:
        int maxlen = 0, i
        list result_list, result, poly, tpoly, t
        
    result_list = []
    
    for i in range(len(poly2)):
        t = [poly2[i]*x for x in poly1] + [0]*(len(poly2)-i-1)
        result_list.append(t)
        if len(t) > maxlen:
            maxlen = len(t)
    
    result = [0]*maxlen
    
    for poly in result_list:
        tpoly = [0]*(maxlen-len(poly)) + poly
        result = [result[i] + tpoly[i] for i in range(maxlen)]
    
    return result


# calculate product of two polynomials by convolution
cpdef list GF_polynomial_product(list a, list b, int modulo=0):
    cdef:
        int degree_a, degree_b, support, k, i, maxlen
        list result, result_k, t, tlist, alphas_list
    
    degree_a = len(a)-1
    degree_b = len(b)-1
    
    # degree of result is degree of a + degree of b
    support = degree_a+degree_b
    result = []
    
    # k=0 is the highest degree
    for k in range(support+1):
        
        maxlen = 0
        tlist = []
        for i in range(k+1):
            if i < len(a) and k-i < len(b):
                alphas_list = polynomial_product(a[i], b[k-i])
                tlist.append(alphas_list)
                if len(alphas_list) > maxlen:
                    maxlen = len(alphas_list)
                
        result_k = [0]*maxlen
        for t in tlist:
            t = [0]*(maxlen-len(t)) + t
            result_k = [result_k[j] + t[j] for j in range(maxlen)]
            
        if modulo !=0:
            result_k = [i%modulo for i in result_k]
        
        result.append(result_k)
    return result


###############################################################################

# Find the remainder of polynomial division, not restricted to GF(2^n) and
# does not require GF tables.
# Note that the coefficients are not in powers of alpha, just regular integers.
cpdef list GF_polynomial_div_remainder(list dividend, list divisor,
                                      int returnlen=0, int modulo=0):
    
    if len(dividend) < len(divisor):
        return dividend
    
    cdef:
        int quot, i, j, lendiv, lenrem
        list remainder
    
    remainder = dividend[:]
    lendiv = len(divisor)
    lenrem = len(remainder)
    
    i = 0
    while lenrem > i and remainder[i] == 0:
        i += 1
    
    while lenrem >= i+lendiv:
        quot = remainder[i] / divisor[0]
        
        for j in range(lendiv):
            if modulo != 0:
                remainder[i+j] = (remainder[i+j] - quot*divisor[j]) % modulo
            else:
                remainder[i+j] = remainder[i+j] - quot*divisor[j]
        
        # remove leading zero of the remainder
        while lenrem > i and remainder[i] == 0:
            i += 1
        
    # cut or pad the array to the desired length
    if returnlen:
        if returnlen > len(remainder):
            return [0]*(returnlen-len(remainder))+remainder
        else:
            return remainder[len(remainder)-returnlen:]
    # or return the array as long as the highest degree with leading 1
    else:
        return remainder


###############################################################################




#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import math
from auxiliary import *
from cpython.array cimport array, clone
import numpy as np
cimport numpy as np

cdef array array_int_template = array('i')

"""
By convention, polynomials are written from the lowest exponent in increasing
order. For example, [1,2,3] represents x^0 + 2*x^1 + 3*x^2.
On the other hand, symbols are written with the most significant bit as first
element. In other words, the highest exponent goes first.
"""

# Multiply a and b, where both are integers form of some powers of alpha.
# Note that a and b themselves are not the exponents of an alpha.
# Returns an integer which represents \alpha^{log a + log b}.
# If either a or b is 0, it means multiply by 0, or to be precise 
# multiplying with alpha with an exponent of negative infinity.
cpdef int GF_product(int a, int b, 
                     int n, int[::1] exptable, int[::1] logtable):
    
    # multiplied by x^{-\infty}
    if a==0 or b==0:
        return 0
    
    cdef:
        int expnt
    
    if a < n+1 and b < n+1:
        expnt = (logtable[a] + logtable[b]) % n
        return exptable[expnt]
    else:
        return -1


# Given a binary polynomial (in integer), find its inverse in the
# exponential table -> x * x^{-1} = 1
# Returns the integer form of a binary polynomial
cpdef int GF_inverse(int x, 
                     int n, int[::1] exptable, int[::1] logtable):
    cdef:
        int expnt
    
    # inverse of 0 (x^{-\infty}) is not defined
    if x == 0:
        return -1
    expnt = (-logtable[x]) % n
    return exptable[expnt]


# Multiply poly_a and poly_b, where both are the integer form of two
# polynomials in Galois Field.
# Returns an list which represents the resulting polynomial.
cpdef array[int] GF2_poly_product(int[::1] poly_a, int[::1] poly_b, 
                                  int[:,::1] multiplication_table):
    
    # similar to convolution but look up table whenever multiplication occurs
    cdef:
        int i, j, total, support_end, lena, lenb
        array[int] result
    
    lena = len(poly_a)
    lenb = len(poly_b)
    support_end = lena+ lenb - 2
    
    result = clone(array_int_template, support_end+1, False)
    
    for i in range(0, support_end+1):
        total = 0
        for j in range(0, i+1):
            if j < lena and i-j < lenb:
                total ^= multiplication_table[poly_a[j], poly_b[i-j]]
                
        result.data.as_ints[i] = total
    return result


# The input arguments dividend and divisor are polynomials where
# each element is a symbol in the form of an integer. A symbol is represents
# a polynomial that is some power of alpha, encoded as binary number before
# conversion to integer.
# While the input polynomials dividend and divisor start with the lowest
# exponent as their first element (i.e. x^0 + x^1 + x^2 + ...), in symbols
# the lowest exponent is on the least significant (last) bit.
cpdef array[int] GF2_div_remainder(int[::1] dividend, int[::1] divisor, int n,
                                   int[::1] exptable, int[::1] logtable,
                                   int returnlen=0):
    
    cdef:
        array[int] remainder, temparray
        int lenrem, lendiv, i, j, divisor_lead_x_exponent, quot, expnt, t
    
    lenrem = len(dividend)
    lendiv = len(divisor)
    
    if lenrem < lendiv:
        if returnlen and returnlen > lenrem:
            remainder = clone(array_int_template, returnlen, True)
            for i in range(lenrem):
                remainder.data.as_ints[i] = dividend[i]
                return remainder
        elif returnlen and returnlen <= lenrem:
            return dividend[lenrem - returnlen:]
        else:
            return dividend
    
    remainder = clone(array_int_template, lenrem, False)
    for i in range(lenrem):
        remainder.data.as_ints[i] = dividend[i]
    
    i = lenrem-1
    # trim leading zeros before first iteration
    while i >= 0 and remainder.data.as_ints[i] == 0:
        i -= 1
    divisor_lead_x_exponent = logtable[divisor[lendiv-1]]
    
    while i+1 >= lendiv:
        
        # find how much the divisor should be multiplied by, in terms of
        # power of alpha
        quot = (logtable[remainder.data.as_ints[i]] - 
                divisor_lead_x_exponent) % n
        
        # the result will reduce the degree of remainder by at least one
        for j in range(lendiv):
            
            t = divisor[lendiv-1-j]
            # check to see if the coefficient is 0 because log(0) is -infinity
            # exp(-infinity) is equal to 0, and XOR with 0 does not change the
            # result, so we can skip
            if t:
                expnt = (logtable[t] + quot) % n
                remainder.data.as_ints[i-j] = (remainder.data.as_ints[i-j] ^
                                               exptable[expnt])
        
        # remove leading zero of the remainder
        while i >= 0 and remainder.data.as_ints[i] == 0:
            i -= 1
    
    # result from the loop above should have length of at most len(generator)-1
    # so we can add 0s in front until the desired length
    # in cases where remainder's length is more than the generator's
    # (usually when codeword is all 0), we take the tail with length as needed
    if returnlen:
        if returnlen > lenrem:
            temparray = clone(array_int_template, returnlen, True)
            for i in range(lenrem):
                temparray.data.as_ints[i] = remainder.data.as_ints[i]
                
            return temparray
        else:
            return remainder[:returnlen]
    else:
        return remainder


# Functionally a faster version of GF2_div_remainder, but limited to only
# monic polynomial as divisor.
cpdef array[int] GF2_remainder_monic_divisor(int[::1] dividend,
                                             int[::1] divisor, 
                                             int[:,::1] multiplication_table, 
                                             int returnlen=0):
    
    cdef:
        int lendiv = len(divisor), lenrem = len(dividend), \
            lead_remainder, i, j
        array[int] remainder, temparray
    
    if lenrem < lendiv:
        if returnlen and returnlen > lenrem:
            remainder = clone(array_int_template, returnlen, True)
            for i in range(lenrem):
                remainder.data.as_ints[i] = dividend[i]
                return remainder
        elif returnlen and returnlen <= lenrem:
            return dividend[lenrem - returnlen:]
        else:
            return dividend
    
    remainder = clone(array_int_template, lenrem, False)
    for i in range(lenrem):
        remainder.data.as_ints[i] = dividend[i]
    
    i = lenrem-1
    # trim leading zeros before first iteration
    while i >= 0 and remainder.data.as_ints[i] == 0:
        i -= 1
    
    while i+1 >= lendiv:
        lead_remainder = remainder.data.as_ints[i]
        
        # XOR with the remainder
        # the result will reduce the degree of remainder by at least one
        # i.e. lead_remainder will be 0 after the loop, and may be followed
        # by some zeros
        for j in range(lendiv):
            remainder.data.as_ints[i-j] = (remainder.data.as_ints[i-j] ^
                    multiplication_table[divisor[lendiv-1-j], lead_remainder])
        
        while i >= 0 and remainder.data.as_ints[i] == 0:
            i -= 1
    
    # result from the loop above should have length of at most len(generator)-1
    # so we can add 0s in front until the desired length
    # in cases where remainder's length is more than the generator's
    # (usually when codeword is all 0), we take the tail with length as needed
    if returnlen:
        if returnlen > lenrem:
            temparray = clone(array_int_template, returnlen, True)
            for i in range(lenrem):
                temparray.data.as_ints[i] = remainder.data.as_ints[i]
                
            return temparray
        else:
            return remainder[:returnlen]
    else:
        return remainder


# Evaluate the input polynomial px for each element in alphaexps.
# px is a polynomial where each element is the integer representation of some
# binary polynomial for a power of x (so we look up the entry in logtable to 
# know what power of alpha it is).
# On the other hand, each element of alphaexps is a power of alpha for which 
# we wish to evaluate, i.e. substitute x with \alpha^{exp}.
cpdef array[int] GF2_poly_eval(int[::1] px, int n, int k, int[::1] alphaexps, 
                         int[::1] exptable, int[:,::1] multiplication_table,
                         rootsonly = False):
    cdef:
        array[int] results, temp
        int degree=len(px)-1, reps=len(alphaexps), \
            i, j, result_i, expnt, numroots = 0, t, t2
    
    results = clone(array_int_template, reps, False)
    
    for i in range(reps):
        result_i = 0
        
        # \alpha^{expnt} is what x is substituted with
        expnt = alphaexps[i]
        t = 0
        for j in range(degree+1):
        # each iteration counts result_i + \alpha^{log(px[j])} * x^{degree-j}
            
            t2 = t%n
            # \alpha^{expnt * power of x}
            t2 = exptable[t2]
            
            result_i ^= (multiplication_table[px[j], t2])
            
            # it is faster to keep incrementing t and use another variable
            # to modulo it with n than to modulo t with n at every step
            t += expnt
        
        results.data.as_ints[i] = result_i
        # increment if result_i == 0:
        numroots += result_i==0
    
    if rootsonly:
        temp = clone(array_int_template, numroots, False)
        j = 0
        i = 0
        while j < numroots:
            if results.data.as_ints[i] == 0:
                temp.data.as_ints[j] = alphaexps[i]
                j += 1
            i += 1
        return temp
    else:
        return results


# Add two polynomials and store the result in the first polynomial, returns an
# integer that indicates the length of resulting polynomial.
# To ensure correctness, have to make sure length of array a >= b before
# passing them to the function, as result is stored in a.
# Note that length of a polynomial is not necessarily the same as the length
# of the array. A polynomial can be shorter than the array it is contained in;
# in calculation, their length is defined by the parameter integers.
# Thus, length of polynomial b can be larger than polynomial a, as long as
# length of array a is greater than or equal to length of polynomial b. 
# This way, the first polynomial can be reused as buffer for successive uses.
cpdef int GF2_poly_add(int[::1] poly_a, int len_a,
                       int[::1] poly_b, int len_b):
    cdef:
        int i, lendiff
    
    if len_a > len_b:
        lendiff = len_a - len_b
        for i in range(len_b):
            poly_a[i] = poly_a[i] ^ poly_b[i]
        
        return len_a

    else:
        lendiff = len_b - len_a
        for i in range(len_a):
            poly_a[i] = poly_a[i] ^ poly_b[i]
        for i in range(lendiff):
            poly_a[len_a + i] = poly_b[len_a + i]

        return len_b
    

###############################################################################

cdef list polynomial_product(list poly1, list poly2):
    cdef:
        int maxlen = 0, i
        list result_list, result, poly, tpoly, t
        
    result_list = []
    
    for i in range(len(poly2)):
        t = [0]*(len(poly2)-i-1) + [poly2[i]*x for x in poly1]
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
            t = t + [0]*(maxlen-len(t))
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
        if returnlen:
            return dividend + [0]*(returnlen-len(dividend))
        else:
            return dividend
    
    cdef:
        int quot, i, j, lendiv, lenrem
        list remainder
    
    remainder = dividend[:]
    lendiv = len(divisor)
    lenrem = len(remainder)
    
    i = lenrem-1
    while i >= 0 and remainder[i] == 0:
        i -= 1
    
    while i+1 >= lendiv:
        quot = remainder[i] / divisor[lendiv-1]
        
        for j in range(lendiv):
            if modulo != 0:
                remainder[i-j] = (remainder[i-j] - quot*divisor[lendiv-1-j]) % modulo
            else:
                remainder[i-j] = remainder[i-j] - quot*divisor[lendiv-1-j]
        
        # remove leading zero of the remainder
        while i >= 0 and remainder[i] == 0:
            i -= 1
        
    # cut or pad the array to the desired length
    if returnlen:
        if returnlen > len(remainder):
            return remainder + [0]*(returnlen-len(remainder))
        else:
            return remainder[:returnlen]
    # or return the full array
    else:
        return remainder


###############################################################################




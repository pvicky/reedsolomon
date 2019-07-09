#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import math
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
    expnt = (n-logtable[x]) % n
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


# Calculate remainder from given dividend and divisor polynomials over GF(2^m).
# In a polynomial, each coefficient of x is represented by an integer
# from exponential table, which itself is some power of an element alpha in
# the Galois field.
# While the input polynomials dividend and divisor start with the lowest
# exponent as their first element (i.e. x^0 + x^1 + x^2 + ...), in symbols
# the lowest exponent is on the least significant (last) bit.
cpdef array[int] GF2_div_remainder(int[::1] dividend, int[::1] divisor, int n,
                                   int[::1] exptable, int[::1] logtable,
                                   int returnlen=0):
    
    cdef:
        array[int] remainder, temparray
        int lendividend, lendivisor, i, j, \
            divisor_lead_x_exponent, quotient, expnt, t
    
    lendividend = len(dividend)
    lendivisor = len(divisor)
    
    # degree of dividend is already smaller than divisor
    # no need for calculation, just allocate space for result and return
    if lendividend < lendivisor:
        if returnlen:
            remainder = clone(array_int_template, returnlen, True)
            if returnlen > lendividend:
                t = lendividend
            else:
                t = returnlen
            
            for i in range(t):
                remainder.data.as_ints[i] = dividend[i]
            return remainder
            
        else:
            remainder = clone(array_int_template, lendividend, True)
            for i in range(lendividend):
                remainder.data.as_ints[i] = dividend[i]
            return remainder
    
    remainder = clone(array_int_template, lendividend, True)
    for i in range(lendividend):
        remainder.data.as_ints[i] = dividend[i]
    
    # i denotes the index of leading coefficient (poly in ascending order)
    i = lendividend-1
    # trim leading zeros before first iteration
    while i >= 0 and remainder.data.as_ints[i] == 0:
        i -= 1
    divisor_lead_x_exponent = logtable[divisor[lendivisor-1]]
    
    # while the degree of divisor is smaller, keep dividing
    while i+1 >= lendivisor:
        
        # find how much the divisor should be multiplied by to get rid of
        # leading coefficient in remainder
        # quotient is an integer representing the exponent part
        # of a power of alpha
        quotient = (logtable[remainder.data.as_ints[i]] - 
                divisor_lead_x_exponent) % n
        
        # we multiply the divisor by quotient, and subtract it from remainder
        # the result will reduce the degree of remainder by at least one
        for j in range(lendivisor):
            
            t = divisor[lendivisor-1-j]
            # t here is a coefficient of divisor
            # subtraction is done by XORing the coefficient in remainder with
            # exp(log(t) + quotient)
            
            # check to see if the coefficient is 0 because log(0) is -infinity
            # exp(-infinity) is equal to 0
            # since XOR with 0 does nothing, so we can skip
            if t != 0:
                expnt = (logtable[t] + quotient) % n
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
        if returnlen > lendividend:
            temparray = clone(array_int_template, returnlen, True)
            for i in range(lendividend):
                temparray.data.as_ints[i] = remainder.data.as_ints[i]
                
            return temparray
        else:
            return remainder[:returnlen]
    else:
        return remainder


# Calculate remainder from given dividend and divisor polynomials over GF(2^m).
# Functionally similar of GF2_div_remainder, but limited to only monic
# polynomial (1 as leading coefficient) as divisor.
cpdef array[int] GF2_remainder_monic_divisor(int[::1] dividend,
                                             int[::1] divisor, 
                                             int[:,::1] multiplication_table, 
                                             int returnlen=0):
    
    cdef:
        int lendivisor = len(divisor), lendividend = len(dividend), \
            lead_remainder, i, j, t
        array[int] remainder, temparray
    
    # degree of dividend is already smaller than divisor
    # no need for calculation, just allocate space for result and return
    if lendividend < lendivisor:
        if returnlen:
            remainder = clone(array_int_template, returnlen, True)
            if returnlen > lendividend:
                t = lendividend
            else:
                t = returnlen
            
            for i in range(t):
                remainder.data.as_ints[i] = dividend[i]
            return remainder
            
        else:
            remainder = clone(array_int_template, lendividend, True)
            for i in range(lendividend):
                remainder.data.as_ints[i] = dividend[i]
            return remainder
    
    remainder = clone(array_int_template, lendividend, True)
    for i in range(lendividend):
        remainder.data.as_ints[i] = dividend[i]
    
    # i denotes the index of leading coefficient (poly in ascending order)
    i = lendividend-1
    # trim leading zeros before first iteration
    while i >= 0 and remainder.data.as_ints[i] == 0:
        i -= 1
    
    while i+1 >= lendivisor:
        # monic polynomial has 1 as leading coefficient, so we multiply
        # divisor by the leading coefficient of remainder
        lead_remainder = remainder.data.as_ints[i]
        
        # XOR with the remainder
        # the result will reduce the degree of remainder by at least one
        # i.e. lead_remainder will be 0 after the loop, and may be followed
        # by some zeros
        for j in range(lendivisor):
            remainder.data.as_ints[i-j] = (remainder.data.as_ints[i-j] ^
                    multiplication_table[divisor[lendivisor-1-j], lead_remainder])
        
        while i >= 0 and remainder.data.as_ints[i] == 0:
            i -= 1
    
    # result from the loop above should have length of at most len(generator)-1
    # so we can add 0s in front until the desired length
    # in cases where remainder's length is more than the generator's
    # (usually when codeword is all 0), we take the tail with length as needed
    if returnlen:
        if returnlen > lendividend:
            temparray = clone(array_int_template, returnlen, True)
            for i in range(lendividend):
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
cpdef array[int] GF2_poly_eval(int[::1] px, int n, int[::1] alphaexps, 
                         int[::1] exptable, int[:,::1] multiplication_table,
                         rootsonly = False):
    cdef:
        array[int] results, temp
        int degree=len(px)-1, reps=len(alphaexps), \
            i, j, result_i, expnt, numroots = 0, t, t2
    
    results = clone(array_int_template, reps, True)
    
    for i in range(reps):
        result_i = 0
        
        # \alpha^{expnt} is what x is substituted with
        expnt = alphaexps[i]
        t = 0
        
        # each iteration counts result_i + \alpha^{log(px[j])} * x^{degree-j}
        # t is a temp variable and we increment it by expnt at each iteration
        # to indicate the exponent of a multiplied by exponent of x
        # suppose we want to evaluate 1 + x + x^2 + x^3 for a^5
        # which could be written as 1 + (a^5) + (a^5)^2 + (a^5)^3
        # so for first iteration, t=0; second t=5; third t=10; fourth t=15
        for j in range(degree+1):
            
            t2 = t%n
            # \alpha^{expnt * power of x}
            t2 = exptable[t2]
            
            # add product of poly coefficient at position j with a^(expnt*j)
            result_i ^= (multiplication_table[px[j], t2])
            
            # it is faster to keep incrementing t and use another variable
            # to modulo it with n than to modulo t with n at every step
            t += expnt
        
        results.data.as_ints[i] = result_i
        # increment if result_i == 0:
        numroots += result_i==0
    
    if rootsonly:
        temp = clone(array_int_template, numroots, True)
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
        # in case degree of a is less than b 
        # we still need to have enough space on a
        for i in range(lendiff):
            poly_a[len_a + i] = poly_b[len_a + i]

        return len_b
    

###############################################################################


# Calculate the product of two polynomials.
# The input poly1 and poly2 are lists where coefficients are ordered from
# lowest exponent to highest, i.e. first element is a^0, second is a^1, etc.
# Result will be stored in array[int].
cpdef array[int] polynomial_product(list poly1, list poly2):
    cdef:
        int maxlen = 0, lenp1 = len(poly1), lenp2 = len(poly2), i, j, k
        array[int] result
    
    maxlen = lenp1+lenp2-1
    result = clone(array_int_template, maxlen, True)
    
    for i in range(lenp2):
        k = i+lenp1
        for j in range(lenp1):
            result.data.as_ints[i+j] += poly2[i] * poly1[j]
    
    return result


# Calculate product of two polynomials a and b, where each element of a and b
# is a polynomial by itself (so they are in list of lists).
# The resulting polynomial is ordered from lowest exponent to highest,
# stored as nested list with depth = 2.
cpdef list nested_polynomial_product(list a, list b, int modulo=0):
    cdef:
        int len_a, len_b, k, i, j, maxlen
        list result, result_k, coef_k_list
        array[int] coef_k
    
    len_a = len(a)
    len_b = len(b)
    
    # degree of result is degree of a + degree of b
    result = [None] * (len_a+len_b-1)
    
    # start from the lowest exponent (x^0)
    for k in range(len_a+len_b-1):
        
        maxlen = 0
        coef_k_list = []
        for i in range(k+1):
            if i < len(a) and k-i < len(b):
                # backwards as in convolution
                coef_k = polynomial_product(a[i], b[k-i])
                coef_k_list.append(coef_k)
                if len(coef_k) > maxlen:
                    maxlen = len(coef_k)
        
        result_k = [0]*maxlen
        for coef_k in coef_k_list:
            j = len(coef_k)
            for i in range(j):
                result_k[i] = (result_k[i] + coef_k.data.as_ints[i])
            
        if modulo !=0:
            for i in range(maxlen):
                result_k[i] = result_k[i] % modulo
        
        result[k] = result_k
    return result


# Find the remainder of polynomial division, not restricted to GF(2^n) and
# does not require GF tables.
# Note that the coefficients are not in powers of alpha, just regular integers.
# The first element represents x^0.
cpdef array[int] GF_polynomial_div_remainder(int[::1] dividend, 
                                             int[::1] divisor,
                                             int returnlen=0, int modulo=0):
    
    cdef:
        int quotient, i, j, t, lendivisor, lendividend
        array[int] remainder
    
    lendivisor = len(divisor)
    lendividend = len(dividend)
    
    if lendividend < lendivisor:
        if returnlen:
            remainder = clone(array_int_template, returnlen, True)
            if returnlen > lendividend:
                t = lendividend
            else:
                t = returnlen
            
            for i in range(t):
                remainder.data.as_ints[i] = dividend[i]
            return remainder
            
        else:
            remainder = clone(array_int_template, lendividend, True)
            for i in range(lendividend):
                remainder.data.as_ints[i] = dividend[i]
            return remainder
    
    # copy the contents of dividend to remainder
    remainder = clone(array_int_template, lendividend, True)
    for i in range(lendividend):
        remainder.data.as_ints[i] = dividend[i]
    
    # i denotes the index of leading coefficient (poly in ascending order)
    i = lendividend-1
    while i >= 0 and remainder.data.as_ints[i] == 0:
        i -= 1
    
    # repeat while degree of divisor >= degree of remainder
    while i+1 >= lendivisor:
        quotient = remainder.data.as_ints[i] / divisor[lendivisor-1]
        
        for j in range(lendivisor):
            remainder.data.as_ints[i-j] -= quotient*divisor[lendivisor-1-j]
            if modulo != 0:
                t = remainder.data.as_ints[i-j] % modulo
                remainder.data.as_ints[i-j] = t
                if t < 0:
                    remainder.data.as_ints[i-j] = t + modulo
        
        # remove leading zero of the remainder
        while i >= 0 and remainder.data.as_ints[i] == 0:
            i -= 1
        
    # cut or pad the array to the desired length
    if returnlen:
        if returnlen > lendividend:
            temparray = clone(array_int_template, returnlen, True)
            for i in range(lendividend):
                temparray.data.as_ints[i] = remainder.data.as_ints[i]
                
            return temparray
        else:
            return remainder[:returnlen]
    else:
        return remainder


# Evaluate result of given polynomial for each exponent in alphaexps.
# Not restricted to binary.
cpdef array[int] GF_poly_eval(int[::1] px, int n, int[::1] alphaexps, 
                         int[::1] exptable, int[:,::1] multiplication_table,
                         int modulo=0, rootsonly = False):
    cdef:
        array[int] results, temp
        int degree=len(px)-1, reps=len(alphaexps), \
            i, j, result_i, expnt, numroots = 0, t, t2
    
    results = clone(array_int_template, reps, True)
    
    for i in range(reps):
        result_i = 0
        
        expnt = alphaexps[i]
        t = 0
        for j in range(degree+1):
            
            t2 = t%n
            t2 = exptable[t2]
            
            result_i += (multiplication_table[px[j], t2])
            
            t += expnt
        
        results.data.as_ints[i] = result_i % modulo
        numroots += result_i==0
    
    if rootsonly:
        temp = clone(array_int_template, numroots, True)
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


# Addition of two polynomials, store result in poly_a.
# Not restricted to binary.
cpdef int GF_poly_add(int[::1] poly_a, int len_a,
                      int[::1] poly_b, int len_b, int modulo=0):
    cdef:
        int i, lendiff
    
    if len_a > len_b:
        lendiff = len_a - len_b
        for i in range(len_b):
            poly_a[i] = (poly_a[i] + poly_b[i]) % modulo
        
        return len_a

    else:
        lendiff = len_b - len_a
        for i in range(len_a):
            poly_a[i] = (poly_a[i] + poly_b[i]) % modulo
        for i in range(lendiff):
            poly_a[len_a + i] = poly_b[len_a + i]

        return len_b
    

###############################################################################

# f(x) = (x-\beta) (x-\beta^q) ... (x-\beta^{q^{r-1}} for GF(q^m)
# Calculates minimal polynomial for \alpha^i, i = 0, 1, 2, ..., n-1
cpdef dict GF2_minimal_polynomial(int n, int[::1] exptable,
                                  int[:,::1] multiplication_table):
    cdef:
        dict min_poly = {}
        int i, j
        set conjugates
    
    # 1 + x
    min_poly[0] = array('i', [1, 1])
    
    for i in range(1,n):
        if i not in min_poly:
            conjugates = set([i])
            j = (2*i) % n
            while i != j:
                conjugates.add(j)
                j = (j*2) % n
            
            temp = array('i', [1])
            for j in conjugates:
                temp = GF2_poly_product(temp, array('i', [exptable[j], 1]),
                                        multiplication_table)
            
            for j in conjugates:
                min_poly[j] = temp
    return min_poly


def Euler_totient(q):
    if q<=0:
        return -1
    
    if q==1:
        return 1
    
    primes = [2]
    i = 3
    
    while i < q:
        i_prime = 1
        for j in primes:
            if i%j == 0:
                i_prime = 0
                break
        if i_prime == 1:
            primes.append(i)
        i += 1
    
    q_prime = 1
    for i in primes:
        if q%i==0:
            q_prime = 0
            break
    
    # if q is prime return q-1
    if q_prime == 1:
        return q-1
    # break down into factors
    else:
        factors = []
        for i in primes:
            j = 0
            while q%i == 0:
                q = q/i
                j += 1
            if j>0:
                factors.append((i, j))
        
        acc = 1
        for prime,exponent in factors:
            acc *= (prime-1)*prime**(exponent-1)
        return acc

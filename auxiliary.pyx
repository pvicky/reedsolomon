#cython: language_level=3, boundscheck=False, wraparound=False
import math
from cpython.array cimport array, clone

cdef array array_int_template = array('i')
cdef array array_char_template = array('b')

# Converts an integer into a binary number as a list of integers {0,1}.
# Negative integers are not handled, we simply return None.
# alternative method: [int(x) for x in bin(integer)[2:]]
cpdef array[char] int2bin(int integer, 
                  int returnlen=0):
    
    cdef:
        int i, t = 0, lendiff
        array[char] r, r2
        array[char] temp
        
    if integer < 0:
        r = clone(array_char_template, 0, False)
    
    elif integer == 0:
        r = clone(array_char_template, 1, True)
    
    else:
        temp = clone(array_char_template, 32, False)
        while integer>0:
            temp.data.as_schars[t] = integer&1
            t += 1
            integer = integer>>1
        
        r = clone(array_char_template, t, True)
        for i in range(t):
            r.data.as_schars[i] = temp.data.as_schars[t-i-1]
        
    if returnlen:
        if returnlen > len(r):
            
            # r = [0]*(returnlen-len(r)) + r
            lendiff = returnlen-t
            r2 = clone(array_char_template, returnlen, True)
            for i in range(t):
                r2.data.as_schars[lendiff+i] = r.data.as_schars[i]
            r = r2
        else:
            r = r[:returnlen]
    
    return r
    
    
cpdef str int2binstr(int integer, 
                    int returnlen=0):
    
    cdef:
        str r, t
    
    if integer < 0:
        r = ''
    elif integer == 0:
        r = '0'
    else:
        t = ''
        while integer>0:
            if integer&1:
                t+='1'
            else:
                t+='0'
            integer = integer>>1
        r = t[::-1]
        
    if returnlen:
        if returnlen > len(r):
            r = '0'*(returnlen-len(r)) + r
        else:
            r = r[:returnlen]
    return r
    

# Converts input (binary number in array of 0s and 1s) into integer.
# Most significant bit is at the start of input.
# It is assumed the length of input is equal to bitslen. If it is smaller,
# then we assume it is padded with 0s (but we don't need to do the padding).
cpdef int bin2int(int[::1] x, int bitslen):
    cdef:
        int i, t = 0
    
    for i in range(bitslen):
        if x[i]==1:
            t += 1<<(bitslen-i-1)
    return t
    

# Converts the input string in {0,1} into substrings of equal length z, then
# transform each substring into its integer representation.
cpdef array[int] binstr2int_eqlen(str binstr, int z):
    cdef:
        array[int] mx
        int x, lenstr = len(binstr), lenresult, ctr=0
    lenresult = lenstr//z
    mx = clone(array_int_template, lenresult, False)
    for x in range(lenresult):
        mx.data.as_ints[x] = int(binstr[ctr:ctr+z], base=2)
        ctr += z
    #mx = [int(binstr[x:x+z],base=2) for x in range(0,lenstr,z)]
    return mx


# Converts a continuous array of binary numbers {0,1} (as signed char) into
# an array of n integers.
# bitpi is the number of bits per integer.
cpdef array[int] binarray2intarray(char[::1] ar, int n, int bitpi):
    cdef:
        array[int] mx
        int x, i, t, real_len, ctr=0
    
    real_len = len(ar)//bitpi
    mx = clone(array_int_template, n, True)
    
    for x in range(real_len):
        t = 0
        for i in range(bitpi):
            t += (ar[ctr + i] & 1) << (bitpi-i-1)
            
        mx.data.as_ints[x] = t
        ctr += bitpi
    
    return mx


# Calculates hamming distance of two array of integers.
cpdef int hamming_distance(int[::1] str1, int[::1] str2):
    cdef:
        int i, len1, len2, dist
        
    len1 = len(str1)
    len2 = len(str2)
    
    # not defined for not equal length
    if len1 != len2:
        return -1
    else:
        dist = 0
        for i in range(len1):
            if str1[i] != str2[i]:
                dist += 1
        return dist
    

###############################################################################

# Find the derivative of a given input polynomial for GF(2^m) for an integer m.
cpdef array[int] GF2_formal_derivative(int[::1] poly):
    
    cdef:
        int i, multiplier, degree = len(poly)-1
        array[int] derivative = clone(array_int_template, degree, True)
    
    for i in range(degree):
        multiplier = i+1
        
        derivative.data.as_ints[i] = (multiplier&1) * poly[i+1]
        
    return derivative


cpdef array[int] formal_derivative(int[::1] poly, int modulo):
    
    cdef:
        int i, t, multiplier, degree = len(poly)-1
        array[int] derivative = clone(array_int_template, degree, True)
    
    for i in range(degree):
        multiplier = i+1
        t = (multiplier * poly[i+1]) % modulo
        derivative.data.as_ints[i] = t
        
    return derivative


# Convolution of two lists of integers.
cpdef list convolve(list lst_a, list lst_b):
    
    cdef:
        list result
        int i, j, lena, lenb, support_end, total
    
    lena, lenb = len(lst_a), len(lst_b)
    
    support_end = len(lst_a) + len(lst_b) - 2
    result = [0] * (support_end+1)
    
    for i in range(0, support_end+1):
        total = 0
        for j in range(0, i+1):
            if j < lena and i-j < lenb:
                total += lst_a[j] * lst_b[i-j]
        result[i] = total
    return result 

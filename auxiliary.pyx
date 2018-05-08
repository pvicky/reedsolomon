#cython: language_level=3, boundscheck=False, wraparound=False
import math
from cpython.array cimport array, clone, resize_smart

cdef array array_int_template = array('i')

# Converts an integer into a binary number as a list of integers {0,1}.
# Negative integers are not handled, we simply return None.
# alternative method: [int(x) for x in bin(integer)[2:]]
cpdef array[int] int2bin(int integer, 
                  int returnlen=0):
    
    cdef:
        int i, t = 0, lendiff
        array[int] r, temp
        
    if integer < 0:
        r = clone(array_int_template, 0, False)
    
    elif integer == 0:
        r = clone(array_int_template, 1, True)
    
    else:
        temp = clone(array_int_template, 0 ,False)
        while integer>0:
            resize_smart(temp, t+1)
            temp.data.as_ints[t] = integer&1
            t += 1
            integer = integer>>1
        
        r = clone(array_int_template, t, True)
        for i in range(t):
            r.data.as_ints[i] = temp.data.as_ints[t-i-1]
        
    if returnlen:
        if returnlen > len(r):
            
            # r = [0]*(returnlen-len(r)) + r
            lendiff = returnlen-t
            temp = clone(array_int_template, returnlen, True)
            for i in range(t):
                temp.data.as_ints[lendiff+i] = r.data.as_ints[i]
            r = temp
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
    

# converts input (binary number in array of 0s and 1s) into integer
cpdef int bin2int(int[::1] x):
    #return np.sum((2**np.arange(len(x)-1,-1,-1))*x)
    cdef:
        int i, t = 0, lenx = len(x)
    
    for i in range(lenx):
        if x[i]==1:
            t += 1<<(lenx-i-1)
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


cpdef int hamming_distance(str str1, str str2):
    cdef:
        str x,y
    return len([1 for x,y in zip(str1,str2) if x!=y])

###############################################################################

cpdef array[int] GF2_polynomial_derivative(int[::1] poly):
    
    cdef:
        int i, multiplier, degree = len(poly)-1
        array[int] derivative = clone(array_int_template, degree, False)
    
    for i in range(degree):
        multiplier = degree-i
        
        # if multiplier is odd take the coefficient as it is
        if (multiplier & 1):
            derivative.data.as_ints[i] = poly[i]
        # if multiplier is even the coefficient of x becomes 0
        else:
            derivative.data.as_ints[i] = 0
    return derivative


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

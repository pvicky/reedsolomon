import math


# Converts an integer into a binary number as a list of integers {0,1}.
# Negative integers are not handled, we simply return None.
# alternative method: [int(x) for x in bin(integer)[2:]]
cpdef list int2bin(int integer, 
                  int returnlen=0):
    
    cdef:
        list r, t
        
    if integer < 0:
        r = []
    elif integer == 0:
        r = [integer]
    else:
        t = []
        while integer>0:
            t+=[integer%2]
            integer //= 2
        r = t[::-1]
        
    if returnlen:
        if returnlen > len(r):
            r = [0]*(returnlen-len(r)) + r
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
            if integer%2==0:
                t+='0'
            else:
                t+='1'
            integer //= 2
        r = t[::-1]
        
    if returnlen:
        if returnlen > len(r):
            r = '0'*(returnlen-len(r)) + r
        else:
            r = r[:returnlen]
    return r
    

# converts input (binary number in array of 0s and 1s) into integer
cpdef int bin2int(list x):
    #return np.sum((2**np.arange(len(x)-1,-1,-1))*x)
    cdef:
        int i, t = 0, lenx = len(x)
    
    for i in range(lenx):
        if x[i]==1:
            t += 1<<(lenx-i-1)
    return t
    

# Converts the input string in {0,1} into substrings of equal length z, then
# transform each substring into its integer representation.
cpdef list binstr2int_eqlen(str binstr, int z):
    cdef:
        list mx
        int x, lenstr = len(binstr)
    mx = [int(binstr[x:x+z],base=2) for x in range(0,lenstr,z)]
    return mx


cpdef int hamming_distance(str str1, str str2):
    cdef:
        str x,y
    return len([1 for x,y in zip(str1,str2) if x!=y])

###############################################################################

cpdef list polynomial_derivative(list poly):
    
    cdef:
        int i, r, j, degree = len(poly)-1
        list derivative = []
    
    for i in range(degree):
        r = bin2int( [((degree-i)*j)%2 for j in int2bin(poly[i])] )
        derivative.append(r)
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

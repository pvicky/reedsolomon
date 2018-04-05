import math


# Converts an integer into a binary number as a list of integers {0,1}.
# Negative integers are not handled, we simply return None.
# alternative method: [int(x) for x in bin(integer)[2:]]
def int2bin(integer, returnlen=0, as_str=False):
    if integer < 0:
        r = None
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
    
    if as_str:
        return ''.join([str(i) for i in r])
    else:
        return r
    
    
def int2binstr(integer, returnlen=0):
    if integer < 0:
        r = None
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
def bin2int(x):
    #return np.sum((2**np.arange(len(x)-1,-1,-1))*x)
    t = 0
    lenx = len(x)
    for i in range(lenx):
        if x[i]==1:
            t += 1<<(lenx-i-1)
    return t
    

# binary string to integer
binstr2int = lambda x: int(x, base=2)


# Converts the input string in {0,1} into substrings of equal length z, then
# transform each substring into its integer representation.
def binstr2int_eqlen(binstr, z):
    
    mx = [int(binstr[x:x+z],base=2) for x in range(0,len(binstr),z)]
    return mx


def hamming_distance(str1, str2):
    return len([1 for x,y in zip(str1,str2) if x!=y])

###############################################################################

def polynomial_derivative(poly):
    
    degree = len(poly)-1
    derivative = []
    
    for i in range(degree):
        r = bin2int( [((degree-i)*j)%2 for j in int2bin(poly[i])] )
        derivative.append(r)
    return derivative


def convolve(lst_a, lst_b):
    support_end = len(lst_a) + len(lst_b) - 2
    result = [0] * (support_end+1)
    
    for i in range(0, support_end+1):
        total = 0
        for j in range(0, i+1):
            if j < len(lst_a) and i-j < len(lst_b):
                total += lst_a[j] * lst_b[i-j]
        result[i] = total
    return result 

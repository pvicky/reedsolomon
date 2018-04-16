import math
from auxiliary import *

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
def GF_product(a, b, gf_tables):
    
    # multiplied by x^{-\infty}
    if a==0 or b==0:
        return 0
    
    exptable, logtable = gf_tables
    order = len(exptable)
    
    if a in logtable and b in logtable:
        return exptable[(logtable[a] + logtable[b]) % order]
    else:
        return None


# Given a binary polynomial (in integer), find its inverse in the
# exponential table -> x * x^{-1} = 1
# Returns the integer form of a binary polynomial
def GF_inverse(x, gf_tables):
    exptable, logtable = gf_tables
    order = len(exptable)
    
    # inverse of 0 (x^{-\infty}) is not defined
    if x not in logtable or logtable[x] is None:
        return None
    
    return exptable[(order - logtable[x]) % order]


# Multiply poly_a and poly_b, where both are the integer form of two
# polynomials in Galois Field.
# Returns an list which represents the resulting polynomial.
def GF2_poly_product(poly_a, poly_b, multiplication_table):
    
    # similar to convolution but look up table whenever multiplication occurs
    support_end = len(poly_a) + len(poly_b) - 2
    result = [0] * (support_end+1)
    
    for i in range(0, support_end+1):
        total = 0
        for j in range(0, i+1):
            if j < len(poly_a) and i-j < len(poly_b):
                total ^= multiplication_table[poly_a[j]][poly_b[i-j]]
                
        result[i] = total
    return result


# The input arguments dividend and divisor are polynomials where
# each element is integer representation of a binary number that itself
# represent a polynomial of x.
# For example, 11 represent 1011 meaning x^3 + x + 1
# If you want the polynomials represented as some power as alpha for each
# power of x, then take the logtable of each element.
def GF2_div_remainder(dividend, divisor, gf_tables, returnlen=0):
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
def GF2_remainder_monic_divisor(dividend, divisor, 
                                multiplication_table, returnlen=0):
    if len(dividend) < len(divisor):
        if returnlen > len(dividend):
            return [0]*(returnlen-len(dividend)) + dividend
        else:
            return dividend[len(dividend) - returnlen:]
    
    lendiv = len(divisor)
    remainder = dividend[:]
    
    i = 0
    # trim leading zeros before first iteration
    while len(remainder)>i and remainder[i] == 0:
        i += 1
    
    while len(remainder) >= i+lendiv:
        lead_remainder = remainder[i]

        # XOR with the remainder
        # the result will reduce the degree of remainder by at least one
        for j in range(lendiv):
            remainder[i+j] = (remainder[i+j] ^
                            multiplication_table[divisor[j]][lead_remainder])
        
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


# Evaluate the input polynomial px for each element in alphaexps.
# px is a polynomial where each element is the integer representation of some
# binary polynomial for a power of x (so we look up the entry in logtable to 
# know what power of alpha it is).
# On the other hand, each element of alphaexps is a power of alpha for which 
# we wish to evaluate, i.e. substitute x with \alpha^{exp}.
def GF2_poly_eval(px, n, k, gf_tables, alphaexps, rootsonly=False):
    exptable, multiplication_table = gf_tables
    
    degree = len(px)-1
    order = len(exptable)
    
    results = [0]*len(alphaexps)
    roots = []
    for i in range(len(alphaexps)):
        r_i = 0
        
        # expnt is what x is substituted with
        expnt = alphaexps[i]
        for j in range(len(px)):
            
            # r_i + \alpha^{log(px[j])} * x^{degree-j}
            r_i ^= (multiplication_table[px[j]]
                        [exptable[expnt*(degree-j)%order]])
        results[i] = r_i
        if r_i == 0:
            roots += [expnt]
    
    if rootsonly:
        return roots
    else:
        return results


def GF2_poly_add(poly_a, poly_b):
    lena, lenb = len(poly_a), len(poly_b)
    maxlen = max(lena, lenb)
    
    if lena > lenb:
        poly_b = [0]*(maxlen-lenb) + poly_b
    else:
        poly_a = [0]*(maxlen-lena) + poly_a
    return [poly_a[i] ^ poly_b[i] for i in range(maxlen)]
    

###############################################################################


def polynomial_product(poly1, poly2):
    result_list = []
    maxlen = 0
    
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
def GF_polynomial_product(a, b, modulo=0):
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
def GF_polynomial_div_remainder(dividend, divisor, returnlen=0, modulo=0):
    
    if len(dividend) < len(divisor):
        return dividend
    
    remainder = dividend[:]
    lendiv = len(divisor)
    
    while len(remainder)>0 and remainder[0] == 0:
        remainder = remainder[1:]
    
    while len(remainder) >= lendiv:
        q = int(remainder[0] / divisor[0])
        
        remainder = ([remainder[i]-q*divisor[i] for i in range(lendiv)] 
                     + remainder[lendiv:])
        
        if modulo !=0:
            remainder = [i%modulo for i in remainder]
        
        # remove leading zero of the remainder
        while len(remainder)>0 and remainder[0] == 0:
            remainder = remainder[1:]
        
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




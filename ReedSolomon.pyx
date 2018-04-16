"""
This file contains the functions necessary to encode a message with Reed
Solomon error-correcting code, and to decode an RS codeword into its original
message, with error checking and correction.

The primitive polynomial, n and k have to be known to build the generator
tables required for the encoding and decoding.

"""

from GaloisField import * 
import itertools, math

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
# keys in exptable, plus an extra entry {0:None}.
cdef tuple RS_generate_tables(int n, list primitive_poly):
    
    cdef:
        dict exptable = {}, logtable = {}, int2binstr_dict = {}
        int polydegree, order, bits_per_symbol, exponent, remainder_int, i
        list x_coefs, remainder, multiplication_table, t, logkeys
         
    polydegree = len(primitive_poly)-1
    order = 2**(polydegree)-1
    bits_per_symbol = math.ceil(math.log2(n))
         
    # log(0) = -infinity
    logtable[0] = None
    # x^(-infinity) = 0 so we don't put in exptable
    
    # x**0 = 1, log(1) = 0
    exptable[0] = 1
    logtable[1] = 0
    
    
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
        
        remainder_int = bin2int(remainder)
        exptable[exponent] = remainder_int
        logtable[remainder_int] = exponent
        int2binstr_dict[exponent] = int2binstr(exponent, 
                                               returnlen=bits_per_symbol)
        exponent += 1
    int2binstr_dict[exponent] = int2binstr(exponent, 
                                           returnlen=bits_per_symbol)
    
    logkeys = sorted(logtable.keys())
    for i in logkeys:
        if i==0:
            multiplication_table = [[0]*len(logkeys)]
            continue
        
        t = []
        for j in logkeys:
            if i==0 or j==0:
                t += [0]
            else:
                t += [GF_product(i, j, (exptable,logtable))]
        multiplication_table += [t]
    
    return exptable, logtable, int2binstr_dict, multiplication_table


# Calls the functions to create the generator polynomial and relevant tables.
# Also converts the calculated generator polynomial into simpler form
# by use of the lookup table.
cpdef tuple RS_generator(int n, int k, list primitive_poly):

    cdef:
        tuple gf_tables
        dict exptable, logtable, int2binstr_dict
        list multiplication_table, generator_poly_alpha, generator_poly_int, \
             x_coef
        int i, j, t, alpha_degree, order

    gf_tables = RS_generate_tables(n, primitive_poly)
    exptable, logtable, int2binstr_dict, multiplication_table = gf_tables
    
    
    generator_poly_alpha = RS_generator_polynomial(1,n-k)
    generator_poly_int = [0]*len(generator_poly_alpha)
    order = len(exptable)
    
    for i in range(len(generator_poly_alpha)):
        x_coef = generator_poly_alpha[i]
        alpha_degree = len(x_coef)-1
        t = 0
        
        for j in range(alpha_degree+1):
            if x_coef[j] != 0:
                t ^= exptable[(alpha_degree-j) % order]
        
        generator_poly_int[i] = t
    
    return gf_tables, generator_poly_int


###############################################################################
# A few functions for Reed-Solomon decoder


# calculating distance of Berlekamp-Massey algorithm at kth iteration
cdef int BM_delta(list syndromes, list cx, list multiplication_table, int k):
    cdef int i, d = 0, lensyn = len(syndromes), lencx = len(cx)
    
    for i in range(k):
        if k-i-1 < lensyn and i < lencx:
            d ^= multiplication_table[syndromes[k-i-1]][cx[-i-1]]
    return d

# Berlekamp-Massey algorithm, finding the error locator polynomial of 
# a codeword given syndromes.
cdef list Berlekamp_Massey(list syndromes, int n, int k, 
                           tuple gf_tables):
    
    cdef:
        dict exptable, logtable
        list multiplication_table, cx, px, tx, subtractby
        int N, double_t, L, l, dm, delta, dx_dm_inv
    
    exptable, logtable, multiplication_table = gf_tables
    
    N = len(syndromes)
    double_t = n-k
    
    # c(x) and p(x) is initialized to [1]
    cx, px = [1], [1]
    L, l, dm = 0, 1, 1
    
    # k=0 is used for initial state so we start from k=1
    for k in range(1,N+1):
        delta = BM_delta(syndromes, cx, multiplication_table, k)
        
        if delta == 0:
            l += 1
        
        else:
            if 2*L >= k:
                
                # cx = cx - d * dm^{-1} * px * x^{l}
                dx_dm_inv = (multiplication_table[delta]
                                    [GF_inverse(dm,(exptable,logtable))])
                subtractby = [multiplication_table[dx_dm_inv][px[i]] 
                              for i in range(len(px))] + [0]*l
                cx = GF2_poly_add(cx, subtractby)
                l += 1
                
            # update L,px,dm and reset l after finding cx for this iteration
            else:
                tx = cx
                
                dx_dm_inv = (multiplication_table[delta]
                                    [GF_inverse(dm,(exptable,logtable))])
                subtractby = [multiplication_table[dx_dm_inv][px[i]] 
                              for i in range(len(px))] + [0]*l
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
cdef list Forney(list Lambda_poly, list Syndromes, int n, int k, 
                 tuple gf_tables):
    
    cdef:
        dict exptable, logtable
        list multiplication_table, Synd_poly, Syndrome_x_Lambda_poly, \
             x_exp_2t, Omega_poly, Lambda_poly_ddx, Lx_roots, \
             error_locations, roots_Omega_eval, roots_Lambda_ddx_eval, \
             error_poly
        int i, order, ome, lam, error, lenerr
    
    exptable, logtable, multiplication_table = gf_tables
    order = len(exptable)
    
    
    # reverse the syndrome list since the syndrome written as polynomial
    # starts with S1 * x^0 + S2 * x^1 + ...
    Synd_poly = Syndromes[::-1]
    Syndrome_x_Lambda_poly = GF2_poly_product(Lambda_poly, Synd_poly,
                                             multiplication_table)
    
    # divide by x^{2t} and take the remainder
    x_exp_2t = [1] + [0]*(n-k)
    Omega_poly = GF2_remainder_monic_divisor(Syndrome_x_Lambda_poly, 
                                             x_exp_2t, multiplication_table)
    
    Lambda_poly_ddx = polynomial_derivative(Lambda_poly)
    
    # roots of L(x) are all exp where L(\alpha^{exp}) evaluates to 0
    Lx_roots = GF2_poly_eval(Lambda_poly, n, k,
                             (exptable,multiplication_table), list(range(0, order)),
                             rootsonly=True)
    
    # error locations are given by the exponent of inverse of L(x) roots
    error_locations = [(order-i)%order for i in Lx_roots]

    roots_Omega_eval = GF2_poly_eval(Omega_poly, n, k,
                               (exptable,multiplication_table), Lx_roots)
    
    roots_Lambda_ddx_eval = GF2_poly_eval(Lambda_poly_ddx, n, k,
                                    (exptable,multiplication_table), Lx_roots)
    
    # 0 is no error
    error_poly = [0]*n
    
    lenerr = len(error_locations)
    
    for i in range(lenerr):
        ome = roots_Omega_eval[i]
        lam = roots_Lambda_ddx_eval[i]
        
        # unreachable?
        if ome==0 or lam==0:
            error = 0
        else:
            error = exptable[(logtable[ome] - logtable[lam]) % order]
        
        error_poly[error_locations[i]] = error

    # reverse the list as the error locator puts lowest exponent of x
    # at the start
    error_poly = error_poly[::-1]
    
    return error_poly


###############################################################################
# ENCODER AND DECODER

# Encoding function for Reed-Solomon code. In systematic encoding, the 
# resulting codeword is the input message (m) of length k concatenated with
# n-k parity bits.
# Input and output are in binary strings (0s and 1s).

cpdef str RS_encode(str m, int n, int k, 
              tuple gf_tables, list gx, 
              systematic=True):
    cdef:
        dict int2binstr_dict
        list multiplication_table, rx, mx
        int s, error_cap, lenm
    
    int2binstr_dict, multiplication_table = gf_tables
    # symbol size, bits per symbol
    s = math.ceil(math.log2(n))
    # n-k = 2t
    error_cap = n-k
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
    rx = GF2_remainder_monic_divisor(mx + [0]*error_cap, gx,
                                     multiplication_table, returnlen=n-k)
    
    rx_binstr = ''.join([int2binstr_dict[x] for x in rx])
    
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
        dict exptable, logtable, int2binstr_dict
        list multiplication_table, cx, ex, Lx, recvx, Syn
        int order, i, x, s, sumsyn, lenrecv
    
    exptable, logtable, int2binstr_dict, multiplication_table = gf_tables
    
    # s represents the number of bits per symbol.
    order = len(exptable)
    s = math.ceil(math.log2(n))
    
    recvx = binstr2int_eqlen(recv, s)
    
    # Calculate RS syndromes by evaluating the codeword polynomial.
    # Generator poly assumed to be (x-alpha)*(x-alpha^2)*...*(x-alpha^(n-k)),
    # so we evaluate cx starting from alpha^1 for S_1 to alpha^(n-k)
    # for the last entry.
    Syn = GF2_poly_eval(recvx, n, k, 
                      (exptable, multiplication_table), 
                      list(range(1, (n-k)+1)))
    
    sumsyn = sum(Syn)
    lenrecv = len(recvx)
    
    if sumsyn != 0:
        # find error locator polynomial with Berlekamp-Massey
        # then use Forney to calculate the error polynomial
        Lx = Berlekamp_Massey(Syn, n, k, 
                              (exptable, logtable, multiplication_table))
        ex = Forney(Lx, Syn, n, k, (exptable, logtable, multiplication_table))

        # c(x) + e(x) = r(x)
        # c(x) = r(x) - e(x) ---> r(x) ^ e(x)
        # length not checked for speed, ensure they are same before this point
        cx = [recvx[i] ^ ex[i] for i in range(lenrecv)]

        # we only need k symbols out of n, so we cut the tail after k
        # if using systematic encoding
        
        decoded = ''.join([int2binstr_dict[x] for x in cx[:k]])
        
        return decoded
    else:
        return recv[:k*s]
    
    # TODO non-systematic decoding

###############################################################################

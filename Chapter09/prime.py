# to install logpy use:
# pip install logic

import itertools as it
import logpy.core as lc
from sympy.ntheory.generate import prime, isprime

# Check if the elements of x are prime 
def check_prime(x):
    if lc.isvar(x):
        return lc.condeseq([(lc.eq, x, p)] for p in map(prime, it.count(1)))
    else:
        return lc.success if isprime(x) else lc.fail

# Declate the variable
x = lc.var()

# Print first 7 prime numbers
print('\nList of first 7 prime numbers:')
print(lc.run(20, x, check_prime(x)))

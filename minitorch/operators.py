"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# Implementation Task 0.1:

def mul(a, b): return a * b

def id(a): return a

def add(a, b): return a + b

def neg(a): return -a

def lt(a, b): return a < b

def eq(a, b): return a == b

def max(a, b): return a if a > b else b

def is_close(a, b): return -1e-2 < a - b < 1e-2

def sigmoid(a): return 1/(1+math.exp(-a)) if a >= 0 else (math.exp(a)) / (1 + math.exp(a))

def relu(a): return max(a, 0)

def log(a): return math.log(a)

def exp(a): return math.exp(a)

def log_back(a, b): return b / a

def inv(a): return 1 / a

def inv_back(a, b): return -b/a**2

def relu_back(a, b):return 0 if a < 0 else b


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(l, f): return [f(i) for i in l]

def zipWith(l1, l2, f): return [f(l1[i], l2[i]) for i in range(len(l1))]

def reduce(iter_obj, f, s):
    for i in iter_obj:
        s = f(s, i)
    return s

def negList(l): return map(l, neg)

def addLists(l1, l2): return zipWith(l1, l2, add)

def sum(l): return reduce(l, add, 0)

def prod(l): return reduce(l, mul, 1)
import numpy as np
from math import gamma

"""Computation of gauss quadratures via eigenvalue decomposition. 
Ref: Orthogonal Polynomials: Computation and Approximation, Walter Gautschi"""
def rjacobi(n, a, b):
    ra, rb = np.zeros(n), np.zeros(n) 

    apbp2 = 2. + a + b;    
    ra[0] = (b-a)/apbp2;
    rb[0] = np.power(2., a+b+1)*(  
            gamma(a+1.)*gamma(b+1.)/gamma(apbp2)
        ); 
    rb[1] = (4.*(a+1.)*(b+1.)/((apbp2+1.)*apbp2*apbp2));        

    # Compute other terms        
    apbp2 += 2;
    for i in range(1, n-1):
        ra[i] = (b*b-a*a)/((apbp2-2.)*apbp2)
        rb[i+1] = (
            4.*(i+1)*(i+1+a)*(i+1+b)*(i+1+a+b)/((apbp2*apbp2-1)*apbp2*apbp2)
        );
        apbp2 += 2
        
    ra[n-1] = (b*b-a*a)/((apbp2-2.)*apbp2);
    return ra, rb

def gauss(n, ra, rb):
    scal = rb[0];

    rb[:-1] = np.sqrt(rb[1:]);
    z, V = np.linalg.eigh(np.diag(ra) + np.diag(rb[:-1],-1))
    zidx = np.argsort(z); z.sort();
    V = V[:,zidx];

    w = V[0,:];
    w = scal*(w**2);
    return z, w

def zwgj(n, a, b):
    ra, rb = rjacobi(n, a, b)
    z, w = gauss(n, ra, rb)
    return z, w

def zwglj(n, a, b):
    N = n - 2;
    z, w = rjacobi(n, a, b);

    apb1 = a + b + 1.;
    z[n-1] = (a-b)/(2.*N+apb1+1.);
    w[n-1] = (4.*(N+a+1.)*(N+b+1.)*(N+apb1)
                    /((2.*N+apb1)*np.power(2*N+apb1+1, 2.)));

    z, w = gauss(n, z, w);
    return z, w
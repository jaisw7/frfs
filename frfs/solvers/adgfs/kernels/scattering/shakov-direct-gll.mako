<%include file="linear-common-direct-gll"/>
<% import math %>

// Construct shakov
__global__ void cmaxwellian
(
    const int nupts,
    const int ldim, 
    const int nvars, 
    const int neles,
    const scalar* cx, const scalar* cy, const scalar* cz,
    scalar* M, const scalar* U
)
{
    int idt = blockIdx.x*blockDim.x + threadIdx.x;

    if(idt<nvars*neles*nupts) 
    {
        int upt = int(idt/(nvars*neles));
        int elem = int((idt%(nvars*neles))/nvars);
        int idx = int((idt%(nvars*neles))%nvars);

        <%
        ke = "upt*neles + elem";
        %>

        M[upt*ldim + SOA_IX(elem, idx, nvars)] = U[(${ke})*${nalph}+0]/pow(${math.pi}*U[(${ke})*${nalph}+4], 1.5)*exp(
                -(
                    (cx[idx]-U[(${ke})*${nalph}+1])*(cx[idx]-U[(${ke})*${nalph}+1])
                  + (cy[idx]-U[(${ke})*${nalph}+2])*(cy[idx]-U[(${ke})*${nalph}+2])
                  + (cz[idx]-U[(${ke})*${nalph}+3])*(cz[idx]-U[(${ke})*${nalph}+3])
                )/U[(${ke})*${nalph}+4]
            );

        M[upt*ldim + SOA_IX(elem, idx, nvars)] *= (
            1. 
            + ${2*(1-Pr)/5.}*(
                  U[(${ke})*${nalph}+11]*(cx[idx]-U[(${ke})*${nalph}+1])
                + U[(${ke})*${nalph}+12]*(cy[idx]-U[(${ke})*${nalph}+2])
                + U[(${ke})*${nalph}+13]*(cz[idx]-U[(${ke})*${nalph}+3])
                )/(U[(${ke})*${nalph}+0]*U[(${ke})*${nalph}+4]*U[(${ke})*${nalph}+4])
                *(
                    2*(
                        (cx[idx]-U[(${ke})*${nalph}+1])*(cx[idx]-U[(${ke})*${nalph}+1])
                      + (cy[idx]-U[(${ke})*${nalph}+2])*(cy[idx]-U[(${ke})*${nalph}+2])
                      + (cz[idx]-U[(${ke})*${nalph}+3])*(cz[idx]-U[(${ke})*${nalph}+3])
                    )/U[(${ke})*${nalph}+4]
                    - 5
                )
            );
    }
}


// Construct maxwellian
__global__ void momentNorm
(
    const int lda,
    const scalar* U0,
    scalar *U
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<lda) 
    {   
        // copy
        %for iter in range(nalph):
            U[idx*${nalph}+${iter}] = U0[idx*${nalph}+${iter}];
        %endfor        

        U[idx*${nalph}+0] = U[idx*${nalph}+0]; // density
        U[idx*${nalph}+1] /= U[idx*${nalph}+0]; // x-velocity
        U[idx*${nalph}+2] /= U[idx*${nalph}+0]; // y-velocity
        U[idx*${nalph}+3] /= U[idx*${nalph}+0]; // z-velocity

        // T = mom4/1.5/rho - (u*u+v*v+w*w)/1.5
        U[idx*${nalph}+4] = U[idx*${nalph}+4]/(1.5*U[idx*${nalph}+0])
          - (
                U[idx*${nalph}+1]*U[idx*${nalph}+1]
                + U[idx*${nalph}+2]*U[idx*${nalph}+2]
                + U[idx*${nalph}+3]*U[idx*${nalph}+3]
            )/1.5;
        
        // qx, qy, qz
        % for iter, (qq1, qq2, qq3) in enumerate(['035', '314', '542']):
            <% q1, q2, q3 = map(int, (qq1, qq2, qq3)) %>
            U[idx*${nalph}+${iter}+11] += 
                - 2*(
                        U[idx*${nalph}+5+${q1}]*U[idx*${nalph}+1]
                      + U[idx*${nalph}+5+${q2}]*U[idx*${nalph}+2]
                      + U[idx*${nalph}+5+${q3}]*U[idx*${nalph}+3]
                       
                    )
                + U[idx*${nalph}+0]*U[idx*${nalph}+1+${iter}]*(
                    U[idx*${nalph}+1]*U[idx*${nalph}+1]
                    + U[idx*${nalph}+2]*U[idx*${nalph}+2]
                    + U[idx*${nalph}+3]*U[idx*${nalph}+3]
                )
                - 1.5*U[idx*${nalph}+1+${iter}]*U[idx*${nalph}+0]*U[idx*${nalph}+4];
        %endfor
    }
}


%for ar in nbdf:
__global__ void ${"updateMom{0}_BDF".format(ar)}
(
    const scalar prefac,
    const int lda,
    const scalar dt,
    ${" ".join([
      "const scalar a{0}, const scalar* U{0}, const scalar g{0}, const scalar* LU{0},".format(i) 
      for i in range(ar)
    ])}
    const scalar a${ar}, scalar* U, const scalar b
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar T;

    if(idx<lda) 
    {   
        // density, x-velocity, y-velocity, z-velocity, energy
        %for iter in [0, 1, 2, 3, 4]:
            U[idx*${nalph}+${iter}] = (-(${"+ ".join([
                        "a{0}*U{0}[idx*{1}+{2}] + g{0}*dt*LU{0}[idx*{1}+{2}]".format(
                            i, nalph, iter) for i in range(ar)
                    ])}))/a${ar};    
        %endfor

        T = (U[idx*${nalph}+4]
            - (
                + U[idx*${nalph}+1]*U[idx*${nalph}+1]
                + U[idx*${nalph}+2]*U[idx*${nalph}+2]
                + U[idx*${nalph}+3]*U[idx*${nalph}+3]
              )/U[idx*${nalph}+0]
          )/(1.5*U[idx*${nalph}+0]);

        // diagonal/normal components of the stress
        %for iter in [5, 6, 7]:
            U[idx*${nalph}+${iter}] = (-(${"+ ".join([
                        "a{0}*U{0}[idx*{1}+{2}] + g{0}*dt*LU{0}[idx*{1}+{2}]".format(
                            i, nalph, iter) for i in range(ar)
                    ])}) 
              + b*dt*prefac*U[idx*${nalph}+0]*pow(T,${1-omega})
                *(
                    0.5*U[idx*${nalph}+0]*T
                  + U[idx*${nalph}+${iter-4}]*U[idx*${nalph}+${iter-4}]
                      /U[idx*${nalph}+0]
                )
            )/(a${ar}+b*dt*prefac*U[idx*${nalph}+0]*pow(T,${1.-omega}));
        %endfor

        // off-diagonal components of the stress
        %for iter in [8, 9, 10]:
            U[idx*${nalph}+${iter}] = (-(${"+ ".join([
                        "a{0}*U{0}[idx*{1}+{2}] + g{0}*dt*LU{0}[idx*{1}+{2}]".format(
                            i, nalph, iter) for i in range(ar)
                    ])})
              + b*dt*prefac*U[idx*${nalph}+0]*pow(T,${1-omega})
                *(
                    U[idx*${nalph}+${iter-7}]*U[idx*${nalph}+${(iter-7)%3+1}]
                      /U[idx*${nalph}+0]
                  )
            )/(a${ar}+b*dt*prefac*U[idx*${nalph}+0]*pow(T,${1.-omega}));
        %endfor


        // Evolve heat flux
        % for iter, (qq1, qq2, qq3) in enumerate(['035', '314', '542']):
            <% q1, q2, q3 = map(int, (qq1, qq2, qq3)) %>
            U[idx*${nalph}+${iter}+11] = (-(${"+ ".join([
                        "a{0}*U{0}[idx*{1}+{2}] + g{0}*dt*LU{0}[idx*{1}+{2}]".format(
                            i, nalph, 11+iter) for i in range(ar)
                    ])})
                + b*dt*prefac*U[idx*${nalph}+0]*pow(T,${1-omega})
                  *(
                    ${1.-Pr}*(
                      -2.*(
                          U[idx*${nalph}+5+${q1}]*U[idx*${nalph}+1]
                        + U[idx*${nalph}+5+${q2}]*U[idx*${nalph}+2]
                        + U[idx*${nalph}+5+${q3}]*U[idx*${nalph}+3]
                      )/U[idx*${nalph}+0]
                      + U[idx*${nalph}+1+${iter}]*(
                          U[idx*${nalph}+1]*U[idx*${nalph}+1]
                        + U[idx*${nalph}+2]*U[idx*${nalph}+2]
                        + U[idx*${nalph}+3]*U[idx*${nalph}+3]
                      )/pow(U[idx*${nalph}+0], 2.)
                      - 1.5*U[idx*${nalph}+1+${iter}]*T
                    )
                  + (
                      U[idx*${nalph}+1+${iter}]*(
                          U[idx*${nalph}+1]*U[idx*${nalph}+1]
                        + U[idx*${nalph}+2]*U[idx*${nalph}+2]
                        + U[idx*${nalph}+3]*U[idx*${nalph}+3]
                      )/pow(U[idx*${nalph}+0], 2.)
                      + 2.5*U[idx*${nalph}+1+${iter}]*T
                    )
                  )
                )/(
                    a${ar} + ${Pr}*b*dt*prefac*U[idx*${nalph}+0]*pow(T, ${1-omega})
                );                
        %endfor
    }
}
%endfor



%for ar in nars:
__global__ void ${"updateMom{0}_ARS".format(ar)}
(
    const scalar prefac,
    const int lda, 
    const double dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(ar)])}
    ${" ".join(["const scalar* LU{0},".format(i) for i in range(ar)])}
    ${" ".join(["const scalar* U{0},".format(i) for i in range(ar)])}
    ${"scalar *U{0}".format(ar)}
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    scalar ${", ".join(["T{0}".format(i) for i in range(1, ar+1)])};

    if(idx<lda) 
    {   
        // density, x-velocity, y-velocity, z-velocity, energy
        %for iter in [0, 1, 2, 3, 4]:
            U${ar}[idx*${nalph}+${iter}] = U0[idx*${nalph}+${iter}]
                + 
                ${"+ ".join(
                    [
                        "dt*_a{0}{1}*LU{1}[idx*{2}+{3}]".format(
                            ar, i, nalph, iter) for i in range(ar)
                    ])};    
        %endfor

        %for i in range(1, ar+1):
            T${i} = (  
                  U${i}[idx*${nalph}+4]
                  - (
                      U${i}[idx*${nalph}+1]*U${i}[idx*${nalph}+1]
                      + U${i}[idx*${nalph}+2]*U${i}[idx*${nalph}+2]
                      + U${i}[idx*${nalph}+3]*U${i}[idx*${nalph}+3]
                    )/U${i}[idx*${nalph}+0]
                )/(1.5*U${i}[idx*${nalph}+0]);  // Temperature
        %endfor

        // diagonal/normal components of the stress
        %for iter in [5, 6, 7]:
            U${ar}[idx*${nalph}+${iter}] = (U0[idx*${nalph}+${iter}] +
              + ${"+ ".join([
                        "_a{0}{1}*dt*LU{1}[idx*{2}+{3}]".format(
                            ar, i, nalph, iter) for i in range(ar)
                    ])} 
              %for i in range(1, ar+1):
              + a${ar}${i}*dt*prefac*U${i}[idx*${nalph}+0]*pow(T${i},${1-omega})
                *(
                    0.5*U${i}[idx*${nalph}+0]*T${i}
                  + U${i}[idx*${nalph}+${iter-4}]*U${i}[idx*${nalph}+${iter-4}]
                      /U${i}[idx*${nalph}+0]
                  %if i!=ar:
                    - U${i}[idx*${nalph}+${iter}]
                  %endif
                )
              %endfor
            )/(1+a${ar}${ar}*dt*prefac*U${ar}[idx*${nalph}+0]*pow(T${ar},${1.-omega}));
        %endfor

        // off-diagonal components of the stress
        %for iter in [8, 9, 10]:
            U${ar}[idx*${nalph}+${iter}] = (U0[idx*${nalph}+${iter}] +
              + ${"+ ".join([
                        "_a{0}{1}*dt*LU{1}[idx*{2}+{3}]".format(
                            ar, i, nalph, iter) for i in range(ar)
                    ])} 
              %for i in range(1, ar+1):
              + a${ar}${i}*dt*prefac*U${i}[idx*${nalph}+0]*pow(T${i},${1-omega})
                *(
                    U${i}[idx*${nalph}+${iter-7}]*U${i}[idx*${nalph}+${(iter-7)%3+1}]
                      /U${i}[idx*${nalph}+0]
                  %if i!=ar:
                    - U${i}[idx*${nalph}+${iter}]
                  %endif
                )
              %endfor
            )/(1+a${ar}${ar}*dt*prefac*U${ar}[idx*${nalph}+0]*pow(T${ar},${1.-omega}));
        %endfor


        // Evolve heat flux
        % for iter, (qq1, qq2, qq3) in enumerate(['035', '314', '542']):
            <% q1, q2, q3 = map(int, (qq1, qq2, qq3)) %>
            U${ar}[idx*${nalph}+${iter}+11] = (U0[idx*${nalph}+${iter}+11]
                + ${"+ ".join([
                        "_a{0}{1}*dt*LU{1}[idx*{2}+{3}]".format(
                            ar, i, nalph, iter+11) for i in range(ar)])}
                + 
                (
                  % for i in range(1, ar+1):
                    + a${ar}${i}*dt*prefac*U${i}[idx*${nalph}+0]*pow(T${i}, ${1-omega})
                    *(
                      ${(1-Pr)}*(
                        -2.*(
                            U${i}[idx*${nalph}+5+${q1}]*U${i}[idx*${nalph}+1]
                          + U${i}[idx*${nalph}+5+${q2}]*U${i}[idx*${nalph}+2]
                          + U${i}[idx*${nalph}+5+${q3}]*U${i}[idx*${nalph}+3]
                        )/U${i}[idx*${nalph}+0]
                        + U${i}[idx*${nalph}+1+${iter}]*(
                            U${i}[idx*${nalph}+1]*U${i}[idx*${nalph}+1]
                          + U${i}[idx*${nalph}+2]*U${i}[idx*${nalph}+2]
                          + U${i}[idx*${nalph}+3]*U${i}[idx*${nalph}+3]
                        )/pow(U${i}[idx*${nalph}+0], 2.)
                        - 1.5*U${i}[idx*${nalph}+1+${iter}]*T${i}
                      )
                      + (
                          U${i}[idx*${nalph}+1+${iter}]*(
                              U${i}[idx*${nalph}+1]*U${i}[idx*${nalph}+1]
                            + U${i}[idx*${nalph}+2]*U${i}[idx*${nalph}+2]
                            + U${i}[idx*${nalph}+3]*U${i}[idx*${nalph}+3]
                          )/pow(U${i}[idx*${nalph}+0], 2.)
                        + 2.5*U${i}[idx*${nalph}+1+${iter}]*T${i}
                      )
                      %if i!=ar:
                      - ${Pr}*U${i}[idx*${nalph}+${iter}+11]
                      %endif
                    )
                  %endfor  
                )
              )/(
                    1. + ${Pr}*a${ar}${ar}*dt*prefac*U${ar}[idx*${nalph}+0]*pow(T${ar}, ${1-omega})
                );                
        %endfor
    }
}
%endfor


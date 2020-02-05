<%include file="linear-common-direct-gll"/>
<% import math %>

// Construct gaussian
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
    scalar detT;

    if(idt<nvars*neles*nupts) 
    {
        int upt = int(idt/(nvars*neles));
        int elem = int((idt%(nvars*neles))/nvars);
        int idx = int((idt%(nvars*neles))%nvars);

        <%
        ke = "upt*neles + elem";
        %>

        detT = (-U[(${ke})*${nalph}+7]*U[(${ke})*${nalph}+8]*U[(${ke})*${nalph}+8]
                    + 2*U[(${ke})*${nalph}+8]*U[(${ke})*${nalph}+9]*U[(${ke})*${nalph}+10] 
                    - U[(${ke})*${nalph}+5]*U[(${ke})*${nalph}+9]*U[(${ke})*${nalph}+9] 
                    - U[(${ke})*${nalph}+6]*U[(${ke})*${nalph}+10]*U[(${ke})*${nalph}+10] 
                    + U[(${ke})*${nalph}+5]*U[(${ke})*${nalph}+6]*U[(${ke})*${nalph}+7]);

        M[upt*ldim + SOA_IX(elem, idx, nvars)] = U[(${ke})*${nalph}+0]/(${(math.pi)**1.5}*sqrt(detT))*exp(
                -(
                    + (U[(${ke})*${nalph}+6]*U[(${ke})*${nalph}+7] - U[(${ke})*${nalph}+9]*U[(${ke})*${nalph}+9])*((cx[idx]-U[(${ke})*${nalph}+1])*(cx[idx]-U[(${ke})*${nalph}+1]))
                    + (U[(${ke})*${nalph}+5]*U[(${ke})*${nalph}+7] - U[(${ke})*${nalph}+10]*U[(${ke})*${nalph}+10])*((cy[idx]-U[(${ke})*${nalph}+2])*(cy[idx]-U[(${ke})*${nalph}+2]))
                    + (U[(${ke})*${nalph}+5]*U[(${ke})*${nalph}+6] - U[(${ke})*${nalph}+8]*U[(${ke})*${nalph}+8])*((cz[idx]-U[(${ke})*${nalph}+3])*(cz[idx]-U[(${ke})*${nalph}+3]))
                    + 2*(U[(${ke})*${nalph}+9]*U[(${ke})*${nalph}+10] - U[(${ke})*${nalph}+7]*U[(${ke})*${nalph}+8])*((cx[idx]-U[(${ke})*${nalph}+1])*(cy[idx]-U[(${ke})*${nalph}+2]))
                    + 2*(U[(${ke})*${nalph}+8]*U[(${ke})*${nalph}+10] - U[(${ke})*${nalph}+5]*U[(${ke})*${nalph}+9])*((cy[idx]-U[(${ke})*${nalph}+2])*(cz[idx]-U[(${ke})*${nalph}+3]))
                    + 2*(U[(${ke})*${nalph}+8]*U[(${ke})*${nalph}+9] - U[(${ke})*${nalph}+10]*U[(${ke})*${nalph}+6])*((cz[idx]-U[(${ke})*${nalph}+3])*(cx[idx]-U[(${ke})*${nalph}+1]))
                )/(detT)
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
    scalar T;

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

        // T = 1/1.5/rho*(mom5 + mom6 + mom7) - (u*u+v*v+w*w)/1.5
        T = (
                U[idx*${nalph}+4]
            )/(1.5*U[idx*${nalph}+0])
          - 
            (
                U[idx*${nalph}+1]*U[idx*${nalph}+1]
                + U[idx*${nalph}+2]*U[idx*${nalph}+2]
                + U[idx*${nalph}+3]*U[idx*${nalph}+3]
            )/1.5;
        
        // Non-dimensionalization factors
        <% nF = 2.0 %> 
        <% nF2 = 1.0 %>

        // Temperature
        U[idx*${nalph}+4] = T;

        // T_xx
        U[idx*${nalph}+5] = ${nF2}*${1./Pr}*T
                + ${nF}*${1.-1./Pr}*U[idx*${nalph}+5]/U[idx*${nalph}+0]
                - ${nF}*${1.-1./Pr}*U[idx*${nalph}+1]*U[idx*${nalph}+1];

        // T_yy
        U[idx*${nalph}+6] = ${nF2}*${1./Pr}*T
                + ${nF}*${1.-1./Pr}*U[idx*${nalph}+6]/U[idx*${nalph}+0]
                - ${nF}*${1.-1./Pr}*U[idx*${nalph}+2]*U[idx*${nalph}+2];

        // T_zz
        U[idx*${nalph}+7] = ${nF2}*${1./Pr}*T
                + ${nF}*${1.-1./Pr}*U[idx*${nalph}+7]/U[idx*${nalph}+0]
                - ${nF}*${1.-1./Pr}*U[idx*${nalph}+3]*U[idx*${nalph}+3];

        // T_xy
        U[idx*${nalph}+8] = ${nF}*${1.-1./Pr}*U[idx*${nalph}+8]/U[idx*${nalph}+0]
                        - ${nF}*${1.-1./Pr}*U[idx*${nalph}+1]*U[idx*${nalph}+2];

        // T_yz
        U[idx*${nalph}+9] = ${nF}*${1.-1./Pr}*U[idx*${nalph}+9]/U[idx*${nalph}+0]
                        - ${nF}*${1.-1./Pr}*U[idx*${nalph}+2]*U[idx*${nalph}+3];

        // T_xz
        U[idx*${nalph}+10] = ${nF}*${1.-1./Pr}*U[idx*${nalph}+10]/U[idx*${nalph}+0]
                        - ${nF}*${1.-1./Pr}*U[idx*${nalph}+3]*U[idx*${nalph}+1];
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
              + b/${Pr}*dt*prefac*U[idx*${nalph}+0]*pow(T,${1-omega})
                *(
                    0.5*U[idx*${nalph}+0]*T
                  + U[idx*${nalph}+${iter-4}]*U[idx*${nalph}+${iter-4}]
                      /U[idx*${nalph}+0]
                )
            )/(a${ar}+b/${Pr}*dt*prefac*U[idx*${nalph}+0]*pow(T,${1.-omega}));
        %endfor

        // off-diagonal components of the stress
        %for iter in [8, 9, 10]:
            U[idx*${nalph}+${iter}] = (-(${"+ ".join([
                        "a{0}*U{0}[idx*{1}+{2}] + g{0}*dt*LU{0}[idx*{1}+{2}]".format(
                            i, nalph, iter) for i in range(ar)
                    ])})
            + b/${Pr}*dt*prefac*U[idx*${nalph}+0]*pow(T,${1-omega})
                *(
                    U[idx*${nalph}+${iter-7}]*U[idx*${nalph}+${(iter-7)%3+1}]
                      /U[idx*${nalph}+0]
                  )
            )/(a${ar}+b/${Pr}*dt*prefac*U[idx*${nalph}+0]*pow(T,${1.-omega}));
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
              + a${ar}${i}/${Pr}*dt*prefac*U${i}[idx*${nalph}+0]*pow(T${i},${1-omega})
                *(
                    0.5*U${i}[idx*${nalph}+0]*T${i}
                  + U${i}[idx*${nalph}+${iter-4}]*U${i}[idx*${nalph}+${iter-4}]
                      /U${i}[idx*${nalph}+0]
                  %if i!=ar:
                    - U${i}[idx*${nalph}+${iter}]
                  %endif
                )
              %endfor
            )/(1+a${ar}${ar}/${Pr}*dt*prefac*U${ar}[idx*${nalph}+0]*pow(T${ar},${1.-omega}));
        %endfor

        // off-diagonal components of the stress
        %for iter in [8, 9, 10]:
            U${ar}[idx*${nalph}+${iter}] = (U0[idx*${nalph}+${iter}] +
              + ${"+ ".join([
                        "_a{0}{1}*dt*LU{1}[idx*{2}+{3}]".format(
                            ar, i, nalph, iter) for i in range(ar)
                    ])} 
              %for i in range(1, ar+1):
              + a${ar}${i}/${Pr}*dt*prefac*U${i}[idx*${nalph}+0]*pow(T${i},${1-omega})
                *(
                    U${i}[idx*${nalph}+${iter-7}]*U${i}[idx*${nalph}+${(iter-7)%3+1}]
                      /U${i}[idx*${nalph}+0]
                  %if i!=ar:
                    - U${i}[idx*${nalph}+${iter}]
                  %endif
                )
              %endfor
            )/(1+a${ar}${ar}/${Pr}*dt*prefac*U${ar}[idx*${nalph}+0]*pow(T${ar},${1.-omega}));
        %endfor
    }
}
%endfor
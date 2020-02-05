<%include file="linear-common-direct-gll"/>
<% import math %>

// Construct maxwellian
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
    scalar T;

    if(idt<nvars*neles*nupts) 
    {
        int upt = int(idt/(nvars*neles));
        int elem = int((idt%(nvars*neles))/nvars);
        int idx = int((idt%(nvars*neles))%nvars);

        //out[upt*nvars*neles + elem*nvars + idx] = in[upt*ldim + SOA_IX(elem, idx, nvars)];

        <%
        ke = "upt*neles + elem";
        %>

        T = (
                U[(${ke})*${nalph}+4]
                + (
                    - U[(${ke})*${nalph}+1]*U[(${ke})*${nalph}+1]
                    - U[(${ke})*${nalph}+2]*U[(${ke})*${nalph}+2]
                    - U[(${ke})*${nalph}+3]*U[(${ke})*${nalph}+3]
                )/U[(${ke})*${nalph}+0]
            )/(1.5*U[(${ke})*${nalph}+0]);

        M[upt*ldim + SOA_IX(elem, idx, nvars)] = U[(${ke})*${nalph}+0]/pow(${math.pi}*T, 1.5)
                *exp(
                    -(
                        (cx[idx]-U[(${ke})*${nalph}+1]/U[(${ke})*${nalph}+0])
                        *(cx[idx]-U[(${ke})*${nalph}+1]/U[(${ke})*${nalph}+0])
                        + (cy[idx]-U[(${ke})*${nalph}+2]/U[(${ke})*${nalph}+0])
                        *(cy[idx]-U[(${ke})*${nalph}+2]/U[(${ke})*${nalph}+0])
                        + (cz[idx]-U[(${ke})*${nalph}+3]/U[(${ke})*${nalph}+0])
                            *(cz[idx]-U[(${ke})*${nalph}+3]/U[(${ke})*${nalph}+0])
                    )/T
                ); 
    }
}


%for ar in nbdf:
// compute the moment of the bgk kernel
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

    if(idx<lda) 
    {   
        // density, x-velocity, y-velocity, z-velocity, energy
        %for iter in [0, 1, 2, 3, 4]:
            U[idx*${nalph}+${iter}] = (-(${"+ ".join([
                        "a{0}*U{0}[idx*{1}+{2}] + g{0}*dt*LU{0}[idx*{1}+{2}]".format(
                            i, nalph, iter) for i in range(ar)
                    ])}))/a${ar};    
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
    }
}
%endfor

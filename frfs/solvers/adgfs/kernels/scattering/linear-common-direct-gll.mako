%if dtype == 'double':
    #define scalar double
    #define Exp exp
%elif dtype == 'float':
    #define scalar float
    #define Exp expf
%else:
    #error "undefined floating point data type"
%endif

<%!
import math 
%>

#define SOA_SZ ${soasz}
#define SOA_IX(a, v, nv) ((((a) / SOA_SZ)*(nv) + (v))*SOA_SZ + (a) % SOA_SZ)

__global__ void swap_axes
(
    const int nupts,
    const int ldim, 
    const int nvars, 
    const int neles,
    const scalar* in,
    scalar* out
)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<nvars*neles*nupts) 
    {
        int upt = int(idx/(nvars*neles));
        int elem = int((idx%(nvars*neles))/nvars);
        int idv = int((idx%(nvars*neles))%nvars);

        out[upt*nvars*neles + elem*nvars + idv] = in[upt*ldim + SOA_IX(elem, idv, nvars)];
    }
}


%for ar in nbdf:
__global__ void ${"updateDistribution{0}_BDF".format(ar)}
(
    const scalar prefac,
    const int nupts,
    const int ldim, 
    const int nvars, 
    const int neles,
    const scalar dt,
    ${" ".join([
      "const scalar a{0}, const scalar* f{0}, const scalar g{0}, const scalar* L{0},".format(i) 
      for i in range(ar)
    ])}
    const scalar b, const scalar* M,
    const scalar a${ar}, const scalar* U, scalar* f    
)
{
    int idt = blockIdx.x*blockDim.x + threadIdx.x;
    scalar nu;

    if(idt<nvars*neles*nupts) 
    {   
        int upt = int(idt/(nvars*neles));
        int elem = int((idt%(nvars*neles))/nvars);
        int idx = int((idt%(nvars*neles))%nvars);
        int id =  upt*ldim + SOA_IX(elem, idx, nvars);

        <%
        ke = "upt*neles + elem";
        %>

        nu = (  
              U[(${ke})*${nalph}+4]
              - (
                  U[(${ke})*${nalph}+1]*U[(${ke})*${nalph}+1]
                  + U[(${ke})*${nalph}+2]*U[(${ke})*${nalph}+2]
                  + U[(${ke})*${nalph}+3]*U[(${ke})*${nalph}+3]
                )/U[(${ke})*${nalph}+0]
            )/(1.5*U[(${ke})*${nalph}+0]);  // Temperature
            nu = prefac*U[(${ke})*${nalph}+0]*pow(nu, ${1.-omega});
          
        f[id] = (-(${"+ ".join(["a{0}*f{0}[id] + g{0}*dt*L{0}[id]".format(i) 
                                for i in range(ar)
                  ])}) + b*dt*nu*M[id])/(a${ar}+b*dt*nu);
    }
}
%endfor


%for ar in nars:
__global__ void ${"updateDistribution{0}_ARS".format(ar)}
(
    const scalar prefac,
    const int nupts,
    const int ldim, 
    const int nvars, 
    const int neles,
    const scalar dt,
    ${" ".join(["const scalar _a{0}{1},".format(ar,i) for i in range(ar)])}
    ${" ".join(["const scalar a{0}{1},".format(ar,i+1) for i in range(ar)])}
    ${" ".join(["const scalar* L{0},".format(i) for i in range(ar)])}
    ${" ".join(["const scalar* U{0},".format(i) for i in range(ar+1)])}
    ${" ".join(["const scalar* M{0},".format(i) for i in range(1,ar+1)])}
    ${" ".join(["const scalar* f{0},".format(i) for i in range(ar)])}
    scalar *f${ar}
)
{
    int idt = blockIdx.x*blockDim.x + threadIdx.x;
    scalar ${", ".join(["nu{0}".format(i) for i in range(1, ar+1)])};

    if(idt<nvars*neles*nupts) 
    {   
        int upt = int(idt/(nvars*neles));
        int elem = int((idt%(nvars*neles))/nvars);
        int idx = int((idt%(nvars*neles))%nvars);
        int id =  upt*ldim + SOA_IX(elem, idx, nvars);

        <%
        ke = "upt*neles + elem";
        %>

        %for i in range(1, ar+1):
            nu${i} = (  
              U${i}[(${ke})*${nalph}+4]
              - (
                  U${i}[(${ke})*${nalph}+1]*U${i}[(${ke})*${nalph}+1]
                  + U${i}[(${ke})*${nalph}+2]*U${i}[(${ke})*${nalph}+2]
                  + U${i}[(${ke})*${nalph}+3]*U${i}[(${ke})*${nalph}+3]
                )/U${i}[(${ke})*${nalph}+0]
            )/(1.5*U${i}[(${ke})*${nalph}+0]);  // Temperature
            nu${i} = prefac*U${i}[(${ke})*${nalph}+0]*pow(nu${i}, ${1.-omega});
        %endfor
          
        f${ar}[id] = (f0[id] 
            + ${"+ ".join("_a{0}{1}*dt*L{1}[id]".format(ar, i) for i in range(ar))}
        %if ar>1:
            + ${"+ ".join("a{0}{1}*dt*nu{1}*(M{1}[id]-f{1}[id])".format(
                            ar, i) for i in range(1, ar))}
        %endif
            + a${ar}${ar}*dt*nu${ar}*(M${ar}[id])                    
        )
        /(1. + a${ar}${ar}*dt*nu${ar});
    }
}
%endfor

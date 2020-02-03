# -*- coding: utf-8 -*-
<%namespace module='frfs.backends.base.makoutil' name='frfs'/>

// AoSoA macros
#define SOA_SZ ${soasz}
#define SOA_IX(a, v, nv) ((((a) / SOA_SZ)*(nv) + (v))*SOA_SZ + (a) % SOA_SZ)

// Typedefs
typedef ${frfs.npdtype_to_ctype(fpdtype)} fpdtype_t;

${next.body()}

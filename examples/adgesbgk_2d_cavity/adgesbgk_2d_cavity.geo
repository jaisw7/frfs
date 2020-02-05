nVertexX = 9;  // Should be odd
nVertexY = 9;

H = 1;
L = H;

Point(1) = {0, 0, 0, 1e+22};
Point(2) = {L, 0, 0, 1e+22};
Point(3) = {L, H, 0, 1e+22};
Point(4) = {0, H, 0, 1e+22};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};
a=1;
Transfinite Line {4} = nVertexX Using Progression a;
Transfinite Line {2} = nVertexX Using Progression (1.0/a);
Transfinite Line {3} = nVertexY Using Progression a;
Transfinite Line {1} = nVertexY Using Progression (1.0/a);
Transfinite Surface{6};
Recombine Surface {6};


Physical Surface("Fluid") = {6};
Physical Line("bottom") = {1};    //- Bottom
Physical Line("right") = {2};    //- Right
Physical Line("top") = {3};    //- Top
Physical Line("left") = {4};    //- Left

//Physical Line("periodic_0_l") = {1};    //- Bottom
//Physical Line("periodic_1_r") = {2};    //- Right
//Physical Line("periodic_0_r") = {3};    //- Top
//Physical Line("periodic_1_l") = {4};    //- Left

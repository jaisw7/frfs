_H = 1e-6;  // dimensional half channel height

// non-dimensional quantities
H = _H/_H;
hH = 0.5*H; 
Li = H*2;
Lo = H*2;
Hi = H*2;
Ho = H*2;
L =  H*5;

lc = 1e22;

Point(1) = {0, 0, 0, lc};
Point(2) = {Li, 0, 0, lc};
Point(3) = {Li+L, 0, 0, lc};
Point(4) = {Li+L+Lo, 0, 0, lc};
Point(5) = {Li+L+Lo, hH, 0, lc};
Point(6) = {Li+L+Lo, Ho, 0, lc};
Point(7) = {Li+L, Ho, 0, lc};
Point(8) = {Li+L, hH, 0, lc};
Point(9) = {Li, hH, 0, lc};
Point(10) = {Li, Hi, 0, lc};
Point(11) = {0, Hi, 0, lc};
Point(12) = {0, hH, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 1};
Line(13) = {9, 12};
Line(14) = {2, 9};
Line(15) = {3, 8};
Line(16) = {5, 8};

Line Loop(1) = {1, 14, 13, 12};
Line Loop(2) = {2, 15, 8, -14};
Line Loop(3) = {3, 4, 16, -15};
Line Loop(4) = {5, 6, 7, -16};
Line Loop(5) = {9, 10, 11, -13};

Plane Surface(6) = {1};
Plane Surface(7) = {2};
Plane Surface(8) = {3};
Plane Surface(9) = {4};
Plane Surface(10) = {5};

a=0.8;
a1=1.5;
a2=1; 
a3=1;
nCellsTubeY = 4;
nCellsTubeX = 20;
nCellsInOutX = 6;
nCellsInOutY = 5;
Transfinite Line {1,-10,-13} = nCellsInOutX Using Progression a2;
Transfinite Line {3,-16,-6} = nCellsInOutX Using Progression 1/a2;
Transfinite Line {2,-8} = nCellsTubeX Using Bump a;
Transfinite Line {-12,14} = nCellsTubeY Using Bump a1;
Transfinite Line {15,4} = nCellsTubeY Using Bump a1;
Transfinite Line {5,-7,9,-11} = nCellsInOutY Using Progression a3;
Transfinite Surface{6,7,8,9,10};
Recombine Surface {6,7,8,9,10};

Physical Surface("Fluid") = {6,7,8,9,10};
Physical Line("symmetry") = {1,2,3};    //- symmetry
Physical Line("left") = {-11,-12,-10};    //- left
Physical Line("right") = {4,5,6};    //- right
//Physical Line("wall-left") = {9,8,-7};    //- wall
Physical Line("wall-left") = {9};    //- wall-left
Physical Line("wall-middle") = {8};    //- wall-middle
Physical Line("wall-right") = {-7};    //- wall-right



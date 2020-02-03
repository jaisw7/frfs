H0 = 1e-6;  // non-dimensional length [m]
H = 1e-6/H0;  // height of the domain non-dimensionalized by H0
L = 0.2*H;  //  length of the domain

Point(1) = {0, 0, 0, 1e+22};  // bottom-left end of the box
Point(2) = {L, 0, 0, 1e+22};  // bottom-right end of the box
Point(3) = {L, H, 0, 1e+22};  // top-right end of the box
Point(4) = {0, H, 0, 1e+22};  // top-left end of the box
Line(1) = {1, 2};  // bottom edge
Line(2) = {2, 3};  // right edge
Line(3) = {3, 4};  // top edge
Line(4) = {4, 1};  // left edge

nVertexX = 2;  // nVertexX-1 = Number of cells in the x-direction
nVertexY = 3;  // nVertexY-1 = Number of cells in the y-direction

Line Loop(5) = {1, 2, 3, 4};  // create a line loop with four lines
Plane Surface(6) = {5};  // define a surface using the line loop

r=1;  // ratio of the geometric progression

// Place nVertexY points on line 4 using geoemtric progression with ratio: r
Transfinite Line {4} = nVertexY Using Progression r;   

// Place nVertexY points on line 2 using geoemtric progression with ratio: 1/r
Transfinite Line {2} = nVertexY Using Progression (1.0/r);

// Place nVertexX points on line 3 using geoemtric progression with ratio: r
Transfinite Line {3} = nVertexX Using Progression r;

// Place nVertexX points on line 1 using geoemtric progression with ratio: 1/r
Transfinite Line {1} = nVertexX Using Progression (1.0/r);

// The next two lines ensure that the mesh contains only quadrilateral elements
Transfinite Surface{6};
Recombine Surface {6};

// Define the domain of the simulatio
Physical Surface("Fluid") = {6};

// The bottom boundary
Physical Line("bottom") = {1};    //- Bottom

// The top boundary
Physical Line("top") = {3};    //- Top

// First periodic boundary: right edge of the period 
Physical Line("periodic_0_r") = {2};    //- Right

// First periodic boundary: left edge of the period
Physical Line("periodic_0_l") = {4};    //- Left

// The periodic boundaries have special nomenclature: 
//   periodic_{n}_{t}
//     Here {n} is a unique id for the periodic boundary
//          {t} is either "l" or "r"

nVertex = 5;

Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};
Point(5) = {0, 0, 1};
Point(6) = {1, 0, 1};
Point(7) = {1, 1, 1};
Point(8) = {0, 1, 1};
Line(1) = {8, 7};
Line(2) = {7, 6};
Line(3) = {6, 5};
Line(4) = {5, 8};
Line(5) = {3, 2};
Line(6) = {2, 1};
Line(7) = {1, 4};
Line(8) = {4, 3};
Line(9) = {3, 7};
Line(10) = {2, 6};
Line(11) = {8, 4};
Line(12) = {5, 1};
Line Loop(13) = {9, 2, -10, -5};
Plane Surface(14) = {13};
Line Loop(15) = {1, -9, -8, -11};
Plane Surface(16) = {15};
Line Loop(17) = {8, 5, 6, 7};
Plane Surface(18) = {17};
Line Loop(19) = {3, 12, -6, 10};
Plane Surface(20) = {19};
Line Loop(21) = {12, 7, -11, -4};
Plane Surface(22) = {21};
Line Loop(23) = {2, 3, 4, 1};
Plane Surface(24) = {-23};
Surface Loop(25) = {24, 14, 16, 18, 20, 22};
Volume(26) = {25};

Transfinite Line {1,2,3,4,5,6,7,8,9,10,11,12} = nVertex;
Transfinite Surface {14,16,18,20,22,24};
Transfinite Volume {26};

Recombine Surface {14,16,18,20,22,24};

Physical Surface("left") = {22};   // left
Physical Surface("bottom") = {20}; // bottom
Physical Surface("right") = {14};  // right
Physical Surface("top") = {16};    // top
Physical Surface("back") = {18};   // back
Physical Surface("front") = {24};  // front

Physical Volume("fluid") = {26};          

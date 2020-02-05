nx1 = 15;
nx2 = 5;
nx3 = 4;
nx4 = 5;
nx5 = 15;
ax1 = 1/1.1;
ax2 = 1.0;
ax3 = 1.0;
ax4 = 1.0;
ax5 = 1.1;

ny1 = 2;
ny2 = 7;
ny3 = 17;
ay1 = 1.0;
ay2 = 1.0;
ay3 = 1.1;

H0 = 20e-6;

// non-dimensionalized
L = 600e-6 / H0;  // Length of the domain
H = 300e-6 / H0;  // Height of the domain
hG = 10e-6 / H0;  // half of the gap between the two heaters
hW = 4e-6 / H0;   // location of heater from the wall

Lh = 50e-6 / H0;  // length of the heater
Hh = 50e-6 / H0;  // width of the heater

lc = 1; //100;
lcw = 1; //5;
lcb = 1; //12;

Point(1) = {0, 0, 0, lcw};
Point(2) = {L/2-hG-Lh, 0, 0, lcw};
Point(3) = {L/2-hG, 0, 0, lcw};
Point(4) = {L/2+hG, 0, 0, lcw};
Point(5) = {L/2+hG+Lh, 0, 0, lcw};
Point(6) = {L, 0, 0, lcw};
Point(7) = {L, hW, 0, lcb};
Point(8) = {L, hW+Hh, 0, lcb};
Point(9) = {L, H, 0, lc};
Point(10) = {L/2+hG+Lh, H, 0, lcb};
Point(11) = {L/2+hG, H, 0, lcb};
Point(12) = {L/2-hG, H, 0, lcb};
Point(13) = {L/2-hG-Lh, H, 0, lcb};
Point(14) = {0, H, 0, lc};
Point(15) = {0, hW+Hh, 0, lcb};
Point(16) = {0, hW, 0, lcb};
Point(17) = {L/2-hG-Lh, hW, 0, lcw};
Point(18) = {L/2-hG, hW, 0, lcw};
Point(19) = {L/2-hG, hW+Hh, 0, lcb};
Point(20) = {L/2-hG-Lh, hW+Hh, 0, lcb};
Point(21) = {L/2+hG, hW, 0, lcw};
Point(22) = {L/2+hG+Lh, hW, 0, lcw};
Point(23) = {L/2+hG+Lh, hW+Hh, 0, lcb};
Point(24) = {L/2+hG, hW+Hh, 0, lcb};

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
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 15};
Line(15) = {15, 16};
Line(16) = {16, 1};

Line(17) = {2, 17};
Line(18) = {3, 18};
Line(19) = {4, 21};
Line(20) = {5, 22};

Line(21) = {16, 17};
Line(22) = {17, 18};
Line(23) = {18, 21};
Line(24) = {21, 22};
Line(25) = {22, 7};

Line(26) = {17, 20};
Line(27) = {18, 19};
Line(28) = {21, 24};
Line(29) = {22, 23};

Line(30) = {15, 20};
Line(31) = {20, 19};
Line(32) = {19, 24};
Line(33) = {24, 23};
Line(34) = {23, 8};

Line(35) = {20, 13};
Line(36) = {19, 12};
Line(37) = {24, 11};
Line(38) = {23, 10};


Line Loop(1) = {1, 17, -21, 16};
Line Loop(2) = {2, 18, -22, -17};
Line Loop(3) = {3, 19, -23, -18};
Line Loop(4) = {4, 20, -24, -19};
Line Loop(5) = {5, 6, -25, -20};

Line Loop(6) = {21, 26, -30, 15};
Line Loop(7) = {22, 27, -31, -26};
Line Loop(8) = {23, 28, -32, -27};
Line Loop(9) = {24, 29, -33, -28};
Line Loop(10) = {25, 7, -34, -29};

Line Loop(11) = {30, 35, 13, 14};
Line Loop(12) = {31, 36, 12, -35};
Line Loop(13) = {32, 37, 11, -36};
Line Loop(14) = {33, 38, 10, -37};
Line Loop(15) = {34, 8, 9, -38};


Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};
//Plane Surface(7) = {7};
Plane Surface(8) = {8};
//Plane Surface(9) = {9};
Plane Surface(10) = {10};
Plane Surface(11) = {11};
Plane Surface(12) = {12};
Plane Surface(13) = {13};
Plane Surface(14) = {14};
Plane Surface(15) = {15};

//Plane Surface(16) = {1, 2, 3, 4, 5, 6, -7, 8, -9, 10, 11, 12, 13, 14, 15};


Transfinite Line {1,21,30,-13} = nx1 Using Progression ax1;
Transfinite Line {2,22,31,12} = nx2 Using Progression ax2;
Transfinite Line {3,23,32,11} = nx3 Using Progression ax3;
Transfinite Line {4,24,33,10} = nx4 Using Progression ax4;
Transfinite Line {5,25,34,-9} = nx5 Using Progression ax5;

Transfinite Line {-16,17,18,19,20,6} = ny1 Using Progression ay1;
Transfinite Line {-15,26,27,28,29,7} = ny2 Using Progression ay2;
Transfinite Line {-14,35,36,37,38,8} = ny3 Using Progression ay3;

Transfinite Surface {1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15};
Recombine Surface {1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15};


Physical Surface("Fluid") = {1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15};
Physical Line("substrate") = {1,2,3,4,5};   
Physical Line("symmRight") = {6,7,8};
Physical Line("freestream") = {9,10,11,12,13};     
Physical Line("symmLeft") = {14,15,16};    
Physical Line("leftHeater") = {22,27,31,26};
Physical Line("rightHeater") = {24,29,33,28};


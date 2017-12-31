/* Rectangle Geometry File */

gridsize = .50; // prescribed mesh element size
    
Lx = 20.0; // width
Ly = 10.0;   // height
   
Point(1) = {0, -Ly/2, 0, gridsize};
Point(2) = {Lx, -Ly/2, 0, gridsize};
Point(3) = {Lx, Ly/2, 0, gridsize};
Point(4) = {0, Ly/2, 0, gridsize};
 
Line(1) = {1, 2}; 
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {4, 1, 2, 3};
Plane Surface(1) = {1};
    
Periodic Line {2} = {-4};
Periodic Line {1} = {-3};

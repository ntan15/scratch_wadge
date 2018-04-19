/* Rectangle Geometry File */

h = .25; // prescribed mesh element size
Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = 2*h;
    
Lx = 20.0; // width
Ly = 10.0;   // height
   
Point(1) = {0, -Ly/2, 0, h};
Point(2) = {Lx, -Ly/2, 0, h};
Point(3) = {Lx, Ly/2, 0, h};
Point(4) = {0, Ly/2, 0, h};
 
Line(1) = {1, 2}; 
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {4, 1, 2, 3};
Plane Surface(1) = {1};
    

//Periodic Line {2} = {-4};
//Periodic Line {1} = {-3};

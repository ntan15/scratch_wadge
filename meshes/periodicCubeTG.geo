// This is from the demos distributed with gmsh
SetFactory("OpenCASCADE");

R = 3.14159265359;

h = R/2; //10;
Mesh.Algorithm = 6;
Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = h;

Box(1) = {-R,-R,-R, 2*R,2*R,2*R}; // x0,y0,z0, dx, dy, dz

Periodic Surface{2} = {1} Translate{2*R,0,0};
Periodic Surface{4} = {3} Translate{0,2*R,0};
Periodic Surface{6} = {5} Translate{0,0,2*R};
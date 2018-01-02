// This is from the demos distributed with gmsh
SetFactory("OpenCASCADE");

h = 10;
Mesh.Algorithm = 6;
Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = h;

R = 10;
Box(1) = {0,0,0, R,R,R};

Periodic Surface{2} = {1} Translate{R,0,0};
Periodic Surface{4} = {3} Translate{0,R,0};
Periodic Surface{6} = {5} Translate{0,0,R};
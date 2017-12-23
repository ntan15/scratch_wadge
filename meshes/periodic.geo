// This is from the demos distributed with gmsh
SetFactory("OpenCASCADE");

Mesh.Algorithm = 6;
Mesh.CharacteristicLengthMin = 2;
Mesh.CharacteristicLengthMax = 2;

R = 2;
Box(1) = {0,0,0, R,R,R};

Periodic Surface{2} = {1} Translate{R,0,0};

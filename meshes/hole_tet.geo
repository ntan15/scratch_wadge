SetFactory("OpenCASCADE");

Mesh.Algorithm = 6;
Mesh.CharacteristicLengthMin = 0.25;
Mesh.CharacteristicLengthMax = 0.25;


Block(1) = {0,0,0, 1,1,1};
Cylinder(2) = {0, 0.5, 0.5, 1, 0, 0, 0.25, 2*Pi};
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

SetFactory("OpenCASCADE");

Mesh.Algorithm = 6;
Mesh.CharacteristicLengthMin = 2;
Mesh.CharacteristicLengthMax = 2;

Block(1) = {-1, -1, 0, 2, 2, 2};

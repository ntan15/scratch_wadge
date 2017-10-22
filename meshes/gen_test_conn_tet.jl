perms = [1 2 3 4
         3 1 2 4
         2 3 1 4
         1 4 2 3
         2 1 4 3
         4 2 1 3
         3 4 1 2
         1 3 4 2
         4 1 3 2
         4 3 2 1
         2 4 3 1
         3 2 4 1];
nodes = [-1 -1 -1
          1 -1 -1
         -1  1 -1
         -1 -1  1
          1  1  1];
elems = [1 2 3 4
         3 4 2 5];

Nperms = size(perms)[1]

println("\$MeshFormat")
println("2.2 0 8")
println("\$EndMeshFormat")
println("\$Nodes")
println(Nperms^2 * size(nodes)[1])
for p1 = 1:Nperms
  for p2 = 1:Nperms
    p = (p1-1) * Nperms + (p2-1)
    for r = 1:size(nodes)[1]
      @printf("%3d", r + p * size(nodes)[1])
      for c = 1:size(nodes)[2]
        @printf(" %3d", nodes[r,c] + 3*p)
      end
      @printf("\n")
    end
  end
end
println("\$EndNodes")
println("\$Elements")
println(Nperms^2 * size(elems)[1])
for p1 = 1:Nperms
  for p2 = 1:Nperms
    p = (p1-1) * Nperms + (p2-1)
    for r = 1:size(elems)[1]
      pp = (r % 2 == 1)  ? p1 : p2
      @printf("%3d 4 2 0 1", r + p * size(elems)[1])
      for c = 1:size(elems)[2]
        @printf(" %3d", elems[r,perms[pp,c]] + p * size(nodes)[1])
      end
      @printf("\n")
    end
  end
end
println("\$EndElements")

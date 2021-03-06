/*********************************************************************
 *
 *  Gmsh tutorial 1
 *
 *  Variables, elementary entities (points, lines, surfaces), physical
 *  entities (points, lines, surfaces)
 *
 *********************************************************************/

// The simplest construction in Gmsh's scripting language is the
// `affectation'. The following command defines a new variable `lc':

lc = .25;

// This variable can then be used in the definition of Gmsh's simplest
// `elementary entity', a `Point'. A Point is defined by a list of four numbers:
// three coordinates (X, Y and Z), and a characteristic length (lc) that sets
// the target element size at the point:

// The distribution of the mesh element sizes is then obtained by interpolation
// of these characteristic lengths throughout the geometry. Another method to
// specify characteristic lengths is to use a background mesh (see `t7.geo' and
// `bgmesh.pos').

// We can then define some additional points as well as our first curve.  Curves
// are Gmsh's second type of elementery entities, and, amongst curves, straight
// lines are the simplest. A straight line is defined by a list of point
// numbers. In the commands below, for example, the line 1 starts at point 1 and
// ends at point 2:
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0,  0, lc} ;
Point(3) = {1, 1, 0, lc} ;
Point(4) = {0, 1, 0, lc} ;

Line(1) = {1,2} ;
Line(2) = {3,2} ;
Line(3) = {3,4} ;
Line(4) = {4,1} ;

// The third elementary entity is the surface. In order to define a simple
// rectangular surface from the four lines defined above, a line loop has first
// to be defined. A line loop is a list of connected lines, a sign being
// associated with each line (depending on the orientation of the line):

Line Loop(1) = {4,1,-2,3} ;

// We can then define the surface as a list of line loops (only one here, since
// there are no holes--see `t4.geo'):

Plane Surface(1) = {1} ;

// At this level, Gmsh knows everything to display the rectangular surface 6 and
// to mesh it. An optional step is needed if we want to associate specific
// region numbers to the various elements in the mesh (e.g. to the line segments
// discretizing lines 1 to 4 or to the triangles discretizing surface 1). This
// is achieved by the definition of `physical entities'. Physical entities will
// group elements belonging to several elementary entities by giving them a
// common number (a region number).

// We can for example group the points 1 and 2 into the physical entity 1:

//Physical Point(1) = {1,2} ;


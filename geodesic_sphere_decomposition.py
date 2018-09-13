import numpy as np 
import math

def icosahedron():
    """Construct a 20-sided polyhedron"""
    faces = [ 
        (0,1,2),
        (0,2,3),
        (0,3,4),
        (0,4,5),
        (0,5,1),
        (11,6,7),
        (11,7,8),
        (11,8,9),
        (11,9,10),
        (11,10,6),
        (1,2,6),
        (2,3,7),
        (3,4,8),
        (4,5,9),
        (5,1,10),
        (6,7,2),
        (7,8,3),
        (8,9,4),
        (9,10,5),
        (10,6,1) ]
    verts = [ 
        ( 0.000,  0.000,  1.000 ),
        ( 0.894,  0.000,  0.447 ),
        ( 0.276,  0.851,  0.447 ),
        (-0.724,  0.526,  0.447 ),
        (-0.724, -0.526,  0.447 ),
        ( 0.276, -0.851,  0.447 ),
        ( 0.724,  0.526, -0.447 ),
        (-0.276,  0.851, -0.447 ),
        (-0.894,  0.000, -0.447 ),
        (-0.276, -0.851, -0.447 ),
        ( 0.724, -0.526, -0.447 ),
        ( 0.000,  0.000, -1.000 ) ]
    return verts, faces
    
def subdivide(verts, faces):
    """Subdivide each triangle on the face of the polyhedron into four triangles
    in this way generate a finer geodesic sphere from the icosahedron.
    each time you do this the number of vertices increases by a factor of 4"""
    triangles = len(faces)
    for faceIndex in range(triangles):
    
        # Create three new verts at the midpoints of each edge:
        face = faces[faceIndex]
        a,b,c = (Vector3(*verts[vertIndex]) for vertIndex in face)
        verts.append((a + b).normalized()[:])
        verts.append((b + c).normalized()[:])
        verts.append((a + c).normalized()[:])

        # Split the current triangle into four smaller triangles:
        i = len(verts) - 3
        j, k = i+1, i+2
        faces.append((i, j, k))
        faces.append((face[0], i, k))
        faces.append((i, face[1], j))
        faces[faceIndex] = (k, j, face[2])

    return verts, faces
##############################################################################3 all that follows stolen from euclid 

class Vector3: #stole this from euclid 
    __slots__ = ['x', 'y', 'z']
    __hash__ = None

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __copy__(self):
        return self.__class__(self.x, self.y, self.z)

    copy = __copy__

    def __repr__(self):
        return 'Vector3(%.2f, %.2f, %.2f)' % (self.x,
                                              self.y,
                                              self.z)

    def __eq__(self, other):
        if isinstance(other, Vector3):
            return self.x == other.x and \
                   self.y == other.y and \
                   self.z == other.z
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return self.x == other[0] and \
                   self.y == other[1] and \
                   self.z == other[2]

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return self.x != 0 or self.y != 0 or self.z != 0

    def __len__(self):
        return 3

    def __getitem__(self, key):
        return (self.x, self.y, self.z)[key]

    def __setitem__(self, key, value):
        l = [self.x, self.y, self.z]
        l[key] = value
        self.x, self.y, self.z = l

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __getattr__(self, name):

        return tuple([(self.x, self.y, self.z)['xyz'.index(c)] \
                      for c in name])





    def __add__(self, other):
        if isinstance(other, Vector3):
            # Vector + Vector -> Vector
            # Vector + Point -> Point
            # Point + Point -> Vector
            if self.__class__ is other.__class__:
                _class = Vector3
            else:
                _class = Point3
            return _class(self.x + other.x,
                          self.y + other.y,
                          self.z + other.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return Vector3(self.x + other[0],
                           self.y + other[1],
                           self.z + other[2])
    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Vector3):
            self.x += other.x
            self.y += other.y
            self.z += other.z
        else:
            self.x += other[0]
            self.y += other[1]
            self.z += other[2]
        return self

    def __sub__(self, other):
        if isinstance(other, Vector3):
            # Vector - Vector -> Vector
            # Vector - Point -> Point
            # Point - Point -> Vector
            if self.__class__ is other.__class__:
                _class = Vector3
            else:
                _class = Point3
            return Vector3(self.x - other.x,
                           self.y - other.y,
                           self.z - other.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return Vector3(self.x - other[0],
                           self.y - other[1],
                           self.z - other[2])

   
    def __rsub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(other.x - self.x,
                           other.y - self.y,
                           other.z - self.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return Vector3(other.x - self[0],
                           other.y - self[1],
                           other.z - self[2])

    def __mul__(self, other):
        if isinstance(other, Vector3):
            # TODO component-wise mul/div in-place and on Vector2; docs.
            if self.__class__ is Point3 or other.__class__ is Point3:
                _class = Point3
            else:
                _class = Vector3
            return _class(self.x * other.x,
                          self.y * other.y,
                          self.z * other.z)
        else: 
            assert type(other) in (int, long, float)
            return Vector3(self.x * other,
                           self.y * other,
                           self.z * other)

    __rmul__ = __mul__

    def __imul__(self, other):
        assert type(other) in (int, long, float)
        self.x *= other
        self.y *= other
        self.z *= other
        return self

    def __div__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.div(self.x, other),
                       operator.div(self.y, other),
                       operator.div(self.z, other))


    def __rdiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.div(other, self.x),
                       operator.div(other, self.y),
                       operator.div(other, self.z))

    def __floordiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.floordiv(self.x, other),
                       operator.floordiv(self.y, other),
                       operator.floordiv(self.z, other))


    def __rfloordiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.floordiv(other, self.x),
                       operator.floordiv(other, self.y),
                       operator.floordiv(other, self.z))

    def __truediv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.truediv(self.x, other),
                       operator.truediv(self.y, other),
                       operator.truediv(self.z, other))


    def __rtruediv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.truediv(other, self.x),
                       operator.truediv(other, self.y),
                       operator.truediv(other, self.z))
    
    def __neg__(self):
        return Vector3(-self.x,
                        -self.y,
                        -self.z)

    __pos__ = __copy__
    
    def __abs__(self):
        return math.sqrt(self.x ** 2 + \
                         self.y ** 2 + \
                         self.z ** 2)

    magnitude = __abs__

    def magnitude_squared(self):
        return self.x ** 2 + \
               self.y ** 2 + \
               self.z ** 2

    def normalize(self):
        d = self.magnitude()
        if d:
            self.x /= d
            self.y /= d
            self.z /= d
        return self

    def normalized(self):
        d = self.magnitude()
        if d:
            return Vector3(self.x / d, 
                           self.y / d, 
                           self.z / d)
        return self.copy()

    def dot(self, other):
        assert isinstance(other, Vector3)
        return self.x * other.x + \
               self.y * other.y + \
               self.z * other.z

    def cross(self, other):
        assert isinstance(other, Vector3)
        return Vector3(self.y * other.z - self.z * other.y,
                       -self.x * other.z + self.z * other.x,
                       self.x * other.y - self.y * other.x)

    def reflect(self, normal):
        # assume normal is normalized
        assert isinstance(normal, Vector3)
        d = 2 * (self.x * normal.x + self.y * normal.y + self.z * normal.z)
        return Vector3(self.x - d * normal.x,
                       self.y - d * normal.y,
                       self.z - d * normal.z)

    def rotate_around(self, axis, theta):
        """Return the vector rotated around axis through angle theta. Right hand rule applies"""

        # Adapted from equations published by Glenn Murray.
        # http://inside.mines.edu/~gmurray/ArbitraryAxisRotation/ArbitraryAxisRotation.html
        x, y, z = self.x, self.y,self.z
        u, v, w = axis.x, axis.y, axis.z

        # Extracted common factors for simplicity and efficiency
        r2 = u**2 + v**2 + w**2
        r = math.sqrt(r2)
        ct = math.cos(theta)
        st = math.sin(theta) / r
        dt = (u*x + v*y + w*z) * (1 - ct) / r2
        return Vector3((u * dt + x * ct + (-w * y + v * z) * st),
                       (v * dt + y * ct + ( w * x - u * z) * st),
                       (w * dt + z * ct + (-v * x + u * y) * st))

    def angle(self, other):
        """Return the angle to the vector other"""
        return math.acos(self.dot(other) / (self.magnitude()*other.magnitude()))

    def project(self, other):
        """Return one vector projected on the vector other"""
        n = other.normalized()
        return self.dot(n)*n
  ########################################################################### end of euclid theft 

def asSpherical(xyz):
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arccos(z/r)*180/ pi #to degrees
    phi     =  np.atan2(y,x)*180/ pi
    return [r,theta,phi]    


def unit_geodesic(num_subdivisions=3):
    verts, faces = icosahedron()
    for x in range(num_subdivisions):
        verts, faces = subdivide(verts, faces)
    verts = np.array([[*v] for v in verts])
    faces = np.array([[*f] for f in faces])
    return verts,faces

# example usage 

# verts,faces = unit_geodesic(num_subdivisions=5)
# this will generate a geodesic sphere with 20472 vertices 
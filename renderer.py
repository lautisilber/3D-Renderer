import numpy as np
from PIL import Image, ImageDraw
from typing import Union


arraylike = Union[list, tuple, np.ndarray]


class Mesh:
    def __init__(self) -> None:
        self.vertices = None
        self.triangles = None
        self.color = None
        self.n_tris = None
        self.normals = None

        self.scale_m = np.identity(4, dtype=float)
        self.translation = np.identity(4, dtype=float)
        self.rotation = np.identity(4, dtype=float)

    def set_verts_tris(self, vertices: arraylike, triangles: arraylike) -> None:
        self.vertices = vertices
        self.triangles = triangles

    def set_color(self, color: arraylike) -> None:
        self.color = color

    def recalc_normals(self) -> None:
        self.n_tris = len(self.triangles) // 3
        self.normals = np.zeros((self.n_tris, 3), dtype=float)
        for i in range(self.n_tris):
            t1 = self.triangles[i*3]
            t2 = self.triangles[i*3+1]
            t3 = self.triangles[i*3+2]
            a = self.vertices[t1][:3]
            b = self.vertices[t2][:3]
            c = self.vertices[t3][:3]
            v1 = b-a
            v2 = c-a
            norm = np.cross(v1, v2) # np.array([v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]], dtype=float)
            self.normals[i] = norm/np.linalg.norm(norm)

    def translate(self, x, y, z):
        self.translation[0,3] += x
        self.translation[1,3] += y
        self.translation[2,3] += z

    def scale(self, x, y, z):
        self.scale_m[0, 0] = x
        self.scale_m[1, 1] = y
        self.scale_m[2, 2] = z


class Cube(Mesh):
    def __init__ (self, color: arraylike=(255, 255, 255),  scale: int=1) -> None:
        super().__init__()
        self.vertices = np.array([
            [-1, 1, 1, 1], [1, 1, 1, 1], [-1, -1, 1, 1], [1, -1, 1, 1],
            [-1, 1, -1, 1], [1, 1, -1, 1], [-1, -1, -1, 1], [1, -1, -1, 1]
        ], dtype=float) * scale
        self.triangles = np.array([
            0, 2, 1, 1, 2, 3,
            4, 5, 6, 5, 7, 6,
            0, 4, 6, 0, 6, 2,
            1, 7, 5, 1, 3, 7,
            0, 5, 4, 0, 1, 5,
            2, 6, 7, 2, 7, 3
        ])
        self.color = np.array(color)
        self.recalc_normals()


class OrthograficProjection:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)
        self.meshes = []
        self.screen = np.zeros((width, height, 3), dtype=np.uint8)
        self.translation = np.identity(4, dtype=float)
        self.rotation = np.identity(4, dtype=float)

        left, right = 0, width
        up, bottom = 0, height
        far, near = -10, 100
        scale_x = 2.0 / (right - left)
        scale_y = 2.0 / (up - bottom)
        scale_z = 2.0 / (far - near)
        mid_x = (left + right) / 2.0
        mid_y = (bottom + up) / 2.0
        mid_z = (-near + -far) / 2.0
        self.projection = np.array([
            [scale_x, 0, 0, -mid_x],
            [0, scale_y, 0, -mid_y],
            [0, 0, scale_z, -mid_z],
            [0, 0, 0, 1]
        ])

        '''theta = np.pi / 2
        z_near = 0.01
        z_far = 1000
        self.projection = np.array([
            [(height/width)*1/np.tan(theta/2), 0, 0, 0],
            [0, 1/np.tan(theta/2), 0, 0],
            [0, 0, z_far/(z_far-z_near), 1],
            [0, 0, -z_far*z_near/(z_far-z_near), 0]

        ])'''

    def add_mesh(self, mesh: Mesh) -> None:
        self.meshes.append(mesh)

    def render(self) -> np.ndarray:
        img = Image.new(mode='RGB', size=self.size, color='black')
        draw = ImageDraw.Draw(img)
        for mesh in self.meshes:
            for i in range(mesh.n_tris):
                v = mesh.vertices
                t = mesh.triangles
                n = mesh.normals

                t1 = t[i*3]
                t2 = t[i*3+1]
                t3 = t[i*3+2]

                a = v[t1]
                b = v[t2]
                c = v[t3]

                # local scale
                a = np.dot(mesh.scale_m, a)
                b = np.dot(mesh.scale_m, b)
                c = np.dot(mesh.scale_m, c)

                # local rotation
                a = np.dot(mesh.rotation, a)
                b = np.dot(mesh.rotation, b)
                c = np.dot(mesh.rotation, c)

                # local translation
                a = np.dot(mesh.translation, a)
                b = np.dot(mesh.translation, b)
                c = np.dot(mesh.translation, c)

                # global translation
                a = np.dot(self.translation, a)
                b = np.dot(self.translation, b)
                c = np.dot(self.translation, c)

                # global rotation
                a = np.dot(self.rotation, a)
                b = np.dot(self.rotation, b)
                c = np.dot(self.rotation, c)

                # global projection
                a = np.dot(self.projection, a)
                b = np.dot(self.projection, b)
                c = np.dot(self.projection, c)

                a = a[:2]
                b = b[:2]
                c = c[:2]

                draw.polygon([*a, *b, *c], fill='green')
        img.show()


cube = Cube()
cube.translate(5, 0, -1)
cube.scale(100, 100, 100)

op = OrthograficProjection(400, 400)
op.add_mesh(cube)
op.render()
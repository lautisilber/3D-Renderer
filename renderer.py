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
        self.rotation_x = np.identity(4, dtype=float)
        self.rotation_y = np.identity(4, dtype=float)
        self.rotation_z = np.identity(4, dtype=float)

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

    def rotate_x(self, theta):
        self.rotation_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

    def rotate_y(self, theta):
        self.rotation_x = np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

    def rotate_z(self, theta):
        self.rotation_x = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])


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
        self.color = tuple(color)
        self.recalc_normals()


class Triangle(Mesh):
    def __init__(self) -> None:
        super().__init__()
        self.vertices = np.array([
            [-.5, -.5, 0, 1], [.5, -.5, 0, 1], [0, 2**(1/2)-1, 0, 1]
        ])
        self.triangles = np.array([0, 1, 2])
        self.color = (255, 0, 255)
        self.recalc_normals()



class OrthograficProjection:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)
        self.meshes = []
        self.screen = np.zeros((width, height, 3), dtype=np.uint8)
        self.translation = np.identity(4, dtype=float)
        self.rotation = np.identity(4, dtype=float)

        self.light_direction = np.array([1, 1, -1], dtype=float)

        left, right = -1, 1#width
        up, bottom = -1, 1#height
        far, near = 10, -10
        self.projection = np.array([
            [2/(right-left), 0, 0, 2/2],
            [0, 2/(up-bottom), 0, 2/2],
            [0, 0, -2/(far-near), 0],
            [-(right + left)/(right-left), -(up+bottom)/(up-bottom), -(far+near)/(far-near), 1]
        ])

        self.projection = np.array([
            [width, 0, 0, width/2],
            [0, -height, 0, height/2],
            [0, 0, 1, 0],
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

    def set_light_direction(self, x: float, y: float, z: float) -> None:
        self.light_direction = np.array([x, y, z], dtype=float)

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

                '''# local rotation x
                a = np.dot(mesh.rotation_x, a)
                b = np.dot(mesh.rotation_x, b)
                c = np.dot(mesh.rotation_x, c)

                # local rotation y
                a = np.dot(mesh.rotation_y, a)
                b = np.dot(mesh.rotation_y, b)
                c = np.dot(mesh.rotation_y, c)

                # local rotation z
                a = np.dot(mesh.rotation_z, a)
                b = np.dot(mesh.rotation_z, b)
                c = np.dot(mesh.rotation_z, c)'''

                # local rotation
                a = np.dot(np.dot(mesh.rotation_z, np.dot(mesh.rotation_y, mesh.rotation_x)), a)
                b = np.dot(np.dot(mesh.rotation_z, np.dot(mesh.rotation_y, mesh.rotation_x)), b)
                c = np.dot(np.dot(mesh.rotation_z, np.dot(mesh.rotation_y, mesh.rotation_x)), c)

                # local translation
                a = np.dot(mesh.translation, a)
                b = np.dot(mesh.translation, b)
                c = np.dot(mesh.translation, c)

                # lighting
                intensity = np.dot(np.cross(b[:3]-a[:3], c[:3]-a[:3]), self.light_direction)
                intensity = int(intensity*255)

                # global translation
                a = np.dot(self.translation, a)
                b = np.dot(self.translation, b)
                c = np.dot(self.translation, c)

                # global rotation
                a = np.dot(self.rotation, a)
                b = np.dot(self.rotation, b)
                c = np.dot(self.rotation, c)

                print(self.projection, a, b, c)

                # global projection
                a = np.dot(self.projection, a)
                b = np.dot(self.projection, b)
                c = np.dot(self.projection, c)

                a = a[:2]
                b = b[:2]
                c = c[:2]

                print(a, b, c)

                draw.polygon([*a, *b, *c], fill=(intensity, intensity, intensity))
        img.show()


cube = Cube()
cube.scale(.25, .25, .25)
cube.rotate_y(np.pi/6)
cube.rotate_x(np.pi/8)


op = OrthograficProjection(400, 400)
op.add_mesh(cube)
op.render()
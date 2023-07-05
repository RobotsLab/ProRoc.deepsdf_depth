import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def look_at_matrix(point, target, up):
    z = np.array(point) - np.array(target)
    z = z / la.norm(z)

    x = np.cross(up, z)
    x = x / la.norm(x)
    y = np.cross(z, x)

    r = np.array([x, y, z]).transpose()

    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = point

    return m


def sphere_points(n_per_level, zs):
    points = []
    rs = np.sqrt(1 - np.power(zs, 2))
    angs = np.linspace(0, 2 * np.pi, n_per_level + 1)[:-1]
    print(angs)
    for r, z in zip(rs, zs):
        for ang in angs:
            x = r * np.sin(ang)
            y = r * np.cos(ang)
            p = [x, y, z]
            points.append(p)

    return np.array(points)


def rot_x(angle):
    rot = np.zeros((3, 3))
    s, c = np.sin(angle), np.cos(angle)
    rot[1, 1] = c
    rot[1, 2] = -s
    rot[2, 1] = s
    rot[2, 2] = c
    rot[0, 0] = 1
    return rot


def rot_y(angle):
    rot = np.zeros((3, 3))
    s, c = np.sin(angle), np.cos(angle)
    rot[0, 0] = c
    rot[0, 2] = s
    rot[2, 0] = -s
    rot[2, 2] = c
    rot[1, 1] = 1
    return rot


def rot_z(angle):
    rot = np.zeros((3, 3))
    s, c = np.sin(angle), np.cos(angle)
    rot[0, 0] = c
    rot[0, 1] = -s
    rot[1, 0] = s
    rot[1, 1] = c
    rot[2, 2] = 1
    return rot


def get_eye(x):
    return x[:3, 2]


def get_up(x):
    return x[:3, 1]


def get_right(x):
    return x[:3, 0]


point = [3.0, 0.0, 3.0]
target = [0.0, 0.0, 0.0]
up = [0.0, 0.0, 1.0]

mat = look_at_matrix(point, target, up)
eye = get_eye(mat)

sample_points = sphere_points(10, [0.0, 0.3, 0.5, 0.9])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for p in sample_points:
    ax.scatter(*p)

plt.show()

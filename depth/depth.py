import os
import json
import numpy as np

class BaseFile:
    def __init__(self, source_path, destination_dir=''):
        self.source_path = source_path
        self.destination_dir = destination_dir
        self.name = self.get_name()
        self.o_c_transformation = np.zeros(6)
        self.pixels = []
        self.ds = 0

    def get_name(self):
        return os.path.split(self.source_path)[1].split('.')[0]

    def get_version(self):
        if self.destination_dir:
            dir_files = os.listdir(self.destination_dir)
            file_number = len([x for x in dir_files if x.startswith(self.name) and x.endswith('.txt')]) + 1
            return file_number

    def save(self):
        raise NotImplementedError("Subclasses should implement this method")

    def load(self):
        raise NotImplementedError("Subclasses should implement this method")

class ViewsFile(BaseFile):
    def __init__(self, source_path, destination_dir=''):
        super().__init__(source_path, destination_dir)
        self.scale = 0
        self.s_o_transformation = np.zeros(6)
        self.frames = []
        if destination_dir:
            self.version = self.get_version()

    def save(self):
        with open(os.path.join(self.destination_dir, self.name + '.txt'), 'w') as f:
            f.write(f'{self.scale}\n')
            f.write(f"{' '.join(map(str, self.s_o_transformation))}\n")
            f.write(f"{' '.join(map(str, self.o_c_transformation))}\n")
            for frame in self.frames:
                f.write(f"{' '.join(map(str, frame))}\n")
        print(f"Saved: {os.path.join(self.destination_dir, self.name + '.txt')}")

    def load(self):
        with open(self.source_path, 'r') as f:
            lines = f.readlines()
            self.scale = float(lines[0].strip())
            self.s_o_transformation = np.array(list(map(float, lines[1].strip().split())))
            self.o_c_transformation = np.array(list(map(float, lines[2].strip().split())))
            self.frames = [np.array(list(map(float, line.strip().split()))) for line in lines[3:]]
        print(f"Loaded: {self.source_path}")

class DepthFile(BaseFile):
    def __init__(self, source_path, destination_dir=''):
        super().__init__(source_path, destination_dir)
        if destination_dir:
            self.version = self.get_version()
        self.f = None
        self.cx = None
        self.cy = None
        self.Ndx = None
        self.Ndy = None
        self.nx = None
        self.ny = None
        self.z = None
        self.ndx = None
        self.ndy = None
        self.dz = None
        self.dz2 = None

    def get_camera_parameters(self, f, cx, cy):
        self.f = f
        self.cx = cx
        self.cy = cy

    def get_image_resolution(self, Ndx, Ndy):
        self.Ndx = Ndx
        self.Ndy = Ndy

    def get_outliar_factor(self, ds):
        self.ds = ds

    def get_bounding_box_coords(self, nx, ny, z):
        self.nx = int(nx)
        self.ny = int(ny)
        self.z = z

    def get_bounding_box_size(self, ndx, ndy, dz, dz2):
        self.ndx = ndx
        self.ndy = ndy
        self.dz = dz
        self.dz2 = dz2

    def save(self, view, GT=False, POWER_FACTOR=1.0):
        if GT:
            filename = f"{self.name}_{view}_gt.txt"
        else:
            filename = f"{self.name}_{view}_a{POWER_FACTOR}.txt"

        with open(os.path.join(self.destination_dir, filename), 'w') as f:
            f.write(f"{' '.join(map(str, self.o_c_transformation))}\n")
            f.write(f'{self.f} {self.cx} {self.cy}\n')
            f.write(f'{self.Ndx} {self.Ndy}\n')
            f.write(f'{self.ds}\n')
            f.write(f'{self.nx} {self.ny} {self.z}\n')
            f.write(f'{self.ndx} {self.ndy} {self.dz} {self.dz2}\n')
            for image in self.pixels:
                for row in image:
                    for pixel in row:
                        f.write(f"{' '.join(map(str, pixel))}\n")
        print("FILE SAVED:", os.path.join(self.destination_dir, filename))

    def load(self, view, GT=False, POWER_FACTOR=1.0):
        if GT:
            filename = f"{self.name}_{view}_gt.txt"
        else:
            filename = f"{self.name}_{view}_a{POWER_FACTOR}.txt"

        with open(os.path.join(self.source_path, filename), 'r') as f:
            lines = f.readlines()
            self.o_c_transformation = np.array(list(map(float, lines[0].strip().split())))
            self.f, self.cx, self.cy = map(float, lines[1].strip().split())
            self.Ndx, self.Ndy = map(int, lines[2].strip().split())
            self.ds = float(lines[3].strip())
            self.nx, self.ny, self.z = map(int, lines[4].strip().split())
            self.ndx, self.ndy, self.dz, self.dz2 = map(float, lines[5].strip().split())
            self.pixels = []
            for i in range(6, len(lines)):
                pixel_row = [list(map(float, pixel.split())) for pixel in lines[i].strip().split('\n')]
                self.pixels.append(pixel_row)
        print(f"Loaded: {os.path.join(self.source_path, filename)}")

class TrainFile(BaseFile):
    def __init__(self, source_path, destination_dir):
        super().__init__(source_path, destination_dir)
        self.version = self.get_version()
        self.f = None
        self.cx = None
        self.cy = None
        self.Ndx = None
        self.Ndy = None
        self.nx = None
        self.ny = None
        self.z = None
        self.ndx = None
        self.ndy = None
        self.dz = None
        self.dz2 = None

    def get_camera_parameters(self, f, cx, cy):
        self.f = f
        self.cx = cx
        self.cy = cy

    def get_image_resolution(self, Ndx, Ndy):
        self.Ndx = Ndx
        self.Ndy = Ndy

    def get_outliar_factor(self, ds):
        self.ds = ds

    def get_bounding_box_coords(self, nx, ny, z):
        self.nx = nx
        self.ny = ny
        self.z = z

    def get_bounding_box_size(self, ndx, ndy, dz, dz2):
        self.ndx = ndx
        self.ndy = ndy
        self.dz = dz
        self.dz2 = dz2

    def save(self, dictionary, K=1):
        filename = f"{self.name}_k{K}_inp_train.json"
        with open(os.path.join(self.destination_dir, filename), "w") as outfile:
            json.dump(dictionary, outfile)
        print("Saved:", os.path.join(self.destination_dir, filename))

    def load(self, K=1):
        filename = f"{self.name}_k{K}_inp_train.json"
        with open(os.path.join(self.source_path, filename), 'r') as infile:
            dictionary = json.load(infile)
            self.o_c_transformation = np.array(dictionary.get('o_c_transformation', np.zeros(6)))
            self.f = dictionary.get('f', None)
            self.cx = dictionary.get('cx', None)
            self.cy = dictionary.get('cy', None)
            self.Ndx = dictionary.get('Ndx', None)
            self.Ndy = dictionary.get('Ndy', None)
            self.ds = dictionary.get('ds', 0)
            self.nx = dictionary.get('nx', None)
            self.ny = dictionary.get('ny', None)
            self.z = dictionary.get('z', None)
            self.ndx = dictionary.get('ndx', None)
            self.ndy = dictionary.get('ndy', None)
            self.dz = dictionary.get('dz', None)
            self.dz2 = dictionary.get('dz2', None)
            self.pixels = dictionary.get('pixels', [])
        print(f"Loaded: {os.path.join(self.source_path, filename)}")

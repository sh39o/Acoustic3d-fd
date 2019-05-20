class Model:
    def __init__(self, nx, ny, nz, dx, dy, dz, vel_file=None, rho_file=None, delta_file=None, epsilon_file=None):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.rho_file = rho_file
        self.vel_file = vel_file
        self.delta_file = delta_file
        self.epsilon_file = epsilon_file
        self.shape = (self.nx, self.ny, self.nz)

    def write_model(self, filename, value):
        import struct
        if filename is None:
            print(filename, "cannot be None")
            raise NameError

        f = open(filename, 'wb')

        for _ in range(self.nx * self.ny * self.nz):
            data = struct.pack('f', value)
            f.write(data)
        f.close()

    def read_model(self, filename):
        import struct
        import numpy as np
        f = open(filename, 'rb')
        res = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        for j in range(self.ny):
            for i in range(self.nx):
                for k in range(self.nz):
                    data1 = f.read(np.nbytes[np.float32])
                    data = struct.unpack(data1, 'f')[0]
                    res[i, j, k] = data
        f.close()
        return res

    def __str__(self):
        res = "*" * 30 + '\n'
        res += "the size of model is:\n"
        res += "nx, ny, nz = {}, {}, {}\n".format(self.nx, self.ny, self.nz)
        res += "dx, dy, dz = {}, {}, {}\n".format(self.dx, self.dy, self.dz)
        res += "*" * 30 + '\n'
        res += "velocity file : " + self.vel_file + '\n'
        if self.rho_file:
            res += "density file : " + self.rho_file + '\n'
        if self.delta_file:
            res += "delta file : " + self.delta_file + '\n'
        if self.epsilon_file:
            res += "epsilon file: " + self.epsilon_file + '\n'
        res += "*" * 30 + '\n'
        return res


class ShotReceiver:

    def __init__(self, sxbeg, sybeg, szbeg, jsx, jsy, jsz, sxend, syend, szend, dgx, dgy, dgt, shotfolder):
        self.sxbeg = sxbeg
        self.sybeg = sybeg
        self.szbeg = szbeg
        self.jsx = jsx
        self.jsy = jsy
        self.jsz = jsz
        self.sxend = sxend
        self.syend = syend
        self.szend = szend
        self.shotfile = shotfolder
        self.shotlist = self.generateShotList()
        self.nshot = len(self.shotlist)
        self.nodetask = self.allocateNode()
        self.receiver_dx = dgx
        self.receiver_dy = dgy
        self.receiver_dt = dgt

    def generateShotList(self):
        shotlist = []
        for isx in range(self.sxbeg, self.sxend + 1, self.jsx):
            for isy in range(self.sybeg, self.syend + 1, self.jsy):
                for isz in range(self.szbeg, self.szend + 1, self.jsz):
                    shotlist.append((isx, isy, isz, self.jsx, self.jsy, self.jsz))
        self.shotlist = shotlist
        return shotlist

    def allocateNode(self, nnode=1):
        nodetask = []
        for nodei in range(nnode):
            nodetask.append(self.shotlist[nodei::nnode])
        self.nodetask = nodetask
        return nodetask

    def __str__(self):
        res = '*' * 30 + '\n'
        res += "the shot begin: {}, {}, {}\n".format(self.sxbeg, self.sybeg, self.szbeg)
        res += "the shot inverval: {}, {}, {}\n".format(self.jsx, self.jsy, self.jsz)
        res += "the geophone interval: {}, {}\n".format(self.receiver_dx, self.receiver_dy)
        res += "the geophone sampling time: {}\n".format(self.receiver_dt)
        res += "shot shotfolder is " + self.shotfile + '\n'
        res += "*" * 30 + '\n'
        return res


def hello():
    print("hello")

class FD:
    def __init__(self, model, sg, nt, dt, fpeak, snap_folder, snap_interval, wtype=1, device_i=0, savesnap=False,
                 nodei=0,
                 cut_directwave=False):
        self.model = model
        self.nt = nt
        self.dt = dt
        self.sg = sg
        self.fpeak = fpeak
        self.snap_folder = snap_folder
        self.snap_interval = snap_interval
        self.kernel = None
        self.savesnap = savesnap
        self.nodei = nodei
        self.cut_directwave = cut_directwave
        self.wtype = wtype
        assert (self.sg.receiver_dx % self.model.dx == 0)
        assert (self.sg.receiver_dy % self.model.dy == 0)
        assert (self.sg.receiver_dt % self.dt == 0)

    def __str__(self):
        res = "the Model Property: \n"
        res += str(self.model)
        res += "the Receiver and Shot setting:\n"
        res += str(self.sg)
        res += "the dominant frequency: {}\n".format(self.fpeak)
        res += "nt : {}, dt : {}\n".format(self.nt, self.dt)
        res += "save snapshot : {}\n".format(self.savesnap)
        if self.savesnap:
            res += "snap interval : {}\n".format(self.snap_interval)
            res += "snap folder : {}\n".format(self.snap_folder)
        res += "cut direct wave : {}\n".format(self.cut_directwave)
        return res

    def available_GPU(self):
        import subprocess
        import numpy as np
        nDevice = int(subprocess.getoutput("nvidia-smi -L | grep GPU |wc -l"))
        total_GPU_str = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total | grep -o '[0-9]\+'")
        total_GPU = total_GPU_str.split('\n')
        total_GPU = np.array([int(device_i) for device_i in total_GPU])
        avail_GPU_str = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free | grep -o '[0-9]\+'")
        avail_GPU = avail_GPU_str.split('\n')
        avail_GPU = np.array([int(device_i) for device_i in avail_GPU])
        avail_GPU = avail_GPU / total_GPU
        return np.argmax(avail_GPU)

    def parallel_run(self):
        from multiprocessing import freeze_support, Pool
        import subprocess
        import time
        nDevice = int(subprocess.getoutput("nvidia-smi -L | grep GPU |wc -l"))
        freeze_support()
        pool = Pool(nDevice)
        taskList = self.sg.nodetask[self.nodei]
        print("there are {} task on this node".format(len(taskList)))
        print("the nDevice is {}".format(nDevice))
        for device_i in range(nDevice):
            task = taskList.pop(0)
            pool.apply_async(func=self.run, args=(task, device_i))
            time.sleep(3)
        for task in taskList:
            pool.apply_async(func=self.run, args=(task, None))
            time.sleep(3)
        pool.close()
        pool.join()



class Acoustic3d1order(FD):

    def load_kernel(self):
        from ctypes import cdll
        try:
            self.kernel = cdll.LoadLibrary('./lib/acoustic1order.so')
        except OSError:
            print("cannot open ./lib/acoustic1order.so")

    def run(self, ishot, device):
        from ctypes import c_char_p, c_int, c_float, c_bool
        self.load_kernel()
        if device is None:
            device = super().available_GPU()
        print("device: {}, shot: ".format(device), ishot)
        shotname = self.sg.shotfile + '{0}.{1}.bin'.format(str(ishot[0]), str(ishot[1]))
        if 0:
            print(shotname + " exists, skip")
        else:

            self.kernel.cuda_3dfd_1order(
                c_char_p(bytes(self.model.vel_file, 'utf-8')),
                c_char_p(bytes(self.model.rho_file, 'utf-8')),
                c_char_p(bytes(self.snap_folder, 'utf-8')),
                c_char_p(bytes(shotname, 'utf-8')),
                c_int(0),
                c_int(1),
                c_int(self.model.nx), c_int(self.model.ny), c_int(self.model.nz),
                c_float(self.model.dx), c_float(self.model.dy), c_float(self.model.dz),
                c_int(ishot[0]), c_int(ishot[1]), c_int(ishot[2]),
                c_int(ishot[3]), c_int(ishot[4]), c_int(ishot[5]),
                c_float(self.sg.receiver_dx), c_float(self.sg.receiver_dy), c_float(self.sg.receiver_dt),
                c_int(self.nt), c_float(self.dt), c_int(self.wtype),
                c_float(self.fpeak), c_bool(self.savesnap), c_bool(self.cut_directwave),
                c_int(self.snap_interval),
                c_int(device)
            )

    def __str__(self):
        res = "Stencil Acoustic 3D first order\n"
        res += "*" * 30 + '\n'
        res += super().__str__()
        return res


class Acoustic3dvti(FD):

    def __str__(self):
        res = "Stencil Acoustic vti 3D \n"
        res += "*" * 30 + '\n'
        res += super().__str__()
        return res

    def load_kernel(self):
        from ctypes import cdll
        try:
            self.kernel = cdll.LoadLibrary('./lib/acoustic1vti.so')
        except OSError:
            print("cannot open ./lib/acoustic1vti.so")

    def run(self, ishot, device):
        from ctypes import c_char_p, c_int, c_float, c_bool
        self.load_kernel()
        if device is None:
            device = super().available_GPU()
        print("device: {}, shot: ".format(device), ishot)
        shotname = self.sg.shotfile + '{0}.{1}.bin'.format(str(ishot[0]), str(ishot[1]))
        if 0:
            print(shotname + " exists, skip")
        else:
            self.kernel.cuda_3dfd_vti(
                c_char_p(bytes(self.model.vel_file, 'utf-8')),
                c_char_p(bytes(self.model.rho_file, 'utf-8')),
                c_char_p(bytes(self.model.epsilon_file, 'utf-8')),
                c_char_p(bytes(self.model.delta_file, 'utf-8')),
                c_char_p(bytes(self.snap_folder, 'utf-8')),
                c_char_p(bytes(shotname, 'utf-8')),
                c_int(0),
                c_int(1),
                c_int(self.model.nx), c_int(self.model.ny), c_int(self.model.nz),
                c_float(self.model.dx), c_float(self.model.dy), c_float(self.model.dz),
                c_int(ishot[0]), c_int(ishot[1]), c_int(ishot[2]),
                c_int(ishot[3]), c_int(ishot[4]), c_int(ishot[5]),
                c_float(self.sg.receiver_dx), c_float(self.sg.receiver_dy), c_float(self.sg.receiver_dt),
                c_int(self.nt), c_float(self.dt),
                c_float(self.fpeak), c_bool(self.savesnap), c_bool(self.cut_directwave),
                c_int(self.snap_interval),
                c_int(device)
            )


class Acoustic3d2order(FD):

    def __str__(self):
        res = "Stencil Acoustic 3D second order\n"
        res += "*" * 30 + '\n'
        res += str(super().__str__())
        return res

    def load_kernel(self):
        from ctypes import cdll
        try:
            self.kernel = cdll.LoadLibrary('./lib/acoustic2order.so')
        except OSError:
            print("cannot open ./lib/acoustic2order.so")

    def run(self, ishot, device):
        from ctypes import c_char_p, c_int, c_float, c_bool
        self.load_kernel()
        if device is None:
            device = super().available_GPU()
        print("device: {}, shot: ".format(device), ishot)
        shotname = self.sg.shotfile + '{0}.{1}.bin'.format(str(ishot[0]), str(ishot[1]))
        if 0:
            print(shotname + " exists, skip")
        else:
            self.kernel.cuda_3dfd(
                c_char_p(bytes(self.model.vel_file, 'utf-8')),
                c_char_p(bytes(self.snap_folder, 'utf-8')),
                c_char_p(bytes(shotname, 'utf-8')),
                c_int(0),
                c_int(1),
                c_int(self.model.nx), c_int(self.model.ny), c_int(self.model.nz),
                c_float(self.model.dx), c_float(self.model.dy), c_float(self.model.dz),
                c_int(ishot[0]), c_int(ishot[1]), c_int(ishot[2]),
                c_int(ishot[3]), c_int(ishot[4]), c_int(ishot[5]),
                c_float(self.sg.receiver_dx), c_float(self.sg.receiver_dy), c_float(self.sg.receiver_dt),
                c_int(self.nt), c_int(300), c_float(self.dt),
                c_float(self.fpeak), c_bool(self.savesnap),
                c_int(self.snap_interval), c_bool(self.cut_directwave),
                c_int(device)
            )

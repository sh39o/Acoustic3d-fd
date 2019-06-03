__author__ = "Sun Hui"
__doc__ = """
************************************************************************************
  The finite-difference acoustic wave simulation with CUDA implementation,
  CUDA version >= 7.5 and Python3 are required.
  The program supports 3 types of simulation:
      1. 2nd order acoustic wave simulation. Only scalar wave velocity is required
      2. 1st order acoustic pressure-particle velocity wave simulation. Scalar wave
         velocity and density files are required
      3. 1rd order vti acoustic wave simulation. Scalar wave velocity, density,
         epsilon, delta are required.
      4. 1rd order tti acoustic wave simulation. Scalar wave velocity, density,
         epsilon, delta, theta and psi are required.
  The program supports a preliminary implementation of multi-node allocation, which
  only supports the platform all GPUs have the same computing capability and may
  cause unexpectly abortion. For further development, I will try to make it more stable.
************************************************************************************
  Installation:
      make
************************************************************************************
  Three basic components make the architecture of the program, they are:
      Model       : describe the model
      ShotReceiver: describe the shot and recorder settings
      FD          : describe the finite-difference operators
       |__Acoustic3d1order: the 1st order 3d acoustic wave equation
       |__Acoustic3dvti   : the 1st order 3d vti acoustic wave equation
       |__Acoustic3dtti   : the 1st order 3d tti acoustic wave equation
       |__Acoustic3d2order: the 2nd order 3d acoustic wave equation
  The details of the three components are described below in the example code
"""
from readmodel import *
wavelet = {'ricker': 1, 'dgaussian': 2, 'gaussian': 3}

#***********************************************************************************
#Model:
#  Describe the velocity (density, etc) model parameters, 6 parameters are required
#    nx, ny, nz  : the size of the velocity (density) model in three directions
#    dx, dy, dz  : the spatial interval of the velocity (density, etc) model
#  The following are optional parameters:
#    vel_file    : if it is not given, you can use "write_model" function to produce a
#                  homogeneous (vel, den, etc)model, the usage of the "write_model" is:
#                  zjj_model.write_model('homo_vel.bin', 3000)
#                  where the first parameter is filename of the model file and the second
#                  parameter is the value of the homogeneous (vel, den, etc) model.
#    rho_file    : it is required in the 1st order equations (1st iso and 1st order vti)
#    delta_file  : required in the 1st order vti equation
#    epsilon_file: required in the 1st order vti equation
#    theta_file  : required in the 1st order tti equation
#    psi_file    : required in the 1st order tti equation
#
#  You can use print function to check your model settings
#***********************************************************************************

#  establish a model named "zjj_double"
zjj_double = Model(nx=201, ny=201, nz=201, dx=10, dy=10, dz=10,
                  vel_file='./model/vel.bin', rho_file='./model/den.bin',
                  epsilon_file='./model/epsilon.bin', delta_file='./model/delta.bin',
                  theta_file='./model/theta.bin', psi_file='./model/psi.bin')
print(zjj_double)
#  if homo_vel.bin is not exist, you can use write_model to produce one, the vel is 3000
#  in this example
zjj_double.generate_model(vel=3000, rho=1200, epsilon=0.3, delta=0.1, theta=1.1, psi=1.1)


#***********************************************************************************
#ShotReceiver:
#  set the shot and receiver settings named sg
#    sxbeg, sybeg, szbeg: the beginning shot in three directions
#    jsx, jsy, jsz      : the shot intervals in three directions
#    sxend, syend, szend: the endding shot in three directions
#    dgx, dgy           : the receiver intervals in x, y directions
#    dgt                : the recoder intervals
#    shotfolder         : the folder to save the shot recoder
#
#  Also, you can use print function to check your shot receiver settings, e.g. print(sg)
#***********************************************************************************

sg = ShotReceiver(sxbeg=101, sybeg=101, szbeg=101, jsx=10, jsy=10, jsz=1,
                  sxend=101, syend=101, szend=101,
                  dgx=10.0, dgy=10.0, dgt=0.002,
                  shotfolder='./shot_record/')
print(sg)
#  The allocateNode function is to allocate the tasks into different nodes, nnode is
#  the number of devices
sg.allocateNode(nnode=1)
#  the node id of the current host
nodei = 0


#***********************************************************************************
#  Here we use Acoustic3d1order to explain the usage of the FD stencil
#  model: which model to be simulated
#  sg   : the shot and receiver setting
#  nt   : the time sampling number
#  dt   : the temperal interval
#  fpeak: the dominant frequency of the source wavelet
#  wtype: the source type, 1 for ricker wavelet,
#                          2 for one derivative of gaussian wavelet
#                          3 for gaussian wavelet
#  snap_folder   : the folder to save snapshots. if savesnap is False, snap_folder=None
#  nodei         : the node id of the current host, it decides the task pool of this host
#  cut_directwave: whether cut the directwave, True or False
#
#  Just like discussed before, you can use print to check the stencil settings
#***********************************************************************************

stencil = Acoustic3dtti(model=zjj_double, sg=sg, nt=1001, dt=0.001, fpeak=15.0,
                        wtype=wavelet['ricker'],
                        snap_folder='./snapshots/', snap_interval=200,
                        savesnap=True, nodei=nodei, cut_directwave=False)
print(stencil)
#  After the FD stencil is settled, it won't run at once. use parallel_run to run the task
#stencil.parallel_run()
stencil.run((51,51,51,10,10,1),0)

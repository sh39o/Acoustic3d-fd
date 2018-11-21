from readmodel import *
homo = Model(nx=101, ny=101, nz=101, dx=10, dy=10, dz=10,
             vel_file='./model/vel.bin', rho_file='./model/den.bin',
             delta_file='./model/delta.bin',
             epsilon_file='./model/epsilon.bin')
homo.write_model(homo.vel_file, 2000.0)
homo.write_model(homo.rho_file, 1000.0)
homo.write_model(homo.delta_file, 0.1)
homo.write_model(homo.epsilon_file, 0.3)

sg = ShotReceiver(sxbeg=51, sybeg=51, szbeg=0, jsx=16, jsy=8, jsz=1,
                  sxend=51, syend=51, szend=0,
                  dgx=10.0, dgy=10.0, dgt=0.002,
                  shotfolder='./shots/')
sg.allocateNode(nnode=1)
nodei = 0
stencil = Acoustic3d1order(model=homo, sg=sg, nt=1001, dt=0.0010, fpeak=20.0,
                           snap_folder='./snapshots/', snap_interval=200,
                           savesnap=False, nodei=nodei, cut_directwave=False)
print(stencil)
#stencil.parallel_run()

from readmodel import *

homo = Model(nx=411, ny=411, nz=241, dx=10, dy=10, dz=10,
             vel_file='./model/vel.bin', rho_file='./model/den.bin')

sg = ShotReceiver(sxbeg=51, sybeg=51, szbeg=0, jsx=16, jsy=8, jsz=1,
                  sxend=51, syend=51, szend=0,
                  dgx=20.0, dgy=40.0, dgt=0.002,
                  shotfolder='/data2/sunhui/')
sg.allocateNode(nnode=4)
nodei = 0 #unode101
stencil = Acoustic3d1order(model=homo, sg=sg, nt=3001, dt=0.0010, fpeak=20.0,
                        snap_folder='./snapshots/', snap_interval=200,
                        savesnap=False, nodei=nodei, cut_directwave=True)
stencil.parallel_run()

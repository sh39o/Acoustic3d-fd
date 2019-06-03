#include<stdio.h>
#include<malloc.h>
#include<math.h>
#include<stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define pi 3.141592653

#define BlockSize1 16// tile size in 1st-axis
#define BlockSize2 16// tile size in 2nd-axis

#define mm 4

__device__ float d0;

__constant__ float c[mm] = {1.196289, -0.0797526, 0.009570313, -0.0006975447};

//a################################################################################
void check_gpu_error(const char *msg)
/*< check GPU errors >*/
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        printf("Cuda error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(0);
    }
}

//a################################################################################
__global__ void
add_source(float pfac, int fsx, int fsy, int sz, int nx, int ny, int nz, int nnx, int nny, int nnz, float dt, float t,
           float favg, int wtype, int npml, int is, int dsx, int dsy, float *P, float *Q, int nsx)
/*< generate ricker wavelet with time deley >*/
{
    int ixs, iys, izs;
    float x_, xx_, tdelay, ts, source = 0.0, sx, sy;

    tdelay = 1.0 / favg;
    ts = t - tdelay;

    sx = fsx + is % nsx * dsx;
    sy = fsy + is / nsx * dsy;

    if (wtype == 1)//ricker wavelet
    {
        x_ = favg * ts;
        xx_ = x_ * x_;
        source = (1 - 2 * pi * pi * (xx_)) * exp(-(pi * pi * xx_));
    } else if (wtype == 2) {//derivative of gaussian
        x_ = (-4) * favg * favg * pi * pi / log(0.1);
        source = (-2) * pi * pi * ts * exp(-x_ * ts * ts);
    } else if (wtype == 3) {//derivative of gaussian
        x_ = (-1) * favg * favg * pi * pi / log(0.1);
        source = exp(-x_ * ts * ts);
    }

    if (t <= 2 * tdelay) {
        ixs = sx + npml - 1;
        iys = sy + npml - 1;
        izs = sz + npml - 1;
        P[izs + ixs * nnz + iys * nnz * nnx] += pfac * source;
        Q[izs + ixs * nnz + iys * nnz * nnx] += pfac * source;
    }
}

/*******************func*********************/
__global__ void
update_vel(int nx, int ny, int nz, int nnx, int nny, int nnz, int npml, float dt, float dx, float dy, float dz,
           float *u0, float *v0, float *w0, float *u1, float *v1, float *w1, float *P, float *Q,
           float *coffx1, float *coffx2, float *coffy1, float *coffy2, float *coffz1, float *coffz2, float *theta,
           float *phi) {
    const int iz = blockIdx.x * blockDim.x + threadIdx.x;//0--nz's thread:iz
    const int ix = blockIdx.y * blockDim.y + threadIdx.y;//0--nx's thread:ix

    int id, iy, im;
    float dtx, dty, dtz, Px, Py, Pz, Qx, Qy, Qz, xx, yy, zz;

    dtx = dt / dx;
    dty = dt / dy;
    dtz = dt / dz;

    for (iy = 0; iy < nny; iy++) {
        id = iz + ix * nnz + iy * nnz * nnx;
        if (id >= mm && id < nnx * nny * nnz - mm) {
            if (ix >= mm && ix < (nnx - mm) && iy >= mm && iy < (nny - mm) && iz >= mm && iz < (nnz - mm)) {
                Px = 0.0;
                Qx = 0.0;
                Py = 0.0;
                Qy = 0.0;
                Pz = 0.0;
                Qz = 0.0;
                for (im = 0; im < mm; im++) {
                    Py += c[im] * (P[id + (im + 1) * nnz * nnx] - P[id - im * nnz * nnx]);
                    Px += c[im] * (P[id + (im + 1) * nnz] - P[id - im * nnz]);
                    Pz += c[im] * (P[id + im + 1] - P[id - im]);
                    Qy += c[im] * (Q[id + (im + 1) * nnz * nnx] - Q[id - im * nnz * nnx]);
                    Qx += c[im] * (Q[id + (im + 1) * nnz] - Q[id - im * nnz]);
                    Qz += c[im] * (Q[id + im + 1] - Q[id - im]);
                }
                xx = cos(theta[id]) * cos(phi[id]) * Px - cos(theta[id]) * sin(phi[id]) * Py - sin(theta[id]) * Pz;
                yy = sin(phi[id]) * Px + cos(phi[id]) * Py;
                zz = sin(theta[id]) * cos(phi[id]) * Qx - sin(theta[id]) * sin(phi[id]) * Qy + cos(theta[id]) * Qz;
                u1[id] = coffx2[ix] * u0[id] - coffx1[ix] * dtx * xx;
                v1[id] = coffy2[iy] * v0[id] - coffy1[iy] * dty * yy;
                w1[id] = coffz2[iz] * w0[id] - coffz1[iz] * dtz * zz;
            }
        }
    }


}

/*******************func***********************/
__global__ void update_stress(int nx, int ny, int nz, int nnx, int nny, int nnz, float dt, float dx, float dy, float dz,
                              float *u1, float *v1, float *w1, float *P, float *Q, float *vp, int npml,
                              float *px1, float *px0, float *py1, float *py0, float *pz1, float *pz0,
                              float *qx1, float *qx0, float *qy1, float *qy0, float *qz1, float *qz0,
                              float *acoffx1, float *acoffx2, float *acoffy1, float *acoffy2, float *acoffz1,
                              float *acoffz2,
                              float *delta, float *epsilon, int fsx, int dsx, int fsy, int dsy, int zs, int is, int nsx,
                              bool SV, float *theta, float *phi) {
    const int iz = blockIdx.x * blockDim.x + threadIdx.x;//0--nz's thread:iz
    const int ix = blockIdx.y * blockDim.y + threadIdx.y;//0--nx's thread:ix

    int id, iy, im, rx, ry, rz, R = 15, r = 4;
    float dtx, dty, dtz, xx, yy, zz, ee, dd;
    float ux, uy, uz, vx, vy, vz, wx, wy, wz;

    dtx = dt / dx;
    dty = dt / dy;
    dtz = dt / dz;

    for (iy = 0; iy < nny; iy++) {
        id = iz + ix * nnz + iy * nnz * nnx;
        if (id >= mm && id < nnx * nnz * nny - mm) {
/************************i****************************************/
/************************iso circle start*************************/
            rx = ix - (fsx + is % nsx * dsx + npml - 1);
            ry = iy - (fsy + is / nsx * dsy + npml - 1);
            rz = iz - (zs + npml - 1);
            if (SV) {
                if ((rx * rx + ry * ry + rz * rz) <= R * R) {
                    if ((rx * rx + ry * ry + rz * rz) <= r * r) {
                        ee = 0.0;
                        dd = 0.0;
                    } else {
                        ee = 0.5 * (1 - cos(pi * ((sqrtf(rx * rx + ry * ry + rz * rz) - r) * 4.0 / (R * 3.0 - 1)))) *
                             epsilon[id];
                        dd = 0.5 * (1 - cos(pi * ((sqrtf(rx * rx + ry * ry + rz * rz) - r) * 4.0 / (R * 3.0 - 1)))) *
                             delta[id];
                    }
                } else {
                    ee = epsilon[id];
                    dd = delta[id];
                }
            } else {
                ee = epsilon[id];
                dd = delta[id];
            }
/************************ iso circle end *************************/
/************************i****************************************/
            if (ix >= mm && ix < (nnx - mm) && iy >= mm && iy < (nny - mm) && iz >= mm && iz < (nnz - mm)) {
                ux = 0.0;
                uy = 0.0;
                uz = 0.0;
                vx = 0.0;
                vy = 0.0;
                vz = 0.0;
                wx = 0.0;
                wy = 0.0;
                wz = 0.0;
                for (im = 0; im < mm; im++) {
                    uy += c[im] * (u1[id + im * nnz * nnx] - u1[id - (im + 1) * nnz * nnx]);
                    ux += c[im] * (u1[id + im * nnz] - u1[id - (im + 1) * nnz]);
                    uz += c[im] * (u1[id + im] - u1[id - im - 1]);
                    vy += c[im] * (v1[id + im * nnz * nnx] - v1[id - (im + 1) * nnz * nnx]);
                    vx += c[im] * (v1[id + im * nnz] - v1[id - (im + 1) * nnz]);
                    vz += c[im] * (v1[id + im] - v1[id - im - 1]);
                    wy += c[im] * (w1[id + im * nnz * nnx] - w1[id - (im + 1) * nnz * nnx]);
                    wx += c[im] * (w1[id + im * nnz] - w1[id - (im + 1) * nnz]);
                    wz += c[im] * (w1[id + im] - w1[id - im - 1]);
                }
                xx = cos(theta[id]) * cos(phi[id]) * ux - cos(theta[id]) * sin(phi[id]) * uy - sin(theta[id]) * uz;
                yy = sin(phi[id]) * vx + cos(phi[id]) * vy;
                zz = sin(theta[id]) * cos(phi[id]) * wx - sin(theta[id]) * sin(phi[id]) * wy + cos(theta[id]) * wz;

                px1[id] = acoffx2[ix] * px0[id] - acoffx1[ix] * vp[id] * vp[id] * (1 + 2 * ee) * dtx * xx;
                py1[id] = acoffy2[iy] * py0[id] - acoffy1[iy] * vp[id] * vp[id] * (1 + 2 * ee) * dty * yy;
                pz1[id] = acoffz2[iz] * pz0[id] - acoffz1[iz] * vp[id] * vp[id] * sqrtf(1 + 2 * dd) * dtz * zz;

                qx1[id] = acoffx2[ix] * qx0[id] - acoffx1[ix] * vp[id] * vp[id] * sqrtf(1 + 2 * dd) * dtx * xx;
                qy1[id] = acoffy2[iy] * qy0[id] - acoffy1[iy] * vp[id] * vp[id] * sqrtf(1 + 2 * dd) * dty * yy;
                qz1[id] = acoffz2[iz] * qz0[id] - acoffz1[iz] * vp[id] * vp[id] * dtz * zz;

                P[id] = px1[id] + py1[id] + pz1[id];
                Q[id] = qx1[id] + qy1[id] + qz1[id];
            }
        }
    }
}

/********************func**********************/
__global__ void get_d0(float dx, float dy, float dz, int nnx, int nny, int nnz, int npml, float *vp) {
    d0 = 10.0 * vp[nny * nnx * nnz / 2] * log(100000.0) / (2.0 * npml * ((dx + dy + dz) / 3.0));
}

/*************func*******************/
void pad_vv(int nx, int ny, int nz, int nnx, int nny, int nnz, int npml, float *ee) {
    int ix, iy, iz, id;

    for (iy = 0; iy < nny; iy++)
        for (ix = 0; ix < nnx; ix++) {
            for (iz = 0; iz < nnz; iz++) {
                id = iz + ix * nnz + iy * nnz * nnx;

                if (ix < npml) {
                    ee[id] = ee[iz + npml * nnz + iy * nnz * nnx];  //left
                } else if (ix >= nnx - npml) {
                    ee[id] = ee[iz + (nnx - npml - 1) * nnz + iy * nnz * nnx];//right
                }
            }
        }
    for (iy = 0; iy < nny; iy++)
        for (ix = 0; ix < nnx; ix++) {
            for (iz = 0; iz < nnz; iz++) {
                id = iz + ix * nnz + iy * nnz * nnx;

                if (iy < npml) {
                    ee[id] = ee[iz + ix * nnz + npml * nnz * nnx];  //front
                } else if (iy >= nny - npml) {
                    ee[id] = ee[iz + ix * nnz + (nny - npml - 1) * nnz * nnx];//back
                }
            }
        }
    for (iy = 0; iy < nny; iy++)
        for (ix = 0; ix < nnx; ix++) {
            for (iz = 0; iz < nnz; iz++) {
                id = iz + ix * nnz + iy * nnz * nnx;

                if (iz < npml) {
                    ee[id] = ee[npml + ix * nnz + iy * nnz * nnx];  //up
                } else if (iz >= nnz - npml) {
                    ee[id] = ee[nnz - npml - 1 + ix * nnz + iy * nnz * nnx];//down
                }
            }
        }

}

/*************func*******************/
void read_file(char FN1[], char FN2[], char FN3[], char FN4[], char FN5[], int nx, int ny, int nz, int nnx, int nny, int nnz, float *vv,
               float *epsilon, float *delta, float *theta, float *phi, int npml) {
    int ix, iy, iz, id;

    FILE *fp1, *fp2, *fp3, *fp4, *fp5;
    if ((fp1 = fopen(FN1, "rb")) == NULL) printf("error open <%s>!\n", FN1);
    if ((fp2 = fopen(FN2, "rb")) == NULL) printf("error open <%s>!\n", FN2);
    if ((fp3 = fopen(FN3, "rb")) == NULL) printf("error open <%s>!\n", FN3);
    if ((fp4 = fopen(FN4, "rb")) == NULL) printf("error open <%s>!\n", FN4);
    if ((fp5 = fopen(FN5, "rb")) == NULL) printf("error open <%s>!\n", FN5);

    for (iy = npml; iy < ny + npml; iy++)
        for (ix = npml; ix < nx + npml; ix++) {
            for (iz = npml; iz < nz + npml; iz++) {
                id = iz + ix * nnz + iy * nnz * nnx;
                fread(&vv[id], 4L, 1, fp1);//
                fread(&epsilon[id], 4L, 1, fp2);//
                fread(&delta[id], 4L, 1, fp3);//
                fread(&theta[id], 4L, 1, fp4);
                fread(&phi[id], 4L, 1, fp5);
            }
        }
    fclose(fp1);
    fclose(fp2);
    fclose(fp3);
    fclose(fp4);
    fclose(fp5);
}

/*************func*******************/
__global__ void initial_coffe(float dt, int nn, float *coff1, float *coff2, float *acoff1, float *acoff2, int npml) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id < nn + 2 * npml) {
        if (id < npml) {
            coff1[id] = 1.0 / (1.0 + (dt * d0 * pow((npml - 0.5 - id) / npml, 2.0)) / 2.0);
            coff2[id] = coff1[id] * (1.0 - (dt * d0 * pow((npml - 0.5 - id) / npml, 2.0)) / 2.0);

            acoff1[id] = 1.0 / (1.0 + (dt * d0 * pow(((npml - id) * 1.0) / npml, 2.0)) / 2.0);
            acoff2[id] = acoff1[id] * (1.0 - (dt * d0 * pow(((npml - id) * 1.0) / npml, 2.0)) / 2.0);

        } else if (id >= npml && id < npml + nn) {

            coff1[id] = 1.0;
            coff2[id] = 1.0;

            acoff1[id] = 1.0;
            acoff2[id] = 1.0;

        } else {

            coff1[id] = 1.0 / (1.0 + (dt * d0 * pow((0.5 + id - nn - npml) / npml, 2.0)) / 2.0);
            coff2[id] = coff1[id] * (1.0 - (dt * d0 * pow((0.5 + id - nn - npml) / npml, 2.0)) / 2.0);

            acoff1[id] = 1.0 / (1.0 + (dt * d0 * pow(((id - nn - npml) * 1.0) / npml, 2.0)) / 2.0);
            acoff2[id] = acoff1[id] * (1.0 - (dt * d0 * pow(((id - nn - npml) * 1.0) / npml, 2.0)) / 2.0);
        }
    }
}

/*************func*******************/
__global__ void
shot_record(int nnx, int nny, int nnz, int nx, int ny, int nz, int npml, int it, int nt, float *P, float *shot) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    int ix = id % nx;
    int iy = id / nx;

    if (id < nx * ny) {
        shot[it + nt * ix + nt * nx * iy] = P[npml + nnz * (ix + npml) + nnz * nnx * (iy + npml)];
    }
}

/*************func**************/
void window3d(float *a, float *b, int nz, int nx, int ny, int nnz, int nnx, int npml)
/*< window a 3d subvolume >*/
{
    int iz, ix, iy;

    for (iy = 0; iy < ny; iy++)
        for (ix = 0; ix < nx; ix++)
            for (iz = 0; iz < nz; iz++) {
                a[iz + nz * ix + nz * nx * iy] = b[(iz + npml) + nnz * (ix + npml) + nnz * nnx * (iy + npml)];
            }
}

/*************func**************/
__global__ void
mute_directwave(int nx, int ny, int nt, float dt, float favg, float dx, float dy, float dz, int fsx, int fsy, int dsx,
                int dsy,
                int zs, int is, float *vp, float *epsilon, float *shot, int tt, int nsx) {

    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int id, it;
    int mu_t, mu_nt;
    float mu_x, mu_y, mu_z, mu_t0;

    for (it = 0; it < nt; it++) {
        id = it + ix * nt + iy * nx * nt;
        if (ix < nx && iy < ny && it < nt) {
            mu_x = dx * abs(ix - fsx - (is % nsx) * dsx);
            mu_y = dy * abs(iy - fsy - (is / nsx) * dsy);
            mu_z = dz * zs;
            mu_t0 = sqrtf(pow(mu_x, 2) + pow(mu_y, 2) + pow(mu_z, 2)) / (vp[1] * sqrtf(1 + 2 * epsilon[1]));
            mu_t = (int) (2.0 / (dt * favg));
            mu_nt = (int) (mu_t0 / dt) + mu_t + tt;

            if (it < mu_nt)
                shot[id] = 0.0;
        }
    }
/*    int id=threadIdx.x+blockDim.x*blockIdx.x;

    int mu_t,mu_nt;
    float mu_x,mu_y,mu_z,mu_t0;

    int ix=(id/nt)%nx;
    int iy=(id/nt)/nx;
    int it=id%nt;

   if(id<nx*ny*nt)
   {
        mu_x=dx*abs(ix-fsx-(is%nsx)*dsx);
        mu_y=dy*abs(iy-fsy-(is/nsx)*dsy);
        mu_z=dz*zs;
        mu_t0=sqrtf(pow(mu_x,2)+pow(mu_y,2)+pow(mu_z,2))/(vp[1]*sqrtf(1+2*epsilon[1]));
        mu_t=(int)(2.0/(dt*favg));
        mu_nt=(int)(mu_t0/dt)+mu_t+tt;

           if(it<mu_nt)
              shot[id]=0.0;
   }  */
}

//a########################################################################

extern "C" int cuda_3dfd_tti(char *FNvel, char *FNrho,
        char *FNepsilon, char *FNdelta,
        char *FNtheta, char *FNpsi,
        char *FNsnap, char *FNshot, int is, int ns,
        int nx, int ny, int nz, float dx, float dy, float dz,
        int sxbeg, int sybeg, int szbeg, int jsx, int jsy, int jsz,
        float dgx, float dgy, float dgt,
        int nt, float dt, float fm, bool show_snapshot, bool cut_directwave,
        int snap_interval, int cudaDevicei){

    int it, nnx, nny, nnz, wtype;
    int nsx, dsx, fsx, dsy, fsy, zs, npml;
    float t, pfac, favg;

    float *v, *e, *d, *theta0, *phi0;
    float *vp, *epsilon, *delta, *theta, *phi;
    float *s_u0, *s_u1, *s_px0, *s_qx0, *s_px1, *s_qx1;
    float *s_v0, *s_v1, *s_py0, *s_qy0, *s_py1, *s_qy1;
    float *s_w0, *s_w1, *s_pz0, *s_qz0, *s_pz1, *s_qz1;
    float *s_P, *s_Q, *shot_Dev, *shot_Hos;

    float *coffx1, *coffx2, *coffy1, *coffy2, *coffz1, *coffz2;
    float *acoffx1, *acoffx2, *acoffy1, *acoffy2, *acoffz1, *acoffz2;
    char snapname[300], snapid[300];

//    clock_t start, end;
/*************wavelet\boundary**************/
    wtype = 1;
    npml = 40;
/********** dat document ***********/
//    char FN1[250] = {"model/model1/vp.bin"};
//    char FN2[250] = {"model/model1/epsilon.bin"};
//    char FN3[250] = {"model/model1/delta.bin"};
//    char FN4[250] = {"output/shot.bin"};
//    char FN5[250];

/********aaa************/
    FILE *fpsnap, *fpshot;

    if ((fpshot = fopen(FNshot, "wb")) == NULL){
        printf("cannot write shot file");
        return 1;
    }

/********* parameters *************/

    favg = fm;
    pfac = 10.0;

    ns = 1;
    nsx = ns;

    fsx = sxbeg;
    fsy = sybeg;
    dsx = jsx;
    dsy = jsy;
    zs = szbeg;
    //printf("fsx, fsy=%d,%d\n", fsx, fsy);
/*************v***************/
    nnx = nx + 2 * npml;
    nny = ny + 2 * npml;
    nnz = nz + 2 * npml;
/************a*************/


    v = (float *) malloc(nnz * nnx * nny * sizeof(float));
    e = (float *) malloc(nnz * nnx * nny * sizeof(float));
    d = (float *) malloc(nnz * nnx * nny * sizeof(float));
    theta0 = (float *) malloc(nnz * nnx * nny * sizeof(float));
    phi0 = (float *) malloc(nnz * nnx * nny * sizeof(float));
    shot_Hos = (float *) malloc(nt * nx * ny * sizeof(float));
    read_file(FNvel, FNepsilon, FNdelta, FNtheta, FNpsi, nx, ny, nz, nnx, nny, nnz, v, e, d, theta0, phi0, npml);
/****************************/
    pad_vv(nx, ny, nz, nnx, nny, nnz, npml, e);
    pad_vv(nx, ny, nz, nnx, nny, nnz, npml, d);
    pad_vv(nx, ny, nz, nnx, nny, nnz, npml, v);
    pad_vv(nx, ny, nz, nnx, nny, nnz, npml, theta0);
    pad_vv(nx, ny, nz, nnx, nny, nnz, npml, phi0);

    cudaSetDevice(cudaDevicei);// initialize device, default device=0;
    check_gpu_error("Failed to initialize device!");

    dim3 Xdimg, dimg, dimb;
    Xdimg.x = (nnx + BlockSize1 - 1) / BlockSize1;
    Xdimg.y = (nny + BlockSize2 - 1) / BlockSize2;
    dimg.x = (nnz + BlockSize1 - 1) / BlockSize1;
    dimg.y = (nnx + BlockSize2 - 1) / BlockSize2;
    dimb.x = BlockSize1;
    dimb.y = BlockSize2;
/****************************/
    cudaMalloc(&vp, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&epsilon, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&delta, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&theta, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&phi, nnz * nnx * nny * sizeof(float));
    cudaMemcpy(vp, v, nnz * nnx * nny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(epsilon, e, nnz * nnx * nny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(delta, d, nnz * nnx * nny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(theta, theta0, nnz * nnx * nny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(phi, phi0, nnz * nnx * nny * sizeof(float), cudaMemcpyHostToDevice);
/****************************/
    cudaMalloc(&s_u0, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_u1, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_v0, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_v1, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_w0, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_w1, nnz * nnx * nny * sizeof(float));

    cudaMalloc(&s_P, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_Q, nnz * nnx * nny * sizeof(float));

    cudaMalloc(&s_px0, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_px1, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_py0, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_py1, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_pz0, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_pz1, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_qx0, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_qx1, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_qy0, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_qy1, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_qz0, nnz * nnx * nny * sizeof(float));
    cudaMalloc(&s_qz1, nnz * nnx * nny * sizeof(float));

    cudaMalloc(&coffx1, nnx * sizeof(float));
    cudaMalloc(&coffx2, nnx * sizeof(float));
    cudaMalloc(&coffy1, nnx * sizeof(float));
    cudaMalloc(&coffy2, nnx * sizeof(float));
    cudaMalloc(&coffz1, nnz * sizeof(float));
    cudaMalloc(&coffz2, nnz * sizeof(float));
    cudaMalloc(&acoffx1, nnx * sizeof(float));
    cudaMalloc(&acoffx2, nnx * sizeof(float));
    cudaMalloc(&acoffy1, nnx * sizeof(float));
    cudaMalloc(&acoffy2, nnx * sizeof(float));
    cudaMalloc(&acoffz1, nnz * sizeof(float));
    cudaMalloc(&acoffz2, nnz * sizeof(float));

    cudaMalloc(&shot_Dev, nx * ny * nt * sizeof(float));
/******************************/
    check_gpu_error("Failed to allocate memory for variables!");

    get_d0 << < 1, 1 >> > (dx, dy, dz, nnx, nny, nnz, npml, vp);
    initial_coffe << < (nnx + 511) / 512, 512 >> > (dt, nx, coffx1, coffx2, acoffx1, acoffx2, npml);
    initial_coffe << < (nny + 511) / 512, 512 >> > (dt, ny, coffy1, coffy2, acoffy1, acoffy2, npml);
    initial_coffe << < (nnz + 511) / 512, 512 >> > (dt, nz, coffz1, coffz2, acoffz1, acoffz2, npml);


//    printf("--------------------------------------------------------\n");
//    printf("---   \n");
//    start = clock();
/**********IS Loop start*******/
    for (is = 0; is < ns; is++) {
        //  printf("---   IS=%3d  \n",is);

        cudaMemset(s_u0, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_u1, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_v0, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_v1, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_w0, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_w1, 0, nnz * nnx * nny * sizeof(float));

        cudaMemset(s_P, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_Q, 0, nnz * nnx * nny * sizeof(float));

        cudaMemset(s_px0, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_px1, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_py0, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_py1, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_pz0, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_pz1, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_qx0, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_qx1, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_qy0, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_qy1, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_qz0, 0, nnz * nnx * nny * sizeof(float));
        cudaMemset(s_qz1, 0, nnz * nnx * nny * sizeof(float));

        cudaMemset(shot_Dev, 0, nt * nx * ny * sizeof(float));

        for (it = 0, t = dt; it < nt; it++, t += dt) {
            //if (it % 100 == 0) printf("---   IS===%d   it===%d\n", is, it);
            add_source << < 1, 1 >> >
                               (pfac, fsx, fsy, zs, nx, ny, nz, nnx, nny, nnz, dt, t, favg, wtype, npml, is, dsx, dsy, s_P, s_Q, nsx);
            update_vel << < dimg, dimb >> >
                                  (nx, ny, nz, nnx, nny, nnz, npml, dt, dx, dy, dz, s_u0, s_v0, s_w0, s_u1, s_v1, s_w1, s_P, s_Q, coffx1, coffx2, coffy1, coffy2, coffz1, coffz2, theta, phi);
            update_stress << < dimg, dimb >> >
                                     (nx, ny, nz, nnx, nny, nnz, dt, dx, dy, dz, s_u1, s_v1, s_w1, s_P, s_Q, vp, npml,
                                             s_px1, s_px0, s_py1, s_py0, s_pz1, s_pz0, s_qx1, s_qx0, s_qy1, s_qy0, s_qz1, s_qz0,
                                             acoffx1, acoffx2, acoffy1, acoffy2, acoffz1, acoffz2, delta, epsilon,
                                             fsx, dsx, fsy, dsy, zs, is, nsx, true, theta, phi);
            s_u0 = s_u1;
            s_v0 = s_v1;
            s_w0 = s_w1;
            s_px0 = s_px1;
            s_py0 = s_py1;
            s_pz0 = s_pz1;
            s_qx0 = s_qx1;
            s_qy0 = s_qy1;
            s_qz0 = s_qz1;

            shot_record << < (nx * ny + 511) / 512, 512 >> > (nnx, nny, nnz, nx, ny, nz, npml, it, nt, s_P, shot_Dev);


            if ((is == 0) && (it != 0 && it % 100 == 0)) {
                cudaMemcpy(e, s_P, nnz * nnx * nny * sizeof(float), cudaMemcpyDeviceToHost);
                    strcpy(snapname, FNsnap);
                    sprintf(snapid, "ishot_%d_", is);
                    strcat(snapname, snapid);
                    sprintf(snapid, "it_%d", it);
                    strcat(snapname, snapid);
                    strcat(snapname, ".bin");
                    if ((fpsnap = fopen(snapname, "wb")) == NULL) {
                        printf("cannot write snapfile\n");
                    }

                window3d(v, e, nz, nx, ny, nnz, nnx, npml);
                fwrite(v, 4L, nx * nz * ny, fpsnap);
                fclose(fpsnap);
            }
        }//it loop end
        if (cut_directwave) {
            mute_directwave << < Xdimg, dimb >> >
                                    (nx, ny, nt, dt, favg, dx, dy, dz, fsx, fsy, dsx, dsy, zs, is, vp, epsilon, shot_Dev, 60, nsx);
            check_gpu_error("1!");
        }
        cudaMemcpy(shot_Hos, shot_Dev, nt * nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
        check_gpu_error("2!");
        fseek(fpshot, is * nt * nx * ny * sizeof(float), 0);
        for (int iy = 0; iy < ny; iy += int(dgy / dy)) {
            for (int ix = 0; ix < nx; ix += int(dgx / dx)) {
                for (it = 0; it < nt; it += int(dgt / dt)) {
                    fwrite(&shot_Hos[iy * nx * nt + ix * nt + it], sizeof(float), 1, fpshot);
                }
            }
        }
    }//is loop end
//    end = clock();
/*********IS Loop end*********/
//    printf("---   The forward is over    \n");
//    printf("---   Complete!!!!!!!!! \n");
//    printf("total %d shots: %f (s)\n", ns, ((float) (end - start)) / CLOCKS_PER_SEC);



/***********close************/
    fclose(fpshot);
/***********free*************/
    cudaFree(coffx1);
    cudaFree(coffx2);
    cudaFree(coffz1);
    cudaFree(coffz2);
    cudaFree(acoffx1);
    cudaFree(acoffx2);
    cudaFree(acoffz1);
    cudaFree(acoffz2);

    cudaFree(s_u0);
    cudaFree(s_u1);
    cudaFree(s_v0);
    cudaFree(s_v1);
    cudaFree(s_w0);
    cudaFree(s_w1);

    cudaFree(s_P);
    cudaFree(s_Q);

    cudaFree(s_px0);
    cudaFree(s_px1);
    cudaFree(s_py0);
    cudaFree(s_py1);
    cudaFree(s_pz0);
    cudaFree(s_pz1);
    cudaFree(s_qx0);
    cudaFree(s_qx1);
    cudaFree(s_qy0);
    cudaFree(s_qy1);
    cudaFree(s_qz0);
    cudaFree(s_qz1);

    cudaFree(shot_Dev);
/***************host free*****************/
    free(v);
    free(e);
    free(d);
    free(shot_Hos);
    return 0;
}


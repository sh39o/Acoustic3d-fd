//a#########################################################
//a##    3D iso acoustic fd :MPI + CUDA 
//a##                       code by Rong Tao
//a#########################################################
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ volatile int vint = 0;

#define PI 	3.141592653

#define BlockSize1 16// tile size in 1st-axis
#define BlockSize2 16// tile size in 2nd-axis
#define mm      4    // half of the order in space
#define npd     50   // absorbing boundry condition wield


void check_gpu_error (const char *msg) 
/*< check GPU errors >*/
{
    cudaError_t err = cudaGetLastError ();
    if (cudaSuccess != err) { 
	printf("Cuda error: %s: %s\n", msg, cudaGetErrorString (err)); 
	exit(0);   
    }
}

__constant__ float stencil[mm+1]={-205.0/72.0,8.0/5.0,-1.0/5.0,8.0/315.0,-1.0/560.0};

__global__ void cuda_ricker_wavelet(float *wlt, float fm, float dt, int nt)
/*< generate ricker wavelet with time deley >*/
{
	int it=threadIdx.x+blockDim.x*blockIdx.x;
    if (it<nt){
	  float tmp = PI*fm*fabsf(it*dt-1.0/fm);//delay the wavelet to exhibit all waveform
	  tmp *=tmp;
	  wlt[it]= (1.0-2.0*tmp)*expf(-tmp);// ricker wavelet at time: t=nt*dt
	}
}

__global__ void cuda_set_s(int *szxy, int szbeg, int sxbeg, int sybeg, int jsz, int jsx, int jsy, int ns, int nz, int nx, int ny)
/*< set the positions of sources  in whole domain >*/
{
	int id=threadIdx.x+blockDim.x*blockIdx.x;
	int nnz=nz+2*mm+2*npd;
	int nnx=nx+2*mm+2*npd;
    	if (id<ns) szxy[id]=(szbeg+id*jsz+mm+npd)+nnz*(sxbeg+id*jsx+mm+npd)+nnz*nnx*(sybeg+id*jsy+mm+npd);
}

__global__ void cuda_set_g(int *gzxy, int ng, int nz, int nx, int ny)
/*< set the positions of  geophones in whole domain >*/
{
	int id=threadIdx.x+blockDim.x*blockIdx.x;
	int nnz=nz+2*mm+2*npd;
	int nnx=nx+2*mm+2*npd;
       int iy=id/nx;
       int ix=id%nx;
    	if (id<ng) gzxy[id]=(mm+npd)+nnz*(ix*1+mm+npd)+nnz*nnx*(iy*1+mm+npd);
       
}
__global__ void cuda_trans_xy2txy(float *xy, float *txy, int it, int nt, int ng)
/*< set the positions of  geophones in whole domain >*/
{
	int id=threadIdx.x+blockDim.x*blockIdx.x;
    	if (id<ng) txy[it+id*nt]+=xy[id];
       
}
//mute_directwave<<<Xdimg,dimb>>>(nx,ny,nt,dt,fm,dx,dy,dz,sxbeg,sybeg,jsx,jsy,szbeg,is,v_cut_directwave,d_dcal_device_txy,60,ns);
__global__ void
mute_directwave(int nx, int ny, int nt, float dt, float favg, float dx, float dy, float dz, int fsx, int fsy, int dsx,
                int dsy,
                int zs, int is, float v_cut_directwave, float *shot, int tt, int nsx) {

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
            mu_t0 = sqrtf(pow(mu_x, 2) + pow(mu_y, 2) + pow(mu_z, 2)) / v_cut_directwave;
            mu_t = (int) (2.0 / (dt * favg));
            mu_nt = (int) (mu_t0 / dt) + mu_t + tt;

            if (it < mu_nt){
                shot[id] = 0.0;
            }
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

__global__ void cuda_absorb_bndr(float *d_p,int nz,int nx,int ny,float qp)
/*< absorb boundry condition >*/
{
    const int iz = blockIdx.x * blockDim.x + threadIdx.x;//0--nz's thread:iz
    const int ix = blockIdx.y * blockDim.y + threadIdx.y;//0--nx's thread:ix

       int id,iy;
	int nnz=nz+2*mm+2*npd;
	int nnx=nx+2*mm+2*npd;
	int nny=ny+2*mm+2*npd;

       for(iy=0;iy<nny;iy++)
        {
          id=iz+ix*nnz+iy*nnz*nnx;
            /*< front & back (0<y<ny) >*/
             if ( iy < npd )
               d_p[id]=( qp*pow((npd-iy)/(1.0*npd),2) + 1 )*d_p[id];
             else if ( iy >= 2*mm + npd + ny )
               d_p[id]=( qp*pow((iy-2*mm-npd-ny)/(1.0*npd),2) + 1 )*d_p[id];
            /*< left & right (0<x<nx) >*/
             if ( ix < npd )
               d_p[id]=( qp*pow((npd-ix)/(1.0*npd),2) + 1 )*d_p[id];
             else if ( ix >= 2*mm + npd + nx )
               d_p[id]=( qp*pow((ix-2*mm-npd-nx)/(1.0*npd),2) + 1 )*d_p[id];
            /*< up & down (0<z<nz) >*/
             if ( iz < npd )
               d_p[id]=( qp*pow((npd-iz)/(1.0*npd),2) + 1 )*d_p[id];
             else if ( iz >= 2*mm + npd + nz )
               d_p[id]=( qp*pow((iz-2*mm-npd-nz)/(1.0*npd),2) + 1 )*d_p[id];
        }
       
}
__global__ void cuda_record(float *p, float *seis, int *gxz, int ng)//++++++++++++
/*< record the seismogram at time it >*/
{
	int id=threadIdx.x+blockDim.x*blockIdx.x;
    	if (id<ng) seis[id]=p[gxz[id]];
}


__global__ void cuda_add_source(bool add, float *p, float *source, int *szxy, int ns)
/*< add/subtract sources: length of source[]=ns, index stored in szxy[] >*/
{
  int id=threadIdx.x+blockIdx.x*blockDim.x;

  if(id<ns){
    if(add){
      p[szxy[id]]+=source[id];
    }else{
      p[szxy[id]]-=source[id];
    }
  }
}

__global__ void cuda_step_fd3d(float *p0, float *p1, float *vv, float _dz2, float _dx2, float _dy2, int n1, int n2, int n3)
/*< step forward: 3-D FD, order=8 >*/
{
    bool validr = true;
    bool validw = true;
    const int gtid1 = blockIdx.x * blockDim.x + threadIdx.x;//0--nz's thread:iz
    const int gtid2 = blockIdx.y * blockDim.y + threadIdx.y;//0--nx's thread:ix
    const int ltid1 = threadIdx.x;//ithreadz
    const int ltid2 = threadIdx.y;//ithreadx
    const int work1 = blockDim.x;//nblockz
    const int work2 = blockDim.y;//nblockx
    __shared__ float tile[BlockSize2 + 2 * mm][BlockSize1 + 2 * mm];//tile[16+2*mm][16+2*mm]

    const int stride2 = n1 + 2 * mm + 2 * npd;//n1=nz
    const int stride3 = stride2 * (n2 + 2 * mm + 2 * npd);//n2=nx   stride3=(nz+2*mm)*(nx+2*mm)

    int inIndex = 0;
    int outIndex = 0;

    // Advance inputIndex to start of inner volume
    inIndex += (mm ) * stride2 + mm ;// inIndex=mm*(nz+2*mm+2*npd)+mm;

    // Advance inputIndex to target element
    inIndex += gtid2 * stride2 + gtid1; // inIndex=mm*(nz+2*mm)+mm+ix*(nz+2*mm+2*npd)+iz;:igrid

    float infront[mm];
    float behind[mm];
    float current;

    const int t1 = ltid1 + mm;
    const int t2 = ltid2 + mm;

    // Check in bounds
    if ((gtid1 >= n1 + mm + 2*npd) ||(gtid2 >= n2 + mm + 2*npd)) validr = false;
    if ((gtid1 >= n1 + 2*npd) ||(gtid2 >= n2 + 2*npd)) validw = false;

    // Preload the "infront" and "behind" data
    for (int i = mm -2 ; i >= 0 ; i--)//change 'mm-2' to 'mm-1'+++++++++++++++++++
    {
        if (validr) behind[i] = p1[inIndex];
        inIndex += stride3;//stride3=(nz+2*mm)*(nx+2*mm)
    }

    if (validr)	current = p1[inIndex];

    outIndex = inIndex;
    inIndex += stride3;//stride3=(nz+2*mm)*(nx+2*mm)

    for (int i = 0 ; i < mm ; i++)
    {
	if (validr) infront[i] = p1[inIndex];
        inIndex += stride3;//stride3=(nz+2*mm)*(nx+2*mm)
    }

    // Step through the zx-planes

    for (int i3 = mm ; i3 < n3 + 2*npd + mm ; i3++)
    {
        // Advance the slice (move the thread-front)
        for (int i = mm - 1 ; i > 0 ; i--) behind[i] = behind[i - 1];

        behind[0] = current;
        current = infront[0];

        for (int i = 0 ; i < mm - 1 ; i++) infront[i] = infront[i + 1];

        if (validr) infront[mm - 1] = p1[inIndex];

        inIndex += stride3;
        outIndex += stride3;
        __syncthreads();

        // Update the data slice in the local tile
        // Halo above & below
        if (ltid2 < mm)
        {
          /*   tile[ithread][ithread+mm]=p1[igrid - mm*(nz+2*mm)]  */
            tile[ltid2][t1]                  = p1[outIndex - mm * stride2];//t1 = ltid1 + mm;
            tile[ltid2 + work2 + mm][t1] = p1[outIndex + work2 * stride2];
        }

        // Halo left & right
        if (ltid1 < mm)
        {
            tile[t2][ltid1]                  = p1[outIndex - mm];
            tile[t2][ltid1 + work1 + mm] = p1[outIndex + work1];
        }

        tile[t2][t1] = current;
        __syncthreads();

        // Compute the output value
		float c1, c2, c3;
			c1=c2=c3=stencil[0]*current;        

        for (int i=1; i <= mm ; i++)
        {
			c1 +=stencil[i]*(tile[t2][t1-i]+ tile[t2][t1+i]);//z
			c2 +=stencil[i]*(tile[t2-i][t1]+ tile[t2+i][t1]);//x
			c3 +=stencil[i]*(infront[i-1]  + behind[i-1]  ); //y
        }
			c1*=_dz2;	
			c2*=_dx2;
			c3*=_dy2;
	
	
        if (validw) p0[outIndex]=2.0*p1[outIndex]-p0[outIndex]+vv[outIndex]*(c1+c2+c3);
    }

}

void velocity_transform(float *v0, float*vv, float dt, int n1, int n2, int n3)
 /*< velocit2 transform: vv=v0*dt; vv<--vv^2 >*/
{
  int i1, i2, i3, nn1, nn2, nn3;
  float tmp;

  nn1=n1+2*mm+2*npd;
  nn2=n2+2*mm+2*npd;
  nn3=n3+2*mm+2*npd;

  // inner zone
  for(i3=0; i3<n3; i3++){//y
    for(i2=0; i2<n2; i2++){//x
      for(i1=0; i1<n1; i1++){//z
	tmp=v0[i1+n1*i2+n1*n2*i3]*dt;
	vv[(i1+mm+npd)+nn1*(i2+mm+npd)+nn1*nn2*(i3+mm+npd)]=tmp*tmp;
      }
    }
  }  
    //top & down 
    for(i3=0; i3<nn3; i3++){//y
	    for(i2=0; i2<nn2; i2++){//x
	        for (i1=0; i1<mm+npd; i1++){//z
		    vv[i1+nn1*i2+nn1*nn2*i3]=vv[mm+npd+nn1*i2+nn1*nn2*i3];
		    vv[(nn1-i1-1)+nn1*i2+nn1*nn2*i3]=vv[(nn1-mm-npd-1)+nn1*i2+nn1*nn2*i3];
	        }
	    }
    }

    //left & right
    for(i3=0; i3<nn3; i3++){//y
	    for(i2=0; i2<mm+npd; i2++){//x
	        for (i1=0; i1<nn1; i1++){//z
		    vv[i1+nn1*i2+nn1*nn2*i3]=vv[i1+nn1*(mm+npd)+nn1*nn2*i3];
		    vv[i1+nn1*(nn2-i2-1)+nn1*nn2*i3]=vv[i1+nn1*(nn2-mm-npd-1)+nn1*nn2*i3];
	        }
	    }
    }
    //front & back
    for(i3=0; i3<mm+npd; i3++){//y
	    for(i2=0; i2<nn2; i2++){//x
	        for(i1=0; i1<nn1; i1++){//z
		    vv[i1+nn1*i2+nn1*nn2*i3]=vv[i1+nn1*i2+nn1*nn2*(mm+npd)];
		    vv[i1+nn1*i2+nn1*nn2*(nn3-1-i3)]=vv[i1+nn1*i2+nn1*nn2*(nn3-mm-npd-1)];
	        }
	    }
    }
}


void window3d(float *a, float *b, int n1, int n2, int n3)
/*< window a 3d subvolume >*/
{
	int i1, i2, i3, nn1, nn2;
	nn1=n1+2*mm+ 2*npd;//z
	nn2=n2+2*mm+ 2*npd;//x
	
	for(i3=0; i3<n3; i3++)
	for(i2=0; i2<n2; i2++)
	for(i1=0; i1<n1; i1++)
	{
		a[i1+n1*i2+n1*n2*i3]=b[(i1+mm+npd)+nn1*(i2+mm+npd)+nn1*nn2*(i3+mm+npd)];
	}
}


extern "C"  void cuda_3dfd(char* FNvel, char *FNsnap, char* FNshot, int is, int ns, int nx, int ny, int nz, 
                          float dx, float dy, float dz, int sxbeg, int sybeg, int szbeg, int jsx, int jsy, int jsz, 
                          float dgx, float dgy, float dgt,
                          int nt, int kt, float dt, float fm, bool show_snapshot, bool cut_directwave,
                          int snap_interval, int cudaDevicei)
{
    FILE *fpvel, *fpsnap, *fpshot;
	int  nnz, nnx, nny, it, ng, ix, iy;
	int *d_szxy,*d_gzxy;
	float _dz2, _dx2, _dy2;
	float *v0, *vv, *d_wlt, *d_vv, *d_p0, *d_p1, *ptr;
    float *d_dcal_device_xy,*d_dcal_device_txy,*d_dcal_host;
    char snapname[300], snapid[300];
    float v_cut_directwave;
    cudaError_t error;
    
    fpvel = fopen(FNvel,"rb");
    fpshot = fopen(FNshot, "wb");

	clock_t t0, t1;
	

	_dz2=1.0/(dz*dz);
	_dx2=1.0/(dx*dx);
	_dy2=1.0/(dy*dy);

	nnz=nz+2*mm+2*npd;
	nnx=nx+2*mm+2*npd;
	nny=ny+2*mm+2*npd;

    ng=nx*ny;

    v0=(float*)malloc(nz*nx*ny*sizeof(float));
    vv=(float*)malloc(nnz*nnx*nny*sizeof(float));
    d_dcal_host=(float*)malloc(ng*nt*sizeof(float));

    fread(v0, sizeof(float), nz*nx*ny, fpvel);
    velocity_transform(v0, vv, dt, nz, nx, ny);
    v_cut_directwave = v0[1];

    cudaSetDevice(cudaDevicei);
	check_gpu_error("CUDA:Failed to initialize device!");

	dim3 Xdimg, dimg, dimb;
	dimg.x=(nz+2*npd+2*mm+BlockSize1-1)/BlockSize1;
	dimg.y=(nx+2*npd+2*mm+BlockSize2-1)/BlockSize2;
	dimb.x=BlockSize1;
	dimb.y=BlockSize2;
    Xdimg.x = (nnx + BlockSize1 - 1) / BlockSize1;
    Xdimg.y = (nny + BlockSize2 - 1) / BlockSize2;

	/* allocate memory on device */
	cudaMalloc(&d_wlt, nt*sizeof(float));
	cudaMalloc(&d_vv, nnz*nnx*nny*sizeof(float));
	cudaMalloc(&d_p0, nnz*nnx*nny*sizeof(float));
	cudaMalloc(&d_p1, nnz*nnx*nny*sizeof(float));
	cudaMalloc(&d_szxy, ns*sizeof(int));
	cudaMalloc(&d_gzxy, ng*sizeof(int));
	cudaMalloc(&d_dcal_device_xy, ng*sizeof(float));	
	cudaMalloc(&d_dcal_device_txy, ng*nt*sizeof(float)); 

	check_gpu_error("CUDA: Failed to allocate memory for variables!");

	cuda_ricker_wavelet<<<(nt+511)/512, 512>>>(d_wlt, fm, dt, nt);
	cudaMemcpy(d_vv, vv, nnz*nnx*nny*sizeof(float), cudaMemcpyHostToDevice);
	cuda_set_s<<<1, ns>>>(d_szxy, szbeg, sxbeg, sybeg, jsz, jsx, jsy, ns, nz, nx, ny);
	cuda_set_g<<<(ng+511)/512,512>>>(d_gzxy, ng, nz, nx, ny);

	cudaMemset(d_p0, 0, nnz*nnx*nny*sizeof(float));
	cudaMemset(d_p1, 0, nnz*nnx*nny*sizeof(float));
	cudaMemset(d_dcal_device_xy, 0, ng*sizeof(float));
	cudaMemset(d_dcal_device_txy, 0, ng*nt*sizeof(float));

       t0 = clock();
	for(it=0; it<nt; it++)
        {
         if(it%200==0)printf("   cuda: is=%2d >> it= %d\n",is,it);
	  cuda_add_source<<<1,1>>>(true, d_p1, &d_wlt[it], &d_szxy[is], 1);
	  cuda_step_fd3d<<<dimg,dimb>>>(d_p0, d_p1, d_vv, _dz2, _dx2, _dy2, nz, nx, ny);
	  ptr=d_p0; 
	  d_p0=d_p1; 
	  d_p1=ptr;

     cuda_absorb_bndr<<<dimg,dimb>>>(d_p0, nz, nx, ny, -0.25);
     cuda_absorb_bndr<<<dimg,dimb>>>(d_p1, nz, nx, ny, -0.25);

	  cuda_record<<<(ng+511)/512, 512>>>(d_p0, d_dcal_device_xy, d_gzxy, ng);
         cuda_trans_xy2txy<<<(ng+511)/512, 512>>>(d_dcal_device_xy, d_dcal_device_txy, it, nt, ng);
        
        if(show_snapshot) {
	        if(it % snap_interval == 0){
                cudaMemcpy(vv, d_p0, nnz*nnx*nny*sizeof(float), cudaMemcpyDeviceToHost);
                strcpy(snapname,FNsnap);
                sprintf(snapid,"ishot_%d_",is);
                strcat(snapname,snapid);
                sprintf(snapid,"it_%d",it);
                strcat(snapname,snapid);
                strcat(snapname,".bin");
                if((fpsnap=fopen(snapname,"wb"))==NULL){
                    printf("cannot write snapfile\n");
                }
                window3d(v0, vv, nz, nx, ny);
                fwrite(v0, sizeof(float),nz*nx*ny, fpsnap);
                fclose(fpsnap);
            }
        }
	 }
	 t1 = clock();
    if (cut_directwave) {
        printf("cutting direct wave\n");
        mute_directwave<<<Xdimg,dimb>>>(nx,ny,nt,dt,fm,dx,dy,dz,sxbeg,sybeg,jsx,jsy,szbeg,is,v_cut_directwave,d_dcal_device_txy,60,ns);
    }
    
	 printf("   cudafd:%.3f (s)\n", ((float)(t1-t0))/CLOCKS_PER_SEC); 

       t0 = clock();
	cudaMemcpy(d_dcal_host, d_dcal_device_txy, ng*nt*sizeof(float), cudaMemcpyDeviceToHost);
       fseek(fpshot,is*ng*nt*sizeof(float),0);
        for(iy=0;iy<ny;iy+=int(dgy/dy)){
            for(ix=0;ix<nx;ix+=int(dgx/dx)){
                  for(it=0;it<nt;it+=int(dgt/dt)){
                        fwrite(&d_dcal_host[iy*nx*nt+ix*nt+it],sizeof(float),1,fpshot);
                  }
            }
        }
       t1 = clock();
 	
 	printf("   cuda:save P(r,t):is=%d, %.3f (s), P=%f\n", is, ((float)(t1-t0))/CLOCKS_PER_SEC, d_dcal_host[200]); 

	/* free memory on device */
	cudaFree(d_wlt);
	cudaFree(d_vv);
	cudaFree(d_p0);
	cudaFree(d_p1);
	cudaFree(d_szxy);
	cudaFree(d_gzxy);
	cudaFree(d_dcal_device_xy);
	cudaFree(d_dcal_device_txy);
	free(v0);
	free(vv);
	free(d_dcal_host);

}

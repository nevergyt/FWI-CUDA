#include "fwi_propagator.cuh"


integer IDX(const integer z,
            const integer x,
            const integer y,
            const integer dimmz,
            const integer dimmx) {
    return (y * dimmx * dimmz) + (x * dimmz) + (z);
};

__device__
integer IDX_GPU(const integer z,
                const integer x,
                const integer y,
                const integer dimmz,
                const integer dimmx) {
    return (y * dimmx * dimmz) + (x * dimmz) + (z);
};


real stencil_Z(const offset_t off,
               const real *__restrict__ ptr,
               const real dzi,
               const integer z,
               const integer x,
               const integer y,
               const integer dimmz,
               const integer dimmx) {
    return ((C0 * (ptr[IDX(z + off, x, y, dimmz, dimmx)] - ptr[IDX(z - 1 + off, x, y, dimmz, dimmx)]) +
             C1 * (ptr[IDX(z + 1 + off, x, y, dimmz, dimmx)] - ptr[IDX(z - 2 + off, x, y, dimmz, dimmx)]) +
             C2 * (ptr[IDX(z + 2 + off, x, y, dimmz, dimmx)] - ptr[IDX(z - 3 + off, x, y, dimmz, dimmx)]) +
             C3 * (ptr[IDX(z + 3 + off, x, y, dimmz, dimmx)] - ptr[IDX(z - 4 + off, x, y, dimmz, dimmx)])) * dzi);
};

__device__
real stencil_Z_GPU(const offset_t off,
                   const real *__restrict__ ptr,
                   const real dzi,
                   const integer z,
                   const integer x,
                   const integer y,
                   const integer dimmz,
                   const integer dimmx) {
    return ((C0 * (ptr[IDX_GPU(z + off, x, y, dimmz, dimmx)] - ptr[IDX_GPU(z - 1 + off, x, y, dimmz, dimmx)]) +
             C1 * (ptr[IDX_GPU(z + 1 + off, x, y, dimmz, dimmx)] - ptr[IDX_GPU(z - 2 + off, x, y, dimmz, dimmx)]) +
             C2 * (ptr[IDX_GPU(z + 2 + off, x, y, dimmz, dimmx)] - ptr[IDX_GPU(z - 3 + off, x, y, dimmz, dimmx)]) +
             C3 * (ptr[IDX_GPU(z + 3 + off, x, y, dimmz, dimmx)] - ptr[IDX_GPU(z - 4 + off, x, y, dimmz, dimmx)])) *
            dzi);
};

real stencil_X(const offset_t off,
               const real *__restrict__ ptr,
               const real dxi,
               const integer z,
               const integer x,
               const integer y,
               const integer dimmz,
               const integer dimmx) {
    return ((C0 * (ptr[IDX(z, x + off, y, dimmz, dimmx)] - ptr[IDX(z, x - 1 + off, y, dimmz, dimmx)]) +
             C1 * (ptr[IDX(z, x + 1 + off, y, dimmz, dimmx)] - ptr[IDX(z, x - 2 + off, y, dimmz, dimmx)]) +
             C2 * (ptr[IDX(z, x + 2 + off, y, dimmz, dimmx)] - ptr[IDX(z, x - 3 + off, y, dimmz, dimmx)]) +
             C3 * (ptr[IDX(z, x + 3 + off, y, dimmz, dimmx)] - ptr[IDX(z, x - 4 + off, y, dimmz, dimmx)])) * dxi);
};

__device__
real stencil_X_GPU(const offset_t off,
                   const real *__restrict__ ptr,
                   const real dxi,
                   const integer z,
                   const integer x,
                   const integer y,
                   const integer dimmz,
                   const integer dimmx) {
    return ((C0 * (ptr[IDX_GPU(z, x + off, y, dimmz, dimmx)] - ptr[IDX_GPU(z, x - 1 + off, y, dimmz, dimmx)]) +
             C1 * (ptr[IDX_GPU(z, x + 1 + off, y, dimmz, dimmx)] - ptr[IDX_GPU(z, x - 2 + off, y, dimmz, dimmx)]) +
             C2 * (ptr[IDX_GPU(z, x + 2 + off, y, dimmz, dimmx)] - ptr[IDX_GPU(z, x - 3 + off, y, dimmz, dimmx)]) +
             C3 * (ptr[IDX_GPU(z, x + 3 + off, y, dimmz, dimmx)] - ptr[IDX_GPU(z, x - 4 + off, y, dimmz, dimmx)])) *
            dxi);
};

real stencil_Y(const offset_t off,
               const real *__restrict__ ptr,
               const real dyi,
               const integer z,
               const integer x,
               const integer y,
               const integer dimmz,
               const integer dimmx) {
    return ((C0 * (ptr[IDX(z, x, y + off, dimmz, dimmx)] - ptr[IDX(z, x, y - 1 + off, dimmz, dimmx)]) +
             C1 * (ptr[IDX(z, x, y + 1 + off, dimmz, dimmx)] - ptr[IDX(z, x, y - 2 + off, dimmz, dimmx)]) +
             C2 * (ptr[IDX(z, x, y + 2 + off, dimmz, dimmx)] - ptr[IDX(z, x, y - 3 + off, dimmz, dimmx)]) +
             C3 * (ptr[IDX(z, x, y + 3 + off, dimmz, dimmx)] - ptr[IDX(z, x, y - 4 + off, dimmz, dimmx)])) * dyi);
};

__device__
real stencil_Y_GPU(const offset_t off,
                   const real *__restrict__ ptr,
                   const real dyi,
                   const integer z,
                   const integer x,
                   const integer y,
                   const integer dimmz,
                   const integer dimmx) {
    return ((C0 * (ptr[IDX_GPU(z, x, y + off, dimmz, dimmx)] - ptr[IDX_GPU(z, x, y - 1 + off, dimmz, dimmx)]) +
             C1 * (ptr[IDX_GPU(z, x, y + 1 + off, dimmz, dimmx)] - ptr[IDX_GPU(z, x, y - 2 + off, dimmz, dimmx)]) +
             C2 * (ptr[IDX_GPU(z, x, y + 2 + off, dimmz, dimmx)] - ptr[IDX_GPU(z, x, y - 3 + off, dimmz, dimmx)]) +
             C3 * (ptr[IDX_GPU(z, x, y + 3 + off, dimmz, dimmx)] - ptr[IDX_GPU(z, x, y - 4 + off, dimmz, dimmx)])) *
            dyi);
};

/* -------------------------------------------------------------------- */
/*                     KERNELS FOR VELOCITY                             */
/* -------------------------------------------------------------------- */


real rho_BL(const real *__restrict__ rho,
            const integer z,
            const integer x,
            const integer y,
            const integer dimmz,
            const integer dimmx) {
    return (2.0f / (rho[IDX(z, x, y, dimmz, dimmx)] + rho[IDX(z + 1, x, y, dimmz, dimmx)]));
};

__device__
real rho_BL_GPU(const real *__restrict__ rho,
                const integer z,
                const integer x,
                const integer y,
                const integer dimmz,
                const integer dimmx) {
    return (2.0f / (rho[IDX_GPU(z, x, y, dimmz, dimmx)] + rho[IDX_GPU(z + 1, x, y, dimmz, dimmx)]));
};

real rho_TR(const real *__restrict__ rho,
            const integer z,
            const integer x,
            const integer y,
            const integer dimmz,
            const integer dimmx) {
    return (2.0f / (rho[IDX(z, x, y, dimmz, dimmx)] + rho[IDX(z, x + 1, y, dimmz, dimmx)]));
};

__device__
real rho_TR_GPU(const real *__restrict__ rho,
                const integer z,
                const integer x,
                const integer y,
                const integer dimmz,
                const integer dimmx) {
    return (2.0f / (rho[IDX_GPU(z, x, y, dimmz, dimmx)] + rho[IDX_GPU(z, x + 1, y, dimmz, dimmx)]));
};

real rho_BR(const real *__restrict__ rho,
            const integer z,
            const integer x,
            const integer y,
            const integer dimmz,
            const integer dimmx) {
    return (8.0f / (rho[IDX(z, x, y, dimmz, dimmx)] +
                    rho[IDX(z + 1, x, y, dimmz, dimmx)] +
                    rho[IDX(z, x + 1, y, dimmz, dimmx)] +
                    rho[IDX(z, x, y + 1, dimmz, dimmx)] +
                    rho[IDX(z, x + 1, y + 1, dimmz, dimmx)] +
                    rho[IDX(z + 1, x + 1, y, dimmz, dimmx)] +
                    rho[IDX(z + 1, x, y + 1, dimmz, dimmx)] +
                    rho[IDX(z + 1, x + 1, y + 1, dimmz, dimmx)]));
};

__device__
real rho_BR_GPU(const real *__restrict__ rho,
                const integer z,
                const integer x,
                const integer y,
                const integer dimmz,
                const integer dimmx) {
    return (8.0f / (rho[IDX_GPU(z, x, y, dimmz, dimmx)] +
                    rho[IDX_GPU(z + 1, x, y, dimmz, dimmx)] +
                    rho[IDX_GPU(z, x + 1, y, dimmz, dimmx)] +
                    rho[IDX_GPU(z, x, y + 1, dimmz, dimmx)] +
                    rho[IDX_GPU(z, x + 1, y + 1, dimmz, dimmx)] +
                    rho[IDX_GPU(z + 1, x + 1, y, dimmz, dimmx)] +
                    rho[IDX_GPU(z + 1, x, y + 1, dimmz, dimmx)] +
                    rho[IDX_GPU(z + 1, x + 1, y + 1, dimmz, dimmx)]));
};

real rho_TL(const real *__restrict__ rho,
            const integer z,
            const integer x,
            const integer y,
            const integer dimmz,
            const integer dimmx) {
    return (2.0f / (rho[IDX(z, x, y, dimmz, dimmx)] + rho[IDX(z, x, y + 1, dimmz, dimmx)]));
};

__device__
real rho_TL_GPU(const real *__restrict__ rho,
                const integer z,
                const integer x,
                const integer y,
                const integer dimmz,
                const integer dimmx) {
    return (2.0f / (rho[IDX_GPU(z, x, y, dimmz, dimmx)] + rho[IDX_GPU(z, x, y + 1, dimmz, dimmx)]));
};

void compute_component_vcell_TL(real *__restrict__ vptr,
                                const real *__restrict__ szptr,
                                const real *__restrict__ sxptr,
                                const real *__restrict__ syptr,
                                const real *__restrict__ rho,
                                const real dt,
                                const real dzi,
                                const real dxi,
                                const real dyi,
                                const integer nz0,
                                const integer nzf,
                                const integer nx0,
                                const integer nxf,
                                const integer ny0,
                                const integer nyf,
                                const offset_t _SZ,
                                const offset_t _SX,
                                const offset_t _SY,
                                const integer dimmz,
                                const integer dimmx) {
    __assume(nz0 % HALO == 0);
    __assume(nzf % HALO == 0);

//          real* __restrict__ _vptr  __attribute__ ((aligned (64))) = vptr;
//    const real* __restrict__ _szptr __attribute__ ((aligned (64))) = szptr;
//    const real* __restrict__ _sxptr __attribute__ ((aligned (64))) = sxptr;
//    const real* __restrict__ _syptr __attribute__ ((aligned (64))) = syptr;

    real *__restrict__ _vptr = vptr;
    const real *__restrict__ _szptr = szptr;
    const real *__restrict__ _sxptr = sxptr;
    const real *__restrict__ _syptr = syptr;

#pragma omp parallel for
    for (integer y = ny0; y < nyf; y++) {
        for (integer x = nx0; x < nxf; x++) {
#pragma omp simd
            for (integer z = nz0; z < nzf; z++) {
                const real lrho = rho_TL(rho, z, x, y, dimmz, dimmx);

                const real stx = stencil_X(_SX, _sxptr, dxi, z, x, y, dimmz, dimmx);
                const real sty = stencil_Y(_SY, _syptr, dyi, z, x, y, dimmz, dimmx);
                const real stz = stencil_Z(_SZ, _szptr, dzi, z, x, y, dimmz, dimmx);

                _vptr[IDX(z, x, y, dimmz, dimmx)] += (stx + sty + stz) * dt * lrho;
            }
        }
    }
};

__global__
void compute_component_vcell_TL_GPU(real *__restrict__ vptr,
                                    const real *__restrict__ szptr,
                                    const real *__restrict__ sxptr,
                                    const real *__restrict__ syptr,
                                    const real *__restrict__ rho,
                                    const real dt,
                                    const real dzi,
                                    const real dxi,
                                    const real dyi,
                                    const integer nz0,
                                    const integer nzf,
                                    const integer nx0,
                                    const integer nxf,
                                    const integer ny0,
                                    const integer nyf,
                                    const offset_t _SZ,
                                    const offset_t _SX,
                                    const offset_t _SY,
                                    const integer dimmz,
                                    const integer dimmx) {
//    __assume(nz0 % HALO == 0);
//    __assume(nzf % HALO == 0);

//          real* __restrict__ _vptr  __attribute__ ((aligned (64))) = vptr;
//    const real* __restrict__ _szptr __attribute__ ((aligned (64))) = szptr;
//    const real* __restrict__ _sxptr __attribute__ ((aligned (64))) = sxptr;
//    const real* __restrict__ _syptr __attribute__ ((aligned (64))) = syptr;

    real *__restrict__ _vptr = vptr;
    const real *__restrict__ _szptr = szptr;
    const real *__restrict__ _sxptr = sxptr;
    const real *__restrict__ _syptr = syptr;

    int i = blockIdx.x * blockDim.x + threadIdx.x + nx0;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ny0;
    int z = blockIdx.z * blockDim.z + threadIdx.z + nz0;

    if (j < nyf && i < nxf && z < nzf) {
        const real lrho = rho_TL_GPU(rho, z, i, j, dimmz, dimmx);
        const real stx = stencil_X_GPU(_SX, _sxptr, dxi, z, i, j, dimmz, dimmx);
        const real sty = stencil_Y_GPU(_SY, _syptr, dyi, z, i, j, dimmz, dimmx);
        const real stz = stencil_Z_GPU(_SZ, _szptr, dzi, z, i, j, dimmz, dimmx);
        _vptr[IDX_GPU(z, i, j, dimmz, dimmx)] += (stx + sty + stz) * dt * lrho;
    }

};

void compute_component_vcell_TR(real *__restrict__ vptr,
                                const real *__restrict__ szptr,
                                const real *__restrict__ sxptr,
                                const real *__restrict__ syptr,
                                const real *__restrict__ rho,
                                const real dt,
                                const real dzi,
                                const real dxi,
                                const real dyi,
                                const integer nz0,
                                const integer nzf,
                                const integer nx0,
                                const integer nxf,
                                const integer ny0,
                                const integer nyf,
                                const offset_t _SZ,
                                const offset_t _SX,
                                const offset_t _SY,
                                const integer dimmz,
                                const integer dimmx) {
    __assume(nz0 % HALO == 0);
    __assume(nzf % HALO == 0);

//          real* __restrict__ _vptr  __attribute__ ((aligned (64))) = vptr ;
//    const real* __restrict__ _szptr __attribute__ ((aligned (64))) = szptr;
//    const real* __restrict__ _sxptr __attribute__ ((aligned (64))) = sxptr;
//    const real* __restrict__ _syptr __attribute__ ((aligned (64))) = syptr;

    real *__restrict__ _vptr = vptr;
    const real *__restrict__ _szptr = szptr;
    const real *__restrict__ _sxptr = sxptr;
    const real *__restrict__ _syptr = syptr;

#pragma omp parallel for
    for (integer y = ny0; y < nyf; y++) {
        for (integer x = nx0; x < nxf; x++) {
#ifdef __INTEL_COMPILER
#pragma simd
#else
#pragma omp simd
#endif
            for (integer z = nz0; z < nzf; z++) {
                const real lrho = rho_TR(rho, z, x, y, dimmz, dimmx);

                const real stx = stencil_X(_SX, _sxptr, dxi, z, x, y, dimmz, dimmx);
                const real sty = stencil_Y(_SY, _syptr, dyi, z, x, y, dimmz, dimmx);
                const real stz = stencil_Z(_SZ, _szptr, dzi, z, x, y, dimmz, dimmx);

                _vptr[IDX(z, x, y, dimmz, dimmx)] += (stx + sty + stz) * dt * lrho;
            }
        }
    }
};

__global__
void compute_component_vcell_TR_GPU(real *__restrict__ vptr,
                                    const real *__restrict__ szptr,
                                    const real *__restrict__ sxptr,
                                    const real *__restrict__ syptr,
                                    const real *__restrict__ rho,
                                    const real dt,
                                    const real dzi,
                                    const real dxi,
                                    const real dyi,
                                    const integer nz0,
                                    const integer nzf,
                                    const integer nx0,
                                    const integer nxf,
                                    const integer ny0,
                                    const integer nyf,
                                    const offset_t _SZ,
                                    const offset_t _SX,
                                    const offset_t _SY,
                                    const integer dimmz,
                                    const integer dimmx) {
//    __assume(nz0 % HALO == 0);
//    __assume(nzf % HALO == 0);

//          real* __restrict__ _vptr  __attribute__ ((aligned (64))) = vptr ;
//    const real* __restrict__ _szptr __attribute__ ((aligned (64))) = szptr;
//    const real* __restrict__ _sxptr __attribute__ ((aligned (64))) = sxptr;
//    const real* __restrict__ _syptr __attribute__ ((aligned (64))) = syptr;

    real *__restrict__ _vptr = vptr;
    const real *__restrict__ _szptr = szptr;
    const real *__restrict__ _sxptr = sxptr;
    const real *__restrict__ _syptr = syptr;

    int x = blockIdx.x * blockDim.x + threadIdx.x + nx0;
    int y = blockIdx.y * blockDim.y + threadIdx.y + ny0;
    int z = blockIdx.z * blockDim.z + threadIdx.z + nz0;

    if (x < nxf && y < nyf && z < nzf) {
        const real lrho = rho_TR_GPU(rho, z, x, y, dimmz, dimmx);

        const real stx = stencil_X_GPU(_SX, _sxptr, dxi, z, x, y, dimmz, dimmx);
        const real sty = stencil_Y_GPU(_SY, _syptr, dyi, z, x, y, dimmz, dimmx);
        const real stz = stencil_Z_GPU(_SZ, _szptr, dzi, z, x, y, dimmz, dimmx);

        _vptr[IDX_GPU(z, x, y, dimmz, dimmx)] += (stx + sty + stz) * dt * lrho;
    }

};

void compute_component_vcell_BR(real *__restrict__ vptr,
                                const real *__restrict__ szptr,
                                const real *__restrict__ sxptr,
                                const real *__restrict__ syptr,
                                const real *__restrict__ rho,
                                const real dt,
                                const real dzi,
                                const real dxi,
                                const real dyi,
                                const integer nz0,
                                const integer nzf,
                                const integer nx0,
                                const integer nxf,
                                const integer ny0,
                                const integer nyf,
                                const offset_t _SZ,
                                const offset_t _SX,
                                const offset_t _SY,
                                const integer dimmz,
                                const integer dimmx) {
    __assume(nz0 % HALO == 0);
    __assume(nzf % HALO == 0);

//          real* __restrict__ _vptr  __attribute__ ((aligned (64))) = vptr ;
//    const real* __restrict__ _szptr __attribute__ ((aligned (64))) = szptr;
//    const real* __restrict__ _sxptr __attribute__ ((aligned (64))) = sxptr;
//    const real* __restrict__ _syptr __attribute__ ((aligned (64))) = syptr;

    real *__restrict__ _vptr = vptr;
    const real *__restrict__ _szptr = szptr;
    const real *__restrict__ _sxptr = sxptr;
    const real *__restrict__ _syptr = syptr;

#pragma omp parallel for
    for (integer y = ny0; y < nyf; y++) {
        for (integer x = nx0; x < nxf; x++) {
#ifdef __INTEL_COMPILER
#pragma simd
#else
#pragma omp simd
#endif
            for (integer z = nz0; z < nzf; z++) {
                const real lrho = rho_BR(rho, z, x, y, dimmz, dimmx);

                const real stx = stencil_X(_SX, _sxptr, dxi, z, x, y, dimmz, dimmx);
                const real sty = stencil_Y(_SY, _syptr, dyi, z, x, y, dimmz, dimmx);
                const real stz = stencil_Z(_SZ, _szptr, dzi, z, x, y, dimmz, dimmx);

                _vptr[IDX(z, x, y, dimmz, dimmx)] += (stx + sty + stz) * dt * lrho;
            }
        }
    }
};

__global__
void compute_component_vcell_BR_GPU(real *__restrict__ vptr,
                                    const real *__restrict__ szptr,
                                    const real *__restrict__ sxptr,
                                    const real *__restrict__ syptr,
                                    const real *__restrict__ rho,
                                    const real dt,
                                    const real dzi,
                                    const real dxi,
                                    const real dyi,
                                    const integer nz0,
                                    const integer nzf,
                                    const integer nx0,
                                    const integer nxf,
                                    const integer ny0,
                                    const integer nyf,
                                    const offset_t _SZ,
                                    const offset_t _SX,
                                    const offset_t _SY,
                                    const integer dimmz,
                                    const integer dimmx) {
//    __assume(nz0 % HALO == 0);
//    __assume(nzf % HALO == 0);

//          real* __restrict__ _vptr  __attribute__ ((aligned (64))) = vptr ;
//    const real* __restrict__ _szptr __attribute__ ((aligned (64))) = szptr;
//    const real* __restrict__ _sxptr __attribute__ ((aligned (64))) = sxptr;
//    const real* __restrict__ _syptr __attribute__ ((aligned (64))) = syptr;

    real *__restrict__ _vptr = vptr;
    const real *__restrict__ _szptr = szptr;
    const real *__restrict__ _sxptr = sxptr;
    const real *__restrict__ _syptr = syptr;

    int x = blockIdx.x * blockDim.x + threadIdx.x + nx0;
    int y = blockIdx.y * blockDim.y + threadIdx.y + ny0;
    int z = blockIdx.z * blockDim.z + threadIdx.z + nz0;

    if (x < nxf && y < nyf && z < nzf){
        const real lrho = rho_BR_GPU(rho, z, x, y, dimmz, dimmx);

        const real stx = stencil_X_GPU(_SX, _sxptr, dxi, z, x, y, dimmz, dimmx);
        const real sty = stencil_Y_GPU(_SY, _syptr, dyi, z, x, y, dimmz, dimmx);
        const real stz = stencil_Z_GPU(_SZ, _szptr, dzi, z, x, y, dimmz, dimmx);

        _vptr[IDX_GPU(z, x, y, dimmz, dimmx)] += (stx + sty + stz) * dt * lrho;
    }

};

void compute_component_vcell_BL(real *__restrict__ vptr,
                                const real *__restrict__ szptr,
                                const real *__restrict__ sxptr,
                                const real *__restrict__ syptr,
                                const real *__restrict__ rho,
                                const real dt,
                                const real dzi,
                                const real dxi,
                                const real dyi,
                                const integer nz0,
                                const integer nzf,
                                const integer nx0,
                                const integer nxf,
                                const integer ny0,
                                const integer nyf,
                                const offset_t _SZ,
                                const offset_t _SX,
                                const offset_t _SY,
                                const integer dimmz,
                                const integer dimmx) {
    __assume(nz0 % HALO == 0);
    __assume(nzf % HALO == 0);

//          real* __restrict__ _vptr  __attribute__ ((aligned (64))) = vptr ;
//    const real* __restrict__ _szptr __attribute__ ((aligned (64))) = szptr;
//    const real* __restrict__ _sxptr __attribute__ ((aligned (64))) = sxptr;
//    const real* __restrict__ _syptr __attribute__ ((aligned (64))) = syptr;

    real *__restrict__ _vptr = vptr;
    const real *__restrict__ _szptr = szptr;
    const real *__restrict__ _sxptr = sxptr;
    const real *__restrict__ _syptr = syptr;


#pragma omp parallel for
    for (integer y = ny0; y < nyf; y++) {
        for (integer x = nx0; x < nxf; x++) {
#ifdef __INTEL_COMPILER
#pragma simd
#else
#pragma omp simd
#endif
            for (integer z = nz0; z < nzf; z++) {
                const real lrho = rho_BL(rho, z, x, y, dimmz, dimmx);

                const real stx = stencil_X(_SX, _sxptr, dxi, z, x, y, dimmz, dimmx);
                const real sty = stencil_Y(_SY, _syptr, dyi, z, x, y, dimmz, dimmx);
                const real stz = stencil_Z(_SZ, _szptr, dzi, z, x, y, dimmz, dimmx);

                _vptr[IDX(z, x, y, dimmz, dimmx)] += (stx + sty + stz) * dt * lrho;
            }
        }
    }
};

__global__
void compute_component_vcell_BL_GPU(real *__restrict__ vptr,
                                const real *__restrict__ szptr,
                                const real *__restrict__ sxptr,
                                const real *__restrict__ syptr,
                                const real *__restrict__ rho,
                                const real dt,
                                const real dzi,
                                const real dxi,
                                const real dyi,
                                const integer nz0,
                                const integer nzf,
                                const integer nx0,
                                const integer nxf,
                                const integer ny0,
                                const integer nyf,
                                const offset_t _SZ,
                                const offset_t _SX,
                                const offset_t _SY,
                                const integer dimmz,
                                const integer dimmx) {
//    __assume(nz0 % HALO == 0);
//    __assume(nzf % HALO == 0);

//          real* __restrict__ _vptr  __attribute__ ((aligned (64))) = vptr ;
//    const real* __restrict__ _szptr __attribute__ ((aligned (64))) = szptr;
//    const real* __restrict__ _sxptr __attribute__ ((aligned (64))) = sxptr;
//    const real* __restrict__ _syptr __attribute__ ((aligned (64))) = syptr;

    real *__restrict__ _vptr = vptr;
    const real *__restrict__ _szptr = szptr;
    const real *__restrict__ _sxptr = sxptr;
    const real *__restrict__ _syptr = syptr;

    int x = blockIdx.x * blockDim.x + threadIdx.x + nx0;
    int y = blockIdx.y * blockDim.y + threadIdx.y + ny0;
    int z = blockIdx.z * blockDim.z + threadIdx.z + nz0;

    if (x < nxf && y < nyf && z < nzf){
        const real lrho = rho_BL_GPU(rho, z, x, y, dimmz, dimmx);

        const real stx = stencil_X_GPU(_SX, _sxptr, dxi, z, x, y, dimmz, dimmx);
        const real sty = stencil_Y_GPU(_SY, _syptr, dyi, z, x, y, dimmz, dimmx);
        const real stz = stencil_Z_GPU(_SZ, _szptr, dzi, z, x, y, dimmz, dimmx);

        _vptr[IDX_GPU(z, x, y, dimmz, dimmx)] += (stx + sty + stz) * dt * lrho;
    }
};

void velocity_propagator(v_t v,
                         s_t s,
                         coeff_t coeffs,
                         real *rho,
                         v_t           gpu_v,
                         s_t           gpu_s,
                         coeff_t       gpu_coeffs,
                         real          *gpu_rho,
                         const real dt,
                         const real dzi,
                         const real dxi,
                         const real dyi,
                         const integer nz0,
                         const integer nzf,
                         const integer nx0,
                         const integer nxf,
                         const integer ny0,
                         const integer nyf,
                         const integer dimmz,
                         const integer dimmx) {
    print_debug("Integration limits are (z "I"-"I",x "I"-"I",y "I"-"I")\n", nz0, nzf, nx0, nxf, ny0, nyf);

#ifdef __INTEL_COMPILER
#pragma forceinline recursive
#endif
    {

        //cpu
        compute_component_vcell_TL (v.tl.w, s.bl.zz, s.tr.xz, s.tl.yz, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, back_offset, back_offset, forw_offset, dimmz, dimmx);
        compute_component_vcell_TR (v.tr.w, s.br.zz, s.tl.xz, s.tr.yz, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, back_offset, forw_offset, back_offset, dimmz, dimmx);
        compute_component_vcell_BL (v.bl.w, s.tl.zz, s.br.xz, s.bl.yz, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, forw_offset, back_offset, back_offset, dimmz, dimmx);
        compute_component_vcell_BR (v.br.w, s.tr.zz, s.bl.xz, s.br.yz, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, forw_offset, forw_offset, forw_offset, dimmz, dimmx);
        compute_component_vcell_TL (v.tl.u, s.bl.xz, s.tr.xx, s.tl.xy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, back_offset, back_offset, forw_offset, dimmz, dimmx);
        compute_component_vcell_TR (v.tr.u, s.br.xz, s.tl.xx, s.tr.xy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, back_offset, forw_offset, back_offset, dimmz, dimmx);
        compute_component_vcell_BL (v.bl.u, s.tl.xz, s.br.xx, s.bl.xy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, forw_offset, back_offset, back_offset, dimmz, dimmx);
        compute_component_vcell_BR (v.br.u, s.tr.xz, s.bl.xx, s.br.xy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, forw_offset, forw_offset, forw_offset, dimmz, dimmx);
        compute_component_vcell_TL (v.tl.v, s.bl.yz, s.tr.xy, s.tl.yy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, back_offset, back_offset, forw_offset, dimmz, dimmx);
        compute_component_vcell_TR (v.tr.v, s.br.yz, s.tl.xy, s.tr.yy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, back_offset, forw_offset, back_offset, dimmz, dimmx);
        compute_component_vcell_BL (v.bl.v, s.tl.yz, s.br.xy, s.bl.yy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, forw_offset, back_offset, back_offset, dimmz, dimmx);
        compute_component_vcell_BR (v.br.v, s.tr.yz, s.bl.xy, s.br.yy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, forw_offset, forw_offset, forw_offset, dimmz, dimmx);


        //gpu
//        int blockSize = 16;
//        dim3 numBlocks((nxf-nx0 + blockSize-1) / blockSize, (nyf-ny0 + blockSize-1) / blockSize, (nzf-nz0 + blockSize-1) / blockSize);
//
//
//        compute_component_vcell_TL_GPU<<<numBlocks,blockSize>>>(gpu_v.tl.w, gpu_s.bl.zz, gpu_s.tr.xz, gpu_s.tl.yz, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0,
//                                   nyf, back_offset, back_offset, forw_offset, dimmz, dimmx);
//        compute_component_vcell_TR_GPU<<<numBlocks,blockSize>>>(gpu_v.tr.w, gpu_s.br.zz, gpu_s.tl.xz, gpu_s.tr.yz, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0,
//                                   nyf, back_offset, forw_offset, back_offset, dimmz, dimmx);
//        compute_component_vcell_BL_GPU<<<numBlocks,blockSize>>>(gpu_v.bl.w, gpu_s.tl.zz, gpu_s.br.xz, gpu_s.bl.yz, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0,
//                                   nyf, forw_offset, back_offset, back_offset, dimmz, dimmx);
//        compute_component_vcell_BR_GPU<<<numBlocks,blockSize>>>(gpu_v.br.w, gpu_s.tr.zz, gpu_s.bl.xz, gpu_s.br.yz, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0,
//                                   nyf, forw_offset, forw_offset, forw_offset, dimmz, dimmx);
//        compute_component_vcell_TL_GPU<<<numBlocks,blockSize>>>(gpu_v.tl.u, gpu_s.bl.xz, gpu_s.tr.xx, gpu_s.tl.xy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0,
//                                   nyf, back_offset, back_offset, forw_offset, dimmz, dimmx);
//        compute_component_vcell_TR_GPU<<<numBlocks,blockSize>>>(gpu_v.tr.u, gpu_s.br.xz, gpu_s.tl.xx, gpu_s.tr.xy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0,
//                                   nyf, back_offset, forw_offset, back_offset, dimmz, dimmx);
//        compute_component_vcell_BL_GPU<<<numBlocks,blockSize>>>(gpu_v.bl.u, gpu_s.tl.xz, gpu_s.br.xx, gpu_s.bl.xy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0,
//                                   nyf, forw_offset, back_offset, back_offset, dimmz, dimmx);
//        compute_component_vcell_BR_GPU<<<numBlocks,blockSize>>>(gpu_v.br.u, gpu_s.tr.xz, gpu_s.bl.xx, gpu_s.br.xy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0,
//                                   nyf, forw_offset, forw_offset, forw_offset, dimmz, dimmx);
//        compute_component_vcell_TL_GPU<<<numBlocks,blockSize>>>(gpu_v.tl.v, gpu_s.bl.yz, gpu_s.tr.xy, gpu_s.tl.yy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0,
//                                   nyf, back_offset, back_offset, forw_offset, dimmz, dimmx);
//        compute_component_vcell_TR_GPU<<<numBlocks,blockSize>>>(gpu_v.tr.v, gpu_s.br.yz, gpu_s.tl.xy, gpu_s.tr.yy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0,
//                                   nyf, back_offset, forw_offset, back_offset, dimmz, dimmx);
//        compute_component_vcell_BL_GPU<<<numBlocks,blockSize>>>(gpu_v.bl.v, gpu_s.tl.yz, gpu_s.br.xy, gpu_s.bl.yy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0,
//                                   nyf, forw_offset, back_offset, back_offset, dimmz, dimmx);
//        compute_component_vcell_BR_GPU<<<numBlocks,blockSize>>>(gpu_v.br.v, gpu_s.tr.yz, gpu_s.bl.xy, gpu_s.br.yy, rho, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0,
//                                   nyf, forw_offset, forw_offset, forw_offset, dimmz, dimmx);

    }
};






/* ------------------------------------------------------------------------------ */
/*                                                                                */
/*                               CALCULO DE TENSIONES                             */
/*                                                                                */
/* ------------------------------------------------------------------------------ */

void stress_update(real *__restrict__ sptr,
                   const real c1,
                   const real c2,
                   const real c3,
                   const real c4,
                   const real c5,
                   const real c6,
                   const integer z,
                   const integer x,
                   const integer y,
                   const real dt,
                   const real u_x,
                   const real u_y,
                   const real u_z,
                   const real v_x,
                   const real v_y,
                   const real v_z,
                   const real w_x,
                   const real w_y,
                   const real w_z,
                   const integer dimmz,
                   const integer dimmx) {
    sptr[IDX(z, x, y, dimmz, dimmx)] += dt * c1 * u_x;
    sptr[IDX(z, x, y, dimmz, dimmx)] += dt * c2 * v_y;
    sptr[IDX(z, x, y, dimmz, dimmx)] += dt * c3 * w_z;
    sptr[IDX(z, x, y, dimmz, dimmx)] += dt * c4 * (w_y + v_z);
    sptr[IDX(z, x, y, dimmz, dimmx)] += dt * c5 * (w_x + u_z);
    sptr[IDX(z, x, y, dimmz, dimmx)] += dt * c6 * (v_x + u_y);
};

__device__
void stress_update_GPU(real *__restrict__ sptr,
                   const real c1,
                   const real c2,
                   const real c3,
                   const real c4,
                   const real c5,
                   const real c6,
                   const integer z,
                   const integer x,
                   const integer y,
                   const real dt,
                   const real u_x,
                   const real u_y,
                   const real u_z,
                   const real v_x,
                   const real v_y,
                   const real v_z,
                   const real w_x,
                   const real w_y,
                   const real w_z,
                   const integer dimmz,
                   const integer dimmx) {
    sptr[IDX_GPU(z, x, y, dimmz, dimmx)] += dt * c1 * u_x;
    sptr[IDX_GPU(z, x, y, dimmz, dimmx)] += dt * c2 * v_y;
    sptr[IDX_GPU(z, x, y, dimmz, dimmx)] += dt * c3 * w_z;
    sptr[IDX_GPU(z, x, y, dimmz, dimmx)] += dt * c4 * (w_y + v_z);
    sptr[IDX_GPU(z, x, y, dimmz, dimmx)] += dt * c5 * (w_x + u_z);
    sptr[IDX_GPU(z, x, y, dimmz, dimmx)] += dt * c6 * (v_x + u_y);
};

void stress_propagator(s_t s,
                       v_t v,
                       coeff_t coeffs,
                       real *rho,
                       v_t           gpu_v,
                       s_t           gpu_s,
                       coeff_t       gpu_coeffs,
                       real          *gpu_rho,
                       const real dt,
                       const real dzi,
                       const real dxi,
                       const real dyi,
                       const integer nz0,
                       const integer nzf,
                       const integer nx0,
                       const integer nxf,
                       const integer ny0,
                       const integer nyf,
                       const integer dimmz,
                       const integer dimmx) {
    print_debug("Integration limits are (z "I"-"I",x "I"-"I",y "I"-"I")\n", nz0, nzf, nx0, nxf, ny0, nyf);

#ifdef __INTEL_COMPILER
#pragma forceinline recursive
#endif
    {

        //cpu
        compute_component_scell_BR ( s, v.tr, v.bl, v.br, coeffs, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, forw_offset, back_offset, back_offset, dimmz, dimmx);
        compute_component_scell_BL ( s, v.tl, v.br, v.bl, coeffs, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, forw_offset, back_offset, forw_offset, dimmz, dimmx);
        compute_component_scell_TR ( s, v.br, v.tl, v.tr, coeffs, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, back_offset, forw_offset, forw_offset, dimmz, dimmx);
        compute_component_scell_TL ( s, v.bl, v.tr, v.tl, coeffs, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf, back_offset, back_offset, back_offset, dimmz, dimmx);


        //gpu
//        int blockSize = 16;
//        dim3 numBlocks((nxf-nx0 + blockSize-1) / blockSize, (nyf-ny0 + blockSize-1) / blockSize, (nzf-nz0 + blockSize-1) / blockSize);
//
//
//        compute_component_scell_BR_GPU<<<numBlocks,blockSize>>>(gpu_s, gpu_v.tr, gpu_v.bl, gpu_v.br, coeffs, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf,
//                                   forw_offset, back_offset, back_offset, dimmz, dimmx);
//        compute_component_scell_BL_GPU<<<numBlocks,blockSize>>>(gpu_s, gpu_v.tl, gpu_v.br, gpu_v.bl, coeffs, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf,
//                                   forw_offset, back_offset, forw_offset, dimmz, dimmx);
//        compute_component_scell_TR_GPU<<<numBlocks,blockSize>>>(gpu_s, gpu_v.br, gpu_v.tl, gpu_v.tr, coeffs, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf,
//                                   back_offset, forw_offset, forw_offset, dimmz, dimmx);
//        compute_component_scell_TL_GPU<<<numBlocks,blockSize>>>(gpu_s, gpu_v.bl, gpu_v.tr, gpu_v.tl, coeffs, dt, dzi, dxi, dyi, nz0, nzf, nx0, nxf, ny0, nyf,
//                                   back_offset, back_offset, back_offset, dimmz, dimmx);


    }
};

real cell_coeff_BR(const real *__restrict__ ptr,
                   const integer z,
                   const integer x,
                   const integer y,
                   const integer dimmz,
                   const integer dimmx) {
    return (1.0f / (2.5f * (ptr[IDX(z, x, y, dimmz, dimmx)] +
                            ptr[IDX(z, x + 1, y, dimmz, dimmx)] +
                            ptr[IDX(z + 1, x, y, dimmz, dimmx)] +
                            ptr[IDX(z + 1, x + 1, y, dimmz, dimmx)])));
};


__device__
real cell_coeff_BR_GPU(const real *__restrict__ ptr,
                   const integer z,
                   const integer x,
                   const integer y,
                   const integer dimmz,
                   const integer dimmx) {
    return (1.0f / (2.5f * (ptr[IDX_GPU(z, x, y, dimmz, dimmx)] +
                            ptr[IDX_GPU(z, x + 1, y, dimmz, dimmx)] +
                            ptr[IDX_GPU(z + 1, x, y, dimmz, dimmx)] +
                            ptr[IDX_GPU(z + 1, x + 1, y, dimmz, dimmx)])));
};

real cell_coeff_TL(const real *__restrict__ ptr,
                   const integer z,
                   const integer x,
                   const integer y,
                   const integer dimmz,
                   const integer dimmx) {
    return (1.0f / (ptr[IDX(z, x, y, dimmz, dimmx)]));
};

__device__
real cell_coeff_TL_GPU(const real *__restrict__ ptr,
                   const integer z,
                   const integer x,
                   const integer y,
                   const integer dimmz,
                   const integer dimmx) {
    return (1.0f / (ptr[IDX_GPU(z, x, y, dimmz, dimmx)]));
};

real cell_coeff_BL(const real *__restrict__ ptr,
                   const integer z,
                   const integer x,
                   const integer y,
                   const integer dimmz,
                   const integer dimmx) {
    return (1.0f / (2.5f * (ptr[IDX(z, x, y, dimmz, dimmx)] +
                            ptr[IDX(z, x, y + 1, dimmz, dimmx)] +
                            ptr[IDX(z + 1, x, y, dimmz, dimmx)] +
                            ptr[IDX(z + 1, x, y + 1, dimmz, dimmx)])));
};

__device__
real cell_coeff_BL_GPU(const real *__restrict__ ptr,
                   const integer z,
                   const integer x,
                   const integer y,
                   const integer dimmz,
                   const integer dimmx) {
    return (1.0f / (2.5f * (ptr[IDX_GPU(z, x, y, dimmz, dimmx)] +
                            ptr[IDX_GPU(z, x, y + 1, dimmz, dimmx)] +
                            ptr[IDX_GPU(z + 1, x, y, dimmz, dimmx)] +
                            ptr[IDX_GPU(z + 1, x, y + 1, dimmz, dimmx)])));
};

real cell_coeff_TR(const real *__restrict__ ptr,
                   const integer z,
                   const integer x,
                   const integer y,
                   const integer dimmz,
                   const integer dimmx) {
    return (1.0f / (2.5f * (ptr[IDX(z, x, y, dimmz, dimmx)] +
                            ptr[IDX(z, x + 1, y, dimmz, dimmx)] +
                            ptr[IDX(z, x, y + 1, dimmz, dimmx)] +
                            ptr[IDX(z, x + 1, y + 1, dimmz, dimmx)])));
};

__device__
real cell_coeff_TR_GPU(const real *__restrict__ ptr,
                   const integer z,
                   const integer x,
                   const integer y,
                   const integer dimmz,
                   const integer dimmx) {
    return (1.0f / (2.5f * (ptr[IDX_GPU(z, x, y, dimmz, dimmx)] +
                            ptr[IDX_GPU(z, x + 1, y, dimmz, dimmx)] +
                            ptr[IDX_GPU(z, x, y + 1, dimmz, dimmx)] +
                            ptr[IDX_GPU(z, x + 1, y + 1, dimmz, dimmx)])));
};

real cell_coeff_ARTM_BR(const real *__restrict__ ptr,
                        const integer z,
                        const integer x,
                        const integer y,
                        const integer dimmz,
                        const integer dimmx) {
    return ((1.0f / ptr[IDX(z, x, y, dimmz, dimmx)] +
             1.0f / ptr[IDX(z, x + 1, y, dimmz, dimmx)] +
             1.0f / ptr[IDX(z + 1, x, y, dimmz, dimmx)] +
             1.0f / ptr[IDX(z + 1, x + 1, y, dimmz, dimmx)]) * 0.25f);
};

__device__
real cell_coeff_ARTM_BR_GPU(const real *__restrict__ ptr,
                        const integer z,
                        const integer x,
                        const integer y,
                        const integer dimmz,
                        const integer dimmx) {
    return ((1.0f / ptr[IDX_GPU(z, x, y, dimmz, dimmx)] +
             1.0f / ptr[IDX_GPU(z, x + 1, y, dimmz, dimmx)] +
             1.0f / ptr[IDX_GPU(z + 1, x, y, dimmz, dimmx)] +
             1.0f / ptr[IDX_GPU(z + 1, x + 1, y, dimmz, dimmx)]) * 0.25f);
};

real cell_coeff_ARTM_TL(const real *__restrict__ ptr,
                        const integer z,
                        const integer x,
                        const integer y,
                        const integer dimmz,
                        const integer dimmx) {
    return (1.0f / ptr[IDX(z, x, y, dimmz, dimmx)]);
};

__device__
real cell_coeff_ARTM_TL_GPU(const real *__restrict__ ptr,
                        const integer z,
                        const integer x,
                        const integer y,
                        const integer dimmz,
                        const integer dimmx) {
    return (1.0f / ptr[IDX_GPU(z, x, y, dimmz, dimmx)]);
};

real cell_coeff_ARTM_BL(const real *__restrict__ ptr,
                        const integer z,
                        const integer x,
                        const integer y,
                        const integer dimmz,
                        const integer dimmx) {
    return ((1.0f / ptr[IDX(z, x, y, dimmz, dimmx)] +
             1.0f / ptr[IDX(z, x, y + 1, dimmz, dimmx)] +
             1.0f / ptr[IDX(z + 1, x, y, dimmz, dimmx)] +
             1.0f / ptr[IDX(z + 1, x, y + 1, dimmz, dimmx)]) * 0.25f);
};

__device__
real cell_coeff_ARTM_BL_GPU(const real *__restrict__ ptr,
                        const integer z,
                        const integer x,
                        const integer y,
                        const integer dimmz,
                        const integer dimmx) {
    return ((1.0f / ptr[IDX_GPU(z, x, y, dimmz, dimmx)] +
             1.0f / ptr[IDX_GPU(z, x, y + 1, dimmz, dimmx)] +
             1.0f / ptr[IDX_GPU(z + 1, x, y, dimmz, dimmx)] +
             1.0f / ptr[IDX_GPU(z + 1, x, y + 1, dimmz, dimmx)]) * 0.25f);
};

real cell_coeff_ARTM_TR(const real *__restrict__ ptr,
                        const integer z,
                        const integer x,
                        const integer y,
                        const integer dimmz,
                        const integer dimmx) {
    return ((1.0f / ptr[IDX(z, x, y, dimmz, dimmx)] +
             1.0f / ptr[IDX(z, x + 1, y, dimmz, dimmx)] +
             1.0f / ptr[IDX(z, x, y + 1, dimmz, dimmx)] +
             1.0f / ptr[IDX(z, x + 1, y + 1, dimmz, dimmx)]) * 0.25f);
};

__device__
real cell_coeff_ARTM_TR_GPU(const real *__restrict__ ptr,
                        const integer z,
                        const integer x,
                        const integer y,
                        const integer dimmz,
                        const integer dimmx) {
    return ((1.0f / ptr[IDX_GPU(z, x, y, dimmz, dimmx)] +
             1.0f / ptr[IDX_GPU(z, x + 1, y, dimmz, dimmx)] +
             1.0f / ptr[IDX_GPU(z, x, y + 1, dimmz, dimmx)] +
             1.0f / ptr[IDX_GPU(z, x + 1, y + 1, dimmz, dimmx)]) * 0.25f);
};


void compute_component_scell_TR(s_t s,
                                point_v_t vnode_z,
                                point_v_t vnode_x,
                                point_v_t vnode_y,
                                coeff_t coeffs,
                                const real dt,
                                const real dzi,
                                const real dxi,
                                const real dyi,
                                const integer nz0,
                                const integer nzf,
                                const integer nx0,
                                const integer nxf,
                                const integer ny0,
                                const integer nyf,
                                const offset_t _SZ,
                                const offset_t _SX,
                                const offset_t _SY,
                                const integer dimmz,
                                const integer dimmx) {
    __assume(nz0 % HALO == 0);
    __assume(nzf % HALO == 0);

//    real* __restrict__ sxxptr __attribute__ ((aligned (64))) = s.tr.xx;
//    real* __restrict__ syyptr __attribute__ ((aligned (64))) = s.tr.yy;
//    real* __restrict__ szzptr __attribute__ ((aligned (64))) = s.tr.zz;
//    real* __restrict__ syzptr __attribute__ ((aligned (64))) = s.tr.yz;
//    real* __restrict__ sxzptr __attribute__ ((aligned (64))) = s.tr.xz;
//    real* __restrict__ sxyptr __attribute__ ((aligned (64))) = s.tr.xy;
//
//    const real* __restrict__ vxu    __attribute__ ((aligned (64))) = vnode_x.u;
//    const real* __restrict__ vxv    __attribute__ ((aligned (64))) = vnode_x.v;
//    const real* __restrict__ vxw    __attribute__ ((aligned (64))) = vnode_x.w;
//    const real* __restrict__ vyu    __attribute__ ((aligned (64))) = vnode_y.u;
//    const real* __restrict__ vyv    __attribute__ ((aligned (64))) = vnode_y.v;
//    const real* __restrict__ vyw    __attribute__ ((aligned (64))) = vnode_y.w;
//    const real* __restrict__ vzu    __attribute__ ((aligned (64))) = vnode_z.u;
//    const real* __restrict__ vzv    __attribute__ ((aligned (64))) = vnode_z.v;
//    const real* __restrict__ vzw    __attribute__ ((aligned (64))) = vnode_z.w;

    real *__restrict__ sxxptr = s.tr.xx;
    real *__restrict__ syyptr = s.tr.yy;
    real *__restrict__ szzptr = s.tr.zz;
    real *__restrict__ syzptr = s.tr.yz;
    real *__restrict__ sxzptr = s.tr.xz;
    real *__restrict__ sxyptr = s.tr.xy;

    const real *__restrict__ vxu = vnode_x.u;
    const real *__restrict__ vxv = vnode_x.v;
    const real *__restrict__ vxw = vnode_x.w;
    const real *__restrict__ vyu = vnode_y.u;
    const real *__restrict__ vyv = vnode_y.v;
    const real *__restrict__ vyw = vnode_y.w;
    const real *__restrict__ vzu = vnode_z.u;
    const real *__restrict__ vzv = vnode_z.v;
    const real *__restrict__ vzw = vnode_z.w;

#pragma omp parallel for
    for (integer y = ny0; y < nyf; y++) {
        for (integer x = nx0; x < nxf; x++) {
#ifdef __INTEL_COMPILER
#pragma simd
#else
#pragma omp simd
#endif
            for (integer z = nz0; z < nzf; z++) {
                const real c11 = cell_coeff_TR(coeffs.c11, z, x, y, dimmz, dimmx);
                const real c12 = cell_coeff_TR(coeffs.c12, z, x, y, dimmz, dimmx);
                const real c13 = cell_coeff_TR(coeffs.c13, z, x, y, dimmz, dimmx);
                const real c14 = cell_coeff_ARTM_TR(coeffs.c14, z, x, y, dimmz, dimmx);
                const real c15 = cell_coeff_ARTM_TR(coeffs.c15, z, x, y, dimmz, dimmx);
                const real c16 = cell_coeff_ARTM_TR(coeffs.c16, z, x, y, dimmz, dimmx);
                const real c22 = cell_coeff_TR(coeffs.c22, z, x, y, dimmz, dimmx);
                const real c23 = cell_coeff_TR(coeffs.c23, z, x, y, dimmz, dimmx);
                const real c24 = cell_coeff_ARTM_TR(coeffs.c24, z, x, y, dimmz, dimmx);
                const real c25 = cell_coeff_ARTM_TR(coeffs.c25, z, x, y, dimmz, dimmx);
                const real c26 = cell_coeff_ARTM_TR(coeffs.c26, z, x, y, dimmz, dimmx);
                const real c33 = cell_coeff_TR(coeffs.c33, z, x, y, dimmz, dimmx);
                const real c34 = cell_coeff_ARTM_TR(coeffs.c34, z, x, y, dimmz, dimmx);
                const real c35 = cell_coeff_ARTM_TR(coeffs.c35, z, x, y, dimmz, dimmx);
                const real c36 = cell_coeff_ARTM_TR(coeffs.c36, z, x, y, dimmz, dimmx);
                const real c44 = cell_coeff_TR(coeffs.c44, z, x, y, dimmz, dimmx);
                const real c45 = cell_coeff_ARTM_TR(coeffs.c45, z, x, y, dimmz, dimmx);
                const real c46 = cell_coeff_ARTM_TR(coeffs.c46, z, x, y, dimmz, dimmx);
                const real c55 = cell_coeff_TR(coeffs.c55, z, x, y, dimmz, dimmx);
                const real c56 = cell_coeff_ARTM_TR(coeffs.c56, z, x, y, dimmz, dimmx);
                const real c66 = cell_coeff_TR(coeffs.c66, z, x, y, dimmz, dimmx);

                const real u_x = stencil_X(_SX, vxu, dxi, z, x, y, dimmz, dimmx);
                const real v_x = stencil_X(_SX, vxv, dxi, z, x, y, dimmz, dimmx);
                const real w_x = stencil_X(_SX, vxw, dxi, z, x, y, dimmz, dimmx);

                const real u_y = stencil_Y(_SY, vyu, dyi, z, x, y, dimmz, dimmx);
                const real v_y = stencil_Y(_SY, vyv, dyi, z, x, y, dimmz, dimmx);
                const real w_y = stencil_Y(_SY, vyw, dyi, z, x, y, dimmz, dimmx);

                const real u_z = stencil_Z(_SZ, vzu, dzi, z, x, y, dimmz, dimmx);
                const real v_z = stencil_Z(_SZ, vzv, dzi, z, x, y, dimmz, dimmx);
                const real w_z = stencil_Z(_SZ, vzw, dzi, z, x, y, dimmz, dimmx);

                stress_update(sxxptr, c11, c12, c13, c14, c15, c16, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(syyptr, c12, c22, c23, c24, c25, c26, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(szzptr, c13, c23, c33, c34, c35, c36, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(syzptr, c14, c24, c34, c44, c45, c46, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(sxzptr, c15, c25, c35, c45, c55, c56, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(sxyptr, c16, c26, c36, c46, c56, c66, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
            }
        }
    }
};


__global__
void compute_component_scell_TR_GPU(s_t s,
                                point_v_t vnode_z,
                                point_v_t vnode_x,
                                point_v_t vnode_y,
                                coeff_t coeffs,
                                const real dt,
                                const real dzi,
                                const real dxi,
                                const real dyi,
                                const integer nz0,
                                const integer nzf,
                                const integer nx0,
                                const integer nxf,
                                const integer ny0,
                                const integer nyf,
                                const offset_t _SZ,
                                const offset_t _SX,
                                const offset_t _SY,
                                const integer dimmz,
                                const integer dimmx) {
//    __assume(nz0 % HALO == 0);
//    __assume(nzf % HALO == 0);

//    real* __restrict__ sxxptr __attribute__ ((aligned (64))) = s.tr.xx;
//    real* __restrict__ syyptr __attribute__ ((aligned (64))) = s.tr.yy;
//    real* __restrict__ szzptr __attribute__ ((aligned (64))) = s.tr.zz;
//    real* __restrict__ syzptr __attribute__ ((aligned (64))) = s.tr.yz;
//    real* __restrict__ sxzptr __attribute__ ((aligned (64))) = s.tr.xz;
//    real* __restrict__ sxyptr __attribute__ ((aligned (64))) = s.tr.xy;
//
//    const real* __restrict__ vxu    __attribute__ ((aligned (64))) = vnode_x.u;
//    const real* __restrict__ vxv    __attribute__ ((aligned (64))) = vnode_x.v;
//    const real* __restrict__ vxw    __attribute__ ((aligned (64))) = vnode_x.w;
//    const real* __restrict__ vyu    __attribute__ ((aligned (64))) = vnode_y.u;
//    const real* __restrict__ vyv    __attribute__ ((aligned (64))) = vnode_y.v;
//    const real* __restrict__ vyw    __attribute__ ((aligned (64))) = vnode_y.w;
//    const real* __restrict__ vzu    __attribute__ ((aligned (64))) = vnode_z.u;
//    const real* __restrict__ vzv    __attribute__ ((aligned (64))) = vnode_z.v;
//    const real* __restrict__ vzw    __attribute__ ((aligned (64))) = vnode_z.w;

    real *__restrict__ sxxptr = s.tr.xx;
    real *__restrict__ syyptr = s.tr.yy;
    real *__restrict__ szzptr = s.tr.zz;
    real *__restrict__ syzptr = s.tr.yz;
    real *__restrict__ sxzptr = s.tr.xz;
    real *__restrict__ sxyptr = s.tr.xy;

    const real *__restrict__ vxu = vnode_x.u;
    const real *__restrict__ vxv = vnode_x.v;
    const real *__restrict__ vxw = vnode_x.w;
    const real *__restrict__ vyu = vnode_y.u;
    const real *__restrict__ vyv = vnode_y.v;
    const real *__restrict__ vyw = vnode_y.w;
    const real *__restrict__ vzu = vnode_z.u;
    const real *__restrict__ vzv = vnode_z.v;
    const real *__restrict__ vzw = vnode_z.w;

    int x = blockIdx.x * blockDim.x + threadIdx.x + nx0;
    int y = blockIdx.y * blockDim.y + threadIdx.y + ny0;
    int z = blockIdx.z * blockDim.z + threadIdx.z + nz0;

    if (x < nxf && y < nyf && z < nzf){
            const real c11 = cell_coeff_TR_GPU(coeffs.c11, z, x, y, dimmz, dimmx);
            const real c12 = cell_coeff_TR_GPU(coeffs.c12, z, x, y, dimmz, dimmx);
            const real c13 = cell_coeff_TR_GPU(coeffs.c13, z, x, y, dimmz, dimmx);
            const real c14 = cell_coeff_ARTM_TR_GPU(coeffs.c14, z, x, y, dimmz, dimmx);
            const real c15 = cell_coeff_ARTM_TR_GPU(coeffs.c15, z, x, y, dimmz, dimmx);
            const real c16 = cell_coeff_ARTM_TR_GPU(coeffs.c16, z, x, y, dimmz, dimmx);
            const real c22 = cell_coeff_TR_GPU(coeffs.c22, z, x, y, dimmz, dimmx);
            const real c23 = cell_coeff_TR_GPU(coeffs.c23, z, x, y, dimmz, dimmx);
            const real c24 = cell_coeff_ARTM_TR_GPU(coeffs.c24, z, x, y, dimmz, dimmx);
            const real c25 = cell_coeff_ARTM_TR_GPU(coeffs.c25, z, x, y, dimmz, dimmx);
            const real c26 = cell_coeff_ARTM_TR_GPU(coeffs.c26, z, x, y, dimmz, dimmx);
            const real c33 = cell_coeff_TR_GPU(coeffs.c33, z, x, y, dimmz, dimmx);
            const real c34 = cell_coeff_ARTM_TR_GPU(coeffs.c34, z, x, y, dimmz, dimmx);
            const real c35 = cell_coeff_ARTM_TR_GPU(coeffs.c35, z, x, y, dimmz, dimmx);
            const real c36 = cell_coeff_ARTM_TR_GPU(coeffs.c36, z, x, y, dimmz, dimmx);
            const real c44 = cell_coeff_TR_GPU(coeffs.c44, z, x, y, dimmz, dimmx);
            const real c45 = cell_coeff_ARTM_TR_GPU(coeffs.c45, z, x, y, dimmz, dimmx);
            const real c46 = cell_coeff_ARTM_TR_GPU(coeffs.c46, z, x, y, dimmz, dimmx);
            const real c55 = cell_coeff_TR_GPU(coeffs.c55, z, x, y, dimmz, dimmx);
            const real c56 = cell_coeff_ARTM_TR_GPU(coeffs.c56, z, x, y, dimmz, dimmx);
            const real c66 = cell_coeff_TR_GPU(coeffs.c66, z, x, y, dimmz, dimmx);

            const real u_x = stencil_X_GPU(_SX, vxu, dxi, z, x, y, dimmz, dimmx);
            const real v_x = stencil_X_GPU(_SX, vxv, dxi, z, x, y, dimmz, dimmx);
            const real w_x = stencil_X_GPU(_SX, vxw, dxi, z, x, y, dimmz, dimmx);

            const real u_y = stencil_Y_GPU(_SY, vyu, dyi, z, x, y, dimmz, dimmx);
            const real v_y = stencil_Y_GPU(_SY, vyv, dyi, z, x, y, dimmz, dimmx);
            const real w_y = stencil_Y_GPU(_SY, vyw, dyi, z, x, y, dimmz, dimmx);

            const real u_z = stencil_Z_GPU(_SZ, vzu, dzi, z, x, y, dimmz, dimmx);
            const real v_z = stencil_Z_GPU(_SZ, vzv, dzi, z, x, y, dimmz, dimmx);
            const real w_z = stencil_Z_GPU(_SZ, vzw, dzi, z, x, y, dimmz, dimmx);

            stress_update_GPU(sxxptr, c11, c12, c13, c14, c15, c16, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                          w_z, dimmz, dimmx);
            stress_update_GPU(syyptr, c12, c22, c23, c24, c25, c26, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                          w_z, dimmz, dimmx);
            stress_update_GPU(szzptr, c13, c23, c33, c34, c35, c36, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                          w_z, dimmz, dimmx);
            stress_update_GPU(syzptr, c14, c24, c34, c44, c45, c46, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                          w_z, dimmz, dimmx);
            stress_update_GPU(sxzptr, c15, c25, c35, c45, c55, c56, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                          w_z, dimmz, dimmx);
            stress_update_GPU(sxyptr, c16, c26, c36, c46, c56, c66, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                          w_z, dimmz, dimmx);

    }
};



void compute_component_scell_TL(s_t s,
                                point_v_t vnode_z,
                                point_v_t vnode_x,
                                point_v_t vnode_y,
                                coeff_t coeffs,
                                const real dt,
                                const real dzi,
                                const real dxi,
                                const real dyi,
                                const integer nz0,
                                const integer nzf,
                                const integer nx0,
                                const integer nxf,
                                const integer ny0,
                                const integer nyf,
                                const offset_t _SZ,
                                const offset_t _SX,
                                const offset_t _SY,
                                const integer dimmz,
                                const integer dimmx) {
    __assume(nz0 % HALO == 0);
    __assume(nzf % HALO == 0);

//    real* __restrict__ sxxptr __attribute__ ((aligned (64))) = s.tl.xx;
//    real* __restrict__ syyptr __attribute__ ((aligned (64))) = s.tl.yy;
//    real* __restrict__ szzptr __attribute__ ((aligned (64))) = s.tl.zz;
//    real* __restrict__ syzptr __attribute__ ((aligned (64))) = s.tl.yz;
//    real* __restrict__ sxzptr __attribute__ ((aligned (64))) = s.tl.xz;
//    real* __restrict__ sxyptr __attribute__ ((aligned (64))) = s.tl.xy;
//
//    const real* __restrict__ vxu    __attribute__ ((aligned (64))) = vnode_x.u;
//    const real* __restrict__ vxv    __attribute__ ((aligned (64))) = vnode_x.v;
//    const real* __restrict__ vxw    __attribute__ ((aligned (64))) = vnode_x.w;
//    const real* __restrict__ vyu    __attribute__ ((aligned (64))) = vnode_y.u;
//    const real* __restrict__ vyv    __attribute__ ((aligned (64))) = vnode_y.v;
//    const real* __restrict__ vyw    __attribute__ ((aligned (64))) = vnode_y.w;
//    const real* __restrict__ vzu    __attribute__ ((aligned (64))) = vnode_z.u;
//    const real* __restrict__ vzv    __attribute__ ((aligned (64))) = vnode_z.v;
//    const real* __restrict__ vzw    __attribute__ ((aligned (64))) = vnode_z.w;

    real *__restrict__ sxxptr = s.tl.xx;
    real *__restrict__ syyptr = s.tl.yy;
    real *__restrict__ szzptr = s.tl.zz;
    real *__restrict__ syzptr = s.tl.yz;
    real *__restrict__ sxzptr = s.tl.xz;
    real *__restrict__ sxyptr = s.tl.xy;

    const real *__restrict__ vxu = vnode_x.u;
    const real *__restrict__ vxv = vnode_x.v;
    const real *__restrict__ vxw = vnode_x.w;
    const real *__restrict__ vyu = vnode_y.u;
    const real *__restrict__ vyv = vnode_y.v;
    const real *__restrict__ vyw = vnode_y.w;
    const real *__restrict__ vzu = vnode_z.u;
    const real *__restrict__ vzv = vnode_z.v;
    const real *__restrict__ vzw = vnode_z.w;

#pragma omp parallel for
    for (integer y = ny0; y < nyf; y++) {
        for (integer x = nx0; x < nxf; x++) {
#ifdef __INTEL_COMPILER
#pragma simd
#else
#pragma omp simd
#endif
            for (integer z = nz0; z < nzf; z++) {
                const real c11 = cell_coeff_TL(coeffs.c11, z, x, y, dimmz, dimmx);
                const real c12 = cell_coeff_TL(coeffs.c12, z, x, y, dimmz, dimmx);
                const real c13 = cell_coeff_TL(coeffs.c13, z, x, y, dimmz, dimmx);
                const real c14 = cell_coeff_ARTM_TL(coeffs.c14, z, x, y, dimmz, dimmx);
                const real c15 = cell_coeff_ARTM_TL(coeffs.c15, z, x, y, dimmz, dimmx);
                const real c16 = cell_coeff_ARTM_TL(coeffs.c16, z, x, y, dimmz, dimmx);
                const real c22 = cell_coeff_TL(coeffs.c22, z, x, y, dimmz, dimmx);
                const real c23 = cell_coeff_TL(coeffs.c23, z, x, y, dimmz, dimmx);
                const real c24 = cell_coeff_ARTM_TL(coeffs.c24, z, x, y, dimmz, dimmx);
                const real c25 = cell_coeff_ARTM_TL(coeffs.c25, z, x, y, dimmz, dimmx);
                const real c26 = cell_coeff_ARTM_TL(coeffs.c26, z, x, y, dimmz, dimmx);
                const real c33 = cell_coeff_TL(coeffs.c33, z, x, y, dimmz, dimmx);
                const real c34 = cell_coeff_ARTM_TL(coeffs.c34, z, x, y, dimmz, dimmx);
                const real c35 = cell_coeff_ARTM_TL(coeffs.c35, z, x, y, dimmz, dimmx);
                const real c36 = cell_coeff_ARTM_TL(coeffs.c36, z, x, y, dimmz, dimmx);
                const real c44 = cell_coeff_TL(coeffs.c44, z, x, y, dimmz, dimmx);
                const real c45 = cell_coeff_ARTM_TL(coeffs.c45, z, x, y, dimmz, dimmx);
                const real c46 = cell_coeff_ARTM_TL(coeffs.c46, z, x, y, dimmz, dimmx);
                const real c55 = cell_coeff_TL(coeffs.c55, z, x, y, dimmz, dimmx);
                const real c56 = cell_coeff_ARTM_TL(coeffs.c56, z, x, y, dimmz, dimmx);
                const real c66 = cell_coeff_TL(coeffs.c66, z, x, y, dimmz, dimmx);

                const real u_x = stencil_X(_SX, vxu, dxi, z, x, y, dimmz, dimmx);
                const real v_x = stencil_X(_SX, vxv, dxi, z, x, y, dimmz, dimmx);
                const real w_x = stencil_X(_SX, vxw, dxi, z, x, y, dimmz, dimmx);

                const real u_y = stencil_Y(_SY, vyu, dyi, z, x, y, dimmz, dimmx);
                const real v_y = stencil_Y(_SY, vyv, dyi, z, x, y, dimmz, dimmx);
                const real w_y = stencil_Y(_SY, vyw, dyi, z, x, y, dimmz, dimmx);

                const real u_z = stencil_Z(_SZ, vzu, dzi, z, x, y, dimmz, dimmx);
                const real v_z = stencil_Z(_SZ, vzv, dzi, z, x, y, dimmz, dimmx);
                const real w_z = stencil_Z(_SZ, vzw, dzi, z, x, y, dimmz, dimmx);

                stress_update(sxxptr, c11, c12, c13, c14, c15, c16, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(syyptr, c12, c22, c23, c24, c25, c26, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(szzptr, c13, c23, c33, c34, c35, c36, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(syzptr, c14, c24, c34, c44, c45, c46, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(sxzptr, c15, c25, c35, c45, c55, c56, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(sxyptr, c16, c26, c36, c46, c56, c66, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
            }
        }
    }
};

__global__
void compute_component_scell_TL_GPU(s_t s,
                                point_v_t vnode_z,
                                point_v_t vnode_x,
                                point_v_t vnode_y,
                                coeff_t coeffs,
                                const real dt,
                                const real dzi,
                                const real dxi,
                                const real dyi,
                                const integer nz0,
                                const integer nzf,
                                const integer nx0,
                                const integer nxf,
                                const integer ny0,
                                const integer nyf,
                                const offset_t _SZ,
                                const offset_t _SX,
                                const offset_t _SY,
                                const integer dimmz,
                                const integer dimmx) {
//    __assume(nz0 % HALO == 0);
//    __assume(nzf % HALO == 0);

//    real* __restrict__ sxxptr __attribute__ ((aligned (64))) = s.tl.xx;
//    real* __restrict__ syyptr __attribute__ ((aligned (64))) = s.tl.yy;
//    real* __restrict__ szzptr __attribute__ ((aligned (64))) = s.tl.zz;
//    real* __restrict__ syzptr __attribute__ ((aligned (64))) = s.tl.yz;
//    real* __restrict__ sxzptr __attribute__ ((aligned (64))) = s.tl.xz;
//    real* __restrict__ sxyptr __attribute__ ((aligned (64))) = s.tl.xy;
//
//    const real* __restrict__ vxu    __attribute__ ((aligned (64))) = vnode_x.u;
//    const real* __restrict__ vxv    __attribute__ ((aligned (64))) = vnode_x.v;
//    const real* __restrict__ vxw    __attribute__ ((aligned (64))) = vnode_x.w;
//    const real* __restrict__ vyu    __attribute__ ((aligned (64))) = vnode_y.u;
//    const real* __restrict__ vyv    __attribute__ ((aligned (64))) = vnode_y.v;
//    const real* __restrict__ vyw    __attribute__ ((aligned (64))) = vnode_y.w;
//    const real* __restrict__ vzu    __attribute__ ((aligned (64))) = vnode_z.u;
//    const real* __restrict__ vzv    __attribute__ ((aligned (64))) = vnode_z.v;
//    const real* __restrict__ vzw    __attribute__ ((aligned (64))) = vnode_z.w;

    real *__restrict__ sxxptr = s.tl.xx;
    real *__restrict__ syyptr = s.tl.yy;
    real *__restrict__ szzptr = s.tl.zz;
    real *__restrict__ syzptr = s.tl.yz;
    real *__restrict__ sxzptr = s.tl.xz;
    real *__restrict__ sxyptr = s.tl.xy;

    const real *__restrict__ vxu = vnode_x.u;
    const real *__restrict__ vxv = vnode_x.v;
    const real *__restrict__ vxw = vnode_x.w;
    const real *__restrict__ vyu = vnode_y.u;
    const real *__restrict__ vyv = vnode_y.v;
    const real *__restrict__ vyw = vnode_y.w;
    const real *__restrict__ vzu = vnode_z.u;
    const real *__restrict__ vzv = vnode_z.v;
    const real *__restrict__ vzw = vnode_z.w;

    int x = blockIdx.x * blockDim.x + threadIdx.x + nx0;
    int y = blockIdx.y * blockDim.y + threadIdx.y + ny0;
    int z = blockIdx.z * blockDim.z + threadIdx.z + nz0;

    if (x < nxf && y < nyf && z < nzf){
        const real c11 = cell_coeff_TL_GPU(coeffs.c11, z, x, y, dimmz, dimmx);
        const real c12 = cell_coeff_TL_GPU(coeffs.c12, z, x, y, dimmz, dimmx);
        const real c13 = cell_coeff_TL_GPU(coeffs.c13, z, x, y, dimmz, dimmx);
        const real c14 = cell_coeff_ARTM_TL_GPU(coeffs.c14, z, x, y, dimmz, dimmx);
        const real c15 = cell_coeff_ARTM_TL_GPU(coeffs.c15, z, x, y, dimmz, dimmx);
        const real c16 = cell_coeff_ARTM_TL_GPU(coeffs.c16, z, x, y, dimmz, dimmx);
        const real c22 = cell_coeff_TL_GPU(coeffs.c22, z, x, y, dimmz, dimmx);
        const real c23 = cell_coeff_TL_GPU(coeffs.c23, z, x, y, dimmz, dimmx);
        const real c24 = cell_coeff_ARTM_TL_GPU(coeffs.c24, z, x, y, dimmz, dimmx);
        const real c25 = cell_coeff_ARTM_TL_GPU(coeffs.c25, z, x, y, dimmz, dimmx);
        const real c26 = cell_coeff_ARTM_TL_GPU(coeffs.c26, z, x, y, dimmz, dimmx);
        const real c33 = cell_coeff_TL_GPU(coeffs.c33, z, x, y, dimmz, dimmx);
        const real c34 = cell_coeff_ARTM_TL_GPU(coeffs.c34, z, x, y, dimmz, dimmx);
        const real c35 = cell_coeff_ARTM_TL_GPU(coeffs.c35, z, x, y, dimmz, dimmx);
        const real c36 = cell_coeff_ARTM_TL_GPU(coeffs.c36, z, x, y, dimmz, dimmx);
        const real c44 = cell_coeff_TL_GPU(coeffs.c44, z, x, y, dimmz, dimmx);
        const real c45 = cell_coeff_ARTM_TL_GPU(coeffs.c45, z, x, y, dimmz, dimmx);
        const real c46 = cell_coeff_ARTM_TL_GPU(coeffs.c46, z, x, y, dimmz, dimmx);
        const real c55 = cell_coeff_TL_GPU(coeffs.c55, z, x, y, dimmz, dimmx);
        const real c56 = cell_coeff_ARTM_TL_GPU(coeffs.c56, z, x, y, dimmz, dimmx);
        const real c66 = cell_coeff_TL_GPU(coeffs.c66, z, x, y, dimmz, dimmx);

        const real u_x = stencil_X_GPU(_SX, vxu, dxi, z, x, y, dimmz, dimmx);
        const real v_x = stencil_X_GPU(_SX, vxv, dxi, z, x, y, dimmz, dimmx);
        const real w_x = stencil_X_GPU(_SX, vxw, dxi, z, x, y, dimmz, dimmx);

        const real u_y = stencil_Y_GPU(_SY, vyu, dyi, z, x, y, dimmz, dimmx);
        const real v_y = stencil_Y_GPU(_SY, vyv, dyi, z, x, y, dimmz, dimmx);
        const real w_y = stencil_Y_GPU(_SY, vyw, dyi, z, x, y, dimmz, dimmx);

        const real u_z = stencil_Z_GPU(_SZ, vzu, dzi, z, x, y, dimmz, dimmx);
        const real v_z = stencil_Z_GPU(_SZ, vzv, dzi, z, x, y, dimmz, dimmx);
        const real w_z = stencil_Z_GPU(_SZ, vzw, dzi, z, x, y, dimmz, dimmx);

        stress_update_GPU(sxxptr, c11, c12, c13, c14, c15, c16, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(syyptr, c12, c22, c23, c24, c25, c26, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(szzptr, c13, c23, c33, c34, c35, c36, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(syzptr, c14, c24, c34, c44, c45, c46, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(sxzptr, c15, c25, c35, c45, c55, c56, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(sxyptr, c16, c26, c36, c46, c56, c66, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
    }
};


void compute_component_scell_BR(s_t s,
                                point_v_t vnode_z,
                                point_v_t vnode_x,
                                point_v_t vnode_y,
                                coeff_t coeffs,
                                const real dt,
                                const real dzi,
                                const real dxi,
                                const real dyi,
                                const integer nz0,
                                const integer nzf,
                                const integer nx0,
                                const integer nxf,
                                const integer ny0,
                                const integer nyf,
                                const offset_t _SZ,
                                const offset_t _SX,
                                const offset_t _SY,
                                const integer dimmz,
                                const integer dimmx) {
    __assume(nz0 % HALO == 0);
    __assume(nzf % HALO == 0);

//    real* __restrict__ sxxptr __attribute__ ((aligned (64))) = s.br.xx;
//    real* __restrict__ syyptr __attribute__ ((aligned (64))) = s.br.yy;
//    real* __restrict__ szzptr __attribute__ ((aligned (64))) = s.br.zz;
//    real* __restrict__ syzptr __attribute__ ((aligned (64))) = s.br.yz;
//    real* __restrict__ sxzptr __attribute__ ((aligned (64))) = s.br.xz;
//    real* __restrict__ sxyptr __attribute__ ((aligned (64))) = s.br.xy;
//
//    const real* __restrict__ vxu    __attribute__ ((aligned (64))) = vnode_x.u;
//    const real* __restrict__ vxv    __attribute__ ((aligned (64))) = vnode_x.v;
//    const real* __restrict__ vxw    __attribute__ ((aligned (64))) = vnode_x.w;
//    const real* __restrict__ vyu    __attribute__ ((aligned (64))) = vnode_y.u;
//    const real* __restrict__ vyv    __attribute__ ((aligned (64))) = vnode_y.v;
//    const real* __restrict__ vyw    __attribute__ ((aligned (64))) = vnode_y.w;
//    const real* __restrict__ vzu    __attribute__ ((aligned (64))) = vnode_z.u;
//    const real* __restrict__ vzv    __attribute__ ((aligned (64))) = vnode_z.v;
//    const real* __restrict__ vzw    __attribute__ ((aligned (64))) = vnode_z.w;

    real *__restrict__ sxxptr = s.br.xx;
    real *__restrict__ syyptr = s.br.yy;
    real *__restrict__ szzptr = s.br.zz;
    real *__restrict__ syzptr = s.br.yz;
    real *__restrict__ sxzptr = s.br.xz;
    real *__restrict__ sxyptr = s.br.xy;

    const real *__restrict__ vxu = vnode_x.u;
    const real *__restrict__ vxv = vnode_x.v;
    const real *__restrict__ vxw = vnode_x.w;
    const real *__restrict__ vyu = vnode_y.u;
    const real *__restrict__ vyv = vnode_y.v;
    const real *__restrict__ vyw = vnode_y.w;
    const real *__restrict__ vzu = vnode_z.u;
    const real *__restrict__ vzv = vnode_z.v;
    const real *__restrict__ vzw = vnode_z.w;

#pragma omp parallel for
    for (integer y = ny0; y < nyf; y++) {
        for (integer x = nx0; x < nxf; x++) {
#ifdef __INTEL_COMPILER
#pragma simd
#else
#pragma omp simd
#endif
            for (integer z = nz0; z < nzf; z++) {
                const real c11 = cell_coeff_BR(coeffs.c11, z, x, y, dimmz, dimmx);
                const real c12 = cell_coeff_BR(coeffs.c12, z, x, y, dimmz, dimmx);
                const real c13 = cell_coeff_BR(coeffs.c13, z, x, y, dimmz, dimmx);
                const real c22 = cell_coeff_BR(coeffs.c22, z, x, y, dimmz, dimmx);
                const real c23 = cell_coeff_BR(coeffs.c23, z, x, y, dimmz, dimmx);
                const real c33 = cell_coeff_BR(coeffs.c33, z, x, y, dimmz, dimmx);
                const real c44 = cell_coeff_BR(coeffs.c44, z, x, y, dimmz, dimmx);
                const real c55 = cell_coeff_BR(coeffs.c55, z, x, y, dimmz, dimmx);
                const real c66 = cell_coeff_BR(coeffs.c66, z, x, y, dimmz, dimmx);

                const real c14 = cell_coeff_ARTM_BR(coeffs.c14, z, x, y, dimmz, dimmx);
                const real c15 = cell_coeff_ARTM_BR(coeffs.c15, z, x, y, dimmz, dimmx);
                const real c16 = cell_coeff_ARTM_BR(coeffs.c16, z, x, y, dimmz, dimmx);
                const real c24 = cell_coeff_ARTM_BR(coeffs.c24, z, x, y, dimmz, dimmx);
                const real c25 = cell_coeff_ARTM_BR(coeffs.c25, z, x, y, dimmz, dimmx);
                const real c26 = cell_coeff_ARTM_BR(coeffs.c26, z, x, y, dimmz, dimmx);
                const real c34 = cell_coeff_ARTM_BR(coeffs.c34, z, x, y, dimmz, dimmx);
                const real c35 = cell_coeff_ARTM_BR(coeffs.c35, z, x, y, dimmz, dimmx);
                const real c36 = cell_coeff_ARTM_BR(coeffs.c36, z, x, y, dimmz, dimmx);
                const real c45 = cell_coeff_ARTM_BR(coeffs.c45, z, x, y, dimmz, dimmx);
                const real c46 = cell_coeff_ARTM_BR(coeffs.c46, z, x, y, dimmz, dimmx);
                const real c56 = cell_coeff_ARTM_BR(coeffs.c56, z, x, y, dimmz, dimmx);

                const real u_x = stencil_X(_SX, vxu, dxi, z, x, y, dimmz, dimmx);
                const real v_x = stencil_X(_SX, vxv, dxi, z, x, y, dimmz, dimmx);
                const real w_x = stencil_X(_SX, vxw, dxi, z, x, y, dimmz, dimmx);

                const real u_y = stencil_Y(_SY, vyu, dyi, z, x, y, dimmz, dimmx);
                const real v_y = stencil_Y(_SY, vyv, dyi, z, x, y, dimmz, dimmx);
                const real w_y = stencil_Y(_SY, vyw, dyi, z, x, y, dimmz, dimmx);

                const real u_z = stencil_Z(_SZ, vzu, dzi, z, x, y, dimmz, dimmx);
                const real v_z = stencil_Z(_SZ, vzv, dzi, z, x, y, dimmz, dimmx);
                const real w_z = stencil_Z(_SZ, vzw, dzi, z, x, y, dimmz, dimmx);

                stress_update(sxxptr, c11, c12, c13, c14, c15, c16, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(syyptr, c12, c22, c23, c24, c25, c26, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(szzptr, c13, c23, c33, c34, c35, c36, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(syzptr, c14, c24, c34, c44, c45, c46, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(sxzptr, c15, c25, c35, c45, c55, c56, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(sxyptr, c16, c26, c36, c46, c56, c66, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
            }
        }
    }
};

__global__
void compute_component_scell_BR_GPU(s_t s,
                                point_v_t vnode_z,
                                point_v_t vnode_x,
                                point_v_t vnode_y,
                                coeff_t coeffs,
                                const real dt,
                                const real dzi,
                                const real dxi,
                                const real dyi,
                                const integer nz0,
                                const integer nzf,
                                const integer nx0,
                                const integer nxf,
                                const integer ny0,
                                const integer nyf,
                                const offset_t _SZ,
                                const offset_t _SX,
                                const offset_t _SY,
                                const integer dimmz,
                                const integer dimmx) {
//    __assume(nz0 % HALO == 0);
//    __assume(nzf % HALO == 0);

//    real* __restrict__ sxxptr __attribute__ ((aligned (64))) = s.br.xx;
//    real* __restrict__ syyptr __attribute__ ((aligned (64))) = s.br.yy;
//    real* __restrict__ szzptr __attribute__ ((aligned (64))) = s.br.zz;
//    real* __restrict__ syzptr __attribute__ ((aligned (64))) = s.br.yz;
//    real* __restrict__ sxzptr __attribute__ ((aligned (64))) = s.br.xz;
//    real* __restrict__ sxyptr __attribute__ ((aligned (64))) = s.br.xy;
//
//    const real* __restrict__ vxu    __attribute__ ((aligned (64))) = vnode_x.u;
//    const real* __restrict__ vxv    __attribute__ ((aligned (64))) = vnode_x.v;
//    const real* __restrict__ vxw    __attribute__ ((aligned (64))) = vnode_x.w;
//    const real* __restrict__ vyu    __attribute__ ((aligned (64))) = vnode_y.u;
//    const real* __restrict__ vyv    __attribute__ ((aligned (64))) = vnode_y.v;
//    const real* __restrict__ vyw    __attribute__ ((aligned (64))) = vnode_y.w;
//    const real* __restrict__ vzu    __attribute__ ((aligned (64))) = vnode_z.u;
//    const real* __restrict__ vzv    __attribute__ ((aligned (64))) = vnode_z.v;
//    const real* __restrict__ vzw    __attribute__ ((aligned (64))) = vnode_z.w;

    real *__restrict__ sxxptr = s.br.xx;
    real *__restrict__ syyptr = s.br.yy;
    real *__restrict__ szzptr = s.br.zz;
    real *__restrict__ syzptr = s.br.yz;
    real *__restrict__ sxzptr = s.br.xz;
    real *__restrict__ sxyptr = s.br.xy;

    const real *__restrict__ vxu = vnode_x.u;
    const real *__restrict__ vxv = vnode_x.v;
    const real *__restrict__ vxw = vnode_x.w;
    const real *__restrict__ vyu = vnode_y.u;
    const real *__restrict__ vyv = vnode_y.v;
    const real *__restrict__ vyw = vnode_y.w;
    const real *__restrict__ vzu = vnode_z.u;
    const real *__restrict__ vzv = vnode_z.v;
    const real *__restrict__ vzw = vnode_z.w;

    int x = blockIdx.x * blockDim.x + threadIdx.x + nx0;
    int y = blockIdx.y * blockDim.y + threadIdx.y + ny0;
    int z = blockIdx.z * blockDim.z + threadIdx.z + nz0;

    if (x < nxf && y < nyf && z < nzf){
        const real c11 = cell_coeff_BR_GPU(coeffs.c11, z, x, y, dimmz, dimmx);
        const real c12 = cell_coeff_BR_GPU(coeffs.c12, z, x, y, dimmz, dimmx);
        const real c13 = cell_coeff_BR_GPU(coeffs.c13, z, x, y, dimmz, dimmx);
        const real c22 = cell_coeff_BR_GPU(coeffs.c22, z, x, y, dimmz, dimmx);
        const real c23 = cell_coeff_BR_GPU(coeffs.c23, z, x, y, dimmz, dimmx);
        const real c33 = cell_coeff_BR_GPU(coeffs.c33, z, x, y, dimmz, dimmx);
        const real c44 = cell_coeff_BR_GPU(coeffs.c44, z, x, y, dimmz, dimmx);
        const real c55 = cell_coeff_BR_GPU(coeffs.c55, z, x, y, dimmz, dimmx);
        const real c66 = cell_coeff_BR_GPU(coeffs.c66, z, x, y, dimmz, dimmx);

        const real c14 = cell_coeff_ARTM_BR_GPU(coeffs.c14, z, x, y, dimmz, dimmx);
        const real c15 = cell_coeff_ARTM_BR_GPU(coeffs.c15, z, x, y, dimmz, dimmx);
        const real c16 = cell_coeff_ARTM_BR_GPU(coeffs.c16, z, x, y, dimmz, dimmx);
        const real c24 = cell_coeff_ARTM_BR_GPU(coeffs.c24, z, x, y, dimmz, dimmx);
        const real c25 = cell_coeff_ARTM_BR_GPU(coeffs.c25, z, x, y, dimmz, dimmx);
        const real c26 = cell_coeff_ARTM_BR_GPU(coeffs.c26, z, x, y, dimmz, dimmx);
        const real c34 = cell_coeff_ARTM_BR_GPU(coeffs.c34, z, x, y, dimmz, dimmx);
        const real c35 = cell_coeff_ARTM_BR_GPU(coeffs.c35, z, x, y, dimmz, dimmx);
        const real c36 = cell_coeff_ARTM_BR_GPU(coeffs.c36, z, x, y, dimmz, dimmx);
        const real c45 = cell_coeff_ARTM_BR_GPU(coeffs.c45, z, x, y, dimmz, dimmx);
        const real c46 = cell_coeff_ARTM_BR_GPU(coeffs.c46, z, x, y, dimmz, dimmx);
        const real c56 = cell_coeff_ARTM_BR_GPU(coeffs.c56, z, x, y, dimmz, dimmx);

        const real u_x = stencil_X_GPU(_SX, vxu, dxi, z, x, y, dimmz, dimmx);
        const real v_x = stencil_X_GPU(_SX, vxv, dxi, z, x, y, dimmz, dimmx);
        const real w_x = stencil_X_GPU(_SX, vxw, dxi, z, x, y, dimmz, dimmx);

        const real u_y = stencil_Y_GPU(_SY, vyu, dyi, z, x, y, dimmz, dimmx);
        const real v_y = stencil_Y_GPU(_SY, vyv, dyi, z, x, y, dimmz, dimmx);
        const real w_y = stencil_Y_GPU(_SY, vyw, dyi, z, x, y, dimmz, dimmx);

        const real u_z = stencil_Z_GPU(_SZ, vzu, dzi, z, x, y, dimmz, dimmx);
        const real v_z = stencil_Z_GPU(_SZ, vzv, dzi, z, x, y, dimmz, dimmx);
        const real w_z = stencil_Z_GPU(_SZ, vzw, dzi, z, x, y, dimmz, dimmx);

        stress_update_GPU(sxxptr, c11, c12, c13, c14, c15, c16, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(syyptr, c12, c22, c23, c24, c25, c26, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(szzptr, c13, c23, c33, c34, c35, c36, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(syzptr, c14, c24, c34, c44, c45, c46, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(sxzptr, c15, c25, c35, c45, c55, c56, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(sxyptr, c16, c26, c36, c46, c56, c66, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
    }

};

void compute_component_scell_BL(s_t s,
                                point_v_t vnode_z,
                                point_v_t vnode_x,
                                point_v_t vnode_y,
                                coeff_t coeffs,
                                const real dt,
                                const real dzi,
                                const real dxi,
                                const real dyi,
                                const integer nz0,
                                const integer nzf,
                                const integer nx0,
                                const integer nxf,
                                const integer ny0,
                                const integer nyf,
                                const offset_t _SZ,
                                const offset_t _SX,
                                const offset_t _SY,
                                const integer dimmz,
                                const integer dimmx) {
    __assume(nz0 % HALO == 0);
    __assume(nzf % HALO == 0);

//    real* __restrict__ sxxptr __attribute__ ((aligned (64))) = s.bl.xx;
//    real* __restrict__ syyptr __attribute__ ((aligned (64))) = s.bl.yy;
//    real* __restrict__ szzptr __attribute__ ((aligned (64))) = s.bl.zz;
//    real* __restrict__ syzptr __attribute__ ((aligned (64))) = s.bl.yz;
//    real* __restrict__ sxzptr __attribute__ ((aligned (64))) = s.bl.xz;
//    real* __restrict__ sxyptr __attribute__ ((aligned (64))) = s.bl.xy;
//
//    const real* __restrict__ vxu    __attribute__ ((aligned (64))) = vnode_x.u;
//    const real* __restrict__ vxv    __attribute__ ((aligned (64))) = vnode_x.v;
//    const real* __restrict__ vxw    __attribute__ ((aligned (64))) = vnode_x.w;
//    const real* __restrict__ vyu    __attribute__ ((aligned (64))) = vnode_y.u;
//    const real* __restrict__ vyv    __attribute__ ((aligned (64))) = vnode_y.v;
//    const real* __restrict__ vyw    __attribute__ ((aligned (64))) = vnode_y.w;
//    const real* __restrict__ vzu    __attribute__ ((aligned (64))) = vnode_z.u;
//    const real* __restrict__ vzv    __attribute__ ((aligned (64))) = vnode_z.v;
//    const real* __restrict__ vzw    __attribute__ ((aligned (64))) = vnode_z.w;

    real *__restrict__ sxxptr = s.bl.xx;
    real *__restrict__ syyptr = s.bl.yy;
    real *__restrict__ szzptr = s.bl.zz;
    real *__restrict__ syzptr = s.bl.yz;
    real *__restrict__ sxzptr = s.bl.xz;
    real *__restrict__ sxyptr = s.bl.xy;

    const real *__restrict__ vxu = vnode_x.u;
    const real *__restrict__ vxv = vnode_x.v;
    const real *__restrict__ vxw = vnode_x.w;
    const real *__restrict__ vyu = vnode_y.u;
    const real *__restrict__ vyv = vnode_y.v;
    const real *__restrict__ vyw = vnode_y.w;
    const real *__restrict__ vzu = vnode_z.u;
    const real *__restrict__ vzv = vnode_z.v;
    const real *__restrict__ vzw = vnode_z.w;

#pragma omp parallel for
    for (integer y = ny0; y < nyf; y++) {
        for (integer x = nx0; x < nxf; x++) {
#ifdef __INTEL_COMPILER
#pragma simd
#else
#pragma omp simd
#endif
            for (integer z = nz0; z < nzf; z++) {
                const real c11 = cell_coeff_BL(coeffs.c11, z, x, y, dimmz, dimmx);
                const real c12 = cell_coeff_BL(coeffs.c12, z, x, y, dimmz, dimmx);
                const real c13 = cell_coeff_BL(coeffs.c13, z, x, y, dimmz, dimmx);
                const real c14 = cell_coeff_ARTM_BL(coeffs.c14, z, x, y, dimmz, dimmx);
                const real c15 = cell_coeff_ARTM_BL(coeffs.c15, z, x, y, dimmz, dimmx);
                const real c16 = cell_coeff_ARTM_BL(coeffs.c16, z, x, y, dimmz, dimmx);
                const real c22 = cell_coeff_BL(coeffs.c22, z, x, y, dimmz, dimmx);
                const real c23 = cell_coeff_BL(coeffs.c23, z, x, y, dimmz, dimmx);
                const real c24 = cell_coeff_ARTM_BL(coeffs.c24, z, x, y, dimmz, dimmx);
                const real c25 = cell_coeff_ARTM_BL(coeffs.c25, z, x, y, dimmz, dimmx);
                const real c26 = cell_coeff_ARTM_BL(coeffs.c26, z, x, y, dimmz, dimmx);
                const real c33 = cell_coeff_BL(coeffs.c33, z, x, y, dimmz, dimmx);
                const real c34 = cell_coeff_ARTM_BL(coeffs.c34, z, x, y, dimmz, dimmx);
                const real c35 = cell_coeff_ARTM_BL(coeffs.c35, z, x, y, dimmz, dimmx);
                const real c36 = cell_coeff_ARTM_BL(coeffs.c36, z, x, y, dimmz, dimmx);
                const real c44 = cell_coeff_BL(coeffs.c44, z, x, y, dimmz, dimmx);
                const real c45 = cell_coeff_ARTM_BL(coeffs.c45, z, x, y, dimmz, dimmx);
                const real c46 = cell_coeff_ARTM_BL(coeffs.c46, z, x, y, dimmz, dimmx);
                const real c55 = cell_coeff_BL(coeffs.c55, z, x, y, dimmz, dimmx);
                const real c56 = cell_coeff_ARTM_BL(coeffs.c56, z, x, y, dimmz, dimmx);
                const real c66 = cell_coeff_BL(coeffs.c66, z, x, y, dimmz, dimmx);

                const real u_x = stencil_X(_SX, vxu, dxi, z, x, y, dimmz, dimmx);
                const real v_x = stencil_X(_SX, vxv, dxi, z, x, y, dimmz, dimmx);
                const real w_x = stencil_X(_SX, vxw, dxi, z, x, y, dimmz, dimmx);

                const real u_y = stencil_Y(_SY, vyu, dyi, z, x, y, dimmz, dimmx);
                const real v_y = stencil_Y(_SY, vyv, dyi, z, x, y, dimmz, dimmx);
                const real w_y = stencil_Y(_SY, vyw, dyi, z, x, y, dimmz, dimmx);

                const real u_z = stencil_Z(_SZ, vzu, dzi, z, x, y, dimmz, dimmx);
                const real v_z = stencil_Z(_SZ, vzv, dzi, z, x, y, dimmz, dimmx);
                const real w_z = stencil_Z(_SZ, vzw, dzi, z, x, y, dimmz, dimmx);

                stress_update(sxxptr, c11, c12, c13, c14, c15, c16, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(syyptr, c12, c22, c23, c24, c25, c26, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(szzptr, c13, c23, c33, c34, c35, c36, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(syzptr, c14, c24, c34, c44, c45, c46, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(sxzptr, c15, c25, c35, c45, c55, c56, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
                stress_update(sxyptr, c16, c26, c36, c46, c56, c66, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                              w_z, dimmz, dimmx);
            }
        }
    }
};

__global__
void compute_component_scell_BL_GPU(s_t s,
                                point_v_t vnode_z,
                                point_v_t vnode_x,
                                point_v_t vnode_y,
                                coeff_t coeffs,
                                const real dt,
                                const real dzi,
                                const real dxi,
                                const real dyi,
                                const integer nz0,
                                const integer nzf,
                                const integer nx0,
                                const integer nxf,
                                const integer ny0,
                                const integer nyf,
                                const offset_t _SZ,
                                const offset_t _SX,
                                const offset_t _SY,
                                const integer dimmz,
                                const integer dimmx) {
//    __assume(nz0 % HALO == 0);
//    __assume(nzf % HALO == 0);

//    real* __restrict__ sxxptr __attribute__ ((aligned (64))) = s.bl.xx;
//    real* __restrict__ syyptr __attribute__ ((aligned (64))) = s.bl.yy;
//    real* __restrict__ szzptr __attribute__ ((aligned (64))) = s.bl.zz;
//    real* __restrict__ syzptr __attribute__ ((aligned (64))) = s.bl.yz;
//    real* __restrict__ sxzptr __attribute__ ((aligned (64))) = s.bl.xz;
//    real* __restrict__ sxyptr __attribute__ ((aligned (64))) = s.bl.xy;
//
//    const real* __restrict__ vxu    __attribute__ ((aligned (64))) = vnode_x.u;
//    const real* __restrict__ vxv    __attribute__ ((aligned (64))) = vnode_x.v;
//    const real* __restrict__ vxw    __attribute__ ((aligned (64))) = vnode_x.w;
//    const real* __restrict__ vyu    __attribute__ ((aligned (64))) = vnode_y.u;
//    const real* __restrict__ vyv    __attribute__ ((aligned (64))) = vnode_y.v;
//    const real* __restrict__ vyw    __attribute__ ((aligned (64))) = vnode_y.w;
//    const real* __restrict__ vzu    __attribute__ ((aligned (64))) = vnode_z.u;
//    const real* __restrict__ vzv    __attribute__ ((aligned (64))) = vnode_z.v;
//    const real* __restrict__ vzw    __attribute__ ((aligned (64))) = vnode_z.w;

    real *__restrict__ sxxptr = s.bl.xx;
    real *__restrict__ syyptr = s.bl.yy;
    real *__restrict__ szzptr = s.bl.zz;
    real *__restrict__ syzptr = s.bl.yz;
    real *__restrict__ sxzptr = s.bl.xz;
    real *__restrict__ sxyptr = s.bl.xy;

    const real *__restrict__ vxu = vnode_x.u;
    const real *__restrict__ vxv = vnode_x.v;
    const real *__restrict__ vxw = vnode_x.w;
    const real *__restrict__ vyu = vnode_y.u;
    const real *__restrict__ vyv = vnode_y.v;
    const real *__restrict__ vyw = vnode_y.w;
    const real *__restrict__ vzu = vnode_z.u;
    const real *__restrict__ vzv = vnode_z.v;
    const real *__restrict__ vzw = vnode_z.w;

    int x = blockIdx.x * blockDim.x + threadIdx.x + nx0;
    int y = blockIdx.y * blockDim.y + threadIdx.y + ny0;
    int z = blockIdx.z * blockDim.z + threadIdx.z + nz0;

    if (x < nxf && y < nyf && z < nzf){
        const real c11 = cell_coeff_BL_GPU(coeffs.c11, z, x, y, dimmz, dimmx);
        const real c12 = cell_coeff_BL_GPU(coeffs.c12, z, x, y, dimmz, dimmx);
        const real c13 = cell_coeff_BL_GPU(coeffs.c13, z, x, y, dimmz, dimmx);
        const real c14 = cell_coeff_ARTM_BL_GPU(coeffs.c14, z, x, y, dimmz, dimmx);
        const real c15 = cell_coeff_ARTM_BL_GPU(coeffs.c15, z, x, y, dimmz, dimmx);
        const real c16 = cell_coeff_ARTM_BL_GPU(coeffs.c16, z, x, y, dimmz, dimmx);
        const real c22 = cell_coeff_BL_GPU(coeffs.c22, z, x, y, dimmz, dimmx);
        const real c23 = cell_coeff_BL_GPU(coeffs.c23, z, x, y, dimmz, dimmx);
        const real c24 = cell_coeff_ARTM_BL_GPU(coeffs.c24, z, x, y, dimmz, dimmx);
        const real c25 = cell_coeff_ARTM_BL_GPU(coeffs.c25, z, x, y, dimmz, dimmx);
        const real c26 = cell_coeff_ARTM_BL_GPU(coeffs.c26, z, x, y, dimmz, dimmx);
        const real c33 = cell_coeff_BL_GPU(coeffs.c33, z, x, y, dimmz, dimmx);
        const real c34 = cell_coeff_ARTM_BL_GPU(coeffs.c34, z, x, y, dimmz, dimmx);
        const real c35 = cell_coeff_ARTM_BL_GPU(coeffs.c35, z, x, y, dimmz, dimmx);
        const real c36 = cell_coeff_ARTM_BL_GPU(coeffs.c36, z, x, y, dimmz, dimmx);
        const real c44 = cell_coeff_BL_GPU(coeffs.c44, z, x, y, dimmz, dimmx);
        const real c45 = cell_coeff_ARTM_BL_GPU(coeffs.c45, z, x, y, dimmz, dimmx);
        const real c46 = cell_coeff_ARTM_BL_GPU(coeffs.c46, z, x, y, dimmz, dimmx);
        const real c55 = cell_coeff_BL_GPU(coeffs.c55, z, x, y, dimmz, dimmx);
        const real c56 = cell_coeff_ARTM_BL_GPU(coeffs.c56, z, x, y, dimmz, dimmx);
        const real c66 = cell_coeff_BL_GPU(coeffs.c66, z, x, y, dimmz, dimmx);

        const real u_x = stencil_X_GPU(_SX, vxu, dxi, z, x, y, dimmz, dimmx);
        const real v_x = stencil_X_GPU(_SX, vxv, dxi, z, x, y, dimmz, dimmx);
        const real w_x = stencil_X_GPU(_SX, vxw, dxi, z, x, y, dimmz, dimmx);

        const real u_y = stencil_Y_GPU(_SY, vyu, dyi, z, x, y, dimmz, dimmx);
        const real v_y = stencil_Y_GPU(_SY, vyv, dyi, z, x, y, dimmz, dimmx);
        const real w_y = stencil_Y_GPU(_SY, vyw, dyi, z, x, y, dimmz, dimmx);

        const real u_z = stencil_Z_GPU(_SZ, vzu, dzi, z, x, y, dimmz, dimmx);
        const real v_z = stencil_Z_GPU(_SZ, vzv, dzi, z, x, y, dimmz, dimmx);
        const real w_z = stencil_Z_GPU(_SZ, vzw, dzi, z, x, y, dimmz, dimmx);

        stress_update_GPU(sxxptr, c11, c12, c13, c14, c15, c16, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(syyptr, c12, c22, c23, c24, c25, c26, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(szzptr, c13, c23, c33, c34, c35, c36, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(syzptr, c14, c24, c34, c44, c45, c46, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(sxzptr, c15, c25, c35, c45, c55, c56, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
        stress_update_GPU(sxyptr, c16, c26, c36, c46, c56, c66, z, x, y, dt, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y,
                      w_z, dimmz, dimmx);
    }

};
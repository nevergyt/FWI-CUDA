#ifndef _FWI_PROPAGATOR_H_
#define _FWI_PROPAGATOR_H_

#include "fwi_common.cuh"

/* stress point structure */
typedef struct {
    real *zz, *xz, *yz, *xx, *xy, *yy;
} point_s_t;

/* velocity point structure */
typedef struct {
    real *u, *v, *w;
} point_v_t;

/* velocity points on a cell */
typedef struct {
    point_v_t tl, tr, bl, br;
} v_t;

/* stress points on a cell */
typedef struct {
    point_s_t tl, tr, bl, br;
} s_t;

/* coefficients for materials */
typedef struct {
    real *c11, *c12, *c13, *c14, *c15, *c16;
    real *c22, *c23, *c24, *c25, *c26;
    real *c33, *c34, *c35, *c36;
    real *c44, *c45, *c46;
    real *c55, *c56;
    real *c66;
} coeff_t;

#define C0 1.2f
#define C1 1.4f
#define C2 1.6f
#define C3 1.8f

#define ASSUMED_DISTANCE 16

typedef enum {back_offset, forw_offset} offset_t;


integer IDX (const integer z, 
             const integer x, 
             const integer y, 
             const integer dimmz, 
             const integer dimmx);

__device__
integer IDX_GPU (const integer z,
                 const integer x,
                 const integer y,
                 const integer dimmz,
                 const integer dimmx);

real stencil_Z(const offset_t off,
               const real*  ptr,
               const real    dzi,
               const integer z,
               const integer x,
               const integer y,
               const integer dimmz,
               const integer dimmx);

__device__
real stencil_Z_GPU(const offset_t off,
               const real*  ptr,
               const real    dzi,
               const integer z,
               const integer x,
               const integer y,
               const integer dimmz,
               const integer dimmx);

real stencil_X(const offset_t off,
               const real* ptr,
               const real dxi,
               const integer z,
               const integer x,
               const integer y,
               const integer dimmz,
               const integer dimmx);


__device__
real stencil_X_GPU(const offset_t off,
               const real* ptr,
               const real dxi,
               const integer z,
               const integer x,
               const integer y,
               const integer dimmz,
               const integer dimmx);

real stencil_Y(const offset_t off,
               const real*  ptr,
               const real dyi,
               const integer z,
               const integer x,
               const integer y,
               const integer dimmz,
               const integer dimmx);

__device__
real stencil_Y_GPU(const offset_t off,
               const real*  ptr,
               const real dyi,
               const integer z,
               const integer x,
               const integer y,
               const integer dimmz,
               const integer dimmx);


/* ------------------------------------------------------------------------------ */
/*                                                                                */
/*                               CALCULO DE VELOCIDADES                           */
/*                                                                                */
/* ------------------------------------------------------------------------------ */



real rho_BL ( const real*  rho,
              const integer z,
              const integer x,
              const integer y,
              const integer dimmz,
              const integer dimmx);

__device__
real rho_BL_GPU ( const real*  rho,
              const integer z,
              const integer x,
              const integer y,
              const integer dimmz,
              const integer dimmx);

real rho_TR ( const real*  rho,
              const integer z,
              const integer x,
              const integer y,
              const integer dimmz,
              const integer dimmx);

__device__
real rho_TR_GPU ( const real*  rho,
              const integer z,
              const integer x,
              const integer y,
              const integer dimmz,
              const integer dimmx);

real rho_BR ( const real*  rho,
              const integer z,
              const integer x,
              const integer y,
              const integer dimmz,
              const integer dimmx);

__device__
real rho_BR_GPU ( const real*  rho,
              const integer z,
              const integer x,
              const integer y,
              const integer dimmz,
              const integer dimmx);

real rho_TL ( const real*  rho,
              const integer z,
              const integer x,
              const integer y,
              const integer dimmz,
              const integer dimmx);

__device__
real rho_TL_GPU ( const real*  rho,
              const integer z,
              const integer x,
              const integer y,
              const integer dimmz,
              const integer dimmx);

void compute_component_vcell_TL (      real*  _vptr,
                                 const real*  _szptr,
                                 const real*  _sxptr,
                                 const real*  _syptr,
                                 const real*  rho,
                                 const real     dt,
                                 const real     dzi,
                                 const real     dxi,
                                 const real     dyi,
                                 const integer  nz0,
                                 const integer  nzf,
                                 const integer  nx0,
                                 const integer  nxf,
                                 const integer  ny0,
                                 const integer  nyf,
                                 const offset_t _SZ,
                                 const offset_t _SX,
                                 const offset_t _SY,
                                 const integer  dimmz,
                                 const integer  dimmx);

__global__
void compute_component_vcell_TL_GPU (      real*  _vptr,
                                       const real*  _szptr,
                                       const real*  _sxptr,
                                       const real*  _syptr,
                                       const real*  rho,
                                       const real     dt,
                                       const real     dzi,
                                       const real     dxi,
                                       const real     dyi,
                                       const integer  nz0,
                                       const integer  nzf,
                                       const integer  nx0,
                                       const integer  nxf,
                                       const integer  ny0,
                                       const integer  nyf,
                                       const offset_t _SZ,
                                       const offset_t _SX,
                                       const offset_t _SY,
                                       const integer  dimmz,
                                       const integer  dimmx);


void compute_component_vcell_TR (      real*  _vptr,
                                 const real*  _szptr,
                                 const real*  _sxptr,
                                 const real*  _syptr,
                                 const real*  rho,
                                 const real     dt,
                                 const real     dzi,
                                 const real     dxi,
                                 const real     dyi,
                                 const integer  nz0,
                                 const integer  nzf,
                                 const integer  nx0,
                                 const integer  nxf,
                                 const integer  ny0,
                                 const integer  nyf,
                                 const offset_t _SZ,
                                 const offset_t _SX,
                                 const offset_t _SY,
                                 const integer  dimmz,
                                 const integer  dimmx);

__global__
void compute_component_vcell_TR_GPU (      real*  _vptr,
                                       const real*  _szptr,
                                       const real*  _sxptr,
                                       const real*  _syptr,
                                       const real*  rho,
                                       const real     dt,
                                       const real     dzi,
                                       const real     dxi,
                                       const real     dyi,
                                       const integer  nz0,
                                       const integer  nzf,
                                       const integer  nx0,
                                       const integer  nxf,
                                       const integer  ny0,
                                       const integer  nyf,
                                       const offset_t _SZ,
                                       const offset_t _SX,
                                       const offset_t _SY,
                                       const integer  dimmz,
                                       const integer  dimmx);


void compute_component_vcell_BR (      real*  _vptr,
                                 const real*  _szptr,
                                 const real*  _sxptr,
                                 const real*  _syptr,
                                 const real*  rho,
                                 const real     dt,
                                 const real     dzi,
                                 const real     dxi,
                                 const real     dyi,
                                 const integer  ny0,
                                 const integer  nyf,
                                 const integer  nx0,
                                 const integer  nxf,
                                 const integer  nz0,
                                 const integer  nzf,
                                 const offset_t _SZ,
                                 const offset_t _SX,
                                 const offset_t _SY,
                                 const integer  dimmz,
                                 const integer  dimmx);

__global__
void compute_component_vcell_BR_GPU (      real*  _vptr,
                                       const real*  _szptr,
                                       const real*  _sxptr,
                                       const real*  _syptr,
                                       const real*  rho,
                                       const real     dt,
                                       const real     dzi,
                                       const real     dxi,
                                       const real     dyi,
                                       const integer  ny0,
                                       const integer  nyf,
                                       const integer  nx0,
                                       const integer  nxf,
                                       const integer  nz0,
                                       const integer  nzf,
                                       const offset_t _SZ,
                                       const offset_t _SX,
                                       const offset_t _SY,
                                       const integer  dimmz,
                                       const integer  dimmx);

void compute_component_vcell_BL (      real*  _vptr,
                                 const real*  _szptr,
                                 const real*  _sxptr,
                                 const real*  _syptr,
                                 const real*  rho,
                                 const real     dt,
                                 const real     dzi,
                                 const real     dxi,
                                 const real     dyi,
                                 const integer  ny0,
                                 const integer  nyf,
                                 const integer  nx0,
                                 const integer  nxf,
                                 const integer  nz0,
                                 const integer  nzf,
                                 const offset_t _SZ,
                                 const offset_t _SX,
                                 const offset_t _SY,
                                 const integer  dimmz,
                                 const integer  dimmx);

__global__
void compute_component_vcell_BL_GPU (      real*  _vptr,
                                       const real*  _szptr,
                                       const real*  _sxptr,
                                       const real*  _syptr,
                                       const real*  rho,
                                       const real     dt,
                                       const real     dzi,
                                       const real     dxi,
                                       const real     dyi,
                                       const integer  ny0,
                                       const integer  nyf,
                                       const integer  nx0,
                                       const integer  nxf,
                                       const integer  nz0,
                                       const integer  nzf,
                                       const offset_t _SZ,
                                       const offset_t _SX,
                                       const offset_t _SY,
                                       const integer  dimmz,
                                       const integer  dimmx);

void velocity_propagator(v_t       v,
                         s_t       s,
                         coeff_t   coeffs,
                         real      *rho,
                         v_t           gpu_v,
                         s_t           gpu_s,
                         coeff_t       gpu_coeffs,
                         real          *gpu_rho,
                         const real      dt,
                         const real      dzi,
                         const real      dxi,
                         const real      dyi,
                         const integer   nz0,
                         const integer   nzf,
                         const integer   nx0,
                         const integer   nxf,
                         const integer   ny0,
                         const integer   nyf,
                         const integer   dimmz,
                         const integer   dimmx);





/* ------------------------------------------------------------------------------ */
/*                                                                                */
/*                               CALCULO DE TENSIONES                             */
/*                                                                                */
/* ------------------------------------------------------------------------------ */

void stress_update(real*  sptr,
                   const real       c1,
                   const real       c2,
                   const real       c3,
                   const real       c4,
                   const real       c5,
                   const real       c6,
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
                   const integer dimmx);


__device__
void stress_update_GPU(real*  sptr,
                   const real       c1,
                   const real       c2,
                   const real       c3,
                   const real       c4,
                   const real       c5,
                   const real       c6,
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
                   const integer dimmx);

void stress_propagator(s_t           s,
                       v_t           v,
                       coeff_t       coeffs,
                       real          *rho,
                       v_t           gpu_v,
                       s_t           gpu_s,
                       coeff_t       gpu_coeffs,
                       real          *gpu_rho,
                       const real    dt,
                       const real    dzi,
                       const real    dxi,
                       const real    dyi,
                       const integer nz0,
                       const integer nzf,
                       const integer nx0,
                       const integer nxf,
                       const integer ny0,
                       const integer nyf,
                       const integer dimmz,
                       const integer dimmx );

real cell_coeff_BR ( const real*  ptr,
                     const integer z, 
                     const integer x, 
                     const integer y, 
                     const integer dimmz, 
                     const integer dimmx );

__device__
real cell_coeff_BR_GPU ( const real*  ptr,
                     const integer z,
                     const integer x,
                     const integer y,
                     const integer dimmz,
                     const integer dimmx );

real cell_coeff_TL ( const real*  ptr,
                     const integer z, 
                     const integer x, 
                     const integer y, 
                     const integer dimmz, 
                     const integer dimmx );

__device__
real cell_coeff_TL_GPU ( const real*  ptr,
                     const integer z,
                     const integer x,
                     const integer y,
                     const integer dimmz,
                     const integer dimmx );

real cell_coeff_BL ( const real*  ptr,
                     const integer z, 
                     const integer x, 
                     const integer y, 
                     const integer dimmz, 
                     const integer dimmx );

__device__
real cell_coeff_BL_GPU ( const real*  ptr,
                     const integer z,
                     const integer x,
                     const integer y,
                     const integer dimmz,
                     const integer dimmx );

real cell_coeff_TR ( const real*  ptr,
                     const integer z, 
                     const integer x, 
                     const integer y, 
                     const integer dimmz, 
                     const integer dimmx );

__device__
real cell_coeff_TR_GPU ( const real*  ptr,
                     const integer z,
                     const integer x,
                     const integer y,
                     const integer dimmz,
                     const integer dimmx );

real cell_coeff_ARTM_BR ( const real*  ptr,
                          const integer z, 
                          const integer x, 
                          const integer y, 
                          const integer dimmz, 
                          const integer dimmx);

__device__
real cell_coeff_ARTM_BR_GPU ( const real*  ptr,
                          const integer z,
                          const integer x,
                          const integer y,
                          const integer dimmz,
                          const integer dimmx);

real cell_coeff_ARTM_TL ( const real*  ptr,
                          const integer z, 
                          const integer x, 
                          const integer y, 
                          const integer dimmz, 
                          const integer dimmx);

__device__
real cell_coeff_ARTM_TL_GPU ( const real*  ptr,
                          const integer z,
                          const integer x,
                          const integer y,
                          const integer dimmz,
                          const integer dimmx);

real cell_coeff_ARTM_BL ( const real*  ptr,
                          const integer z, 
                          const integer x, 
                          const integer y, 
                          const integer dimmz, 
                          const integer dimmx);

__device__
real cell_coeff_ARTM_BL_GPU ( const real*  ptr,
                          const integer z,
                          const integer x,
                          const integer y,
                          const integer dimmz,
                          const integer dimmx);

real cell_coeff_ARTM_TR ( const real*  ptr,
                          const integer z, 
                          const integer x, 
                          const integer y, 
                          const integer dimmz, 
                          const integer dimmx);

__device__
real cell_coeff_ARTM_TR_GPU ( const real*  ptr,
                          const integer z,
                          const integer x,
                          const integer y,
                          const integer dimmz,
                          const integer dimmx);

void compute_component_scell_TR (s_t             s,
                                 point_v_t       vnode_z,
                                 point_v_t       vnode_x,
                                 point_v_t       vnode_y,
                                 coeff_t         coeffs,
                                 const real      dt,
                                 const real      dzi,
                                 const real      dxi,
                                 const real      dyi,
                                 const integer   nz0,
                                 const integer   nzf,
                                 const integer   nx0,
                                 const integer   nxf,
                                 const integer   ny0,
                                 const integer   nyf,
                                 const offset_t _SZ,
                                 const offset_t _SX,
                                 const offset_t _SY,
                                 const integer  dimmz,
                                 const integer  dimmx);

__global__
void compute_component_scell_TR_GPU (s_t             s,
                                 point_v_t       vnode_z,
                                 point_v_t       vnode_x,
                                 point_v_t       vnode_y,
                                 coeff_t         coeffs,
                                 const real      dt,
                                 const real      dzi,
                                 const real      dxi,
                                 const real      dyi,
                                 const integer   nz0,
                                 const integer   nzf,
                                 const integer   nx0,
                                 const integer   nxf,
                                 const integer   ny0,
                                 const integer   nyf,
                                 const offset_t _SZ,
                                 const offset_t _SX,
                                 const offset_t _SY,
                                 const integer  dimmz,
                                 const integer  dimmx);

void compute_component_scell_TL ( s_t             s,
                                  point_v_t       vnode_z,
                                  point_v_t       vnode_x,
                                  point_v_t       vnode_y,
                                  coeff_t         coeffs,
                                  const real      dt,
                                  const real      dzi,
                                  const real      dxi,
                                  const real      dyi,
                                  const integer   nz0,
                                  const integer   nzf,
                                  const integer   nx0,
                                  const integer   nxf,
                                  const integer   ny0,
                                  const integer   nyf,
                                  const offset_t _SZ,
                                  const offset_t _SX,
                                  const offset_t _SY,
                                  const integer  dimmz,
                                  const integer  dimmx);

__global__
void compute_component_scell_TL_GPU ( s_t             s,
                                  point_v_t       vnode_z,
                                  point_v_t       vnode_x,
                                  point_v_t       vnode_y,
                                  coeff_t         coeffs,
                                  const real      dt,
                                  const real      dzi,
                                  const real      dxi,
                                  const real      dyi,
                                  const integer   nz0,
                                  const integer   nzf,
                                  const integer   nx0,
                                  const integer   nxf,
                                  const integer   ny0,
                                  const integer   nyf,
                                  const offset_t _SZ,
                                  const offset_t _SX,
                                  const offset_t _SY,
                                  const integer  dimmz,
                                  const integer  dimmx);


void compute_component_scell_BR ( s_t             s,
                                  point_v_t       vnode_z,
                                  point_v_t       vnode_x,
                                  point_v_t       vnode_y,
                                  coeff_t         coeffs,
                                  const real      dt,
                                  const real      dzi,
                                  const real      dxi,
                                  const real      dyi,
                                  const integer   nz0,
                                  const integer   nzf,
                                  const integer   nx0,
                                  const integer   nxf,
                                  const integer   ny0,
                                  const integer   nyf,
                                  const offset_t _SZ,
                                  const offset_t _SX,
                                  const offset_t _SY,
                                  const integer  dimmz,
                                  const integer  dimmx);

__global__
void compute_component_scell_BR_GPU ( s_t             s,
                                  point_v_t       vnode_z,
                                  point_v_t       vnode_x,
                                  point_v_t       vnode_y,
                                  coeff_t         coeffs,
                                  const real      dt,
                                  const real      dzi,
                                  const real      dxi,
                                  const real      dyi,
                                  const integer   nz0,
                                  const integer   nzf,
                                  const integer   nx0,
                                  const integer   nxf,
                                  const integer   ny0,
                                  const integer   nyf,
                                  const offset_t _SZ,
                                  const offset_t _SX,
                                  const offset_t _SY,
                                  const integer  dimmz,
                                  const integer  dimmx);

void compute_component_scell_BL ( s_t             s,
                                  point_v_t       vnode_z,
                                  point_v_t       vnode_x,
                                  point_v_t       vnode_y,
                                  coeff_t         coeffs,
                                  const real      dt,
                                  const real      dzi,
                                  const real      dxi,
                                  const real      dyi,
                                  const integer   nz0,
                                  const integer   nzf,
                                  const integer   nx0,
                                  const integer   nxf,
                                  const integer   ny0,
                                  const integer   nyf,
                                  const offset_t _SZ,
                                  const offset_t _SX,
                                  const offset_t _SY,
                                  const integer  dimmz,
                                  const integer  dimmx);

__global__
void compute_component_scell_BL_GPU ( s_t             s,
                                  point_v_t       vnode_z,
                                  point_v_t       vnode_x,
                                  point_v_t       vnode_y,
                                  coeff_t         coeffs,
                                  const real      dt,
                                  const real      dzi,
                                  const real      dxi,
                                  const real      dyi,
                                  const integer   nz0,
                                  const integer   nzf,
                                  const integer   nx0,
                                  const integer   nxf,
                                  const integer   ny0,
                                  const integer   nyf,
                                  const offset_t _SZ,
                                  const offset_t _SX,
                                  const offset_t _SY,
                                  const integer  dimmz,
                                  const integer  dimmx);

#endif /* end of _FWI_PROPAGATOR_H_ definition */

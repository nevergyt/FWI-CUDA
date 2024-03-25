/*
 * =====================================================================================
 *
 *       Filename:  fwi_kernel.c
 *
 *    Description:  kernel propagator implementation
 *
 *        Version:  1.0
 *        Created:  14/12/15 12:10:05
 *       Revision:  none
 *       Compiler:  icc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "fwi_kernel.cuh"


/*
 * Initializes an array of length "length" to a random number.
 */

void write_velocity_datafile(v_t     *v,
                             s_t     *s,
                             coeff_t *c,
                             real    *rho,
                             const integer dimmx,
                             const integer dimmy,
                             const integer dimmz){
    char debug[300];
    const integer cellsInVolume = dimmz * dimmx * dimmy;

    sprintf( debug, "./velocityData.txt");

    FILE *debugFile=fopen(debug,"wb");

    fwrite(v->tl.w, sizeof(real), cellsInVolume,debugFile );
    fwrite(v->tr.w, sizeof(real), cellsInVolume,debugFile );
    fwrite(v->bl.w, sizeof(real), cellsInVolume,debugFile );
    fwrite(v->br.w, sizeof(real), cellsInVolume,debugFile );
    fwrite(v->tl.u, sizeof(real), cellsInVolume,debugFile );
    fwrite(v->tr.u, sizeof(real), cellsInVolume,debugFile );
    fwrite(v->bl.u, sizeof(real), cellsInVolume,debugFile );
    fwrite(v->br.u, sizeof(real), cellsInVolume,debugFile );
    fwrite(v->tl.v, sizeof(real), cellsInVolume,debugFile );
    fwrite(v->tr.v, sizeof(real), cellsInVolume,debugFile );
    fwrite(v->bl.v, sizeof(real), cellsInVolume,debugFile );
    fwrite(v->br.v, sizeof(real), cellsInVolume,debugFile );

    fwrite(s->bl.zz, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->br.zz, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->tl.zz, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->tr.zz, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->bl.xz, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->br.xz, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->tl.xz, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->tr.xz, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->bl.yz, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->br.yz, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->tl.yz, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->tr.yz, sizeof(real), cellsInVolume,debugFile );

    fwrite(s->tr.xx, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->tl.xx, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->br.xx, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->bl.xx, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->tr.xy, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->tl.xy, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->br.xy, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->bl.xy, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->tl.yy, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->tr.yy, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->bl.yy, sizeof(real), cellsInVolume,debugFile );
    fwrite(s->br.yy, sizeof(real), cellsInVolume,debugFile );

    fclose(debugFile);
}



//__global__
void
set_array_to_random_real(real *__restrict__ array, const integer length) {
    const real randvalue = rand() / (1.0 * RAND_MAX);

//
//    int i = blockDim.x *
//            blockIdx.x +
//            threadIdx.x;
//
//    if (i < length)
//    {
//        array[i]= randvalue;
//    }
    for (integer i = 0; i < length; i++)
        array[i] = randvalue;

}

/*
 * Initializes an array of length "length" to a constant floating point value.
 */
//__global__
void set_array_to_constant(real *__restrict__ array, const real value, const integer length) {
//    int i = blockDim.x *
//            blockIdx.x +
//            threadIdx.x;
//
//    if (i < length)
//    {
//        array[i]= value;
//    }

    for (integer i = 0; i < length; i++)
        array[i] = value;
}

void check_memory_shot(const integer dimmz,
                       const integer dimmx,
                       const integer dimmy,
                       coeff_t *c,
                       s_t *s,
                       v_t *v,
                       real *rho) {
#if defined(DEBUG)
    print_debug("Checking memory shot values");

    real UNUSED(value);
    for( int i=0; i < (dimmz * dimmx * dimmy); i++)
    {
        value = c->c11[i];
        value = c->c12[i];
        value = c->c13[i];
        value = c->c14[i];
        value = c->c15[i];
        value = c->c16[i];

        value = c->c22[i];
        value = c->c23[i];
        value = c->c24[i];
        value = c->c25[i];
        value = c->c26[i];

        value = c->c33[i];
        value = c->c34[i];
        value = c->c35[i];
        value = c->c36[i];

        value = c->c44[i];
        value = c->c45[i];
        value = c->c46[i];
        
        value = c->c55[i];
        value = c->c56[i];
        value = c->c66[i];

        value = v->tl.u[i];
        value = v->tl.v[i];
        value = v->tl.w[i];

        value = v->tr.u[i];
        value = v->tr.v[i];
        value = v->tr.w[i];

        value = v->bl.u[i];
        value = v->bl.v[i];
        value = v->bl.w[i];
        
        value = v->br.u[i];
        value = v->br.v[i];
        value = v->br.w[i];

        value = rho[i];
    }
        print_debug("Shot memory is well allocated");
#endif
};


void alloc_memory_shot_gpu(const integer dimmz,
                           const integer dimmx,
                           const integer dimmy,
                           coeff_t *gpu_c,
                           s_t *gpu_s,
                           v_t *gpu_v,
                           real **gpu_rho) {
    const size_t size = dimmz * dimmx * dimmy * sizeof(real);

    cudaMalloc(&(gpu_c->c11), size);
    cudaMalloc(&(gpu_c->c12), size);
    cudaMalloc(&(gpu_c->c13), size);
    cudaMalloc(&(gpu_c->c14), size);
    cudaMalloc(&(gpu_c->c15), size);
    cudaMalloc(&(gpu_c->c16), size);

    cudaMalloc(&(gpu_c->c22), size);
    cudaMalloc(&(gpu_c->c23), size);
    cudaMalloc(&(gpu_c->c24), size);
    cudaMalloc(&(gpu_c->c25), size);
    cudaMalloc(&(gpu_c->c26), size);

    cudaMalloc(&(gpu_c->c33), size);
    cudaMalloc(&(gpu_c->c34), size);
    cudaMalloc(&(gpu_c->c35), size);
    cudaMalloc(&(gpu_c->c36), size);

    cudaMalloc(&(gpu_c->c44), size);
    cudaMalloc(&(gpu_c->c45), size);
    cudaMalloc(&(gpu_c->c46), size);

    cudaMalloc(&(gpu_c->c55), size);
    cudaMalloc(&(gpu_c->c56), size);
    cudaMalloc(&(gpu_c->c66), size);

    cudaMalloc(&(gpu_v->tl.u), size);
    cudaMalloc(&(gpu_v->tl.v), size);
    cudaMalloc(&(gpu_v->tl.w), size);

    cudaMalloc(&(gpu_v->tr.u), size);
    cudaMalloc(&(gpu_v->tr.v), size);
    cudaMalloc(&(gpu_v->tr.w), size);

    cudaMalloc(&(gpu_v->bl.u), size);
    cudaMalloc(&(gpu_v->bl.v), size);
    cudaMalloc(&(gpu_v->bl.w), size);

    cudaMalloc(&(gpu_v->br.u), size);
    cudaMalloc(&(gpu_v->br.v), size);
    cudaMalloc(&(gpu_v->br.w), size);

    cudaMalloc(&(gpu_s->tl.zz), size);
    cudaMalloc(&(gpu_s->tl.xz), size);
    cudaMalloc(&(gpu_s->tl.yz), size);
    cudaMalloc(&(gpu_s->tl.xx), size);
    cudaMalloc(&(gpu_s->tl.xy), size);
    cudaMalloc(&(gpu_s->tl.yy), size);

    cudaMalloc(&(gpu_s->tr.zz), size);
    cudaMalloc(&(gpu_s->tr.xz), size);
    cudaMalloc(&(gpu_s->tr.yz), size);
    cudaMalloc(&(gpu_s->tr.xx), size);
    cudaMalloc(&(gpu_s->tr.xy), size);
    cudaMalloc(&(gpu_s->tr.yy), size);

    cudaMalloc(&(gpu_s->bl.zz), size);
    cudaMalloc(&(gpu_s->bl.xz), size);
    cudaMalloc(&(gpu_s->bl.yz), size);
    cudaMalloc(&(gpu_s->bl.xx), size);
    cudaMalloc(&(gpu_s->bl.xy), size);
    cudaMalloc(&(gpu_s->bl.yy), size);

    cudaMalloc(&(gpu_s->br.zz), size);
    cudaMalloc(&(gpu_s->br.xz), size);
    cudaMalloc(&(gpu_s->br.yz), size);
    cudaMalloc(&(gpu_s->br.xx), size);
    cudaMalloc(&(gpu_s->br.xy), size);
    cudaMalloc(&(gpu_s->br.yy), size);

    cudaMalloc(&(*gpu_rho), size);

}

void alloc_memory_shot(const integer dimmz,
                       const integer dimmx,
                       const integer dimmy,
                       coeff_t *c,
                       s_t *s,
                       v_t *v,
                       real **rho
) {
    const size_t size = dimmz * dimmx * dimmy * sizeof(real);

    print_debug("ptr size = %lu bytes (%lu elements)",
                size,
                (size_t) dimmz * dimmx * dimmy);

    /* allocate coefficients */
    c->c11 = (real *) __malloc(ALIGN_REAL, size);
    c->c12 = (real *) __malloc(ALIGN_REAL, size);
    c->c13 = (real *) __malloc(ALIGN_REAL, size);
    c->c14 = (real *) __malloc(ALIGN_REAL, size);
    c->c15 = (real *) __malloc(ALIGN_REAL, size);
    c->c16 = (real *) __malloc(ALIGN_REAL, size);

    c->c22 = (real *) __malloc(ALIGN_REAL, size);
    c->c23 = (real *) __malloc(ALIGN_REAL, size);
    c->c24 = (real *) __malloc(ALIGN_REAL, size);
    c->c25 = (real *) __malloc(ALIGN_REAL, size);
    c->c26 = (real *) __malloc(ALIGN_REAL, size);

    c->c33 = (real *) __malloc(ALIGN_REAL, size);
    c->c34 = (real *) __malloc(ALIGN_REAL, size);
    c->c35 = (real *) __malloc(ALIGN_REAL, size);
    c->c36 = (real *) __malloc(ALIGN_REAL, size);

    c->c44 = (real *) __malloc(ALIGN_REAL, size);
    c->c45 = (real *) __malloc(ALIGN_REAL, size);
    c->c46 = (real *) __malloc(ALIGN_REAL, size);

    c->c55 = (real *) __malloc(ALIGN_REAL, size);
    c->c56 = (real *) __malloc(ALIGN_REAL, size);
    c->c66 = (real *) __malloc(ALIGN_REAL, size);

    /* allocate velocity components */
    v->tl.u = (real *) __malloc(ALIGN_REAL, size);
    v->tl.v = (real *) __malloc(ALIGN_REAL, size);
    v->tl.w = (real *) __malloc(ALIGN_REAL, size);

    v->tr.u = (real *) __malloc(ALIGN_REAL, size);
    v->tr.v = (real *) __malloc(ALIGN_REAL, size);
    v->tr.w = (real *) __malloc(ALIGN_REAL, size);

    v->bl.u = (real *) __malloc(ALIGN_REAL, size);
    v->bl.v = (real *) __malloc(ALIGN_REAL, size);
    v->bl.w = (real *) __malloc(ALIGN_REAL, size);

    v->br.u = (real *) __malloc(ALIGN_REAL, size);
    v->br.v = (real *) __malloc(ALIGN_REAL, size);
    v->br.w = (real *) __malloc(ALIGN_REAL, size);

    /* allocate stress components   */
    s->tl.zz = (real *) __malloc(ALIGN_REAL, size);
    s->tl.xz = (real *) __malloc(ALIGN_REAL, size);
    s->tl.yz = (real *) __malloc(ALIGN_REAL, size);
    s->tl.xx = (real *) __malloc(ALIGN_REAL, size);
    s->tl.xy = (real *) __malloc(ALIGN_REAL, size);
    s->tl.yy = (real *) __malloc(ALIGN_REAL, size);

    s->tr.zz = (real *) __malloc(ALIGN_REAL, size);
    s->tr.xz = (real *) __malloc(ALIGN_REAL, size);
    s->tr.yz = (real *) __malloc(ALIGN_REAL, size);
    s->tr.xx = (real *) __malloc(ALIGN_REAL, size);
    s->tr.xy = (real *) __malloc(ALIGN_REAL, size);
    s->tr.yy = (real *) __malloc(ALIGN_REAL, size);

    s->bl.zz = (real *) __malloc(ALIGN_REAL, size);
    s->bl.xz = (real *) __malloc(ALIGN_REAL, size);
    s->bl.yz = (real *) __malloc(ALIGN_REAL, size);
    s->bl.xx = (real *) __malloc(ALIGN_REAL, size);
    s->bl.xy = (real *) __malloc(ALIGN_REAL, size);
    s->bl.yy = (real *) __malloc(ALIGN_REAL, size);

    s->br.zz = (real *) __malloc(ALIGN_REAL, size);
    s->br.xz = (real *) __malloc(ALIGN_REAL, size);
    s->br.yz = (real *) __malloc(ALIGN_REAL, size);
    s->br.xx = (real *) __malloc(ALIGN_REAL, size);
    s->br.xy = (real *) __malloc(ALIGN_REAL, size);
    s->br.yy = (real *) __malloc(ALIGN_REAL, size);

    /* allocate density array       */
    *rho = (real *) __malloc(ALIGN_REAL, size);
};

void free_memory_shot_gpu(coeff_t *gpu_c,
                          s_t *gpu_s,
                          v_t *gpu_v,
                          real **gpu_rho) {
    cudaFree((void *) gpu_c->c11);
    cudaFree((void *) gpu_c->c12);
    cudaFree((void *) gpu_c->c13);
    cudaFree((void *) gpu_c->c14);
    cudaFree((void *) gpu_c->c15);
    cudaFree((void *) gpu_c->c16);

    cudaFree((void *) gpu_c->c22);
    cudaFree((void *) gpu_c->c23);
    cudaFree((void *) gpu_c->c24);
    cudaFree((void *) gpu_c->c25);
    cudaFree((void *) gpu_c->c26);
    cudaFree((void *) gpu_c->c33);
    cudaFree((void *) gpu_c->c34);
    cudaFree((void *) gpu_c->c35);
    cudaFree((void *) gpu_c->c36);

    cudaFree((void *) gpu_c->c44);
    cudaFree((void *) gpu_c->c45);
    cudaFree((void *) gpu_c->c46);

    cudaFree((void *) gpu_c->c55);
    cudaFree((void *) gpu_c->c56);

    cudaFree((void *) gpu_c->c66);

    cudaFree((void *) gpu_v->tl.u);
    cudaFree((void *) gpu_v->tl.v);
    cudaFree((void *) gpu_v->tl.w);

    cudaFree((void *) gpu_v->tr.u);
    cudaFree((void *) gpu_v->tr.v);
    cudaFree((void *) gpu_v->tr.w);

    cudaFree((void *) gpu_v->bl.u);
    cudaFree((void *) gpu_v->bl.v);
    cudaFree((void *) gpu_v->bl.w);

    cudaFree((void *) gpu_v->br.u);
    cudaFree((void *) gpu_v->br.v);
    cudaFree((void *) gpu_v->br.w);

    cudaFree((void *) gpu_s->tl.zz);
    cudaFree((void *) gpu_s->tl.xz);
    cudaFree((void *) gpu_s->tl.yz);
    cudaFree((void *) gpu_s->tl.xx);
    cudaFree((void *) gpu_s->tl.xy);
    cudaFree((void *) gpu_s->tl.yy);

    cudaFree((void *) gpu_s->tr.zz);
    cudaFree((void *) gpu_s->tr.xz);
    cudaFree((void *) gpu_s->tr.yz);
    cudaFree((void *) gpu_s->tr.xx);
    cudaFree((void *) gpu_s->tr.xy);
    cudaFree((void *) gpu_s->tr.yy);

    cudaFree((void *) gpu_s->bl.zz);
    cudaFree((void *) gpu_s->bl.xz);
    cudaFree((void *) gpu_s->bl.yz);
    cudaFree((void *) gpu_s->bl.xx);
    cudaFree((void *) gpu_s->bl.xy);
    cudaFree((void *) gpu_s->bl.yy);

    cudaFree((void *) gpu_s->br.zz);
    cudaFree((void *) gpu_s->br.xz);
    cudaFree((void *) gpu_s->br.yz);
    cudaFree((void *) gpu_s->br.xx);
    cudaFree((void *) gpu_s->br.xy);
    cudaFree((void *) gpu_s->br.yy);

    cudaFree((void *) *gpu_rho);

}

void free_memory_shot(coeff_t *c,
                      s_t *s,
                      v_t *v,
                      real **rho) {
    /* deallocate coefficients */
    __free((void *) c->c11);
    __free((void *) c->c12);
    __free((void *) c->c13);
    __free((void *) c->c14);
    __free((void *) c->c15);
    __free((void *) c->c16);

    __free((void *) c->c22);
    __free((void *) c->c23);
    __free((void *) c->c24);
    __free((void *) c->c25);
    __free((void *) c->c26);
    __free((void *) c->c33);

    __free((void *) c->c34);
    __free((void *) c->c35);
    __free((void *) c->c36);

    __free((void *) c->c44);
    __free((void *) c->c45);
    __free((void *) c->c46);

    __free((void *) c->c55);
    __free((void *) c->c56);

    __free((void *) c->c66);

    __free((void *) v->tl.u);
    __free((void *) v->tl.v);
    __free((void *) v->tl.w);

    __free((void *) v->tr.u);
    __free((void *) v->tr.v);
    __free((void *) v->tr.w);

    __free((void *) v->bl.u);
    __free((void *) v->bl.v);
    __free((void *) v->bl.w);

    __free((void *) v->br.u);
    __free((void *) v->br.v);
    __free((void *) v->br.w);

    __free((void *) s->tl.zz);
    __free((void *) s->tl.xz);
    __free((void *) s->tl.yz);
    __free((void *) s->tl.xx);
    __free((void *) s->tl.xy);
    __free((void *) s->tl.yy);

    __free((void *) s->tr.zz);
    __free((void *) s->tr.xz);
    __free((void *) s->tr.yz);
    __free((void *) s->tr.xx);
    __free((void *) s->tr.xy);
    __free((void *) s->tr.yy);

    __free((void *) s->bl.zz);
    __free((void *) s->bl.xz);
    __free((void *) s->bl.yz);
    __free((void *) s->bl.xx);
    __free((void *) s->bl.xy);
    __free((void *) s->bl.yy);

    __free((void *) s->br.zz);
    __free((void *) s->br.xz);
    __free((void *) s->br.yz);
    __free((void *) s->br.xx);
    __free((void *) s->br.xy);
    __free((void *) s->br.yy);


    /* deallocate density array       */
    __free((void *) *rho);
};

/*
 * Loads initial values from coeffs, stress and velocity.
 *
 * dimmz: number of z planes.
 * dimmx: number of x planes
 * FirstYPlane: first Y plane of my local domain (includes HALO)
 * LastYPlane: last Y plane of my local domain (includes HALO)
 */
void load_local_velocity_model(const real waveletFreq,
                               const integer dimmz,
                               const integer dimmx,
                               const integer FirstYPlane,
                               const integer LastYPlane,
                               coeff_t *c,
                               s_t *s,
                               v_t *v,
                               real *rho) {
    /* Local variables */
    double tstart_outer, tstart_inner, tend_outer, tend_inner;
    double iospeed_inner, iospeed_outer;
    char modelname[300];

    const integer cellsInVolume = dimmz * dimmx * (LastYPlane - FirstYPlane);
    const integer bytesForVolume = WRITTEN_FIELDS * cellsInVolume * sizeof(real);

    /*
     * Material, velocities and stresses are initizalized
     * accorting to the compilation flags, either randomly
     * or by reading an input velocity model.
     */

    /* initialize stress arrays */
    set_array_to_constant(s->tl.zz, 0, cellsInVolume);
    set_array_to_constant(s->tl.xz, 0, cellsInVolume);
    set_array_to_constant(s->tl.yz, 0, cellsInVolume);
    set_array_to_constant(s->tl.xx, 0, cellsInVolume);
    set_array_to_constant(s->tl.xy, 0, cellsInVolume);
    set_array_to_constant(s->tl.yy, 0, cellsInVolume);
    set_array_to_constant(s->tr.zz, 0, cellsInVolume);
    set_array_to_constant(s->tr.xz, 0, cellsInVolume);
    set_array_to_constant(s->tr.yz, 0, cellsInVolume);
    set_array_to_constant(s->tr.xx, 0, cellsInVolume);
    set_array_to_constant(s->tr.xy, 0, cellsInVolume);
    set_array_to_constant(s->tr.yy, 0, cellsInVolume);
    set_array_to_constant(s->bl.zz, 0, cellsInVolume);
    set_array_to_constant(s->bl.xz, 0, cellsInVolume);
    set_array_to_constant(s->bl.yz, 0, cellsInVolume);
    set_array_to_constant(s->bl.xx, 0, cellsInVolume);
    set_array_to_constant(s->bl.xy, 0, cellsInVolume);
    set_array_to_constant(s->bl.yy, 0, cellsInVolume);
    set_array_to_constant(s->br.zz, 0, cellsInVolume);
    set_array_to_constant(s->br.xz, 0, cellsInVolume);
    set_array_to_constant(s->br.yz, 0, cellsInVolume);
    set_array_to_constant(s->br.xx, 0, cellsInVolume);
    set_array_to_constant(s->br.xy, 0, cellsInVolume);
    set_array_to_constant(s->br.yy, 0, cellsInVolume);

#if defined(DO_NOT_PERFORM_IO)

    /* initialize material coefficients */
    set_array_to_random_real( c->c11, cellsInVolume);
    set_array_to_random_real( c->c12, cellsInVolume);
    set_array_to_random_real( c->c13, cellsInVolume);
    set_array_to_random_real( c->c14, cellsInVolume);
    set_array_to_random_real( c->c15, cellsInVolume);
    set_array_to_random_real( c->c16, cellsInVolume);
    set_array_to_random_real( c->c22, cellsInVolume);
    set_array_to_random_real( c->c23, cellsInVolume);
    set_array_to_random_real( c->c24, cellsInVolume);
    set_array_to_random_real( c->c25, cellsInVolume);
    set_array_to_random_real( c->c26, cellsInVolume);
    set_array_to_random_real( c->c33, cellsInVolume);
    set_array_to_random_real( c->c34, cellsInVolume);
    set_array_to_random_real( c->c35, cellsInVolume);
    set_array_to_random_real( c->c36, cellsInVolume);
    set_array_to_random_real( c->c44, cellsInVolume);
    set_array_to_random_real( c->c45, cellsInVolume);
    set_array_to_random_real( c->c46, cellsInVolume);
    set_array_to_random_real( c->c55, cellsInVolume);
    set_array_to_random_real( c->c56, cellsInVolume);
    set_array_to_random_real( c->c66, cellsInVolume);
    
    /* initalize velocity components */
    set_array_to_random_real( v->tl.u, cellsInVolume );
    set_array_to_random_real( v->tl.v, cellsInVolume );
    set_array_to_random_real( v->tl.w, cellsInVolume );
    set_array_to_random_real( v->tr.u, cellsInVolume );
    set_array_to_random_real( v->tr.v, cellsInVolume );
    set_array_to_random_real( v->tr.w, cellsInVolume );
    set_array_to_random_real( v->bl.u, cellsInVolume );
    set_array_to_random_real( v->bl.v, cellsInVolume );
    set_array_to_random_real( v->bl.w, cellsInVolume );
    set_array_to_random_real( v->br.u, cellsInVolume );
    set_array_to_random_real( v->br.v, cellsInVolume );
    set_array_to_random_real( v->br.w, cellsInVolume );

    /* initialize density (rho) */
    set_array_to_random_real( rho, cellsInVolume );

#else /* load velocity model from external file */

    /* initialize material coefficients */
    set_array_to_constant(c->c11, 1.0, cellsInVolume);
    set_array_to_constant(c->c12, 1.0, cellsInVolume);
    set_array_to_constant(c->c13, 1.0, cellsInVolume);
    set_array_to_constant(c->c14, 1.0, cellsInVolume);
    set_array_to_constant(c->c15, 1.0, cellsInVolume);
    set_array_to_constant(c->c16, 1.0, cellsInVolume);
    set_array_to_constant(c->c22, 1.0, cellsInVolume);
    set_array_to_constant(c->c23, 1.0, cellsInVolume);
    set_array_to_constant(c->c24, 1.0, cellsInVolume);
    set_array_to_constant(c->c25, 1.0, cellsInVolume);
    set_array_to_constant(c->c26, 1.0, cellsInVolume);
    set_array_to_constant(c->c33, 1.0, cellsInVolume);
    set_array_to_constant(c->c34, 1.0, cellsInVolume);
    set_array_to_constant(c->c35, 1.0, cellsInVolume);
    set_array_to_constant(c->c36, 1.0, cellsInVolume);
    set_array_to_constant(c->c44, 1.0, cellsInVolume);
    set_array_to_constant(c->c45, 1.0, cellsInVolume);
    set_array_to_constant(c->c46, 1.0, cellsInVolume);
    set_array_to_constant(c->c55, 1.0, cellsInVolume);
    set_array_to_constant(c->c56, 1.0, cellsInVolume);
    set_array_to_constant(c->c66, 1.0, cellsInVolume);

    /* initialize density (rho) */
    set_array_to_constant(rho, 1.0, cellsInVolume);

    char *fwipath = read_env_variable("FWIDIR");
    /* open initial model, binary file */
    sprintf(modelname, "%s/InputModels/velocitymodel_(%.2f).bin", fwipath, waveletFreq);
    print_info("Loading input model %s from disk (this could take a while)", modelname);

    /* start clock, take into account file opening */
    tstart_outer = dtime();
    FILE *model = safe_fopen(modelname, (char *) "rb", (char *) __FILE__, __LINE__);

    /* seek to the correct position corresponding to mpi_rank */
    fseek(model, sizeof(real) * WRITTEN_FIELDS * dimmz * dimmx * FirstYPlane, SEEK_SET);

    /* start clock, do not take into account file opening */
    tstart_inner = dtime();

    /* initalize velocity components */
    safe_fread(v->tl.u, sizeof(real), cellsInVolume, model, (char *) __FILE__, __LINE__);
    safe_fread(v->tl.v, sizeof(real), cellsInVolume, model, (char *) __FILE__, __LINE__);
    safe_fread(v->tl.w, sizeof(real), cellsInVolume, model, (char *) __FILE__, __LINE__);
    safe_fread(v->tr.u, sizeof(real), cellsInVolume, model, (char *) __FILE__, __LINE__);
    safe_fread(v->tr.v, sizeof(real), cellsInVolume, model, (char *) __FILE__, __LINE__);
    safe_fread(v->tr.w, sizeof(real), cellsInVolume, model, (char *) __FILE__, __LINE__);
    safe_fread(v->bl.u, sizeof(real), cellsInVolume, model, (char *) __FILE__, __LINE__);
    safe_fread(v->bl.v, sizeof(real), cellsInVolume, model, (char *) __FILE__, __LINE__);
    safe_fread(v->bl.w, sizeof(real), cellsInVolume, model, (char *) __FILE__, __LINE__);
    safe_fread(v->br.u, sizeof(real), cellsInVolume, model, (char *) __FILE__, __LINE__);
    safe_fread(v->br.v, sizeof(real), cellsInVolume, model, (char *) __FILE__, __LINE__);
    safe_fread(v->br.w, sizeof(real), cellsInVolume, model, (char *) __FILE__, __LINE__);

    /* stop inner timer */
    tend_inner = dtime() - tstart_inner;

    /* stop timer and compute statistics */
    safe_fclose(modelname, model, (char *) __FILE__, __LINE__);
    tend_outer = dtime() - tstart_outer;

    iospeed_inner = (bytesForVolume / (1000.f * 1000.f)) / tend_inner;
    iospeed_outer = (bytesForVolume / (1000.f * 1000.f)) / tend_outer;

    print_stats("Initial velocity model loaded (%lf GB)", TOGB(1.f * bytesForVolume));
    print_stats("\tInner time %lf seconds (%lf MiB/s)", tend_inner, iospeed_inner);
    print_stats("\tOuter time %lf seconds (%lf MiB/s)", tend_outer, iospeed_outer);
    print_stats("\tDifference %lf seconds", tend_outer - tend_inner);

#endif /* end of DDO_NOT_PERFORM_IO clause */
};

void copy_velocity_model_ToGpu(const integer dimmz,
                               const integer dimmx,
                               const integer FirstYPlane,
                               const integer LastYPlane,
                               coeff_t *c,
                               s_t *s,
                               v_t *v,
                               real *rho,
                               coeff_t *gpu_c,
                               s_t *gpu_s,
                               v_t *gpu_v,
                               real *gpu_rho) {

    const integer cellsInVolume = dimmz * dimmx * (LastYPlane - FirstYPlane);

    cudaMemcpy(gpu_s->tl.zz, s->tl.zz, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->tl.xz, s->tl.xz, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->tl.yz, s->tl.yz, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->tl.xx, s->tl.xx, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->tl.xy, s->tl.xy, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->tl.yy, s->tl.yy, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->tr.zz, s->tr.zz, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->tr.xz, s->tr.xz, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->tr.yz, s->tr.yz, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->tr.xx, s->tr.xx, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->tr.xy, s->tr.xy, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->tr.yy, s->tr.yy, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->bl.zz, s->bl.zz, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->bl.xz, s->bl.xz, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->bl.yz, s->bl.yz, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->bl.xx, s->bl.xx, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->bl.xy, s->bl.xy, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->bl.yy, s->bl.yy, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->br.zz, s->br.zz, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->br.xz, s->br.xz, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_s->br.yz, s->br.yz, cellsInVolume, cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_c->c11, c->c11, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c12, c->c12, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c13, c->c13, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c14, c->c14, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c15, c->c15, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c16, c->c16, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c22, c->c22, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c23, c->c23, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c24, c->c24, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c25, c->c25, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c26, c->c26, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c33, c->c33, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c34, c->c34, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c35, c->c35, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c36, c->c36, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c44, c->c44, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c45, c->c45, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c46, c->c46, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c55, c->c55, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c56, c->c56, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c->c66, c->c66, cellsInVolume, cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_v->tl.u, v->tl.u, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v->tl.v, v->tl.v, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v->tl.w, v->tl.w, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v->tr.u, v->tr.u, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v->tr.v, v->tr.v, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v->tr.w, v->tr.w, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v->bl.u, v->bl.u, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v->bl.v, v->bl.v, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v->bl.w, v->bl.w, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v->br.u, v->br.u, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v->br.v, v->br.v, cellsInVolume, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v->br.w, v->br.w, cellsInVolume, cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_rho, rho, cellsInVolume, cudaMemcpyHostToDevice);
}


void copy_velocity_data_ToCPU ( v_t     *v,
                                s_t     *s,
                                coeff_t *c,
                                real    *rho,
                                v_t     *gpu_v,
                                s_t     *gpu_s,
                                coeff_t *gpu_c,
                                real    *gpu_rho,
                                const integer dimmx,
                                const integer dimmy,
                                const integer dimmz
){
    const integer cellsInVolume = dimmz * dimmx * dimmy;

    cudaMemcpy(v->tl.w,gpu_v->tl.w,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(v->tr.w,gpu_v->tr.w,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(v->bl.w,gpu_v->bl.w,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(v->br.w,gpu_v->br.w,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(v->tl.u,gpu_v->tl.u,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(v->tr.u,gpu_v->tr.u,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(v->bl.u,gpu_v->bl.u,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(v->br.u,gpu_v->br.u,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(v->tl.v,gpu_v->tl.v,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(v->tr.v,gpu_v->tr.v,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(v->bl.v,gpu_v->bl.v,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(v->br.v,gpu_v->br.v,  cellsInVolume, cudaMemcpyDeviceToHost);

    cudaMemcpy(s->bl.zz,gpu_s->bl.zz,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->br.zz,gpu_s->br.zz,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->tl.zz,gpu_s->tl.zz,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->tr.zz,gpu_s->tr.zz,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->bl.xz,gpu_s->bl.xz,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->br.xz,gpu_s->br.xz,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->tl.xz,gpu_s->tl.xz,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->tr.xz,gpu_s->tr.xz,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->bl.yz,gpu_s->bl.yz,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->br.yz,gpu_s->br.yz,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->tl.yz,gpu_s->tl.yz,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->tr.yz,gpu_s->tr.yz,  cellsInVolume, cudaMemcpyDeviceToHost);


    cudaMemcpy(s->tr.xx,gpu_s->tr.xx,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->tl.xx,gpu_s->tl.xx,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->br.xx,gpu_s->br.xx,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->bl.xx,gpu_s->bl.xx,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->tr.xy,gpu_s->tr.xy,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->tl.xy,gpu_s->tl.xy,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->br.xy,gpu_s->br.xy,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->bl.xy,gpu_s->bl.xy,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->tl.yy,gpu_s->tl.yy,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->tr.yy,gpu_s->tr.yy,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->bl.yy,gpu_s->bl.yy,  cellsInVolume, cudaMemcpyDeviceToHost);
    cudaMemcpy(s->br.yy,gpu_s->br.yy,  cellsInVolume, cudaMemcpyDeviceToHost);
};

/*
 * Saves the complete velocity field to disk.
 */
void write_snapshot(char *folder,
                    int suffix,
                    v_t *v,
                    const integer dimmz,
                    const integer dimmx,
                    const integer dimmy) {
#if defined(DO_NOT_PERFORM_IO)
    print_debug("We are not writing the snapshot here cause IO is not enabled!");
#else
    /* local variables */
    double tstart_outer, tstart_inner;
    double iospeed_outer, iospeed_inner;
    double tend_outer, tend_inner;
    const integer cellsInVolume = dimmz * dimmx * dimmy;
    char fname[300];
    int rank = 0;

#if defined(DISTRIBUTED_MEMORY_IMPLEMENTATION)
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif

    /* open snapshot file and write results */
    sprintf(fname, "%s/snapshot.%03d.%05d", folder, rank, suffix);

    print_debug("[Rank %d] is writting snapshot on %s", rank, fname);

    tstart_outer = dtime();
    FILE *snapshot = safe_fopen(fname, (char *) "wb", (char *) __FILE__, __LINE__);


    tstart_inner = dtime();
    safe_fwrite(v->tr.u, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fwrite(v->tr.v, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fwrite(v->tr.w, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);

    safe_fwrite(v->tl.u, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fwrite(v->tl.v, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fwrite(v->tl.w, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);

    safe_fwrite(v->br.u, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fwrite(v->br.v, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fwrite(v->br.w, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);

    safe_fwrite(v->bl.u, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fwrite(v->bl.v, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fwrite(v->bl.w, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);

    /* stop inner timer */
    tend_inner = dtime();

    /* close file and stop outer timer */
    safe_fclose(fname, snapshot, (char *) __FILE__, __LINE__);
    tend_outer = dtime();

    iospeed_inner = ((cellsInVolume * sizeof(real) * 12.f) / (1000.f * 1000.f)) / (tend_inner - tstart_inner);
    iospeed_outer = ((cellsInVolume * sizeof(real) * 12.f) / (1000.f * 1000.f)) / (tend_outer - tstart_outer);

    print_stats("Write snapshot (%lf GB)", TOGB(cellsInVolume * sizeof(real) * 12));
    print_stats("\tInner time %lf seconds (%lf MB/s)", (tend_inner - tstart_inner), iospeed_inner);
    print_stats("\tOuter time %lf seconds (%lf MB/s)", (tend_outer - tstart_outer), iospeed_outer);
    print_stats("\tDifference %lf seconds", tend_outer - tend_inner);

#endif
};

/*
 * Reads the complete velocity field from disk.
 */
void read_snapshot(char *folder,
                   int suffix,
                   v_t *v,
                   const integer dimmz,
                   const integer dimmx,
                   const integer dimmy) {
#if defined(DO_NOT_PERFORM_IO)
    print_debug("We are not reading the snapshot here cause IO is not enabled!");
#else
    /* local variables */
    double tstart_outer, tstart_inner;
    double iospeed_outer, iospeed_inner;
    double tend_outer, tend_inner;
    const integer cellsInVolume = dimmz * dimmx * dimmy;
    char fname[300];
    int rank = 0;

#if defined(DISTRIBUTED_MEMORY_IMPLEMENTATION)
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif

    /* open snapshot file and read results */
    sprintf(fname, "%s/snapshot.%03d.%05d", folder, rank, suffix);

    print_debug("[Rank %d] is freading snapshot from %s", rank, fname);

    tstart_outer = dtime();
    FILE *snapshot = safe_fopen(fname, (char *) "rb", (char *) __FILE__, __LINE__);

    tstart_inner = dtime();
    safe_fread(v->tr.u, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fread(v->tr.v, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fread(v->tr.w, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);

    safe_fread(v->tl.u, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fread(v->tl.v, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fread(v->tl.w, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);

    safe_fread(v->br.u, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fread(v->br.v, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fread(v->br.w, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);

    safe_fread(v->bl.u, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fread(v->bl.v, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);
    safe_fread(v->bl.w, sizeof(real), cellsInVolume, snapshot, (char *) __FILE__, __LINE__);

    /* stop inner timer */
    tend_inner = dtime() - tstart_inner;

    /* close file and stop outer timer */
    safe_fclose(fname, snapshot, (char *) __FILE__, __LINE__);
    tend_outer = dtime() - tstart_outer;

    iospeed_inner = ((cellsInVolume * sizeof(real) * 12.f) / (1000.f * 1000.f)) / tend_inner;
    iospeed_outer = ((cellsInVolume * sizeof(real) * 12.f) / (1000.f * 1000.f)) / tend_outer;

    print_stats("Read snapshot (%lf GB)", TOGB(cellsInVolume * sizeof(real) * 12));
    print_stats("\tInner time %lf seconds (%lf MiB/s)", tend_inner, iospeed_inner);
    print_stats("\tOuter time %lf seconds (%lf MiB/s)", tend_outer, iospeed_outer);
    print_stats("\tDifference %lf seconds", tend_outer - tend_inner);
#endif
};

void propagate_shot(time_d direction,
                    v_t v,
                    s_t s,
                    coeff_t coeffs,
                    real *rho,
                    v_t gpu_v,
                    s_t gpu_s,
                    coeff_t gpu_coeffs,
                    real *gpu_rho,
                    int timesteps,
                    int ntbwd,
                    real dt,
                    real dzi,
                    real dxi,
                    real dyi,
                    integer nz0,
                    integer nzf,
                    integer nx0,
                    integer nxf,
                    integer ny0,
                    integer nyf,
                    integer stacki,
                    char *folder,
                    real *UNUSED(dataflush),
                    integer dimmz,
                    integer dimmx,
                    integer dimmy) {
    double tglobal_start, tglobal_total = 0.0;
    double tstress_start, tstress_total = 0.0;
    double tvel_start, tvel_total = 0.0;
    double megacells = 0.0;

    int rank = 0, ranksize = 1;

#if defined(DISTRIBUTED_MEMORY_IMPLEMENTATION)
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &ranksize);
#endif


    for (int t = 0; t < timesteps; t++) {
        /* print out some information */
        // print_info("[Rank %d]  Computing %d-th timestep", rank, t);

        /* perform IO */
        if (t % stacki == 0 && direction == BACKWARD) read_snapshot(folder, ntbwd - t, &v, dimmz, dimmx, dimmy);

        tglobal_start = dtime();

        /* ------------------------------------------------------------------------------ */
        /*                      VELOCITY COMPUTATION                                      */
        /* ------------------------------------------------------------------------------ */
        /* Phase 1. Computation of the left-most planes of the domain */
        velocity_propagator(v, s, coeffs, rho, gpu_v, gpu_s, gpu_coeffs, gpu_rho,
                            dt, dzi, dxi, dyi,
                            nz0 + HALO,
                            nzf - HALO,
                            nx0 + HALO,
                            nxf - HALO,
                            ny0 + HALO,
                            ny0 + 2 * HALO,
                            dimmz, dimmx);


        /* Phase 1. Computation of the right-most planes of the domain */
        velocity_propagator(v, s, coeffs, rho,  gpu_v, gpu_s, gpu_coeffs, gpu_rho,
                            dt, dzi, dxi, dyi,
                            nz0 + HALO,
                            nzf - HALO,
                            nx0 + HALO,
                            nxf - HALO,
                            nyf - 2 * HALO,
                            nyf - HALO,
                            dimmz, dimmx);

        /* Boundary exchange for velocity values */
        exchange_velocity_boundaries(v, dimmz * dimmx, rank, ranksize, nyf, ny0);

        /* Phase2. Computation of the central planes. */
        tvel_start = dtime();

        velocity_propagator(v, s, coeffs, rho, gpu_v, gpu_s, gpu_coeffs, gpu_rho,
                            dt, dzi, dxi, dyi,
                            nz0 + HALO,
                            nzf - HALO,
                            nx0 + HALO,
                            nxf - HALO,
                            ny0 + HALO,
                            nyf - HALO,
                            dimmz, dimmx);

        tvel_total += (dtime() - tvel_start);


        //debug files
        copy_velocity_data_ToCPU(&v, &s, &coeffs, rho, &gpu_v, &gpu_s, &gpu_coeffs, gpu_rho,dimmx,dimmy,dimmz);

        write_velocity_datafile(&v, &s, &coeffs, rho,dimmx,dimmy,dimmz);

        /* ------------------------------------------------------------------------------ */
        /*                        STRESS COMPUTATION                                      */
        /* ------------------------------------------------------------------------------ */
        /* Phase 1. Computation of the left-most planes of the domain */
        stress_propagator(s, v, coeffs, rho, gpu_v, gpu_s, gpu_coeffs, gpu_rho,
                          dt, dzi, dxi, dyi,
                          nz0 + HALO,
                          nzf - HALO,
                          nx0 + HALO,
                          nxf - HALO,
                          ny0 + HALO,
                          ny0 + 2 * HALO,
                          dimmz, dimmx);

        /* Phase 1. Computation of the right-most planes of the domain */
        stress_propagator(s, v, coeffs, rho, gpu_v, gpu_s, gpu_coeffs, gpu_rho,
                          dt, dzi, dxi, dyi,
                          nz0 + HALO,
                          nzf - HALO,
                          nx0 + HALO,
                          nxf - HALO,
                          nyf - 2 * HALO,
                          nyf - HALO,
                          dimmz, dimmx);

        /* Boundary exchange for stress values */
        exchange_stress_boundaries(s, dimmz * dimmx, rank, ranksize, nyf, ny0);

        /* Phase 2 computation. Central planes of the domain (maingrid) */
        tstress_start = dtime();
        stress_propagator(s, v, coeffs, rho, gpu_v, gpu_s, gpu_coeffs, gpu_rho,
                          dt, dzi, dxi, dyi,
                          nz0 + HALO,
                          nzf - HALO,
                          nx0 + HALO,
                          nxf - HALO,
                          ny0 + HALO,
                          nyf - HALO,
                          dimmz, dimmx);

        tstress_total += (dtime() - tstress_start);
        tglobal_total += (dtime() - tglobal_start);

        /* perform IO */
        if (t % stacki == 0 && direction == FORWARD) write_snapshot(folder, ntbwd - t, &v, dimmz, dimmx, dimmy);
    }

    /* compute some statistics */
    megacells = ((nzf - nz0) * (nxf - nx0) * (nyf - ny0)) / 1e6;
    tglobal_total /= (double) timesteps;
    tstress_total /= (double) timesteps;
    tvel_total /= (double) timesteps;

    print_stats("Maingrid GLOBAL   computation took %lf seconds - %lf Mcells/s", tglobal_total,
                (2 * megacells) / tglobal_total);
    print_stats("Maingrid STRESS   computation took %lf seconds - %lf Mcells/s", tstress_total,
                megacells / tstress_total);
    print_stats("Maingrid VELOCITY computation took %lf seconds - %lf Mcells/s", tvel_total, megacells / tvel_total);
};

/* --------------- BOUNDARY EXCHANGES ---------------------------------------- */
void exchange_velocity_boundaries(v_t v,
                                  const integer plane_size,
                                  const integer rank,
                                  const integer nranks,
                                  const integer nyf,
                                  const integer ny0) {
    const integer num_planes = HALO;
    const integer nelems = num_planes * plane_size;

    const integer left_recv = ny0;
    const integer left_send = ny0 + HALO;

    const integer right_recv = nyf - HALO;
    const integer right_send = nyf - 2 * HALO;

    if (rank != 0) {
        // [RANK-1] <---> [RANK] communication
        EXCHANGE(&v.tl.u[left_send], &v.tl.u[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.tl.v[left_send], &v.tl.v[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.tl.w[left_send], &v.tl.w[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);

        EXCHANGE(&v.tr.u[left_send], &v.tr.u[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.tr.v[left_send], &v.tr.v[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.tr.w[left_send], &v.tr.w[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);

        EXCHANGE(&v.bl.u[left_send], &v.bl.u[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.bl.v[left_send], &v.bl.v[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.bl.w[left_send], &v.bl.w[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);

        EXCHANGE(&v.br.u[left_send], &v.br.u[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.br.v[left_send], &v.br.v[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.br.w[left_send], &v.br.w[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
    }

    if (rank != nranks - 1)  //task to exchange stress boundaries
    {
        //                [RANK] <---> [RANK+1] communication
        EXCHANGE(&v.tl.u[right_send], &v.tl.u[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.tl.v[right_send], &v.tl.v[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.tl.w[right_send], &v.tl.w[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);

        EXCHANGE(&v.tr.u[right_send], &v.tr.u[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.tr.v[right_send], &v.tr.v[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.tr.w[right_send], &v.tr.w[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);

        EXCHANGE(&v.bl.u[right_send], &v.bl.u[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.bl.v[right_send], &v.bl.v[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.bl.w[right_send], &v.bl.w[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);

        EXCHANGE(&v.br.u[right_send], &v.br.u[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.br.v[right_send], &v.br.v[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&v.br.w[right_send], &v.br.w[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
    }

    print_debug("Velocity boundaries exchanged correctly");
};

void exchange_stress_boundaries(s_t s,
                                const integer plane_size,
                                const integer rank,
                                const integer nranks,
                                const integer nyf,
                                const integer ny0) {
    const integer num_planes = HALO;
    const integer nelems = num_planes * plane_size;

    const integer left_recv = ny0;
    const integer left_send = ny0 + HALO;

    const integer right_recv = nyf - HALO;
    const integer right_send = nyf - 2 * HALO;

    if (rank != 0) {
        // [RANK-1] <---> [RANK] communication
        EXCHANGE(&s.tl.zz[left_send], &s.tl.zz[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tl.xz[left_send], &s.tl.xz[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tl.yz[left_send], &s.tl.yz[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tl.xx[left_send], &s.tl.xx[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tl.xy[left_send], &s.tl.xy[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tl.yy[left_send], &s.tl.yy[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);

        EXCHANGE(&s.tr.zz[left_send], &s.tr.zz[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tr.xz[left_send], &s.tr.xz[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tr.yz[left_send], &s.tr.yz[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tr.xx[left_send], &s.tr.xx[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tr.xy[left_send], &s.tr.xy[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tr.yy[left_send], &s.tr.yy[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);

        EXCHANGE(&s.bl.zz[left_send], &s.bl.zz[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.bl.xz[left_send], &s.bl.xz[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.bl.yz[left_send], &s.bl.yz[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.bl.xx[left_send], &s.bl.xx[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.bl.xy[left_send], &s.bl.xy[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.bl.yy[left_send], &s.bl.yy[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);

        EXCHANGE(&s.br.zz[left_send], &s.br.zz[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.br.xz[left_send], &s.br.xz[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.br.yz[left_send], &s.br.yz[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.br.xx[left_send], &s.br.xx[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.br.xy[left_send], &s.br.xy[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.br.yy[left_send], &s.br.yy[left_recv], rank - 1, rank, nelems, __FILE__, __LINE__);
    }

    if (rank != nranks - 1) {
        //                [RANK] <---> [RANK+1] communication
        EXCHANGE(&s.tl.zz[right_send], &s.tl.zz[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tl.xz[right_send], &s.tl.xz[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tl.yz[right_send], &s.tl.yz[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tl.xx[right_send], &s.tl.xx[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tl.xy[right_send], &s.tl.xy[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tl.yy[right_send], &s.tl.yy[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);

        EXCHANGE(&s.tr.zz[right_send], &s.tr.zz[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tr.xz[right_send], &s.tr.xz[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tr.yz[right_send], &s.tr.yz[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tr.xx[right_send], &s.tr.xx[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tr.xy[right_send], &s.tr.xy[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.tr.yy[right_send], &s.tr.yy[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);

        EXCHANGE(&s.bl.zz[right_send], &s.bl.zz[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.bl.xz[right_send], &s.bl.xz[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.bl.yz[right_send], &s.bl.yz[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.bl.xx[right_send], &s.bl.xx[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.bl.xy[right_send], &s.bl.xy[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.bl.yy[right_send], &s.bl.yy[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);

        EXCHANGE(&s.br.zz[right_send], &s.br.zz[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.br.xz[right_send], &s.br.xz[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.br.yz[right_send], &s.br.yz[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.br.xx[right_send], &s.br.xx[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.br.xy[right_send], &s.br.xy[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
        EXCHANGE(&s.br.yy[right_send], &s.br.yy[right_recv], rank + 1, rank, nelems, __FILE__, __LINE__);
    }

    print_debug("Stress boundaries exchanged correctly");
};

void EXCHANGE(const real *sendbuf,
              real *recvbuf,
              const integer dst,
              const integer src,
              const integer message_size,
              const char *file,
              const integer line) {
#if defined(DISTRIBUTED_MEMORY_IMPLEMENTATION)
    int err;
 int tag = 100;

 print_debug( "         [BEFORE]MPI sendrecv [count:%d][dst:%d][src:%d] %s : %d",
           message_size,  dst, src, file, line);

 MPI_Status  statuses[2];
 MPI_Request requests[2];

 MPI_Irecv( recvbuf, message_size, MPI_FLOAT, dst, tag, MPI_COMM_WORLD, &requests[0] );
 MPI_Isend( sendbuf, message_size, MPI_FLOAT, dst, tag, MPI_COMM_WORLD, &requests[1] );
 err = MPI_Waitall(2, requests, statuses);

 print_debug( "         [AFTER ]MPI sendrecv                          %s : %d",
           file, line);

 if ( err != MPI_SUCCESS )
   {
       print_error("MPI error %d!", err);
       abort();
   }

#endif
};


/*
 * =====================================================================================
 *
 *       Filename:  GenerateInputModel.c
 *
 *    Description:  Generates input velocity model for the FWI code
 *
 *        Version:  1.0
 *        Created:  26/01/16 11:55:02
 *       Revision:  none
 *       Compiler:  icc
 *
 *         Author:  Samuel Rodriguez Bernabeu (samuel.rodriguez@bsc.es)
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */

#include "fwi_sched.cuh"
#include "fwi_kernel.cuh"

/*
 * DISCLAIMER:
 * The model contains (dimmz +2*HALO) * (dimmx +2HALO) * (dimmy +2*HALO) 
 * cells. For now, it assumes that there are no boundary conditions (sponge)
 * implemented.
 * The initial velocity model should be loaded taking into account this
 * criteria.
 */

int main(int argc, const char *argv[])
{
	/* 
	 * Check input arguments
	 */
	if ( argc != 2 ) {
		print_error("Invalid argument count. Schedule file is requiered.");
		abort();
	}

	if ( argv[1] == NULL ) {
		print_error("Invalid argument. Schedule file path is NULL");
		abort();
	}

    /* set seed for random number generator */
    srand(314);
   
		/* Load schedule file */
		schedule_t S = load_schedule(argv[1]);

		/*
	 	* Modify input parameters according to FWIDIR value
	 	*/
		char foldername[500];
		char* fwipath = read_env_variable("FWIDIR");
		sprintf( foldername, "%s/InputModels", fwipath);
		create_folder(foldername);


		/* Generate one velocity model per frequency */
    for(int i=0; i<S.nfreqs; i++)
    {
        real waveletFreq   = S.freq[i];
				integer dimmz      = S.dimmz[i];
				integer dimmy      = S.dimmy[i];
				integer dimmx      = S.dimmx[i];

        print_info("Creating synthetic velocity input model for %f Hz freq", waveletFreq );


				/* generate complete path for output model */
        char modelname[500];
        sprintf( modelname, "%s/velocitymodel_(%.2f).bin", foldername, waveletFreq );

        FILE* model = safe_fopen( modelname, "wb", __FILE__, __LINE__);

				/* Compute number of cells per array */
				const integer cellsInVolume = dimmz * dimmy * dimmx;
        print_info("Number of cells in volume:"I"\n", cellsInVolume );
        real *buffer = __malloc( ALIGN_REAL, sizeof(real) * cellsInVolume);

        /* safe dummy buffer */
        for(int i = 0; i < WRITTEN_FIELDS; i++)
        {
            set_array_to_random_real( buffer, cellsInVolume );
            safe_fwrite( buffer, sizeof(real), cellsInVolume, model, __FILE__, __LINE__);
        }

        /* free buffer */
        __free( buffer );

        /*  close model file */
        safe_fclose( modelname, model, __FILE__, __LINE__);

        print_info("Model %s created correctly", modelname);
    }

		schedule_free(S);

    print_info("End of the program");
    return 0;
}

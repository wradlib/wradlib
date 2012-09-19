/*------------------------------------------------------------------------

    bufr2wradlib interface file 

----------------------------------------------------------------------------

FILE:          BUFR2WRADLIB.C
IDENT:         $Id: apisample.c,v 1.3 2009/05/15 16:09:14 helmutp Exp $

AUTHOR:        Maik Heistermann
               Institute of Earth and Environmental Sciences, 
               University of Potsdam, Germany

STATUS:        UNDER DEVELOPMENT

--------------------------------------------------------------------------- */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "bufrlib.h"
#include "bufr2wradlib.h"
#include "bufr_io.h"
#include "dll.h"

/*===========================================================================*/
/* internal function definitons                                              */
/*===========================================================================*/

#define RIOUTFILE  "img.dec"    /* Name of file for uncompressed radar image */

static int bufr_py_out (varfl val, int ind);
static int bufr_char_to_string (varfl val, int ind);

/*===========================================================================*/
/* internal data                                                             */
/*===========================================================================*/

radar_data_t our_data; /* sturcture holding our decoded data */
radar_data2_t our_data2; /* sturcture holding our decoded data */
char *version = "apisample V3.0, 5-Dec-2007\n";
char descstr[100000]="";
char * bufrfilename;
char * destfilename;

/*===========================================================================*/



void bufr_decoding_sample (bufr_t* msg, radar_data_t* data) {

    sect_1_t s1;
    int ok, desch, ndescs, subsets;
    dd* dds = NULL;

    /* initialize variables */

    memset (&s1, 0, sizeof (sect_1_t));

    /* Here we read our BUFR message from a file */
	bufr_read_file (msg, bufrfilename);

    /* decode section 1 */

    ok = bufr_decode_sections01 (&s1, msg);
	
    /* Write section 1 to ASCII file */

    bufr_sect_1_to_file (&s1, "section.1.out");

    /* read descriptor tables */

    if (ok) ok = (read_tables (NULL, s1.vmtab, s1.vltab, s1.subcent, 
                               s1.gencent) >= 0);
							   
    /* decode data descriptor and data-section now */

    /* open bitstreams for section 3 and 4 */

    desch = bufr_open_descsec_r(msg, &subsets);
    ok = (desch >= 0);
	
    if (ok) ok = (bufr_open_datasect_r(msg) >= 0);
	
    /* calculate number of data descriptors  */
    
    ndescs = bufr_get_ndescs (msg);

    /* allocate memory and read data descriptors from bitstream */

    if (ok) ok = bufr_in_descsec (&dds, ndescs, desch);
	
    /* output data to our global data structure */

    while (ok && subsets--) {
        ok = bufr_parse_out (dds, 0, ndescs - 1, bufr_py_out, 1);
	}

	our_data.desctable = descstr;
    /* get data from global */

    data = &our_data;

    /* close bitstreams and free descriptor array */

    if (dds != (dd*) NULL)
        free (dds);
    bufr_close_descsec_r (desch);
    bufr_close_datasect_r ();

    /* decode data to file also */

    if (ok) ok = bufr_data_to_file ("apisample.src", "apisample.img", msg);
	
    bufr_free_data (msg);
	
    free_descs();
	
/*    exit (EXIT_SUCCESS); */


}


void bufr_decode_via_file (bufr_t* msg, radar_data2_t* data) {

    sect_1_t s1;
    int ok;
	FILE * pFile;
	long lSize;
	size_t result;

    /* initialize variables */

    memset (&s1, 0, sizeof (sect_1_t));

    /* Here we could also read our BUFR message from a file */
	bufr_read_file (msg, bufrfilename);

    /* decode section 1 */

    ok = bufr_decode_sections01 (&s1, msg);
	
    /* Write section 1 to ASCII file */

    bufr_sect_1_to_file (&s1, "section.1.out");

    /* read descriptor tables */

    if (ok) ok = (read_tables (NULL, s1.vmtab, s1.vltab, s1.subcent, 
                               s1.gencent) >= 0);
							   

    /* decode data to file also */
    if (ok) ok = bufr_data_to_file ("apisample.src", "apisample.img", msg);
	
	pFile = fopen("apisample.img", "rb");
    fseek (pFile , 0 , SEEK_END);
	lSize = ftell (pFile);
	rewind (pFile);
	our_data2.data = (unsigned char*) malloc (sizeof(unsigned char)*lSize);
	result = fread (our_data2.data,1,lSize,pFile);
	fclose(pFile);
	printf ("lSize=%d\n",lSize);
	
	pFile = fopen("apisample.src", "r");
    fseek (pFile , 0 , SEEK_END);
	lSize = ftell (pFile);
	rewind (pFile);
	our_data2.desctable = (char*) malloc (sizeof(char)*lSize);
	result = fread (our_data2.desctable,1,lSize,pFile);
	fclose(pFile);
	
    bufr_free_data (msg);
	
    free_descs();
	
/*    exit (EXIT_SUCCESS); */


}


#ifdef __cplusplus
extern "C" {
#endif


EXPORT radar_data_t py2bufr (char* buffile) {

    bufr_t bufr_msg ;   /* structure holding encoded bufr message */
	bufrfilename = buffile;

    /* initialize variables */

    memset (&bufr_msg, 0, sizeof (bufr_t));
    memset (&our_data, 0, sizeof (radar_data_t));
	
    /* sample for encoding to BUFR */

    bufr_decoding_sample (&bufr_msg, &our_data);

    bufr_free_data (&bufr_msg);

    free (our_data.data);
	
	return(our_data);

	/*    exit (EXIT_SUCCESS); */
}


EXPORT radar_data2_t py2bufr2 (char* buffile) {

    bufr_t bufr_msg ;   /* structure holding encoded bufr message */
	bufrfilename = buffile;

    /* initialize variables */

    memset (&bufr_msg, 0, sizeof (bufr_t));
    memset (&our_data2, 0, sizeof (radar_data2_t));
	
    /* sample for encoding to BUFR */
    bufr_decode_via_file (&bufr_msg, &our_data2);

    bufr_free_data (&bufr_msg);

    free (our_data2.data);
	
	return(our_data2);

	/*    exit (EXIT_SUCCESS); */
}



EXPORT int decbufr2py (char* buffile, char* destfile)

{
    /*char destfile[200], buffile[200]*/;
    char *table_dir = NULL;
    char *sect1_file = "section.1.out";
    char imgfile[200];  /* filename of uncompressed image */
    sect_1_t s1;
    bufr_t bufr_msg;    /* structure holding encoded bufr message */
	
	bufrfilename = buffile;
	destfilename = destfile;
	strcpy (imgfile, RIOUTFILE);

    /* initialize variables */

    memset (&bufr_msg, 0, sizeof (bufr_t));
    memset (&s1, 0, sizeof (sect_1_t));

 
    /* read source-file. Therefore allocate memory to hold the complete
       BUFR-message */
	

    if (!bufr_read_file (&bufr_msg, bufrfilename)) {
        bufr_free_data (&bufr_msg);
        return(0);
    }

    /* decode section 1 */

    if (!bufr_decode_sections01 (&s1, &bufr_msg)) {
        bufr_free_data (&bufr_msg);
        return(0);
    }

    /* Write section 1 to ASCII file */

    if (!bufr_sect_1_to_file (&s1, sect1_file)) {
        bufr_free_data (&bufr_msg);
        return(0);
    }

    /* read descriptor tables */

    if (read_tables (table_dir, s1.vmtab, s1.vltab, s1.subcent, 
                     s1.gencent) < 0) {
        bufr_free_data (&bufr_msg);
        free_descs();
        return(0);
    }

    /* decode data descriptor- and data-section now */

    if (!bufr_data_to_file (destfilename, imgfile, &bufr_msg)) {
        fprintf (stderr, "unable to decode BUFR-message !\n");
        bufr_free_data (&bufr_msg);
        free_descs();
        return(0);
    }

#ifdef VERBOSE
    {
        int i;
        for (i = 0; i < 6; i++) {
            fprintf (stderr, "section %d length = %d\n", i, bufr_msg.secl[i]);
        }
    }
#endif

    bufr_free_data (&bufr_msg);
    free_descs();
	/*exit (EXIT_SUCCESS);*/
	return(1);
}



#ifdef __cplusplus
}
#endif

/*===========================================================================*/



/*based on bufr_char_to_file from bufr_io, adapted by heistermann*/
static int bufr_char_to_string (varfl val, int ind)

{
    char* tmp;
	assert (ind == ccitt_special);

    if (val == 0) val = 0x20;
    
    sprintf (tmp, "%c", (int) val);
	strcat (descstr, tmp);
    return 1;
}



/*===========================================================================*/
/** \ingroup cbout
    \brief Writes to a global data structure, mainly based on bufr_file_out

    This function outputs data values and descriptors to an data structure
    opened by \ref bufr_open_output_file.
    In case of CCITT (ASCII) data it calls \ref bufr_parse_out 
    with the callback  \ref bufr_char_to_file for output of the 
    single characters. \n
    In case of sequence descriptors it checks if the descriptor is a special
    OPERA bitmap (currently descriptors 3 21 192 to 3 21 197, 3 21 200 
    and 3 21 202) and in this case writes the data to a special file 
    . For normal sequence descriptors it just
    calls  bufr_parse_out again. \n
    The function also makes use of the global \ref _replicating flag in order
    to decide whether it has to print out the data descriptors or not.

    \param[in] val    Data-value to be output.
    \param[in] ind    Index to the global array \ref des[] holding the 
                      description of known data-descriptors or special 
                      descriptor (\ref ccitt_special, _desc_special,
                      add_f_special).

    \return The function returns 1 on success, 0 on a fault.

    \see bufr_src_in, bufr_open_output_file, bufr_close_output_file,
    bufr_parse_out, _replicating
*/

static int bufr_py_out (varfl val, int ind)

{
    int depth = 1, nv, ok;
    char sval[80];
    char fname[512], tmp[80];
    char* unit;
    dd* d;
    static int nchars = 0;    /* number of characters for ccitt output */
    static int in_seq = 0;    /* flag to indicate sequences */
    static int first_in_seq;  /* flag to indicate first element in sequence */
    static int count = 0;     /* counter for image files */
    bufrval_t* vals;
	
	/*added by heistermann in addtion to bufr_file_out*/
	int nr, nc;
	radar_data_t* b = &our_data;   /* our global data structure */
	char tmpstr[500];
	static char* imgfile_ = NULL;    /* filename for uncompressed bitmap */
	imgfile_ = "dummy.img";

    /* sanity checks */

    if (des[ind] == (desc*) NULL) {
        fprintf (stderr, "Data not available for bufr_py_out!\n");
        return 0;
    }
	
    /* element descriptor */

    if (des[ind]->id == ELDESC) {

        d = &(des[ind]->el->d);

        /* output descriptor if not inside a sequence */

        if (!in_seq && ind != ccitt_special && !_replicating 
            && ind != add_f_special) {
/*            fprintf (fo_, "%2d %2d %3d ", d->f, d->x, d->y);*/
			sprintf (tmpstr, "%2d %2d %3d ", d->f, d->x, d->y);
			strcat(descstr, tmpstr);
		}

        /* descriptor without data (1.x.y, 2.x.y) or ascii) */

        if (ind == _desc_special) {

            unit = des[ind]->el->unit;

            /* special treatment for ASCII data */

            if (unit != NULL && strcmp (unit, "CCITT IA5") == 0) {
/*                fprintf (fo_, "       '");*/
				sprintf (tmpstr, "       '");
				strcat(descstr, tmpstr);
                if (!bufr_parse_out (d, 0, 0, bufr_char_to_string, 0)) {
                    return 0;
                }
/*                fprintf (fo_, "'\n");*/
				sprintf (tmpstr, "'\n");
				strcat(descstr, tmpstr);
                nchars = des[ind]->el->dw / 8;                
            }

            /* only descriptor -> add newline */
            
            else if (!in_seq && !_replicating) {
/*                fprintf (fo_, "\n");*/
				sprintf (tmpstr, "\n");
				strcat(descstr, tmpstr);
            }
        }

        /* "normal" data */

        else { 

            /* check for missing values and flag tables */

            if (val == MISSVAL) {
                strcpy (sval, "      missing");
            }
#if BUFR_OUT_BIN
            else if (desc_is_flagtable (ind)) {
                place_bin_values (val, ind, sval);
            }
#endif
            else {
                sprintf (sval, "%13.5f", val);
            }

            /* do we have a descriptor before the data element? */

            if (!in_seq && !_replicating && ind != add_f_special) {
/*                fprintf (fo_, "%s            %s\n", 
                         sval, des[ind]->el->elname);*/
                sprintf (tmpstr, "%s            %s\n", 
                         sval, des[ind]->el->elname);
				strcat(descstr, tmpstr);
            }
            else {
                if (!first_in_seq) {
/*                    fprintf (fo_, "          ");*/
					sprintf (tmpstr, "          ");
					strcat(descstr, tmpstr);
				}

/*                fprintf (fo_, "%s  %2d %2d %3d %s\n", 
                         sval, d->f, d->x, d->y, des[ind]->el->elname);*/
                sprintf (tmpstr, "%s  %2d %2d %3d %s\n", 
                         sval, d->f, d->x, d->y, des[ind]->el->elname);
				strcat(descstr, tmpstr);
                first_in_seq = 0;
            }
        }
    } /* end if ("Element descriptor") */

    /* sequence descriptor */

    else if (des[ind]->id == SEQDESC) {

        d = &(des[ind]->seq->d);

        /* output descriptor if not inside another sequence descriptor */

        if (!in_seq && !_replicating) {
/*            fprintf (fo_, "%2d %2d %3d ", d->f, d->x, d->y);*/
			sprintf (tmpstr, "%2d %2d %3d ", d->f, d->x, d->y);
			strcat(descstr, tmpstr);
		}

        /* check if bitmap or "normal" sequence descriptor */

        depth = check_bitmap_desc (d);

        /* seqdesc is a special opera bitmap */

        if (depth > 0) {

            strcpy (fname, imgfile_);

            /* Add the counter to the filename */
            
            if (count != 0) {
                sprintf (tmp, "%d", count);
                strcat (fname, tmp);
            }
            count ++;

            /* read bitmap and run length decode to file */

            vals = bufr_open_val_array ();
            if (vals == (bufrval_t*) NULL) return 0;

            _opera_mode = 1;
            if (!bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                                 bufr_val_to_global, 0)) {
                _opera_mode = 0;
                bufr_close_val_array ();
                return 0;
            }
            _opera_mode = 0;
            nv = vals->nvals;

            if (depth == 8)
            {
/*                if (!z_decompress_to_file (fname, vals->vals, &nv)) 
                { 
                    bufr_close_val_array ();
                    fprintf (stderr, "Error during z-compression.\n");
                    return 0;
                }
*/
				bufr_close_val_array ();
				fprintf (stderr, "Cannot use z_decompression - the present BUFR file cannot be decoded, yet.\n");
				return 0;

            } else {

            /* Runlength decode */
/*            
                if (!rldec_to_file (fname, vals->vals, depth, &nv)) 
                { 
                    bufr_close_val_array ();
                    fprintf (stderr, "Error during runlength-compression.\n");
                    return 0;
                }
				
*/
				if (!rldec_to_mem (vals->vals, &(b->data), &nv, &nr, &nc)) { 
					bufr_close_val_array ();
					fprintf (stderr, "Error during runlength-compression.\n");
					return 0;
				}
				printf("Depth=%d\n", depth);
				printf("nvals=%d\nncols=%d\nnrows=%d\n", nv, nc, nr);

            }

            if (in_seq || _replicating) {
/*                fprintf (fo_, "        ");*/
				sprintf (tmpstr, "        ");
				strcat(descstr, tmpstr);
			}

/*            fprintf (fo_, "%s\n", fname);*/
			sprintf (tmpstr, "%s\n", fname);
			strcat(descstr, tmpstr);

            /* free array */

            bufr_close_val_array ();
            return 1;
        } 
        /* normal sequence descriptor - just call bufr_parse_out and
           remember that we are in a sequence */
        
        else {
            if (in_seq == 0)
                first_in_seq = 1;
            in_seq ++;
            ok = bufr_parse_out (des[ind]->seq->del, 0, 
                                 des[ind]->seq->nel - 1, bufr_py_out, 1);
            in_seq --;
            return ok;
        }
    } /* if ("seqdesc") */
    return 1;
}

/* end of file */

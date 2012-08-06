/*------------------------------------------------------------------------

    BUFR encoding and decoding software and library
    Copyright (c) 2007,  Institute of Broadband Communication, TU-Graz
    on behalf of EUMETNET OPERA, http://www.knmi.nl/opera

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; version 2.1 
    of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 

----------------------------------------------------------------------------

FILE:          BUFR_IO.C
IDENT:         $Id: bufr_io.c,v 1.10 2010/05/27 17:36:57 helmutp Exp $

AUTHORS:       Juergen Fuchsberger, Helmut Paulitsch
               Institute of Broadband Communication
               Technical University Graz, Austria

VERSION NUMBER:3.0

DATE CREATED:  30-NOV-2007

STATUS:        DEVELOPMENT FINISHED


AMENDMENT RECORD:

$Log: bufr_io.c,v $
Revision 1.10  2010/05/27 17:36:57  helmutp
allow longer input lines and replace 0x0 characters
in CCITT ascii by 0x20

Revision 1.9  2010/02/15 17:45:10  helmutp
bug fix in byteswap64

Revision 1.8  2010/02/15 11:29:11  helmutp
moved bitmap checking code to desc.c

Revision 1.7  2009/12/18 15:58:51  helmutp
source code improvements and comments

Revision 1.6  2009/11/26 14:11:31  helmutp
added gzip compression

Revision 1.5  2009/09/24 08:25:25  helmutp
chanegd check for special opera bitmaps

Revision 1.4  2009/05/15 15:11:21  helmutp
api change to support subsets

Revision 1.3  2009/04/17 16:07:28  helmutp
decoding of subsets, ascii input for 2 5 y

Revision 1.2  2007/12/18 14:40:13  fuxi
added licence header

Revision 1.1  2007/12/07 08:34:56  fuxi
Initial revision


--------------------------------------------------------------------------- */

/** \file bufr_io.c
    \brief Functions for reading/writing to/from OPERA format ASCII
           BUFR files.
    
    This file contains functions for reading/writing to/from OPERA 
    format ASCII BUFR files.
*/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include "desc.h"
#include "bufr.h"
#include "bitio.h"
#include "rlenc.h"

#define BUFR_OUT_BIN 0 /**< \brief Output to binary format for flag tables */

/* \brief Stucture that holds a decoded (source)
                                bufr message */
typedef struct bufrsrc_s {  

    sect_1_t s1;             /**< \brief section 1 information */
    dd* descs;               /**< \brief array of data descriptors */
    int ndesc;               /**< \brief number of data descriptors */
    int desci;               /**< \brief current index into descs */
    bd_t* data;              /**< \brief array of data elements */
    int ndat;                /**< \brief number of data elements */
    int datai;               /**< \brief current index into data */
} bufrsrc_t;

extern int errno;

static bufrsrc_t* src_ = NULL;   /* structure containing data for encoding */
static int nrows_ = -1;          /* number of rows for bitmap */
static int ncols_ = -1;          /* number of colums for bitmap */
static FILE *fo_ = NULL;         /* File-Pointer for outputfile */
static char* imgfile_ = NULL;    /* filename for uncompressed bitmap */
static char* char_ = NULL;       /* character array for ascii input */
static int   cc_ = 0;            /* index into char array for ascii input */

static int bufr_read_src_file (FILE* fp, bufrsrc_t* data);
#if BUFR_OUT_BIN
static void place_bin_values (varfl val, int ind, char* buf);
#endif
static int bufr_input_char (varfl* val, int ind);
static int desc_to_array (dd* d, bufrsrc_t* data);
static int string_to_array (char* s, bufrsrc_t* bufr);
static int bufr_src_in (varfl* val, int ind);
static int bufr_file_out (varfl val, int ind);
static int bufr_char_to_file (varfl val, int ind);
static bufrsrc_t* bufr_open_src ();
static void bufr_close_src ();
static FILE* bufr_open_output_file (char* name);
static void bufr_close_output_file ();
static char *str_save(char *str);
static void replace_bin_values (char *buf);

static int z_decompress_to_file (char* outfile, varfl* vals, int* nvals);
static int z_compress_from_file (char* infile, varfl* *vals, int* nvals);
static void byteswap64 (unsigned char *buf, int n);

#define MAX_LINE 2000        /* Max. linelength in input file */
#define MAX_DATA  1000000    /* Maximum number of data elements */

/*===========================================================================*/

/** \ingroup operaio
    \brief read data and descriptors from ASCII file and code them into
    sections 3 and 4

    This function reads descriptors and data from an
    ASCII file and codes them into a BUFR data descriptor and data
    section (section 3 and 4).
    Memory for both sections is allocated in 
    this function and must be freed by the calling functions using
    \ref bufr_free_data.

    \param[in] file  Name of the input ASCII file
    \param[in,out] msg  BUFR message to contain the coded sections

    \return 1 on succes, 0 on error

    \see bufr_data_to_file, bufr_create_msg, bufr_free_data

*/

int bufr_data_from_file(char* file, bufr_t* msg)
{
    FILE* fp;
    bufrsrc_t* src;
    int ok = 0, desch = -1;

    /* open file */

    fp = fopen (file, "r");
    if (fp == (FILE*) NULL) {
        fprintf (stderr, "Could not open file %s\n", file);
        return 0;
    }

    /* open global src structure for holding data from file */

    src = bufr_open_src ();
    ok = (src != (bufrsrc_t*) NULL);

    /* read data from file to arrays */

    if (ok) {
        ok = bufr_read_src_file (fp, src);
        fclose (fp);
    }

    /* output descriptors to section 3 */

    if (ok) {
        /* open bitstream */

        desch = bufr_open_descsec_w (1);        
        ok = (desch >= 0);
    }

    if (ok)
        /* write descriptors to bitstream */
        
        ok = bufr_out_descsec (src->descs, src->ndesc, desch);

         /* close bitstream and write data to msg */
    
    bufr_close_descsec_w (msg, desch);

    /* parse descriptors and encode data to section 4 */

    if (ok)
        /* open bitstream */
        
        ok = (bufr_open_datasect_w () >= 0);

    if (ok) 
        /* write data to bitstream */

        ok = bufr_parse_in (src->descs, 0, src->ndesc - 1, bufr_src_in, 1);

        /* close bitstream and write data to msg */

    bufr_close_datasect_w (msg);

    /* free src and cleanup globals */

    bufr_close_src ();

    return ok;
}

/*===========================================================================*/
/** \ingroup operaio
    \brief Decode data and descriptor sections of a BUFR message
    and write them to an ASCII file

    This functions decodes data and descriptor sections of a BUFR message
    and writes them into an ASCII file. 
    If there is an OPERA bitmap (currently descriptors 3 21 192 to 3 21 197,
    3 21 200 and 3 21 202) it is written to a seperate file.

    \param[in] file  Name of the output ASCII file
    \param[in] imgfile  Name of the output bitmap file(s)
    \param[in] msg  BUFR message to contain the coded sections

    \return 1 on succes, 0 on error

    \see bufr_data_from_file, bufr_read_msg

*/

int bufr_data_to_file (char* file, char* imgfile, bufr_t* msg) {

    dd *dds = NULL;
    int ok;
    int ndescs, desch, subsets;

    /* open output-file */

    if (bufr_open_output_file (file) == (FILE*) NULL) {
        fprintf (stderr, "Unable to open outputfile '%s'\n", file);
        return 0;
    }

    /* set image file name */

    imgfile_ = imgfile;

    /* open bitstreams for section 3 and 4 */

    desch = bufr_open_descsec_r(msg, &subsets);

    ok = (desch >= 0);

    if (ok)
        ok = (bufr_open_datasect_r(msg) >= 0);

    /* calculate number of data descriptors  */
    
    ndescs = bufr_get_ndescs (msg);

    /* allocate memory and read data descriptors from bitstream */

    if (ok) 
        ok = bufr_in_descsec (&dds, ndescs, desch);


    /* output data and descriptors */

    if (ok)
      while (subsets--) 
        ok = bufr_parse_out (dds, 0, ndescs - 1, bufr_file_out, 1);
        
    /* close bitstreams and free descriptor array */

    if (dds != (dd*) NULL)
        free (dds);
    bufr_close_descsec_r (desch);
    bufr_close_datasect_r ();
    bufr_close_output_file ();

    return ok;
}


/*===========================================================================*/
/* local functions */
/*===========================================================================*/

/** read data and descriptors from an OPERA BUFR file and store it into
   a bufrsrc_t structure 

   \param[in] fp Pointer to the source file
   \param[in,out] data Is where the data and descriptors are to be stored

   \return 1 on success, 0 on error

*/
static int bufr_read_src_file (FILE* fp, bufrsrc_t* data) {

    dd d;                 /* current descriptor */
    char buf[MAX_LINE];   /* strung containing current line */
    char s[MAX_LINE];     /* string containing data element or filename */
    char* sbuf = NULL;  
    int l, n, ascii_flag = 0;

    if (fp == NULL || data == (bufrsrc_t*) NULL) return 0;

    /* read each line and process it */

    while (fgets (buf, MAX_LINE, fp) != NULL) {

        ascii_flag = 0;
        l = strlen (buf);

        /* delete terminating \n and blanks */

        if (buf[l-1] == '\n') buf[l-1] = 0;
        trim (buf);                        
      
        /* ignore comments */

        if (buf[0] == '#' || strlen (buf) == 0)
            continue;


        /* check for ascii and binary data */

        if (strstr (buf, " '") != NULL) {

            sbuf = strstr (buf, " '");
            if (sbuf == NULL) {
                fprintf (stderr, "Error reading ASCII data from input file\n");
                return 0;
            }
            sbuf++;
            ascii_flag = 1;
        } else {
            
            /* replace binary values by integers */

            replace_bin_values (buf);
        }

        /* check for data descriptor and data */

        n = sscanf (buf, "%d %d %d %s", &d.f, &d.x, &d.y, s);


        /* replication and modification descriptors don't have values */

        if (d.f == 1 || (d.f == 2 && d.x != 5))
            n = sscanf (buf, "%d %d %d", &d.f, &d.x, &d.y);

        /* descriptor and data */

        if (n == 4) {
            if (!desc_to_array (&d, data)) return 0;
            if (ascii_flag) {
                if (!string_to_array (sbuf, data)) return 0;
            } else {
                if (!string_to_array (s, data)) return 0;
            }
        }
        /* only descriptor */

        else if (n == 3) {
            if (!desc_to_array (&d, data)) return 0;
        }
        /* only data */

        else {
            if (ascii_flag) {
                if (!string_to_array (sbuf, data)) return 0;
            } else {
                if (!sscanf (buf, "%s", s)) return 0;
                if (!string_to_array (s, data)) return 0;
            }
        }
    }
    return 1;
}

/*===========================================================================*/
/* \ingroup cbinutl
    \brief Opens bufrsrc structure for function \ref bufr_src_in

    This functions opens a structure to hold BUFR data descriptors and
    data elements from an ASCII file for use by \ref bufr_src_in
    and returns the pointer to this structure.

    \return Pointer to the BUFR src structure or NULL if an error occured.

    \see bufr_close_src, bufr_read_src, bufr_src_in
*/

    
   
static bufrsrc_t* bufr_open_src () {

    if (src_ != (bufrsrc_t*) NULL) {

        fprintf (stderr, "Global src structure not available!\n");
        return (bufrsrc_t*) NULL;
    }

    src_ = malloc (sizeof (bufrsrc_t));
    if (src_ == (bufrsrc_t*) NULL) {

        fprintf (stderr, "Error allocating memory for src structure!\n");
        return (bufrsrc_t*) NULL;
    }

    memset (src_, 0, sizeof (bufrsrc_t));

    return src_;
}

/*===========================================================================*/
/* \ingroup cbinutl
    \brief Closes bufrsrc structure for function \ref bufr_src_in

    This functions closes the structure used by \ref bufr_src_in

    \see bufr_open_src, bufr_src_in
*/

static void bufr_close_src () {

    int i;

    if (src_ == (bufrsrc_t*) NULL) return;

    if (src_->data != (bd_t*) NULL) {
        for (i = 0; i < src_->ndat; i++)
            free (src_->data[i]);
        free (src_->data);
    }

    if (src_->descs != (dd*) NULL)
        free (src_->descs);

    free (src_);
    src_ = (bufrsrc_t*) NULL;

}


/*===========================================================================*/
/* \ingroup cbin

   \brief Gets next data value from BUFR source data.

   This function

   \param[out] val The received value
   \param[in] ind   Index to the global array 
                      \ref des[] holding the description of
                      known data-descriptors.

   \return 1 on success, 0 on error
*/

static int bufr_src_in (varfl* val, int ind) {

    char* line;        /* current string we have to convert */
    int datai;         /* index to current data element */
    dd* d;             /* current descriptor */
    int depth = 0;     /* image depth in bytes per pixel for bitmaps */
    int ok = 0; 
    bufrval_t* vals;

    assert (val != (varfl*) NULL);
    assert (src_ != (bufrsrc_t*) NULL);

    /* get next line frome array */

    datai = src_->datai;
    line = src_->data[datai];
    if (line == NULL) {
        fprintf (stderr, "Data element empty!\n");
        return 0;
    }

    /* element descriptor */

    if (des[ind]->id == ELDESC) {

        d = &(des[ind]->el->d);

        /* special treatment for ASCII data */

        if (ind == _desc_special) {
            char* unit;

            unit = des[ind]->el->unit;
            if (unit != NULL && strcmp (unit, "CCITT IA5") == 0) {
                char_ = line;
                cc_ = 0;
                if (!bufr_parse_in (d, 0, 0, bufr_input_char, 0)) {
                    return 0;
                }
                /* check if we reached end of string */

                if (char_[cc_+1] != '\'') {
                    fprintf (stderr, 
                             "Number of bits missmatch for ascii data!\n");
                    return 0;
                }
                cc_ = 0;
                char_ = NULL;
                src_->datai++;
                return 1;
            }
            else {
                return 1;
            }
        }

        /* "normal" data -> get one value */

        else {

            /* check for missing */

            if (strstr (line, "missing") != NULL ||
                strstr (line, "MISSING") != NULL) {
                *val = MISSVAL;
                src_->datai++;
                return 1;
            }

            /* convert to varfl */

            errno = 0;
            *val = (varfl) strtod (line, NULL);
            src_->datai++;
            if (errno) {
                fprintf (stderr, "Error reading value from bufr_src\n");
                return 0;
            }
            
            /* check for number of rows / columns or
               bins / rays of radar bitmap */

            if (bufr_check_fxy(d, 0, 30, 21) > 0 ||
                bufr_check_fxy(d, 0, 30, 195) > 0)
                ncols_ = (int) *val;
            if (bufr_check_fxy(d, 0, 30, 22) > 0 ||
                bufr_check_fxy(d, 0, 30, 194) > 0)
                nrows_ = (int) *val;

            return 1;
        }
    }
    /* sequence descriptor */

    else if (des[ind]->id == SEQDESC) {

        /* check if bitmap or "normal" sequence descriptor */
        
        d = &(des[ind]->seq->d);
        
        depth = check_bitmap_desc(d);

        /* seqdesc is a special opera run length encoded bitmap */

        if (depth > 0) {

            if (nrows_ <= 0 || ncols_ <= 0) {
                fprintf (stderr, "Unknown number of rows and/or columns\n");
                return 0;
            }

            /* read bitmap and run length encode to memory */

            /* initialize array */

            vals = bufr_open_val_array ();

            if (vals == (bufrval_t*) NULL) return 0;

            if (depth == 8)
            {
                if (!z_compress_from_file (line, &(vals->vals), &(vals->nvals)))
                { 
                    fprintf (stderr, "Error during z-compression.\n");
                    bufr_close_val_array ();
                    return 0;
                }
            } else {
                if (!rlenc_from_file (line, nrows_, ncols_, &(vals->vals), 
                                      &(vals->nvals), depth)) 
                { 
                    fprintf (stderr, "Error during runlength-compression.\n");
                    bufr_close_val_array ();
                    return 0;
                  }
            }
            src_->datai++;

            ok = bufr_parse_in (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                                bufr_val_from_global, 0);

            /* free array */

            bufr_close_val_array ();
            return ok;
        } 
        /* normal sequence descriptor - just call bufr_parse_in with 
           all descriptors in sequence */
        
        else {
            return bufr_parse_in (des[ind]->seq->del, 0, 
                                  des[ind]->seq->nel - 1, bufr_src_in, 1);
        }

    }
    else {
        fprintf (stderr, "Unknown descriptor in bufr_src_in!\n");
        return 0;
    }
}



/*===========================================================================*/
/** \ingroup cboututl
    \brief Opens file for ouput of BUFR data in ASCII format

    This functions opens a file for output to ASCII by \ref bufr_file_out 
    and returns its pointer.

    \param[in] name The name of the output file.
    
    \return Pointer to the file or NULL on error.

    \see bufr_file_out, bufr_close_output_file
*/

static FILE* bufr_open_output_file (char* name) {

     if (fo_ != (FILE*) NULL) {
         fprintf (stderr, "Global output file not available!\n");
         return (FILE*) NULL;
     }

     fo_ = fopen (name, "w");
     return fo_;
 }

/*===========================================================================*/
/** \ingroup cboututl
    \brief Closes ASCII file opened by \ref bufr_open_output_file

    This functions closes the ASCII output file used by \ref bufr_file_out 

    \see bufr_file_out, bufr_open_output_file
*/

static void bufr_close_output_file () {

    if (fo_ == (FILE*) NULL) return;
    fclose (fo_);
    fo_ = (FILE*) NULL;
}


/*===========================================================================*/
/** \ingroup cbout
    \brief Outputs one value + descriptor to an ASCII-file

    This function outputs data values and descriptors to an ASCII file
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

static int bufr_file_out (varfl val, int ind)

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

    /* sanity checks */

    if (des[ind] == (desc*) NULL || fo_ == (FILE*) NULL 
        || imgfile_ == (char* ) NULL) {
        fprintf (stderr, "Data not available for bufr_file_out!\n");
        return 0;
    }

    /* element descriptor */

    if (des[ind]->id == ELDESC) {

        d = &(des[ind]->el->d);

        /* output descriptor if not inside a sequence */

        if (!in_seq && ind != ccitt_special && !_replicating 
            && ind != add_f_special)
            fprintf (fo_, "%2d %2d %3d ", d->f, d->x, d->y);

        /* descriptor without data (1.x.y, 2.x.y) or ascii) */

        if (ind == _desc_special) {

            unit = des[ind]->el->unit;

            /* special treatment for ASCII data */

            if (unit != NULL && strcmp (unit, "CCITT IA5") == 0) {
                fprintf (fo_, "       '");
                if (!bufr_parse_out (d, 0, 0, bufr_char_to_file, 0)) {
                    return 0;
                }
                fprintf (fo_, "'\n");
                nchars = des[ind]->el->dw / 8;                
            }

            /* only descriptor -> add newline */
            
            else if (!in_seq && !_replicating) {
                fprintf (fo_, "\n");
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
                fprintf (fo_, "%s            %s\n", 
                         sval, des[ind]->el->elname);
            }
            else {
                if (!first_in_seq) 
                    fprintf (fo_, "          ");

                fprintf (fo_, "%s  %2d %2d %3d %s\n", 
                         sval, d->f, d->x, d->y, des[ind]->el->elname);
                first_in_seq = 0;
            }
        }
    } /* end if ("Element descriptor") */

    /* sequence descriptor */

    else if (des[ind]->id == SEQDESC) {

        d = &(des[ind]->seq->d);

        /* output descriptor if not inside another sequence descriptor */

        if (!in_seq && !_replicating)
            fprintf (fo_, "%2d %2d %3d ", d->f, d->x, d->y);

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
                if (!z_decompress_to_file (fname, vals->vals, &nv)) 
                { 
                    bufr_close_val_array ();
                    fprintf (stderr, "Error during z-compression.\n");
                    return 0;
                }
            } else {

            /* Runlength decode */
            
                if (!rldec_to_file (fname, vals->vals, depth, &nv)) 
                { 
                    bufr_close_val_array ();
                    fprintf (stderr, "Error during runlength-compression.\n");
                    return 0;
                }
            }

            if (in_seq || _replicating) 
                fprintf (fo_, "        ");

            fprintf (fo_, "%s\n", fname);

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
                                 des[ind]->seq->nel - 1, bufr_file_out, 1);
            in_seq --;
            return ok;
        }
    } /* if ("seqdesc") */
    return 1;
}


/*===========================================================================*/
/** \ingroup cbout Outputs one character of an ASCII string to a file
    \brief Output one CCITT character to an ASCII file.

   This function outputs one CCITT (ASCII) character to a file which was
   opened by \ref bufr_open_output_file. 

   \param[in] val    Data-value to be output.
   \param[in] ind    Index to the global array \ref des[] holding the 
                     description of known data-descriptors.

   \return The function returns 1 on success, 0 on a fault.

   \see bufr_file_out, bufr_open_output_file, bufr_close_output_file
*/

static int bufr_char_to_file (varfl val, int ind)


{
    assert (ind == ccitt_special);

    if (fo_ == (FILE*) NULL) {
        fprintf (stderr, "Global file pointer not available!\n");
        return 0;
    }
    
    if (val == 0) val = 0x20;
    
    fprintf (fo_, "%c", (int) val);
    return 1;
}

/*===========================================================================*/
/** Reads next character from char_ and  stores position in cc_.

   \param[out] val The value of the character
   \param[in]  ind Index to the global array \ref des[] holding the 
               description of known data-descriptors.

   \return 1 on success, 0 on error.

   \see bufr_src_in


*/
static int bufr_input_char (varfl* val, int ind) {

    assert (ind == ccitt_special);

    if (char_ == NULL) {
        fprintf (stderr, "Global char pointer not available!\n");
        return 0;
    }
        
    /* check for correct string */
    
    if (*char_ != '\'') {
        fprintf (stderr, 
                 "Possible number of bits missmatch for ASCII data 1!\n");
        return 0;
    }

    /* check for correct number of characters */

    if (char_[cc_+1] == 0 || char_[cc_+1] == '\'') {
        fprintf (stderr, "Number of bits missmatch for ASCII data\n");
        return 0;
    }

    /* copy character to float */

    *val = (varfl) (unsigned char) char_[cc_+1];
    cc_++;


    return 1;

}

/*===========================================================================*/
/** Add one descriptor to array, allocate memory for array if necessary.
   Memory has to be freed by calling function!

   \param[in] d descriptor to be wrote
   \param[in,out] data The BUFR src structure containing the descriptor array

   \return 1 on success, 0 on error
 */

static int desc_to_array (dd* d, bufrsrc_t* data)

{
    int nd = data->ndesc;       /* number of data descriptors */
    dd* descs = data->descs;    /* array of data descriptors */

    if (nd > MAX_DESCS) {
        fprintf (stderr, "ERROR maximum number of descriptors exceeded!\n");
        return 0;
    }

    /* Allocate memory if not yet done */

    if (descs == (dd*) NULL) {
        descs = (dd *) malloc (MEMBLOCK * sizeof (dd));
        if (descs == (dd*) NULL) {
            fprintf (stderr, 
                     "Could not allocate memory for descriptor array!\n");
            return 0;
        }
		memset (descs, 0, MEMBLOCK * sizeof (dd));
        nd = 0;
    }

    /* Check if memory block is large enough to hold new data 
       and reallocate memory if not */

    if (nd != 0 && nd % MEMBLOCK == 0) {
        descs = (dd *) realloc (descs, (nd + MEMBLOCK) * sizeof (dd));
        if (descs == (dd*) NULL) {
            fprintf (stderr, 
                     "Could not reallocate memory for descriptor array!\n");
            return 0;
        }
		memset ((dd *) (descs + nd), 0, MEMBLOCK * sizeof (dd));
    }

    /* Add descriptor to array */

    memcpy ((dd*) (descs + nd), d, sizeof (dd));
    nd ++;
    data->ndesc = nd;
    data->descs = descs;
    return 1;
}

/*===========================================================================*/
/** Add one data string to array, allocate memory for array if necessary.
   Memory has to be freed by calling function!

   \param[in] s String to be wrote
   \param[in,out] bufr The BUFR src structure containing the data array

   \return 1 on success, 0 on error

 */

static int string_to_array (char* s, bufrsrc_t* bufr)

{
    int ns = bufr->ndat;        /* number of data elements */
    bd_t* data =  bufr->data;     /* array of data elements */
    

    if (ns > MAX_DATA) {
        fprintf (stderr, "ERROR maximum number of data elements exceeded!\n");
        return 0;
    }

    /* Allocate memory if not yet done */

    if (data == (bd_t*) NULL) {
        data = (bd_t*) malloc (MEMBLOCK * sizeof(bd_t));
        if (data == (bd_t*) NULL) {
            fprintf (stderr, 
                     "Could not allocate memory for data array!\n");
            return 0;
        }
		memset (data, 0, MEMBLOCK * sizeof(bd_t));
        ns = 0;
    }

    /* Check if memory block is large enough to hold new data 
       and reallocate memory if not */

    if (ns != 0 && ns % MEMBLOCK == 0) {
        data = (bd_t*) realloc (data, (ns + MEMBLOCK) * sizeof (bd_t));
        if (data == (bd_t*) NULL) {
            fprintf (stderr, 
                     "Could not reallocate memory for data array!\n");
            return 0;
        }
		memset ((bd_t*) (data + ns), 0, MEMBLOCK * sizeof (bd_t));
    }

    /* Add descriptor to array */

    data[ns] = str_save (s);
    ns ++;
    bufr->ndat = ns;
    bufr->data = data;
    return 1;
}
/*===========================================================================*/
/**
   Saves a string into a newly allocated memory area

   \param[in] str         pointer to the string.


   \return the pointer to the start of the string or NULL if a
           memory allocation error occured.
*/
static char *str_save(char *str)
{
    register char *p;
    int l;
    
    /* get the string size */
    
    l = strlen(str) + 1;

    /* allocate memory */
        
    p = malloc(l);
    if (p == NULL) {
        fprintf (stderr, 
                 "Could not allocate memory for string!\n");
        return NULL;
    }

    /* copy string into memory */
    
    memcpy(p, str, l);

    /* return pointer to string */
    
    return p;
}

/*===========================================================================*/
/** replaces binary values given as "b0101001101" by integers. 

    \param[in,out] buf String containing the value

*/

static void replace_bin_values (char *buf)



{
  char *p, *q, *r;
  int bin_val, v, i;
  
  p = buf;
  while ((p = strstr (p + 1, " b")) != NULL) {

    /* check if that is really a binary value and get the end the the beginning of that value */
  
    q = p + 2;
    bin_val = 1;
    while (*q != 0 && *q != ' ') {
      if (*q != '0' && *q != '1') bin_val = 0;
      q ++;
    }

    /* If it is a binary value convert it to an integer */

    if (bin_val) {
      r = q; r --;
      v = 0;
      for (i = 0; r >= p; i ++, r --) {
        if (*r == '1') v |= 1 << i;
      }

      /* Finally replace the binary data be the integer */

      sprintf (p + 1, "%d", v);
    }
  }
}
/*===========================================================================*/
/** Replaces an integer by its binary representation

    \param[in] val The value to be converted to binary format
    \param[in] ind Index to the global array \ref des[] holding the 
                     description of known data-descriptors.
    \param[in,out] buf  Pointer to the output string.

*/
#if BUFR_OUT_BIN
static void place_bin_values (varfl val, int ind, char* buf) {

    int dwi, i, bit;
    assert (buf != NULL);
    dwi = des[ind]->el->dw;

    strcpy (buf, "");
    for (i = 0; i < 13 - 1 - dwi; i ++) strcat (buf, " ");
    strcat (buf, "b");
    for (i = dwi - 1; i >= 0; i --) {
        if (1 << i & (long) val) {
            bit = 1;
        } else {
            bit = 0;
        }
        sprintf (buf + strlen (buf), "%d", bit);
    }
}
#endif

/*===========================================================================*/
/** Byte swap of 64bit values if host platform uses big endian

    \param[in,out] buf  buffer holding 64 bit values
    \param[in]     n    number of 64 bit values in buffer
*/
void byteswap64 (unsigned char *buf, int n)
{
    int i;
    unsigned char c;

    unsigned one = 1;
    unsigned char *test = (unsigned char *) &one;

    if (*test == 1)
        return;

    for (i = 0; i < n; i+= 8)
    {
        c = buf[0];
        buf[0] = buf[7];
        buf[7] = c;
        c = buf[1];
        buf[1] = buf[6];
        buf[6] = c;
        c = buf[2];
        buf[2] = buf[5];
        buf[5] = c;
        c = buf[3];
        buf[3] = buf[4];
        buf[4] = c;
        buf += 8;
    }
}

#include "zlib.h"
#define MAXBLOCK 65534

/*===========================================================================*/
/** z-decompression of array of bufr values with compressed bytes.
 *  Writes 64bit floats in platfrom native form to file 
 *  The float-bytes are swapped if the host representation is different
 *  from the IEEE byte order. 

    \param[in] infile  Name of input file
    \param[in,out] vals    Array of compressed bytes stored as bufr values
    \param[in,out] nvals   Number of values in the array
    \return 1 for success, 0 on error
*/
int z_decompress_to_file (char* outfile, varfl* vals, int* nvals)
{
    FILE *fp;
    unsigned char *cbuf, *buf;
    int i, nv, ncols, nrows, sz;
    z_stream zs;
    
    memset (&zs, 0, sizeof(zs));
    fp = fopen (outfile, "wb");
    if (fp == NULL) {
        fprintf (stderr, "Could not open file %s!\n", outfile);
        return 0;
    }

    cbuf = malloc (MAXBLOCK);
    buf = malloc (MAXBLOCK);
    if (cbuf == NULL || buf == NULL)
    {
        fprintf (stderr, "malloc error\n");
        if (cbuf != NULL) free (cbuf);
        if (buf != NULL) free (buf);
        return 0;
    }
    
    inflateInit (&zs);
    
    sz = 0;
    nv = 0;
    nv++;
    nrows = vals[nv++];
    while (nrows-- > 0)
    {
        ncols = vals[nv++];
        for (i = 0; i < ncols; i++)
        {
            cbuf[i] = (unsigned char) vals[nv++];
        }
        zs.next_in = cbuf;
        zs.avail_in = ncols;
        while (zs.avail_in > 0)
        {
            int err;
            zs.next_out = buf + sz;
            zs.avail_out = MAXBLOCK - sz;
            err = inflate (&zs, Z_SYNC_FLUSH);
            if (err != Z_OK && err != Z_STREAM_END)
                break;

            sz = (MAXBLOCK - zs.avail_out) / 8 * 8;
            byteswap64 (buf, sz); 
            fwrite (buf, 1, sz, fp);
            sz = MAXBLOCK - zs.avail_out - sz;
            if (sz > 0)
                memmove (buf, buf + MAXBLOCK - zs.avail_out - sz, sz);
        }
    }

    inflateEnd (&zs);
    free (buf);
    free (cbuf);
    fclose(fp);
    *nvals = nv;
    return 1;
}


/*===========================================================================*/
/** Reads 64bit floats in platfrom native form from file, apllies 
 *  z-compression and puts the compressed bytes as bufr values in the array.
 *  The float-bytes are swapped if the host representation is different
 *  from the IEEE byte order. 

    \param[in] infile  Name of input file
    \param[in,out] vals    Array of compressed bytes stored as bufr values
    \param[in,out] nvals   Number of values in the array
    \return 1 for success, 0 on error
*/
int z_compress_from_file (char* infile, varfl* *vals, int* nvals)
{
    FILE *fp;
    int nv, sz, n;
    unsigned char *buf, *cbuf;
    unsigned long sz1;
    
    fp = fopen (infile, "rb");
    if (fp == NULL) {
        fprintf (stderr, "error opening '%s'\n", infile);
        return 0;
    }

    fseek (fp, 0, SEEK_END);
    sz = ftell (fp);
    
    if ((buf = malloc (sz)) == NULL)
    {
        fclose (fp);
        fprintf (stderr, "malloc error\n");
        return 0;
    }
    
    fseek (fp, 0, SEEK_SET);
    if (fread (buf, 1, sz, fp) != sz)
    {
        fclose (fp);
        free (buf);
        fprintf (stderr, "read error\n");
        return 0;
    }
    fclose (fp);

    byteswap64 (buf, sz);
    
    sz1 = sz + sz / 1000 + 100 + 12;
    if ((cbuf = malloc (sz1)) == NULL)
    {
        free (buf);
        fprintf (stderr, "malloc error\n");
        return 0;
    }
    if (compress (cbuf, &sz1, buf, sz) != Z_OK)
    {
        free (buf);
        free (cbuf);
        fprintf (stderr, "compress error\n");
        return 0;
    }
    
    free (buf);
    buf = cbuf;
    sz = sz1;
    nv = (sz + MAXBLOCK - 1) / (MAXBLOCK);
    
    bufr_val_to_array (vals, 0, nvals);
    bufr_val_to_array (vals, nv, nvals);

    while (nv-- > 0)
    {
        n = sz < MAXBLOCK ? sz : MAXBLOCK;
        bufr_val_to_array (vals, n, nvals);
        while (n-- > 0)
        {
            if (bufr_val_to_array (vals, *buf++, nvals) == 0)
                break;
            sz--;
        }
        if (n > 0)
            break;
    }
    return sz == 0;
}


#define VV(i) ((i - 50000)/100.0)
void z_test()
{
    bufrval_t* vals;
    varfl v;
    int i, nvals;
    FILE *f;
    
    f = fopen ("test.1", "w");
    for (i = 0; i < 100000; i++)
    {
        v = VV(i);
        fwrite (&v, sizeof(v), 1, f);
    }
    fclose(f);

    vals = bufr_open_val_array();
    z_compress_from_file ("test.1", &vals->vals, &vals->nvals);
    z_decompress_to_file ("test.2", vals->vals, &nvals);
    bufr_close_val_array();

    f = fopen ("test.2", "r");
    for (i = 0; i < 100000; i++)
    {
        fread (&v, sizeof(v), 1, f);
        if (v != VV(i))
            printf ("%6d: %12.6f %12.6f\n", i, v, VV(i));
    }
    fclose(f);
}

/* end of file */


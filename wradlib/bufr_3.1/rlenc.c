/*-------------------------------------------------------------------------

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

FILE:          RLENC.C
IDENT:         $Id: rlenc.c,v 1.11 2010/05/27 17:35:01 helmutp Exp $

AUTHOR:        Konrad Koeck
               Institute of Communication and Wave Propagation, 
               Technical University Graz, Austria

VERSION NUMBER:3.0

DATE CREATED:  18-DEC-2001

STATUS:        DEVELOPMENT FINISHED


FUNCTIONAL DESCRIPTION:
-----------------------

Functions to encode and decode an "n byte per pixel" radar image to
to and from BUFR runlength-code

AMENDMENT RECORD:

ISSUE       DATE            SCNREF      CHANGE DETAILS
-----       ----            ------      --------------
V2.0        18-DEC-2001     Koeck       Initial Issue

$Log: rlenc.c,v $
Revision 1.11  2010/05/27 17:35:01  helmutp
read/write values as float

Revision 1.10  2009/10/21 16:06:06  helmutp
extension to read/write already compressed bytes

Revision 1.9  2009/06/18 17:10:38  helmutp
runlength compression with float values

Revision 1.8  2009/04/10 11:08:05  helmutp
support for PGM P5 format

Revision 1.7  2007/12/18 14:40:13  fuxi
added licence header

Revision 1.6  2007/12/07 08:35:21  fuxi
update to version 3.0

Revision 1.5  2005/04/04 14:57:28  helmutp
update to version 2.3

Revision 1.4  2004/03/04 13:38:30  kon
change the buffer length from 1000 to 2000

Revision 1.3  2003/03/27 17:17:39  helmutp
update to version 2.2

Revision 1.2  2003/03/06 17:12:32  helmutp
update to version 2.1

Revision 1.1  2003/02/28 13:41:12  helmutp
Initial revision

--------------------------------------------------------------------------- */
/** \file rlenc.c
    \brief Functions for run-length encoding and decoding
    
    This file contains all functions used for run-length encoding
    and decoding of image files.
*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "desc.h"
#include "bufr.h"
#include "rlenc.h"

#define LBUFLEN 5000         /**< \brief Size of the internal buffer holding 
                                one uncompressed line */
#define ENCBUFL 5000         /**< \brief Size of the internal buffer holding 
                                one compressed line */

/*===========================================================================*/
/** \ingroup deprecated_g

    \deprecated Use \ref rlenc_from_file instead.
    \brief Runlength-encodes a radar image


    This function encodes a "one byte per pixel" radar image to BUFR runlength-
   code and stores the resulting values by a call to VAL_TO_ARRAY.

   \param[in] infile   File holding the "one byte per pixel" radar image.
   \param[in] ncols    Number of columns of the image.
   \param[in] nrows    Number of rows of the image.
   \param[in,out] vals     Float-array holding the coded image.
   \param[in,out] nvals    Number of values in VALS.

   \return The return-value ist 1 on success, 0 on a fault.
*/

int rlenc (char* infile, int nrows, int ncols, varfl** vals, size_t* nvals)

{
  FILE *fp;
  unsigned char buf[LBUFLEN];
  int i, n;

/* check if the internal buffer is large enough to hold one uncompressed
         line */

  assert (ncols <= LBUFLEN);

/* open file holding the radar image */

  fp = fopen (infile, "rb");
  if (fp == NULL) {
    fprintf (stderr, "error opening '%s'\n", infile);
    return 0;
  }

/* output number of rows */

  val_to_array (vals, (varfl) nrows, nvals);  

/* compress line by line */

  for (i = 0; i < nrows; i ++) {
    n = fread (buf, 1, ncols, fp);
    if (n != ncols) {
      fprintf (stderr, "read error from file '%s'\n", infile);
      goto err;
    }
    if (!rlenc_compress_line (i, buf, ncols, vals, nvals)) goto err;
  }
  fclose (fp);
  return 1;

err:
  fclose (fp);
  return 0;
}


/*===========================================================================*/
/** \ingroup deprecated_g
    \deprecated Use \ref rlenc_compress_line_new instead.

    \brief Encodes one line of a radar image to BUFR runlength-code

    This function encodes one line of a radar image to BUFR runlength-code and
    stores the resulting values by a call to \ref val_to_array.

    \param[in] line     Line number.
    \param[in] src      Is where the uncompressed line is stored.
    \param[in] ncols    Number of pixels per line.
    \param[in,out] dvals     Float-array holding the coded image.
    \param[in,out] nvals    Number of values in VALS.
   
    \return The function returns 1 on success, 0 on a fault.
*/

int rlenc_compress_line (int line, unsigned char* src, int ncols, 
                         varfl** dvals, size_t* nvals)


{
  int count, i, n, npar, lens[LBUFLEN], cw, ncgi, nngi = 0;
  unsigned char val, lval = 0, vals[LBUFLEN];
  varfl encbuf[ENCBUFL];

  /* compress Line into a runlength format */

  count = n = 0;
  for (i = 0; i < ncols; i ++) {
    val = *(src + i);
    if (i != 0 && (val != lval || count >= 255)) {  /* (n >= 255) to ensure that BUFR-descriptor 0 31 001 does not exceed */
      lens[n] = count;
      vals[n] = lval;
      n ++;
      count = 0;
      lval = val;
    }
    lval = val;
    count ++;
  }
  lens[n] = count;
  vals[n] = lval;
  n ++;
  

  /* line is runlength-compressed now to N parts, each of them identified 
     by a length (LENS) and a value (VALS). */
    

  /* Count number of parcels. One parcel is identified by a COUNT of 1
     followed by a COUNT > 1 */

  npar = 0;
  for (i = 0; i < n - 1; i ++) if (lens[i] == 1 && lens[i+1] > 1) npar ++;
  npar ++;

  /* output line-number */

  for (i = 0; i < ENCBUFL; i ++) encbuf[i] = (varfl) 0.0;
  cw = 0;
  encbuf[cw++] = line;  

  /* compress it to parcels */

  encbuf[cw++] = npar;                   /* number of parcels */

  ncgi = cw ++;                          /* is where the number of compressable groups is stored */
  encbuf[ncgi] = (varfl) 0.0;            /* number of compressable groups */

  i = 0;
  for (i = 0; i < n; i ++) {
    if (lens[i] > 1) {                   /* compressable group found */
      if (i > 0 && lens[i-1] == 1) {     /* A new parcel starts here */
        ncgi = cw ++;                    /* is where the number of compressable groups is stored */
        encbuf[ncgi] = (varfl) 0.0;      /* number of compressable groups */
      }
      encbuf[ncgi] += 1.0;
      encbuf[cw++] = lens[i];
      encbuf[cw++] = vals[i];
    }
    else {                               /* non compressable group found */
      if (i == 0 || lens[i-1] != 1) {    /* this is the first uncompressable group in the current parcel */
        nngi = cw ++;                    /* is where the number of non compressable groups is stored */
        encbuf[nngi] = (varfl) 0.0;      /* Number of non compressable groups */
      }
      encbuf[nngi] += 1.0;
      encbuf[cw++] = vals[i];
    }
  }
  if (lens[n-1] != 1) encbuf[cw++] = 0;  /* number of noncompressable groups in the last parcel = 0 */
  assert (cw <= ENCBUFL);

  /* compresson to parcels finished */


  for (i = 0; i < cw; i ++)
      if (!val_to_array (dvals, encbuf[i], nvals)) return 0;

  /* Output data for debugging purposes */

  /*cw = 0;
  printf ("\n\nline no. %d:\n", (int) encbuf[cw++]);
  npar = (int) encbuf[cw];
  printf ("number of parcels: %d\n", (int) encbuf[cw++]);
  for (i = 0; i < npar; i ++) {
    ncgi = (int) encbuf[cw];
    printf ("number of compressable groups: %d\n", (int) encbuf[cw++]);
    for (j = 0; j < ncgi; j ++) {
      printf ("count: %d\n", (int) encbuf[cw++]);
      printf ("val: %d\n", (int) encbuf[cw++]);
    }
    nngi = (int) encbuf[cw];
    printf ("number of uncompressable pixels: %d\n", (int) encbuf[cw++]);
    for (j = 0; j < nngi; j ++) {
      printf ("val: %d\n", (int) encbuf[cw++]);
    }
  }*/

  return 1;
}

/*===========================================================================*/
/** \ingroup deprecated_g
    \deprecated Use \ref rldec_to_file instead.

    \brief Decodes a BUFR-runlength-encoded radar image

    This function decodes a BUFR-runlength-encoded radar image stored at
    \p VALS. The decoded image is stored in a one "byte-per-pixel-format" at
    the file \p OUTFILE.

    \param[in] outfile  Destination-file for the "one byte per pixel" radar 
                        image.
    \param[in] vals     Float-array holding the coded image.
    \param[in] nvals    Number of values needed for the radar image.

    \return The return-value ist 1 on success, 0 on a fault.
*/

int rldec (char* outfile, varfl* vals, size_t* nvals)

{
  FILE *fp;
  int i, j, k, l, ngr, nrows, npar, val, count, nup;
  varfl *ovals;

/* Open destination-file for output */

  fp = fopen (outfile, "wb");
  if (fp == NULL) return 0;

/* decode line by line */

  ovals = vals;
  nrows = (int) *vals ++;               /* number of rows */
  for (i = 0; i < nrows; i ++) {        /* loop for lines */
      vals ++;                            /* skip linenumber */
      npar = (int) *vals ++;              /* number of parcels */
      for (j = 0; j < npar; j ++) {       /* loop for parcels */
          ngr = (int) *vals ++;             /* number of compressable groups */
          for (k = 0; k < ngr; k ++) {      /* loop for compressable groups */
              count = (int) *vals ++;
              val =   (int) *vals ++;
              for (l = 0; l < count; l ++) {  /* loop for length of group */
                  fputc (val, fp);
              }
          }
          nup = (int) *vals ++;             /* number of uncompressable pixels */
          for (k = 0; k < nup; k ++) {      /* loop for uncompressable pixels */
              val = (int) *vals ++;
              fputc (val, fp);
          }
      }
  }

  /* close file */

  fclose (fp);

  /* calculate number of values in VALS occupied by the radar image */

  *nvals = vals - ovals;
  return 1;
}

/*===========================================================================*/
/* New functions */
/*===========================================================================*/

/** \ingroup rlenc_g

    \brief Runlength-encodes a radar image from a file to an array

    This function encodes a radar image file with \p depth bytes per pixel
    to BUFR runlength-code and stores the resulting values into an
    array \p vals by a call to \ref bufr_val_to_array. 

    Currently \p depth can be one or two bytes per pixel.
    In case of two bytes per pixel data is read in 
    "High byte - low byte order". So pixel values 256 257 32000 are represented
    by 0100 0101 7D00 hex.  

    \note In difference to the old \ref rlenc function the
    initial length of \p vals must be given in the parameter
    \p nvals in order to prevent \ref bufr_val_to_array from writing to an
    arbitrary position.

    \param[in] infile   File holding the radar image.
    \param[in] ncols    Number of columns of the image.
    \param[in] nrows    Number of rows of the image.
    \param[in] depth    Image depth in bytes
    \param[in,out] vals     Float-array holding the coded image.
    \param[in,out] nvals    Number of values in VALS.

    \return The return-value ist 1 on success, 0 on a fault.

    \see rlenc_from_mem, rldec_to_file, rlenc_compress_line_new
*/
int rlenc_from_file (char* infile, int nrows, int ncols, varfl* *vals, 
                     int *nvals, int depth)

{
    FILE *fp;
    unsigned char cbuf[LBUFLEN * 2];
    unsigned int ibuf[LBUFLEN];
    float fbuf[LBUFLEN];
    int i, n, j, ok;
/*
    if (depth > 4) {
        fprintf (stderr, 
                 "Unsupported number of bits per bixel!\n");
        return 0;
    }
*/
    /* check if the internal buffer is large enough to hold one uncompressed
       line */

    if (ncols > LBUFLEN) {
        fprintf (stderr, "ERROR: Number of columns larger than %d!\n",
                 LBUFLEN);
        return 0;
    }

    /* open file holding the radar image */

    fp = fopen (infile, "rb");
    if (fp == NULL) {
        fprintf (stderr, "error opening '%s'\n", infile);
        return 0;
    }

    /* read values as float from file */ 

    if (depth > 4) 
    {
        while ((n = fread (fbuf, sizeof(float), LBUFLEN, fp)) > 0)
        {
            for (i = 0; i< n; i++)
                bufr_val_to_array (vals, fbuf[i], nvals);
        }
        fclose(fp);
        return 1;
    }

    /* read P5 header for pgm-format */

    if (strstr (infile, ".pgm") != NULL || strstr (infile, ".PGM"))
    {
        fscanf (fp, "P5 %d %d %d ", &i, &j, &n);
        if (i != nrows || j != ncols || (n > 255 && depth < 2)) 
        {
            fprintf (stderr, "error in pgm file '%s'\n", infile);
            return 0;
        }
    }

    /* output number of rows */

    bufr_val_to_array (vals, (varfl) nrows, nvals);  

    /* compress line by line */

    for (i = 0; i < nrows; i ++) {

        /* read row from file */

        if (depth == 4)
            n = fread (fbuf, 1, ncols * depth, fp);
        else
            n = fread (cbuf, 1, ncols * depth, fp);
        if (n != ncols * depth) {
            fprintf (stderr, "read error from file '%s'\n", infile);
            fclose (fp);
            return 0;
        }
            
        /* convert to integer */
        
        if (depth == 1) {
            for (j = 0; j < ncols; j ++)
                ibuf[j] = (unsigned int) cbuf[j];
        } else if (depth == 2) {
            for (j = 0; j < ncols; j++)
                ibuf[j] = (cbuf[j*2] << 8) + cbuf[j*2+1];
        }

        /* compress line to varfl array */

        if (depth == 4)
            ok =rlenc_compress_line_float (i, fbuf, ncols, vals, nvals);
        else
            ok =rlenc_compress_line_new (i, ibuf, ncols, vals, nvals);
    }
    fclose (fp);
    return ok;
}

/*===========================================================================*/
/** \ingroup rldec_g

    \brief Decodes a BUFR-runlength-encoded radar image to a file

    This function decodes a BUFR-runlength-encoded radar image stored at
    \p vals. The decoded image is stored in a "\p depth byte-per-pixel-format"
    at the file \p outfile.
    Currently \p depth can be one or two bytes per pixel.
    In case of two bytes per pixel data is stored in 
    "High byte - low byte order". So pixel values 256 257 32000 are represented
    by 0100 0101 7D00 hex.  

   \param[in] outfile  Destination-file for the radar image.
   \param[in] vals     Float-array holding the coded image.
   \param[in] depth    Number of bytes per pixel
   \param[out] nvals   Number of \ref varfl values needed for the 
                       compressed radar image.

   \return The return-value ist 1 on success, 0 on a fault.

   \see rldec_to_mem, rldec_decompress_line, rlenc_from_file
*/

int rldec_to_file (char* outfile, varfl* vals, int depth, int* nvals)

{
    FILE *fp;
    int i, j, nrows, ncols, nc, nv;
    unsigned int ibuf[LBUFLEN];
    unsigned char cbuf[LBUFLEN*2];
    float fbuf[LBUFLEN];
    varfl *ovals;
/*
    if (depth > 4) {
        fprintf (stderr, 
                 "Unsupported number of bits per bixel!\n");
        return 0;
    }
*/
    /* Open destination-file for output */

    fp = fopen (outfile, "wb");
    if (fp == NULL) {
        fprintf (stderr, "Could not open file %s!\n", outfile);
        return 0;
    }

    /* write values as float to file */ 
 
    if (depth > 4) 
    {
        nv = *nvals;
        while (nv > 0)
        {
            nc = nv > LBUFLEN ? LBUFLEN : nv;
            for (i = 0; i < nc; i++)
                fbuf[i] = *vals++;
            fwrite (fbuf, sizeof(float), nc, fp);
            nv -= nc;
        }
        fclose(fp);
        return 1;
    }

    ovals = vals;

    /* get number of rows and columns */

    rldec_get_size (vals, &nrows, &ncols);   

    /* check if the buffer is large enough to hold one uncompressed line */

    if (ncols > LBUFLEN) {
        fprintf (stderr, "ERROR: Number of columns larger than %d!\n",
                 LBUFLEN);
        return 0;
    }

    /* write P5 header for pgm-format */

    if (strstr (outfile, ".pgm") != NULL || strstr (outfile, ".PGM"))
    {
        fprintf (fp, "P5\n%d %d\n%5d\n", nrows, ncols, depth == 1 ? 0xff : 0xffff);
    }

    /* skip number of rows */

    *nvals = 0;
    vals ++;
    (*nvals) ++;

    /* decode line by line */

    for (i = 0; i < nrows; i ++) {
        int n;

        /* decompress line */

        if (depth == 4)
        {
            rldec_decompress_line_float (vals, fbuf, &nc, &nv);
        } else {
            rldec_decompress_line (vals, ibuf, &nc, &nv);
        }

        /* check for correct image size */
        
        if (nc != ncols) {
            fprintf (stderr, "Error in run-length decoding!\n");
            fclose (fp);
            return 0;
        }

        /* increase vals pointer */

        vals += nv;
        (*nvals) += nv;

        /* convert to char */

        if (depth == 1) {
            for (j = 0; j < ncols; j ++)
                cbuf[j] = (unsigned char) ibuf[j];
        } else if (depth == 2){
            for (j = 0; j < ncols; j++) {
                cbuf[j*2] = (unsigned char) ((ibuf[j] >> 8) & 0xff);
                cbuf[j*2+1] = (unsigned char) (ibuf[j] & 0xff);
            }
        }

        /* write to file */
        if (depth == 4)
            n = fwrite (fbuf, 1, ncols * depth, fp);
        else
            n = fwrite (cbuf, 1, ncols * depth, fp);
        if (n != (size_t) ncols * depth) {
               fprintf (stderr, "Write error to file '%s'\n", outfile);
               fclose (fp);
               return 0;
        }
    }

    /* close file */

    fclose (fp);

    assert (*nvals == vals - ovals);
    return 1;
}
/*===========================================================================*/
/** \ingroup rlenc_g

   \brief This function encodes a radar image to BUFR runlength-code 

   This function encodes a radar image in memory to BUFR runlength-code 
   and stores the resulting values into an  array \p vals 
   by a call to \ref bufr_val_to_array. 

   \note In difference to the old \ref rlenc function the
   initial length of \p vals must given in the parameter
   \p nvals in order to prevent \ref bufr_val_to_array from writing to an
   arbitrary position.

   \param[in] img      Array holding the uncompressed radar image.
   \param[in] ncols    Number of columns of the image.
   \param[in] nrows    Number of rows of the image.
   \param[in,out] vals    Float-array holding the coded image.
   \param[in,out] nvals   Number of values in \p vals.

   \return The return-value ist 1 on success, 0 on a fault.

   \see rlenc_from_file, rldec_to_mem, rlenc_compress_line_new
*/

int rlenc_from_mem (unsigned short* img, int nrows, int ncols, varfl* *vals, 
                     int *nvals)

{
    unsigned int ibuf[LBUFLEN];
    int i, j;

    if (img == (unsigned short*) NULL) {
        fprintf (stderr, "Image for rlenc not available!\n");
        return 0;
    }

    /* check if the internal buffer is large enough to hold one uncompressed
       line */

    if (ncols > LBUFLEN) {
        fprintf (stderr, "ERROR: Number of columns larger than %d!\n",
                 LBUFLEN);
        return 0;
    }

    /* output number of rows */

    bufr_val_to_array (vals, (varfl) nrows, nvals);  

    /* compress line by line */

    for (i = 0; i < nrows; i ++) {

        /* get row from memory and convert to int */

        for (j = 0; j < ncols; j ++)
            ibuf[j] = (unsigned int) img[i*ncols+j];
        
        /* compress line to varfl array */

        if (!rlenc_compress_line_new (i, ibuf, ncols, vals, nvals)) {
            return 0;
        }
    }
    return 1;
}


/*===========================================================================*/
/** \ingroup rlenc_g

   \brief This function encodes a radar image to BUFR runlength-code 

   This function encodes a radar image in memory to BUFR runlength-code 
   and stores the resulting values into an  array \p vals 
   by a call to \ref bufr_val_to_array. 

   \note In difference to the old \ref rlenc function the
   initial length of \p vals must given in the parameter
   \p nvals in order to prevent \ref bufr_val_to_array from writing to an
   arbitrary position.

   \param[in] img      Array holding the uncompressed radar image.
   \param[in] ncols    Number of columns of the image.
   \param[in] nrows    Number of rows of the image.
   \param[in,out] vals    Float-array holding the coded image.
   \param[in,out] nvals   Number of values in \p vals.

   \return The return-value ist 1 on success, 0 on a fault.

   \see rlenc_from_file, rldec_to_mem, rlenc_compress_line_new, 
   rlenc_to_mem_float
*/

int rlenc_from_mem_float (float* img, int nrows, int ncols, varfl* *vals, 
                     int *nvals)

{
    float fbuf[LBUFLEN];
    int i;

    if (img == (float*) NULL) {
        fprintf (stderr, "Image for rlenc not available!\n");
        return 0;
    }

    /* check if the internal buffer is large enough to hold one uncompressed
       line */

    if (ncols > LBUFLEN) {
        fprintf (stderr, "ERROR: Number of columns larger than %d!\n",
                 LBUFLEN);
        return 0;
    }

    /* output number of rows */

    bufr_val_to_array (vals, (varfl) nrows, nvals);  

    /* compress line by line */

    for (i = 0; i < nrows; i ++) {

        /* get row from memory */

        memcpy (fbuf, img+i*ncols, ncols * sizeof(float));
        
        /* compress line to varfl array */

        if (!rlenc_compress_line_float (i, fbuf, ncols, vals, nvals)) {
            return 0;
        }
    }
    return 1;
}

/*===========================================================================*/
/** \ingroup rldec_g 
    \brief Decodes a BUFR-runlength-encoded radar image to memory

    This function decodes a BUFR-runlength-encoded radar image stored at
    \p vals. The decoded image is stored in an array \p img[] which will be
    allocated by this function if \p img[] = NULL.
    The memory for the image must be freed by the calling function!

    \param[in] vals      Float-array holding the coded image.
    \param[in,out] img  Destination-array for the radar image.
    \param[out] nvals   Number of \ref varfl values needed for the 
                         compressed radar image.
    \param[out] nrows   Number of lines in image
    \param[out] ncols   Number of pixels per line

    \return The return-value ist 1 on success, 0 on a fault.

    \see rlenc_from_mem, rldec_to_file, rldec_decompress_line
*/

int rldec_to_mem (varfl* vals, unsigned short* *img, int* nvals, int* nrows,
                  int* ncols)

{
    int i, j, nc, nv;
    unsigned int ibuf[LBUFLEN];
    varfl *ovals;

    ovals = vals;

    /* get number of rows and columns */

    rldec_get_size (vals, nrows, ncols);   

    /* Allocate memory for image if necessary */

    if (*img == NULL) {
        *img = (unsigned short*) calloc (*nrows * *ncols, 
                                         sizeof (unsigned short));
        if (*img == NULL) {
            fprintf (stderr, "Could not allacote memory for radar image!\n");
            return 0;
        }
    }

    /* check if the buffer is large enough to hold one uncompressed
       line */

    if (*ncols > LBUFLEN) {
        fprintf (stderr, "ERROR: Number of columns larger than %d!\n",
                 LBUFLEN);
        return 0;
    }

    /* skip number of rows */

    *nvals = 0;
    vals ++;
    (*nvals) ++;

    /* decode line by line */

    for (i = 0; i < *nrows; i ++) {

        /* decompress line */

        rldec_decompress_line (vals, ibuf, &nc, &nv);

        /* check for correct image size */
        
        if (nc != *ncols) {
            fprintf (stderr, "Error in run-length decoding!\n");
            return 0;
        }

        /* increase vals pointer */

        vals += nv;
        (*nvals) += nv;

        /* convert to short and write to memory*/

        for (j = 0; j < *ncols; j ++)
            (*img)[i * *ncols + j] = (unsigned short) ibuf[j];
    }

    assert (*nvals == vals - ovals);
    return 1;
}
/*===========================================================================*/
/** \ingroup rldec_g 
    \brief Decodes a BUFR-runlength-encoded float image to memory

    This function decodes a BUFR-runlength-encoded float image stored at
    \p vals. The decoded image is stored in an array \p img[] which will be
    allocated by this function if \p img[] = NULL.
    The memory for the image must be freed by the calling function!

    \param[in] vals      Float-array holding the coded image.
    \param[in,out] img  Destination-array for the radar image.
    \param[out] nvals   Number of \ref varfl values needed for the 
                         compressed radar image.
    \param[out] nrows   Number of lines in image
    \param[out] ncols   Number of pixels per line

    \return The return-value ist 1 on success, 0 on a fault.

    \see rlenc_from_mem_float, rldec_to_file, rldec_decompress_line_float
*/

int rldec_to_mem_float (varfl* vals, float* *img, int* nvals, int* nrows,
                  int* ncols)

{
    int i, nc, nv;
    float fbuf[LBUFLEN];
    varfl *ovals;

    ovals = vals;

    /* get number of rows and columns */

    rldec_get_size (vals, nrows, ncols);   

    /* Allocate memory for image if necessary */

    if (*img == NULL) {
        *img = (float*) calloc (*nrows * *ncols, 
                                         sizeof (float));
        if (*img == NULL) {
            fprintf (stderr, "Could not allacote memory for radar image!\n");
            return 0;
        }
    }

    /* check if the buffer is large enough to hold one uncompressed
       line */

    if (*ncols > LBUFLEN) {
        fprintf (stderr, "ERROR: Number of columns larger than %d!\n",
                 LBUFLEN);
        return 0;
    }

    /* skip number of rows */

    *nvals = 0;
    vals ++;
    (*nvals) ++;

    /* decode line by line */

    for (i = 0; i < *nrows; i ++) {

        /* decompress line */

        rldec_decompress_line_float (vals, fbuf, &nc, &nv);

        /* check for correct image size */
        
        if (nc != *ncols) {
            fprintf (stderr, "Error in run-length decoding!\n");
            return 0;
        }

        /* increase vals pointer */

        vals += nv;
        (*nvals) += nv;

        /* write to memory*/

        memcpy ((*img)+i * *ncols, fbuf, *ncols * sizeof(float));
    }

    assert (*nvals == vals - ovals);
    return 1;
}

/*===========================================================================*/
/** \ingroup rlenc_g 
    \brief Encodes one line of a radar image to BUFR runlength-code

    This function encodes one line of a radar image to BUFR runlength-code and
    stores the resulting values to array \p dvals by a call to 
    \ref bufr_val_to_array.

    \note In difference to the old \ref rlenc_compress_line function the
    initial length of \p vals must given in the parameter
    \p nvals in order to prevent \ref bufr_val_to_array from writing to an
    arbitrary position.

    \param[in] line     Line number.
    \param[in] src      Is where the uncompressed line is stored.
    \param[in] ncols    Number of pixels per line.
    \param[in,out] dvals    Float-array holding the coded image.
    \param[in,out] nvals    Number of values in VALS.
   
    \return The function returns 1 on success, 0 on a fault.

    \see rldec_decompress_line
*/

int rlenc_compress_line_new (int line, unsigned int* src, int ncols, 
                             varfl* *dvals, int *nvals)

{
    int count, i, n, npar, lens[LBUFLEN], cw, ncgi, nngi = 0;
    unsigned int val, lval = 0, vals[LBUFLEN];
    varfl encbuf[ENCBUFL];

    /* line is runlength-compressed now to N parts, each of them identified 
       by a length (LENS) and a value (VALS). */

    count = n = 0;
    for (i = 0; i < ncols; i ++) {
        val = *(src + i);
    
        /* limit length of one part to 255 to ensure that descriptor 0 31 001 
           does not exceed */
        if (i != 0 && (val != lval || count >= 255)) {  
            lens[n] = count;
            vals[n] = lval;
            n ++;
            count = 0;
            lval = val;
        }
        lval = val;
        count ++;
    }
    lens[n] = count;
    vals[n] = lval;
    n ++;

    /* Count number of parcels. One parcel is identified by a COUNT of 1
       followed by a COUNT > 1 */

    npar = 0;
    for (i = 0; i < n - 1; i ++) if (lens[i] == 1 && lens[i+1] > 1) npar ++;
    npar ++;

    /* output line-number */

    for (i = 0; i < ENCBUFL; i ++) encbuf[i] = (varfl) 0.0;
    cw = 0;
    encbuf[cw++] = line;  

    /* compress it to parcels */
  
    encbuf[cw++] = npar;                   /* number of parcels */

    ncgi = cw ++;                          /* is where the number of 
                                              compressable groups is stored */
    encbuf[ncgi] = (varfl) 0.0;            /* number of compressable groups */

    i = 0;
    for (i = 0; i < n; i ++) {
        if (lens[i] > 1) {                   /* compressable group found */
            if (i > 0 && lens[i-1] == 1) {   /* A new parcel starts here */
                ncgi = cw ++;                /* where the number of compress-
                                                able groups is stored */
                encbuf[ncgi] = (varfl) 0.0;  /* number of compressable groups*/
            }
            encbuf[ncgi] += 1.0;
            encbuf[cw++] = lens[i];
            encbuf[cw++] = vals[i];
        }
        else {                               /* non compressable group found */
            if (i == 0 || lens[i-1] != 1) {  /* this is the first 
                                                uncompressable group in 
                                                the current parcel */
                nngi = cw ++;                /* is where the number of non 
                                               compressable groups is stored */
                encbuf[nngi] = (varfl) 0.0;  /* Number of non compressable 
                                                groups */
            }
            encbuf[nngi] += 1.0;
            encbuf[cw++] = vals[i];
        }
    }
    if (lens[n-1] != 1) encbuf[cw++] = 0;  /* number of noncompressable 
                                              groups in the last parcel = 0 */
    assert (cw <= ENCBUFL);
    
    /* compresson to parcels finished, write values to destination array */

    for (i = 0; i < cw; i ++) {
        if (!bufr_val_to_array (dvals, encbuf[i], nvals)) 
            return 0;
    }

    return 1;
}

/*===========================================================================*/
/** \ingroup rlenc_g 
    \brief Encodes one line of a radar image to BUFR runlength-code

    This function encodes one line of a radar image to BUFR runlength-code and
    stores the resulting values to array \p dvals by a call to 
    \ref bufr_val_to_array.

    \note In difference to the old \ref rlenc_compress_line function the
    initial length of \p vals must given in the parameter
    \p nvals in order to prevent \ref bufr_val_to_array from writing to an
    arbitrary position.

    \param[in] line     Line number.
    \param[in] src      Is where the uncompressed line is stored.
    \param[in] ncols    Number of pixels per line.
    \param[in,out] dvals    Float-array holding the coded image.
    \param[in,out] nvals    Number of values in VALS.
   
    \return The function returns 1 on success, 0 on a fault.

    \see rldec_decompress_line, rldec_decompress_line_float
*/

int rlenc_compress_line_float (int line, float* src, int ncols, 
                             varfl* *dvals, int *nvals)

{
    int count, i, n, npar, lens[LBUFLEN], cw, ncgi, nngi = 0;
    float val, lval = 0, vals[LBUFLEN];
    varfl encbuf[ENCBUFL];

    /* line is runlength-compressed now to N parts, each of them identified 
       by a length (LENS) and a value (VALS). */

    count = n = 0;
    for (i = 0; i < ncols; i ++) {
        val = *(src + i);
    
        /* limit length of one part to 255 to ensure that descriptor 0 31 001 
           does not exceed */
        if (i != 0 && (val != lval || count >= 255)) {  
            lens[n] = count;
            vals[n] = lval;
            n ++;
            count = 0;
            lval = val;
        }
        lval = val;
        count ++;
    }
    lens[n] = count;
    vals[n] = lval;
    n ++;

    /* Count number of parcels. One parcel is identified by a COUNT of 1
       followed by a COUNT > 1 */

    npar = 0;
    for (i = 0; i < n - 1; i ++) if (lens[i] == 1 && lens[i+1] > 1) npar ++;
    npar ++;

    /* output line-number */

    for (i = 0; i < ENCBUFL; i ++) encbuf[i] = (varfl) 0.0;
    cw = 0;
    encbuf[cw++] = line;  

    /* compress it to parcels */
  
    encbuf[cw++] = npar;                   /* number of parcels */

    ncgi = cw ++;                          /* is where the number of 
                                              compressable groups is stored */
    encbuf[ncgi] = (varfl) 0.0;            /* number of compressable groups */

    i = 0;
    for (i = 0; i < n; i ++) {
        if (lens[i] > 1) {                   /* compressable group found */
            if (i > 0 && lens[i-1] == 1) {   /* A new parcel starts here */
                ncgi = cw ++;                /* where the number of compress-
                                                able groups is stored */
                encbuf[ncgi] = (varfl) 0.0;  /* number of compressable groups*/
            }
            encbuf[ncgi] += 1.0;
            encbuf[cw++] = lens[i];
            encbuf[cw++] = vals[i];
        }
        else {                               /* non compressable group found */
            if (i == 0 || lens[i-1] != 1) {  /* this is the first 
                                                uncompressable group in 
                                                the current parcel */
                nngi = cw ++;                /* is where the number of non 
                                               compressable groups is stored */
                encbuf[nngi] = (varfl) 0.0;  /* Number of non compressable 
                                                groups */
            }
            encbuf[nngi] += 1.0;
            encbuf[cw++] = vals[i];
        }
    }
    if (lens[n-1] != 1) encbuf[cw++] = 0;  /* number of noncompressable 
                                              groups in the last parcel = 0 */
    assert (cw <= ENCBUFL);
    
    /* compresson to parcels finished, write values to destination array */

    for (i = 0; i < cw; i ++) {
        if (!bufr_val_to_array (dvals, encbuf[i], nvals)) 
            return 0;
    }

    return 1;
}

/*===========================================================================*/
/** \ingroup rldec_g

    \brief Decodes one line of a float image from BUFR runlength-code

    This function decodes one line of a float image from BUFR runlength-code 
    and stores the resulting values to array \p dest which has to be large
    enough to hold a line.

    \param[in]  vals     Float-array holding the coded image.
    \param[out] dest     Is where the uncompressed line is stored.
    \param[out] ncols    Number of pixels per line.
    \param[out] nvals    Number of values needed for compressed line.

    \see rlenc_compress_line_float
   
*/

void rldec_decompress_line_float (varfl* vals, float* dest, int* ncols, 
                                  int* nvals) 
{
    int i = 0, j, k, l, count = 0, npar, ngr, nup;
    float val;
    varfl* ovals;

    ovals = vals;
    vals ++;                          /* skip linenumber */
    npar = (int) *vals ++;            /* number of parcels */
    for (j = 0; j < npar; j ++) {     /* loop for parcels */
        ngr = (int) *vals ++;         /* number of compressable groups */
        for (k = 0; k < ngr; k ++) {  /* loop for compressable groups */
            count = (int) *vals ++;
            if (*vals == MISSVAL) {
                val = MISSVAL;
                vals ++;
            } else {
                val = *vals ++;
            }
            for (l = 0; l < count; l ++) {  /* loop for length of group */
                dest[i++] = val;
            }
        }
        nup = (int) *vals ++;         /* number of uncompressable pixels */
        for (k = 0; k < nup; k ++) {  /* loop for uncompressable pixels */
            if (*vals == MISSVAL) {
                dest[i++] = MISSVAL;
                vals ++;
            } else {
                dest[i++] = *vals ++;
            }
        }
    }


    *nvals = vals - ovals;
    *ncols = i;
}

/*===========================================================================*/
/** \ingroup rldec_g

    \brief Decodes one line of a radar image from BUFR runlength-code

    This function decodes one line of a radar image from BUFR runlength-code 
    and stores the resulting values to array \p dest which has to be large
    enough to hold a line.

    \param[in]  vals     Float-array holding the coded image.
    \param[out] dest     Is where the uncompressed line is stored.
    \param[out] ncols    Number of pixels per line.
    \param[out] nvals    Number of values needed for compressed line.

    \see rlenc_compress_line_new
   
*/

void rldec_decompress_line (varfl* vals, unsigned int* dest, int* ncols, 
                            int* nvals) 
{
    int i = 0, j, k, l, count = 0, npar, ngr, nup;
    unsigned int val;
    varfl* ovals;

    ovals = vals;
    vals ++;                          /* skip linenumber */
    npar = (int) *vals ++;            /* number of parcels */
    for (j = 0; j < npar; j ++) {     /* loop for parcels */
        ngr = (int) *vals ++;         /* number of compressable groups */
        for (k = 0; k < ngr; k ++) {  /* loop for compressable groups */
            count = (int) *vals ++;
            if (*vals == MISSVAL) {
                val = 0xFFFF;
                vals ++;
            } else {
                val =   (unsigned int) *vals ++;
            }
            for (l = 0; l < count; l ++) {  /* loop for length of group */
                dest[i++] = val;
            }
        }
        nup = (int) *vals ++;         /* number of uncompressable pixels */
        for (k = 0; k < nup; k ++) {  /* loop for uncompressable pixels */
            if (*vals == MISSVAL) {
                dest[i++] = 0xFFFF;
                vals ++;
            } else {
                dest[i++] = (unsigned int) *vals ++;
            }
        }
    }


    *nvals = vals - ovals;
    *ncols = i;
}

/*===========================================================================*/
/** \ingroup rldec_g
    \brief Gets the number of rows and columns of a runlength compressed image

    This function gets the number of rows and colums of a runlength compressed
    image stored at array \p vals 

    \param[in] vals     Float-array holding the coded image.
    \param[out] nrows   Number of lines in image.
    \param[out] ncols   Number of pixels per line.

    \see rldec_to_file, rldec_decompress_line
*/
void rldec_get_size (varfl* vals, int* nrows, int* ncols)
{
    int npar, ngr, nup, j, k, l, count;
    
    *nrows = (int) *vals ++;            /* number of rows */
    *ncols = 0;
    vals ++;                            /* skip linenumber */
    npar = (int) *vals ++;              /* number of parcels */
    for (j = 0; j < npar; j ++) {       /* loop for parcels */
        ngr = (int) *vals ++;           /* number of compressable groups */
        for (k = 0; k < ngr; k ++)  {   /* loop for compressable groups */
            count = (int) *vals ++;
            vals ++;			            /* skip pixel value */
            for (l = 0; l < count; l ++) {  /* loop for length of group */
                (*ncols) ++;	  
            }
        }
        nup = (int) *vals ++;             /* number of uncompressable pixels */
        for (k = 0; k < nup; k ++) {      /* loop for uncompressable pixels */
            vals ++;			          /* skip pixel value */
            (*ncols) ++;
        }
    }
}

/* end of file */

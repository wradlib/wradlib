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

FILE:          RLENC.H
IDENT:         $Id: rlenc.h,v 1.7 2009/06/18 17:15:39 helmutp Exp $

AUTHOR:        Konrad Koeck
               Institute of Communication and Wave Propagation, 
               Technical University Graz, Austria

VERSION NUMBER:3.0

DATE CREATED:  18-DEC-2001

STATUS:        DEVELOPMENT FINISHED


FUNCTIONAL DESCRIPTION:
-----------------------
function prototype for rlenc.c

AMENDMENT RECORD:

ISSUE       DATE            SCNREF      CHANGE DETAILS
-----       ----            ------      --------------
V2.0        18-DEC-2001     Koeck       Initial Issue

$Log: rlenc.h,v $
Revision 1.7  2009/06/18 17:15:39  helmutp
added functions for float values

Revision 1.6  2007/12/18 14:40:58  fuxi
added licence header

Revision 1.5  2007/12/07 08:35:33  fuxi
update to version 3.0

Revision 1.4  2005/04/04 14:56:09  helmutp
update to version 2.3

Revision 1.3  2003/03/27 17:17:39  helmutp
update to version 2.2

Revision 1.2  2003/03/06 17:12:32  helmutp
update to version 2.1

Revision 1.1  2003/02/28 13:41:12  helmutp
Initial revision

--------------------------------------------------------------------------- */

/** \file rlenc.h
    \brief Function definitions for run-length encoding and decoding
    
    This file contains all functions used for run-length encoding
    and decoding of image files.
*/

#ifndef RLENC_H_INCLUDED
#define RLENC_H_INCLUDED

int rlenc_from_file (char* infile, int nrows, int ncols, varfl** vals, 
                     int* nvals, int depth);
int rlenc_from_mem (unsigned short* img, int nrows, int ncols, varfl** vals, 
                    int* nvals);
int rldec_to_file (char* outfile, varfl* vals, int depth, int* nvals);
int rldec_to_mem (varfl* vals, unsigned short* *img, int* nvals, int* nrows,
                  int* ncols);
int rlenc_compress_line_new (int line, unsigned int* src, int ncols, 
                             varfl** dvals, int* nvals);
void rldec_decompress_line (varfl* vals, unsigned int* dest, int* ncols, 
                            int* nvals);
void rldec_get_size (varfl* vals, int* nrows, int* ncols);

/* float methods */
int rlenc_from_mem_float (float* img, int nrows, int ncols, varfl** vals, 
                          int* nvals);
int rldec_to_mem_float (varfl* vals, float* *img, int* nvals, int* nrows,
                        int* ncols);
int rlenc_compress_line_float (int line, float* src, int ncols, 
                               varfl** dvals, int* nvals);
void rldec_decompress_line_float (varfl* vals, float* dest, int* ncols, 
                                  int* nvals);

/* old functions */
int rlenc (char *infile, int nrows, int ncols, varfl **vals, size_t *nvals);
int rldec (char *outfile, varfl *vals, size_t *nvals);
int rlenc_compress_line (int line, unsigned char *src, int ncols, 
                         varfl **dvals, size_t *nvals);



#endif

/* end of file */


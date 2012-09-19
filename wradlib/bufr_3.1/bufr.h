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

FILE:          BUFR.H
IDENT:         $Id: bufr.h,v 1.9 2009/05/15 15:10:12 helmutp Exp $

AUTHORS:       Juergen Fuchsberger, Helmut Paulitsch, Konrad Koeck
               Institute of Communication and Wave Propagation, 
               Technical University Graz, Austria

VERSION NUMBER:3.0

DATE CREATED:  18-DEC-2001

STATUS:        DEVELOPMENT FINISHED


FUNCTIONAL DESCRIPTION:
-----------------------
Includefile for BUFR.C. More details can be found there.

AMENDMENT RECORD:

$Log: bufr.h,v $
Revision 1.9  2009/05/15 15:10:12  helmutp
api change to support subsets

Revision 1.8  2009/04/17 13:42:59  helmutp
added subsets

Revision 1.7  2008/03/06 14:19:16  fuxi
changed filenames to const char*

Revision 1.6  2007/12/18 14:40:58  fuxi
added licence header

Revision 1.5  2007/12/07 08:34:43  fuxi
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

/** \file bufr.h
    \brief Definitions of main OPERA BUFR library functions
    
    This file contains declaration of functions used for encoding and 
    decoding data to BUFR format.
*/

/*===========================================================================*/
/* global variables                                                          */
/* If BUFR_MAIN is not defined all variables are declared as external.       */
/* So you sould define BUFR_MAIN only in one function. Otherwise you will    */
/* have this symbols multiple defined.                                       */
/*===========================================================================*/

#ifdef BUFR_MAIN

int _bufr_edition = 3;      /**< bufr edition number */
int _opera_mode = 1;        /* input and output bitmaps to / from file */
int _replicating = 0;       /**< indicates a data replication process */
int _subsets = 1;           /**< number of subsets */

#else
/** \brief global bufr edition number 

   The bufr edition number is stored in section 0 of the BUFR message.
   It is used by the software for determining the format of section 1.

   \see bufr_get_date_time, bufr_encode_sections0125, bufr_decode_sections01,
        bufr_parse_new, bufr_val_from_datasect, bufr_val_to_datasect
*/
extern int _bufr_edition;   
extern int _opera_mode;

/** \brief global replication indicator

    This flag is used to indicate an ongoing data replication and is set
    by \ref bufr_parse_new. It can be used for different output formating 
    when a replication occurs.

    \see bufr_parse_new, bufr_file_out

*/

extern int _replicating;

extern int _subsets;
#endif

#ifndef BUFR_H_INCLUDED
#define BUFR_H_INCLUDED

#define MAX_DESCS 1000       /**< \brief Maximum number of data descriptors
                                in a BUFR message */

#define MEMBLOCK   100       /* The memory-area holding data-strings 
                                and data-descriptors is allocated and 
                                reallocated in blocks of MEMBLOCK elements. */

/*===========================================================================*/
/* structures                                                                */
/*===========================================================================*/

/** \brief Structure that holds the encoded bufr message */
typedef struct bufr_s {      

    char* sec[6];            /**< \brief pointers to sections */
    int   secl[6];           /**< \brief length of sections */
} bufr_t;

typedef char* bd_t;          /**< \brief one bufr data element is a string */

/** \brief Structure holding values for callbacks 
    \ref bufr_val_from_global and \ref bufr_val_to_global */

typedef struct bufrval_s { 
    varfl* vals;            /**< \brief array of values */
    int vali;        /**< \brief current index into array of values */
    int nvals;       /**< \brief number of values */
} bufrval_t;



/*===========================================================================*/
/* protypes of functions in BUFR.C                                           */
/*===========================================================================*/

/* basic functions for encoding to BUFR */

int bufr_create_msg (dd *descs, int ndescs, varfl *vals, void **datasec, 
                        void **ddsec, size_t *datasecl, size_t *ddescl);
int bufr_encode_sections34 (dd* descs, int ndescs, varfl* vals, bufr_t* msg);
int bufr_encode_sections0125 (sect_1_t* s1, bufr_t* msg);
int bufr_write_file (bufr_t* msg, const char* file);

/* basic function for decoding from BUFR */

int bufr_read_file (bufr_t* msg, const char* file);
int bufr_get_sections (char* bm, int len, bufr_t* msg);
int bufr_decode_sections01 (sect_1_t* s1, bufr_t* msg);
int bufr_read_msg (void *datasec, void *ddsec, size_t datasecl, size_t ddescl,
                      dd **desc, int *ndescs, varfl **vals, size_t *nvals);

/* extended functions for encoding to BUFR */

void bufr_sect_1_from_file (sect_1_t* s1, const char* file);
int bufr_open_descsec_w (int subsets);
int bufr_out_descsec (dd *descp, int ndescs, int desch);
void bufr_close_descsec_w(bufr_t* bufr, int desch);
int bufr_parse_in  (dd *descs, int start, int end, 
                    int (*inputfkt) (varfl *val, int ind),
                    int callback_descs);

/* extended functions for decoding from BUFR */

int bufr_open_descsec_r (bufr_t* msg, int *subsets);
int bufr_get_ndescs (bufr_t* msg);
int bufr_in_descsec (dd** descs, int ndescs, int desch);
void bufr_close_descsec_r(int desch);
int bufr_parse_out  (dd *descs, int start, int end, 
                     int (*outputfkt) (varfl val, int ind),
                     int callback_all_descs);
int bufr_sect_1_to_file (sect_1_t* s1, const char* file);

/* utility functions */

void bufr_free_data (bufr_t* d);
int bufr_check_fxy(dd *d, int ff, int xx, int yy);
void bufr_get_date_time (long *year, long *mon, long *day, long *hour,
                            long *min);
int bufr_val_to_array (varfl **vals, varfl v, int *nvals);
int bufr_desc_to_array (dd* descs, dd d, int* ndescs);
int bufr_parse_new (dd *descs, int start, int end, 
                    int (*inputfkt) (varfl *val, int ind),
                    int (*outputfkt) (varfl val, int ind),
                    int callback_all_descs);
int bufr_parse (dd *descs, int start, int end, varfl *vals, unsigned *vali,
                int (*userfkt) (varfl val, int ind));
bufrval_t* bufr_open_val_array ();
void bufr_close_val_array ();
int bufr_open_datasect_w ();
void bufr_close_datasect_w(bufr_t* msg);
int bufr_open_datasect_r (bufr_t* msg);
void bufr_close_datasect_r();

/* callback functions for use with bufr_parse_* */

int bufr_val_from_global (varfl *val, int ind);
int bufr_val_to_global (varfl val, int ind);

/* deprecated functions */

void bufr_clean ();
int val_to_array (varfl **vals, varfl v, size_t *nvals);
int setup_sec0125 (char *sec[], size_t secl[], sect_1_t s1);
int save_sections (char *sec[], size_t secl[], char *buffile);

#endif

/* end of file */

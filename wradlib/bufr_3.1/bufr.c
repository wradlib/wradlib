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

FILE:          BUFR.C
IDENT:         $Id: bufr.c,v 1.17 2010/02/15 11:22:55 helmutp Exp $

AUTHOR:        Konrad Koeck
               Institute of Communication and Wave Propagation, 
               Technical University Graz, Austria

VERSION NUMBER:3.0

DATE CREATED:  18-DEC-2001

STATUS:        DEVELOPMENT FINISHED


FUNCTIONAL DESCRIPTION:
-----------------------
The Functions in this file can be used to encode/decode general BUFR-messages.
The file bufr.h must be included to get the function-prototyping for the
functions in this file. More details can be found in the descriptions of
the functions.

AMENDMENT RECORD:

ISSUE       DATE            SCNREF      CHANGE DETAILS
-----       ----            ------      --------------
V2.0        18-DEC-2001     Koeck       Initial Issue

$Log: bufr.c,v $
Revision 1.17  2010/02/15 11:22:55  helmutp
changed missing value handling

Revision 1.16  2009/05/15 15:34:12  helmutp
api change to support subsets, bug fixes

Revision 1.15  2009/04/17 16:05:02  helmutp
implementation for 2 5 y and 2 6 y descriptors and subsets

Revision 1.14  2009/04/10 12:08:00  helmutp
change of reference value implemented and other modification
descriptors as well (2 21 y ... 2 37 y)
support optional section (for decoding)

Revision 1.13  2008/03/06 14:19:00  fuxi
changed filenames to const char*

Revision 1.12  2007/12/18 15:50:00  fuxi
removed debugging output

Revision 1.11  2007/12/18 14:40:13  fuxi
added licence header

Revision 1.10  2007/12/07 08:34:27  fuxi
update to version 3.0

Revision 1.9  2006/07/20 10:19:44  fuxi
added debugging info

Revision 1.8  2006/07/19 08:59:47  fuxi
added debugging options

Revision 1.7  2005/04/04 15:38:32  helmutp
update to version 2.3
no datawidth or scale change for 0 31 y descriptors
subcenter and generating center

Revision 1.6  2003/03/28 14:03:20  helmutp
fixed missval for pixel values

Revision 1.5  2003/03/27 17:17:39  helmutp
update to version 2.2

Revision 1.4  2003/03/13 17:10:55  helmutp
use descriptor sort function instead of linear search

Revision 1.3  2003/03/06 17:12:32  helmutp
update to version 2.1

Revision 1.2  2003/02/28 14:39:54  helmutp
fixed return value in read_bufr_msg

Revision 1.1  2003/02/28 13:41:12  helmutp
Initial revision


--------------------------------------------------------------------------- */

/** \file bufr.c
    \brief Main OPERA BUFR library functions
    
    This file contains all functions used for encoding and decoding data
    to BUFR format.
*/

#define BUFR_MAIN

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include "desc.h"
#include "bufr.h"
#include "bitio.h"
#include "rlenc.h"

/*===========================================================================*/
/* globals */
/*===========================================================================*/


/*===========================================================================*/
/* default values */
/*===========================================================================*/

/* Define default values for the originating center (OPERA) 
   and the versions of master (WMO) and local (OPERA) table */
#define SUBCENTER 255
#define GENCENTER 255
#define VMTAB 11
#define VLTAB 4

/*===========================================================================*/
/* internal data                                                             */
/*===========================================================================*/

#define MAXREPCOUNT 300      /* Max. replication count */
#define MAX_ADDFIELDS 50     /* Maximum number of nested associated fields */

/* The following variables are used to hold date/time-info of the
   last BUFR-message created. */

static long year_, mon_, day_, hour_, min_;
static int af_[MAX_ADDFIELDS];  /* remember associated fields for nesting */
static int naf_ = 0;            /* current number of associated field */
static int datah_ = -1;          /* bitstream-handle for data-section */
static bufrval_t* vals_ = NULL;  /* structure for holding data values */

static dd cf_spec_des[MAX_ADDFIELDS];    /* remember changed descriptors */
static varfl cf_spec_val[MAX_ADDFIELDS]; /* original referecne values */
static int cf_spec_num = 0;              /* number of changed descriptors */ 

/*===========================================================================*/
/* internal functions                                                        */
/*===========================================================================*/

static int bufr_val_to_datasect (varfl val, int ind);
static int bufr_val_from_datasect (varfl *val, int ind);
static int get_lens (char* buf, long len, int* secl);


/*===========================================================================*/
/* functions */
/*===========================================================================*/

/** \ingroup deprecated_g
    \deprecated use \ref free_descs instead

    This function frees all memory-blocks allocated by \ref read_tables
 */

void bufr_clean (void)


{
  free_descs();
}

/*===========================================================================*/
/** \ingroup deprecated
    \deprecated Use \ref bufr_encode_sections34 instead.

   \brief Creates section 3 and 4 of BUFR message from arrays of data and
  data descriptors.

  This function codes data from an array data descriptors \p descs and an 
  array of varfl-values \p vals to a data section and
  a data descripor section of a BUFR message. Memory for both sections is
  allocated in this function and must be freed by the calling functions.


   \param[in] descs 
     Data-descriptors corresponding to \p vals. 
     For each descriptor there
     must be a data-vaule stored in \p vals. \p descs may also include
     replication factors and sequence descriptors. 
     In that case there must be a larger number of
     \p vals then of \p descs.

   \param[in] ndescs  
     Number of data descriptos contained in \p descs.

   \param[in] vals    
     Data-values to be coded in the data section. For each entry in
     \p descs there must be an entry in \p vals. If there are relication
     factors in \p descs, of course there must be as much \p vals as definded
     by the replication factor.

   \param[out] datasec 
     Is where the data-section (section 4) is stored. The memory-area for the
     data-section is allocated by this function and must be freed by
     the calling function.

   \param[out] ddsec   
     Is where the data-descriptor-section (section 3) in stored. 
     The memory needed is
     allocated by this function and must be freed by the calling 
     function.

   \param[out] datasecl 
     Number of bytes in \p datasec.

   \param[out] ddescl
     Number of bytes in \p ddsec.

   \return The return-value is 1 if data was successfully stored, 0 if not.

   \see bufr_read_msg, bufr_data_from_file

*/


int bufr_create_msg (dd* descs, int ndescs, varfl* vals, void **datasec, 
                     void **ddsec, size_t *datasecl, size_t *ddescl)


{
    bufrval_t* valarray = NULL;
    int ok, desch;
    bufr_t msg;

    memset (&msg, 0, sizeof (bufr_t));

    year_ = mon_ = day_ = hour_ = min_ = 0;

    /* Open two bitstreams, one for data-descriptors, one for data */

    desch = bufr_open_descsec_w (1);

    ok = (desch >= 0);

    if (ok)
        ok = (bufr_open_datasect_w () >= 0);

    /* output data to the data descriptor bitstream */

    if (ok)
        bufr_out_descsec (descs, ndescs, desch);

    /* set global array */

    if (ok) {
        valarray = bufr_open_val_array ();
        ok = (valarray != (bufrval_t*) NULL);
    }

    if (ok) {
        valarray->vals = vals;
        valarray->vali = 0;
    }

    /* output data to the data-section */

    if (ok) {
        ok = bufr_parse_in (descs, 0, ndescs - 1, bufr_val_from_global, 0);
        valarray->vals = (varfl*) NULL;
        bufr_close_val_array ();
    }

    /* close bitstreams and write data to bufr message */

    bufr_close_descsec_w (&msg, desch);

    *ddsec = msg.sec[3];
    *ddescl = (size_t) msg.secl[3];

    bufr_close_datasect_w (&msg);

    *datasec = msg.sec[4];
    *datasecl = (size_t) msg.secl[4];
    
    return ok;
}
/*===========================================================================*/
/** \ingroup basicin
   \brief Creates section 3 and 4 of BUFR message from arrays of data and
  data descriptors.

  This function codes data from an array data descriptors \p descs and an 
  array of varfl-values \p vals to a data section and
  a data descripor section of a BUFR message. Memory for both sections is
  allocated in this function and must be freed by the calling functions.


   \param[in] descs
     Data-descriptors corresponding to \p vals. 
     For each descriptor there
     must be a data-vaule stored in \p vals. \p descs may also include
     replication factors and sequence descriptors. 
     In that case there must be a larger number of
     \p vals then of \p descs.

   \param[in] ndescs  
     Number of data descriptos contained in \p descs.

   \param[in] vals    
     Data-values to be coded in the data section. For each entry in
     \p descs there must be an entry in \p vals. If there are relication
     factors in \p descs, of course there must be as much \p vals as definded
     by the replication factor.

   \param[out] msg The BUFR message where to store the coded descriptor and
                   data sections. The memory-area for both sections
                   is allocated by this function and must be freed by
                   the calling function using \ref bufr_free_data.

   \return The return-value is 1 if data was successfully stored, 0 if not.

   \see bufr_encode_sections0125, bufr_data_from_file, bufr_read_msg

*/


int bufr_encode_sections34 (dd* descs, int ndescs, varfl* vals, bufr_t* msg)


{
    char *datasec = NULL; 
    char *ddsec = NULL;
    size_t datasecl = 0;
    size_t ddescl = 0;
    int ret;

    if (msg == (bufr_t*) NULL) {
        fprintf (stderr, "Error writing data to BUFR message\n");
        return 0;
    }
    
    ret = bufr_create_msg (descs, ndescs, vals, (void**) &datasec, 
                           (void**) &ddsec, &datasecl, &ddescl);

    msg->sec[3] = ddsec;
    msg->sec[4] = datasec;
    msg->secl[3] = ddescl;
    msg->secl[4] = datasecl;

    return ret;
}

/*===========================================================================*/
/** \ingroup basicout
    \brief This functions reads the encoded BUFR-message to a binary file

   This function reads the encoded BUFR message from a binary file,
   calculates the section length and writes each section to a memory
   block.
   Memory for the sections is allocated by this function and must be
   freed by the calling function using \ref bufr_free_data.

   \param[in] msg  The complete BUFR message
   \param[in] file The filename of the binary file

   \return 1 on success, 0 on error

   \see bufr_write_file 
*/

int bufr_read_file (bufr_t* msg, const char* file) {

    FILE* fp;           /* file pointer to bufr file */
    char* bm;           /* pointer to memory holding bufr file */
    int len;

    /* open file */

    fp = fopen (file, "rb");
    if (fp == NULL) {
        fprintf (stderr, "unable to open file '%s'\n", file);
        return 0;
    }

    /* get length of message */

    fseek (fp, 0L, SEEK_END);
    len = ftell (fp);
    fseek (fp, 0L, SEEK_SET);

    /* allocate memory and read message */

    bm = (char *) malloc ((size_t) len);
    if (bm == NULL) {
        fprintf (stderr, 
                 "unable to allocate %d bytes to hold BUFR-message !\n", len);
        fclose (fp);
        return 0;
    }
    if (fread (bm, 1, (size_t) len, fp) != (size_t) len) {
        fprintf (stderr, "Error reading BUFR message from file!\n");
        fclose (fp);
        free (bm);
        return 0;
    }

    fclose (fp);

    /* get raw bufr data */

    if (!bufr_get_sections (bm, len, msg)) {
        free (bm);
        return 0;
    }

    free (bm);
    return 1;
}
/*===========================================================================*/
/** \ingroup basicout
    \brief Calculates the section length of a BUFR message and allocates
    memory for each section.

    This function calculates the sections length of a BUFR message
    and allocates memory for each section. 
    The memory has to be freed by the calling function using 
    \ref bufr_free_data.

    \param[in] bm Pointer to the memory where the raw BUFR message is stored
    \param[in] len Length of \p bm
    \param[in,out] msg The BUFR message containing the single sections and
                       section length

    \return Returns the length of the complete BUFR message or 0 on error.

    \see bufr_free_data, bufr_read_file
*/

int bufr_get_sections (char* bm, int len, bufr_t* msg) 

{
    int co, l;
    char* buf;          /* pointer to beginning of BUFR message */
    char* b7777;        /* pointer to end of BUFR message */
    int i;

    /* Search for "BUFR" */

    buf = NULL;
    for (l = 0; l < len - 4 && buf == NULL; l ++) {
        if (*(bm + l)     == 'B' && 
            *(bm + l + 1) == 'U' &&
            *(bm + l + 2) == 'F' &&
            *(bm + l + 3) == 'R') buf = bm + l;
    }
    if (buf == NULL) {
        fprintf (stderr, "'BUFR' not found in BUFR-message !\n");
        return 0;
    }

    /* Check for the ending "7777" */

    b7777 = NULL;
    for (l = 0; l < len - 3 && b7777 == NULL; l ++) {
        if (*(bm + l)     == '7' && 
            *(bm + l + 1) == '7' &&
            *(bm + l + 2) == '7' &&
            *(bm + l + 3) == '7') b7777 = bm + l;
    }
    if (b7777 == NULL) {
        fprintf (stderr, "'7777' not found in BUFR-message !\n");
        return 0;
    }

    /* Get length of all 6 sections */
    
    if (!get_lens (buf, len, msg->secl)) {
        fprintf (stderr, "unable to read lengths of BUFR-sections !\n");
        return 0;
    }

    /* allocate memory for each section */

    co = 0;
    for (i = 0; i < 6; i ++) {
        msg->sec[i] = (char *) malloc ((size_t) msg->secl[i] + 1);
        if (msg->sec[i] == NULL) {
            fprintf (stderr, 
                     "unable to allocate %d bytes for section %d !\n", 
                     msg->secl[i], i);
            return 0;
        }
        memcpy (msg->sec[i], buf + co, (size_t) msg->secl[i]);
        co += msg->secl[i];
    }
    return co;
}

/*===========================================================================*/
/** \ingroup extin
    \brief Write descriptor section of a BUFR message to the bitsream 

    This function writes the descriptor section of a BUFR message 
    to the section 3 bitstream which has already been opened using
    \ref bufr_open_descsec_w

    \param[in]     descp  Array holding the data descriptors
    \param[in]     ndescs Number of descriptors
    \param[in]     desch  Handle to the bitstream

    \return 1 on success, 0 on error

    \see bufr_open_descsec_w, bufr_out_descsec
*/

int bufr_out_descsec (dd *descp, int ndescs, int desch)

{
    unsigned long l;
    int i;

    /* Append data descriptor to data descriptor section */

    for (i = 0; i < ndescs; i ++) {
        l = (unsigned long) descp->f;
        if (!bitio_o_append (desch, l, 2)) return 0;
        l = (unsigned long) descp->x;
        if (!bitio_o_append (desch, l, 6)) return 0;
        l = (unsigned long) descp->y;
        if (!bitio_o_append (desch, l, 8)) return 0;
        descp ++;
    }

    return 1;
}
/*===========================================================================*/
/** \ingroup extin
    \brief Open bitstream for section 3 for writing and set default values 

    This function opens the bitstream for section 3 and sets default values.
    The bistream must be closed using \ref bufr_close_descsec_w.

    \return Returns handle for the bitstream or -1 on error.

    \see bufr_close_descsec_w, bufr_out_descsec
*/

int bufr_open_descsec_w (int subsets) 
{

    size_t n;
    int desch;

    /* open bitstream */

    desch = bitio_o_open ();
    if (desch == -1) {
        bitio_o_close (desch, &n);
        return -1;
    }

    /* output default data */

    bitio_o_append (desch, 0L, 24);  /* length of descriptor-section, set to 
                                         0. The correct length is set by 
                                         close_descsec_w. */
    bitio_o_append (desch, 0L, 8);   /* reserved octet, set to 0 */
    bitio_o_append (desch, subsets, 16);  /* number of data subsets */
    bitio_o_append (desch, 128L, 8); /* observed non-compressed data */
    return desch;
}
/*===========================================================================*/

/** \ingroup extin
    \brief Write length of section 3 and close bitstream

    This function calculates and writes the length of section 3, then closes
    the bitstream.

   \param[in,out] bufr BUFR message to hold the section.
   \param[in]     desch Handle to the bitstream

   \see bufr_open_descsec_w, bufr_out_descsec

*/

void bufr_close_descsec_w(bufr_t* bufr, int desch) {

    int n;
    size_t st;

    if (desch == -1 || bufr == (bufr_t*) NULL) return;

    /* get current length */

    n = (int)  bitio_o_get_size (desch);

    /* number of bytes must be an even number */

	if (n % 2 != 0) bitio_o_append (desch, 0L, 8);

    /* write length of section to beginning */
    
    n = (int) bitio_o_get_size (desch);
    bitio_o_outp (desch, (long) n, 24, 0L);

    /* close bitstream and return pointer */

    bufr->sec[3] = (char *) bitio_o_close (desch, &st);
    bufr->secl[3] = (int) st;
}


/*===========================================================================*/
/** \ingroup deprecated_g
   \deprecated use \ref bufr_encode_sections0125 instead

   Sets up section 0,1,2,5 in a rather easy fashion and takes Section 1 data
   from structure s1.

   \param[in,out]  sec  Sections 0 - 5
   \param[in,out]  secl Lengths of sections 0 - 5
   \param[in]      s1   Data to be put into Section 1
*/
int setup_sec0125 (char* sec[], size_t secl[], sect_1_t s1)

{
    bufr_t msg;
    int i;

    for (i = 0; i < 6; i++) {
        msg.secl[i] = (int) secl[i];
        msg.sec[i] = sec[i];
    }

    if (!bufr_encode_sections0125 (&s1, &msg))
        return 0;

    for (i = 0; i < 6; i++) {
        secl[i] = (size_t) msg.secl[i];
        sec[i]  = msg.sec[i];
    }

    return 1;

}
/*===========================================================================*/
/** \ingroup deprecated_g
    \deprecated Use \ref bufr_write_file instead.

    Write BUFR message to a binary file.

    \param[in] sec     Poiter-Array to the 6 sections.
    \param[in] secl    Length of the sections.
    \param[in] buffile Output-File

    \return The function returns 1 on success, 0 on a fault.
*/

int save_sections (char** sec, size_t* secl, char* buffile)

{
    FILE *fp;
    int i;

    /* open file */

    fp = fopen (buffile, "wb");
    if (fp == NULL) {
        fprintf (stderr, "Could not open file %s!\n", buffile);
        return 0;
    }

    /* output all sections */

    for (i = 0; i < 6; i ++) {
        if (fwrite (sec[i], 1, secl[i], fp) != secl[i]) {
            fclose (fp);
            fprintf (stderr, 
                     "An error occoured writing '%s'. File is invalid !\n", 
                     buffile);
            return 0;
        }
    }

    /* close file and return */

    fclose (fp);
    return 1;
}


/*===========================================================================*/

/** \ingroup utils_g
    \brief Parse data descriptors and call user defined functions for 
    each data element or for each descriptor

   This function, a more general version of \ref bufr_parse, parses 
   a descriptor or a sequence of descriptors and calls the user defined 
   functions \p inputfkt and \p outputfkt for each data-value 
   corresponding to an element descriptor.
   In case of CCITT (ASCII) data it calls the user-functions for each 
   character of the string.

   Data values are read in using the user-defined function \p inputfkt and
   written out using \p outputfkt.

   Optionally the user-defined functions are called for all descriptors, 
   including sequence descriptors and data modification descriptors.

   \param[in] descs      Pointer to the data-descriptors.
   \param[in] start      First data-descriptor for output.
   \param[in] end        Last data-descriptor for output.
   \param[in] inputfkt   User defined input function to be called for each 
                         data-element or descriptor 
   \param[in] outputfkt  User defined ouput function to be called for each 
                         data-element or descriptor
   \param[in] callback_all_descs Flag that indictes when the user-functions
                         are to be called: \n 
                         \b 0 for normal behaviour 
                         (call user-functions for each
                         element descriptor and each CCITT character) \n
                         \b 1 for extended behaviour (call both user-functions
                         also for sequence descriptors and 
                         CCITT descriptors, \n
                         call \p outputfkt also for replication descriptors
                         and data modification descriptors.)

   \return
   The function returns 1 on success, 0 on error.

   \see bufr_parse, bufr_parse_in, bufr_parse_out, \ref cbin,
   \ref cbout

*/

int bufr_parse_new (dd *descs, int start, int end, 
                    int (*inputfkt) (varfl *val, int ind),
                    int (*outputfkt) (varfl val, int ind),
                    int callback_all_descs) {

    int i, j, nrep, nd;
    int ind;                    /* current descriptor index */
    varfl d;                    /* one float value to process */
    dd descr;                   /* current descriptor */
    static int level = 0;       /* recursion level */
    char* tmp;
    int operator_qual;          /* flag that indicates data descriptor
                                   operator qualifiers (0 31 y) */


    /* increase recursion level */

    level ++;

    /* parse all descriptors */

    for (ind = start; ind <= end; ind++) {

        /* get current descriptor */

        memcpy (&descr, descs + ind, sizeof (dd));

        if (descr.f == 0) {

            /* descriptor is element descriptor */

            if ((i = get_index (ELDESC, &descr)) < 0) {

                /* invalid descriptor */

                fprintf (stderr, 
                       "Unknown data descriptor found: F=%d, X=%d, Y=%d !\n", 
                         descr.f, descr.x, descr.y);
                return 0;
            }

            /* Special Treatment for ASCII data */

            if (strcmp (des[i]->el->unit, "CCITT IA5") == 0) {
                
                /* call outputfkt to ouput descriptor and allow
                   use of proper callback for ascii */
                
                if (callback_all_descs) {
                    varfl v;
                    des[_desc_special]->el->d.f = descr.f;
                    des[_desc_special]->el->d.x = descr.x;
                    des[_desc_special]->el->d.y = descr.y;
                    des[_desc_special]->el->dw = des[i]->el->dw;
                    tmp = des[_desc_special]->el->unit;
                    des[_desc_special]->el->unit = des[i]->el->unit;
                    if (!(*inputfkt) (&v, _desc_special)) return 0;
                    if (!(*outputfkt) (0, _desc_special)) return 0;
                    des[_desc_special]->el->unit = tmp;
                    continue;
                }

                /*loop through all bytes of the character 
                  string and store them using the special descriptor 
                  we have created. */

                for (j = 0; j < des[i]->el->dw / 8; j ++) { 

                    if (!(*inputfkt) (&d, ccitt_special)) return 0;
                    if (!(*outputfkt) (d, ccitt_special)) return 0;
                }
                continue;
            }

            /* Write data to output function. If an "Add associated field" 
               has been set we have to store additional items, 
               except it is a 0 31 y descritor */

            if (_bufr_edition < 3) {
                operator_qual = (des[i]->el->d.x == 31 && 
                                 des[i]->el->d.y == 21);
            } else {
                operator_qual = des[i]->el->d.x == 31;
            }

            if (addfields != 0 && !operator_qual) {
                    
                /* set special descriptor */
                
                des[add_f_special]->el->scale  = 0;
                des[add_f_special]->el->refval = 0;
                des[add_f_special]->el->dw     = addfields;

                /* process data */

                if (!(*inputfkt) (&d, add_f_special)) return 0;
                if (!(*outputfkt) (d, add_f_special)) return 0;
            }

            /* finally process data for the given descriptor */
                
            if (!(*inputfkt) (&d, i)) return 0;
            if (!(*outputfkt) (d, i)) return 0;

            /* Check if this is date/time info and keep this data for 
               further requests in bufr_get_date_time */

            if (descr.x == 4) switch (descr.y)
                {
                case 1: 
                    if (_bufr_edition >= 4) {
                        year_ = (long) d; 
                    }
                    else {
                        year_ = (long) ((int) (d-1) %100 + 1);
                    }
                    break;
                case 2: mon_  = (long) d; break;
                case 3: day_  = (long) d; break;
                case 4: hour_ = (long) d; break;
                case 5: min_  = (long) d; break;
                }
            continue;
        } /* end if (... ELDESC ...) */

        else if (descr.f == 3) {

            /* If data-descriptor is a sequence descriptor -> call this 
               function again for each entry in the sequence descriptor 
               or call user defined callback if parse_seqdescs is not set 
            */

            if ((i = get_index (SEQDESC, &descr)) < 0) {

                /* invalid descriptor */

                fprintf (stderr, 
                       "Unknown data descriptor found: F=%d, X=%d, Y=%d !\n", 
                         descr.f, descr.x, descr.y);
                return 0;
            }

            if (!callback_all_descs) {
                if (!bufr_parse_new (des[i]->seq->del, 0, 
                                     des[i]->seq->nel - 1,
                                     inputfkt, outputfkt, 0)) {
                    return 0;
                }
            }
            else {
                if (!inputfkt (&d, i)) return 0;
                if (!outputfkt (0, i)) return 0;
            }

            continue;
        }

        else if (descr.f == 1) {

            /* replication descriptor */

            nd   = descr.x;
            nrep = descr.y;

            /* output descriptor if not in input mode */
            
            if (callback_all_descs) {

                des[_desc_special]->el->d.f = descr.f;
                des[_desc_special]->el->d.x = descr.x;
                des[_desc_special]->el->d.y = descr.y;
                if (!(*outputfkt) (0, _desc_special)) return 0;
            }

            /* if there is a delayed replication factor */

            if (nrep == 0) {

                /* get number of replications, remember it and write it out*/

                ind++;
                memcpy (&descr, descs + ind, sizeof (dd));
                if ((i = get_index (ELDESC, &descr)) < 0) {
                    fprintf (stderr, 
                        "Unknown data descriptor found: F=%d, X=%d, Y=%d !\n", 
                             descr.f, descr.x, descr.y);
                    return 0;
                }
                if (!(*inputfkt) (&d, i)) return 0;
                nrep = (int) d;
                if (!(*outputfkt) (nrep, i)) return 0;
                
                /* data replication */
                
                if (descr.y == 11 || descr.y == 12)
                    nrep = 1;
            }
            
            /* do the replication now */

            for (i = 0; i < nrep; i ++) {
                if (!bufr_parse_new (descs, ind + 1, ind + nd, inputfkt, 
                                     outputfkt, callback_all_descs))
                    return 0;
                _replicating++;
            }
            _replicating -= nrep;
            ind += nd;
            continue;
        }

        else if (descr.f == 2) {

            /* data modification descriptor */

            if (callback_all_descs) {
            
                /* special treatment for ascii data (2 5 y) */

                if (descr.x == 5)
                {
                    varfl v;
                    des[_desc_special]->el->d.f = descr.f;
                    des[_desc_special]->el->d.x = descr.x;
                    des[_desc_special]->el->d.y = descr.y;
                    des[_desc_special]->el->dw = descr.y * 8;
                    tmp = des[_desc_special]->el->unit;
                    des[_desc_special]->el->unit = des[i]->el->unit;
                    if (!(*inputfkt) (&v, _desc_special)) return 0;
                    if (!(*outputfkt) (0, _desc_special)) return 0;
                    des[_desc_special]->el->unit = tmp;
                    continue;
                }
                des[_desc_special]->el->d.f = descr.f;
                des[_desc_special]->el->d.x = descr.x;
                des[_desc_special]->el->d.y = descr.y;
                if (!(*outputfkt) (0, _desc_special)) return 0;
            }

            switch (descr.x) {

                /* change of datawidth, valid until cancelled by 2 01 000 */
            case 1:   
                if (descr.y == 0) {
                    dw = 128;
                } else {
                    dw = descr.y; 
                }
                continue;
                
                /* change of scale, valid until cancelled by 2 02 000 */
            case 2:
                if (descr.y == 0) {
                    sc = 128;
                } else {
                    sc = descr.y;
                }
                continue;

                /* modyify reference values */
            case 3:

                /* revert all reference value  changes */
                
                if (descr.y == 0)
                {
                    while (cf_spec_num--)
                    {
                        i = get_index (ELDESC, &cf_spec_des[cf_spec_num]);
                        des[i]->el->refval = cf_spec_val[cf_spec_num];
                    }
                }
     
                /* stop reference value change */
                
                else if (descr.y == 255)
                    ;
                     
                /* start reference value change */
                
                else
                {
                    des[cf_special]->el->dw = descr.y;
                    des[cf_special]->el->scale = 0;
                    des[cf_special]->el->refval = 0;
                    
                    /* read new ref. value for all following element descriptors, 
                       until 2 3 255 */
                    
                    ind++;
                    while (ind <= end && ! 
                           (descs[ind].f == 2 && descs[ind].x == 3 && descs[ind].y == 255))
                    {
                        memcpy (&descr, descs + ind, sizeof (dd));
                        if ((i = get_index (ELDESC, &descr)) < 0) {
                            fprintf (stderr, 
                            "Unknown data descriptor found: F=%d, X=%d, Y=%d !\n", 
                                 descr.f, descr.x, descr.y);
                            return 0;
                        }
                        
                        /* get new reference value */

                        des[cf_special]->el->d = descr;
                        if (!(*inputfkt) (&d, cf_special)) return 0;
                        if (!(*outputfkt) (d, cf_special)) return 0;

                        /* save old reference value */
                        
                        if (cf_spec_num < MAX_ADDFIELDS)
                        {
                            cf_spec_des[cf_spec_num] = des[i]->el->d;
                            cf_spec_val[cf_spec_num++] = des[i]->el->refval;
                            des[i]->el->refval = d;
                        }
                        else
                        {
                            fprintf (stderr, 
                                "Maximum number of reference value changes!\n");
                            return 0;
                        }
                        ind++;
                    }
                    
                    /* to allow output stop of ref. value 2 3 255 */

                    if (ind <= end)
                        ind--;
                }
                continue;
 
                /* add associated field, valid until canceled by 2 04 000 */
            case 4:
                if (descr.y == 0) {
                    naf_ --;
                    if (naf_ < 0) {
                        fprintf (stderr, "Illegal call of 2 04 000!\n");
                        return 0;
                    }
                    addfields = af_[naf_];
                }
                else {
                    af_[naf_] = addfields;
                    naf_ ++;
                    if (naf_ > MAX_ADDFIELDS) {
                        fprintf (stderr, 
                            "Maximum number of associated fields reached!\n");
                        return 0;
                    }
                    addfields += descr.y;
                }
                continue;

               
            /* signify character */
            case 5: 
                for (i = 0; i < descr.y; i++) 
                { 
                    if (!(*inputfkt) (&d, ccitt_special)) return 0;
                    if (!(*outputfkt) (d, ccitt_special)) return 0;
                }
                continue;

            case 6: /* signify dw for local desc. */
                if (ind < end && get_index (ELDESC, descs + ind + 1) == -1)
                {
                    ind++;
                    des[cf_special]->el->d = descs[ind];
                    des[cf_special]->el->dw = descr.y;
                    des[cf_special]->el->scale = 0;
                    des[cf_special]->el->refval = 0;
                    if (!(*inputfkt) (&d, cf_special)) return 0;
                    if (!(*outputfkt) (d, cf_special)) return 0;
                }
                continue;
                
            case 21: /* data not present */
            case 22: /* quality info follows */
            case 23: /* substituted values op */
            case 24: /* statistical values */
            case 25: /* statistical values */
            case 32: /* replaced values */
            case 35: /* cancel back reference */
            case 36: /* define data present */
            case 37: /* use data present */
                /* these descriptors don't require special en-/decoding */
                continue;

            /* BUFR edition 4 only */
            /* case 7: increase scale, ref. and width */
            /* case 8: change width of CCITT field */
            /* case 41: event */
            /* case 42: conditioning event */
            /* case 43: categorical forecast */

                /* invalid descriptor */
            default:
                fprintf (stderr, 
                        "Unknown data modification descriptor found: F=%d, X=%d, Y=%d !\n", 
                         descr.f, descr.x, descr.y);
                return 0;
            }
        }
        else {
            
            /* invalid descriptor */
            
            fprintf (stderr, 
                     "Unknown data descriptor found: F=%d, X=%d, Y=%d !\n", 
                     descr.f, descr.x, descr.y);
            return 0;
        }
        
    } /* end for loop over all descriptors */

    /* decrease recursing level */

    level --;
    return 1;

}

/*===========================================================================*/
/** \ingroup utils_g
    \brief Parse data descriptors and call user-function for each element

   This function parses a descriptor or a sequence of
   descriptors and calls the user defined function
   \p userfkt for each data-value corresponding to an element descriptor.
   In case of CCITT (ASCII) data it calls \p userfkt for each character of
   the string.

   Data values are read from an array of floats stored at \p vals.
   
   \param[in] descs      Pointer to the data-descriptors.
   \param[in] start      First data-descriptor for output.
   \param[in] end        Last data-descriptor for output.
   \param[in] vals       Pointer to an array of values.
   \param[in,out] vali   Index for the array \p vals that identifies the 
                         values to be used for output. 
                         \p vali is increased after data-output.
   \param[in] userfkt    User-function to be called for each data-element

   \return
   The function returns 1 on success, 0 if there was an error outputing to the
   bitstreams.
*/


int bufr_parse (dd* descs, int start, int end, varfl *vals, unsigned *vali,
                int (*userfkt) (varfl val, int ind)) {
    int ok;
    bufrval_t* bufrvals;

    bufrvals = bufr_open_val_array ();

    if (bufrvals == (bufrval_t*) NULL) {
        return 0;
    }

    bufrvals->vals = vals;
    bufrvals->vali = *vali;
    ok = bufr_parse_new (descs, start, end, bufr_val_from_global, userfkt, 
                         0);
    *vali = bufrvals->vali;

    bufrvals->vals = (varfl*) NULL;
    bufr_close_val_array ();
    return ok;
}


/*===========================================================================*/
/** \ingroup extin
    \brief Parse data descriptors and call user defined input function for 
    each element or for each descriptor

   This function, derived from \ref bufr_parse_new, parses 
   a descriptor or a sequence of descriptors and calls the user defined 
   function \p inputfkt for reading each data-value corresponding to an 
   element descriptor.
   In case of CCITT (ASCII) data it calls the user-function for each 
   character of the string.

   Data values are wrote out to the global data section bitstream
   (see \ref bufr_open_datasect_w).

   Optionally \p inputfkt is called also for sequence descriptors 
   and ccitt descriptors

   \param[in] descs      Pointer to the data-descriptors.
   \param[in] start      First data-descriptor for output.
   \param[in] end        Last data-descriptor for output.
   \param[in] inputfkt   User defined input function to be called for each 
                         data-element or descriptor 
   \param[in] callback_descs Flag that indictes when the user-functions
                         are to be called: \n
                         \b 0 for normal behaviour 
                         (call \p inputfkt for each
                         element descriptor and each CCITT character) \n
                         \b 1 for extended behaviour 
                         (call \p inputfkt also
                         for sequence descriptors and CCITT descriptors)
   \return
   The function returns 1 on success, 0 on error

   \see bufr_parse, bufr_parse_new, bufr_parse_in, \ref cbin,
   bufr_open_datasect_w
*/


int bufr_parse_in  (dd *descs, int start, int end, 
                    int (*inputfkt) (varfl *val, int ind),
                    int callback_descs) {

  return bufr_parse_new (descs, start, end, inputfkt,  
                         bufr_val_to_datasect, callback_descs); 
}

/*===========================================================================*/
/** \ingroup extout
    \brief Parse data descriptors and call user defined output function for 
    each element or for each descriptor

   This function, derived from \ref bufr_parse_new, parses 
   a descriptor or a sequence of descriptors and calls the user defined 
   function \p outputfkt for each data-value corresponding to an 
   element descriptor.
   In case of CCITT (ASCII) data it calls the user-function for each 
   character of the string.

   Data values are read from the global data section bitstream
   (see \ref bufr_open_datasect_r).

   Optionally \p outputfkt is called for all descriptors 
   including sequence descriptors, repetition descriptors, ...

   \param[in] descs      Pointer to the data-descriptors.
   \param[in] start      First data-descriptor for output.
   \param[in] end        Last data-descriptor for output.
   \param[in] outputfkt  User defined output function to be called for each 
                         data-element or descriptor 
   \param[in] callback_all_descs Flag that indictes when the user-functions
                         are to be called: \n
                         \b 0 for normal behaviour 
                         (call \p outputfkt for each
                         element descriptor and each CCITT character) \n
                         \b 1 for extended behaviour 
                         (call \p outputfkt for all descriptors)
   \return
   The function returns 1 on success, 0 on error

   \see bufr_parse, bufr_parse_new, bufr_parse_in, \ref cbout,
   bufr_open_datasect_r
*/


int bufr_parse_out  (dd *descs, int start, int end, 
                     int (*outputfkt) (varfl val, int ind),
                     int callback_all_descs) {

    return bufr_parse_new (descs, start, end, bufr_val_from_datasect,  
                           outputfkt, callback_all_descs); 
}


/*===========================================================================*/
/** \ingroup extin
    \brief Reads section 1 from a file and stores data read in s1

    This function reads section 1 from an ASCII file and stores the data
    read in a structure \p s1 .
    If the file can not be read, \p s1 is filled with internally defined 
    default values.

    \param[in,out] s1     Structure where section 1 data is stored.
    \param[in]     file   Filename of the input file.

    \see bufr_sect_1_to_file
*/

void bufr_sect_1_from_file (sect_1_t* s1, const char* file)
{
  FILE *fp;
  char buf[200];
  int val, count;

  /* Set section 1 to default vales */

  s1->mtab    = 0;
  s1->subcent  = SUBCENTER;
  s1->gencent  = GENCENTER;
  s1->updsequ = 0;
  s1->opsec   = 0;
  s1->dcat    = 6;
  s1->idcatst = 0;
  s1->dcatst  = 0;
  s1->vmtab   = VMTAB;
  s1->vltab   = VLTAB;
  s1->year    = 999;
  s1->mon     = 999;
  s1->day     = 999;
  s1->hour    = 999;
  s1->min     = 999;
  s1->sec     = 0;

/* open file and read data */

  fp = fopen (file, "r");
  if (fp == NULL) {
      return;
  }

  count = 0;
  while (fgets (buf, 200, fp) != NULL) {
    if (sscanf (buf, "%d", &val) == 1) {
        switch (count) {
        case 0:  s1->mtab    = val; break;
        case 1:  s1->subcent  = val; break;
        case 2:  s1->gencent  = val; break;
        case 3:  s1->updsequ = val; break;
        case 4:  s1->opsec   = val; break;
        case 5:  s1->dcat    = val; break;
        case 6:  s1->dcatst  = val; break;
        case 7:  s1->vmtab   = val; break;
        case 8:  s1->vltab   = val; break;
        case 9:  s1->year    = val; break;
        case 10:  s1->mon     = val; break;
        case 11: s1->day     = val; break;
        case 12: s1->hour    = val; break;
        case 13: s1->min     = val; break;
            /* new fields for edition 4 */
        case 14: s1->sec     = val; break;
        case 15: s1->idcatst = val; break;
        }
        count ++;
    }
  }
  fclose (fp);
}

/*===========================================================================*/
/** \ingroup basicin
    \brief This function creates sections 0, 1, 2 and 5.

    This function creates sections 0, 1, 2 and 5 of a BUFR message.
    Memory for this section is allocated by this function and must be
    freed by the calling function using \ref bufr_free_data. \n
    The total length of the message is calculeted out of the single
    section length, thus sections 3 and 4 must already be present in
    the bufr message when calling this function.
    The BUFR edition is wrote into section 0 and is taken from the global
    \ref _bufr_edition parameter. \n
    If section 1 data and time parameters are set to 999 (no value), the
    current system time is taken for coding date and time information.

    \param[in] s1 \ref sect_1_t structure containing section 1 data
    \param[in,out] msg BUFR message where the sections are to be stored. Must
                       already contain section 3 and 4.

    \return 1 on success, 0 on error.
*/

int bufr_encode_sections0125 (sect_1_t* s1, bufr_t* msg)
{

    char** sec = msg->sec;
    int* secl = msg->secl;

    size_t st;
    int i, hand;
    long len;
    time_t t;
    struct tm t1;

    /* encode section 1. */

    hand = bitio_o_open ();
    if (hand == -1) return 0;
    if (_bufr_edition >= 4) {
         bitio_o_append (hand, 22L, 24);         /* length of section */
    }
    else {
        bitio_o_append (hand, 18L, 24);         /* length of section */
    }
    bitio_o_append (hand, s1->mtab, 8);     /* master table used */
    if (_bufr_edition >= 4) {
        bitio_o_append (hand, s1->gencent, 16);  /* originating/generating 
                                                   center */
        bitio_o_append (hand, s1->subcent, 16);  /* originating/generating
                                                   subcenter */
    }
    else {
        bitio_o_append (hand, s1->subcent, 8);  /* originating subcenter */
        bitio_o_append (hand, s1->gencent, 8);  /* originating/generating 
                                                   center */
    }
    bitio_o_append (hand, s1->updsequ, 8);  /* original BUFR message */
    bitio_o_append (hand, s1->opsec, 8);    /* no optional section */
    bitio_o_append (hand, s1->dcat, 8);     /* message type */
    if (_bufr_edition >= 4)
        bitio_o_append (hand, s1->idcatst, 8);   /* international message 
                                                    subtype */
    bitio_o_append (hand, s1->dcatst, 8);   /* local message subtype */
    bitio_o_append (hand, s1->vmtab, 8);    /* version number of master table*/
    bitio_o_append (hand, s1->vltab, 8);    /* version number of local table */

    /* if not given in section1-file take system time */

    if (s1->year == 999) {   
        time (&t);
        memcpy (&t1, localtime (&t), sizeof (struct tm));
        if (_bufr_edition >= 4) {
            bitio_o_append (hand, (long) t1.tm_year + 1900, 16); /* year */
        }
        else {
            t1.tm_year = (t1.tm_year - 1) % 100 + 1;
            bitio_o_append (hand, (long) t1.tm_year, 8);      /* year */
        }
        bitio_o_append (hand, (long) t1.tm_mon + 1, 8);       /* month */
        bitio_o_append (hand, (long) t1.tm_mday, 8);          /* day */
        bitio_o_append (hand, (long) t1.tm_hour, 8);          /* hour */
        bitio_o_append (hand, (long) t1.tm_min, 8);           /* minute */
        if (_bufr_edition >= 4) 
            bitio_o_append (hand, (long) t1.tm_sec, 8);       /* seconds */
    }
    else {
        if (_bufr_edition >= 4) {
            bitio_o_append (hand, s1->year, 16);              /* year */
        }
        else {
            s1->year = (s1->year - 1) % 100 + 1;
            bitio_o_append (hand, s1->year, 8);                /* year */
        }
        bitio_o_append (hand, s1->mon, 8);                     /* month */
        bitio_o_append (hand, s1->day, 8);                     /* day */
        bitio_o_append (hand, s1->hour, 8);                    /* hour */
        bitio_o_append (hand, s1->min, 8);                     /* minute */
        if (_bufr_edition >= 4)
            bitio_o_append (hand, s1->sec, 8);                 /* second */
    }
    if (_bufr_edition < 4)
        bitio_o_append (hand, 0L, 8);                      /* filler (0) */
    sec[1] = (char *) bitio_o_close (hand, &st);
    secl[1] = (int) st;

    /* there is no section 2 */

    sec[2] = NULL;
    secl[2] = 0;

    /* create section 5 */

    hand = bitio_o_open ();
    for (i = 0; i < 4; i ++) bitio_o_append (hand, (long) '7', 8);
    sec[5] = (char *) bitio_o_close (hand, &st);
    secl[5] = (int) st;

    /* calculate total length of BUFR-message */

    secl[0] = 8;     /* section 0 not yet setup */
    len = 0L;
    for (i = 0; i < 6; i ++) len += (long) secl[i];
  

    /* create section 0 */

    hand = bitio_o_open ();
    if (hand == -1) return 0;
    bitio_o_append (hand, (unsigned long) 'B', 8);
    bitio_o_append (hand, (unsigned long) 'U', 8);
    bitio_o_append (hand, (unsigned long) 'F', 8);
    bitio_o_append (hand, (unsigned long) 'R', 8);
    bitio_o_append (hand, len, 24);          /* length of BUFR-message */
    bitio_o_append (hand, (long) _bufr_edition, 8);  /* BUFR edition number */
    sec[0] = (char *) bitio_o_close (hand, &st);
    secl[0] = (int) st;
    return 1;
}
/*===========================================================================*/
/** \ingroup basicin
    \brief This functions saves the encoded BUFR-message to a binary file

   This function takes the encoded BUFR message and writes it to a binary file.

   \param[in] msg  The complete BUFR message
   \param[in] file The filename of the destination file

   \return 1 on success, 0 on error

   \see bufr_read_file 
*/

int bufr_write_file (bufr_t* msg, const char* file)
{

    char** sec = msg->sec; 
    int* secl = msg->secl;
    FILE *fp;
    int i;

    /* open file */

    fp = fopen (file, "wb");
    if (fp == NULL) {
        fprintf (stderr, "Could not open file %s!\n", file);
        return 0;
    }

    /* output all sections */

    for (i = 0; i < 6; i ++) {
        if (fwrite (sec[i], 1, (size_t) secl[i], fp) != (size_t) secl[i]) {
            fclose (fp);
            fprintf (stderr, 
             "An error occoured during writing '%s'. File is invalid !\n", 
                     file);
            return 0;

        }
    }

    /* close file and return */

    fclose (fp);
    return 1;
}

/*===========================================================================*/

/** \ingroup utils_g
    \brief Frees memory allocated for a BUFR message.

    This function frees all memory allocated for a BUFR message 
    by \ref bufr_data_from_file, \ref   bufr_encode_sections0125, \ref 
    bufr_read_file or \ref bufr_get_sections.

    \param[in] msg The encoded BUFR message

*/


void bufr_free_data (bufr_t* msg) {

    int i;

    if (msg == (bufr_t*) NULL) return;

    for (i = 0; i <= 5; i++) {
        if (msg->sec[i] != NULL) 
            free (msg->sec[i]);
    }
    memset (msg, 0, sizeof (bufr_t));
}



/*===========================================================================*/
/** \ingroup utils_g
    \brief Tests equality of descriptor d with (f,x,y) 

    This functions tests wheter a descriptor equals the given values f, x, y

    \param[in] d The descriptor to be tested
    \param[in] ff, xx, yy The values for testing

    \retval 1 If the descriptor equals the given values
    \retval 0 If the descriptor is different to the given values
*/

int bufr_check_fxy(dd *d, int ff, int xx, int yy) {

    if (d == (dd*) NULL) return -1;
    return (d->f == ff) && (d->x == xx) && (d->y == yy);
}



/*===========================================================================*/
/** \ingroup basicout
    \brief This function decodes sections 0 and 1.

    This function decodes sections 0 and 1 of a BUFR message.
    The BUFR edition is read from section 0 and is written to the global
    \ref _bufr_edition parameter. \n

    \param[in,out] s1 \ref sect_1_t structure to contain section 1 data
    \param[in]     msg BUFR message where the sections are stored.

    \return 1 on success, 0 on error.
*/

int bufr_decode_sections01 (sect_1_t* s1, bufr_t* msg)

{
    int h, edition;
    unsigned long l;

    /* section 0 */
    h = bitio_i_open (msg->sec[0], (size_t) msg->secl[0]);
    if (h == -1) return 0;

    bitio_i_input (h, &l, 32);                  /* BUFR */
    bitio_i_input (h, &l, 24);                  /* length of BUFR-message */
    bitio_i_input (h, &l, 8); edition = l;      /* BUFR edition number */
    bitio_i_close (h);
 
    /* section 1 */

    h = bitio_i_open (msg->sec[1], (size_t) msg->secl[1]);
    if (h == -1) return 0;
  
    bitio_i_input (h, &l, 24);                 /* length of section */

    bitio_i_input (h, &l, 8);  s1->mtab = l;    /* master table used */
    if (edition >= 4) {
        bitio_i_input (h, &l, 16);  s1->gencent = l; /* generating center */
        bitio_i_input (h, &l, 16);  s1->subcent = l; /*originating subcenter */
    }
    else {
        bitio_i_input (h, &l, 8);  s1->subcent = l; /* originating subcenter */
        bitio_i_input (h, &l, 8);  s1->gencent = l; /* generating center */
    }
    bitio_i_input (h, &l, 8);  s1->updsequ = l; /* original BUFR message */
    bitio_i_input (h, &l, 8);  s1->opsec = l;   /* no optional section */
    bitio_i_input (h, &l, 8);  s1->dcat = l;    /* message type */
    if (edition >= 4)
        bitio_i_input (h, &l, 8);  s1->idcatst = l;  /* international message 
                                                        sub type */
    bitio_i_input (h, &l, 8);  s1->dcatst = l;  /* local message subtype */
    bitio_i_input (h, &l, 8);  s1->vmtab = l;   /* version number of master 
                                                   table used */
    bitio_i_input (h, &l, 8);  s1->vltab = l;   /* version number of local 
                                                   table used */
    if (edition >= 4) {
        bitio_i_input (h, &l, 16);  s1->year = l;    /* year */
    } else {
        bitio_i_input (h, &l, 8);  s1->year = l;    /* year */
    }
    bitio_i_input (h, &l, 8);  s1->mon = l;     /* month */
    bitio_i_input (h, &l, 8);  s1->day = l;     /* day */
    bitio_i_input (h, &l, 8);  s1->hour = l;    /* hour */
    bitio_i_input (h, &l, 8);  s1->min = l;     /* minute */
    if (edition >= 4)
        bitio_i_input (h, &l, 8);  s1->sec = l;    /* second */
    bitio_i_close (h);

    /* set edition */

    _bufr_edition = edition;

    return 1;
}

/*===========================================================================*/
/** \ingroup extout
    \brief Writes section 1 data to an ASCII file

    This function writes section 1 data to an ASCII file

    \param[in]     s1     Structure where section 1 data is stored.
    \param[in]     file   Filename of the output file.

    \see bufr_sect_1_from_file
*/

int bufr_sect_1_to_file (sect_1_t* s1, const char* file) {

    FILE* fp;

    fp = fopen (file, "w");
    if (fp == NULL) {
        fprintf (stderr, "unable to open output file for section 1 !\n");
        return 0;
    }

    fprintf (fp, "%5d    master table used                  \n", s1->mtab);
    fprintf (fp, "%5d    originating subcenter              \n", s1->subcent);
    fprintf (fp, "%5d    generating center                  \n", s1->gencent);
    fprintf (fp, "%5d    original BUFR message              \n", s1->updsequ);
    fprintf (fp, "%5d    no optional section                \n", s1->opsec);
    fprintf (fp, "%5d    message type                       \n", s1->dcat);
    fprintf (fp, "%5d    local message subtype              \n", s1->dcatst);
    fprintf (fp, "%5d    version number of master table used\n", s1->vmtab);
    fprintf (fp, "%5d    version number of local table used \n", s1->vltab);
    fprintf (fp, "%5d    year                               \n", s1->year);
    fprintf (fp, "%5d    month                              \n", s1->mon);
    fprintf (fp, "%5d    day                                \n", s1->day);
    fprintf (fp, "%5d    hour                               \n", s1->hour);
    fprintf (fp, "%5d    minute                             \n", s1->min);
    /* new fields for bufr edition 4 */
    if (_bufr_edition >= 4) {
        fprintf (fp, "%5d    second                             \n", s1->sec);
        fprintf (fp, "%5d    international message subtype      \n", 
                 s1->idcatst);
    }

    fclose (fp);

    return 1;
}
 
/*===========================================================================*/
/** \ingroup basicout
    \brief Decode BUFR data and descriptor section and write values and 
    descriptors to arrays

    This function decodes the data and descriptor sections of a BUFR message
    and stored them into arrays \p descr and \p vals.
    Memory for storing descriptor- and data-array is
    allocated by this function and has to be freed by the calling function.

    \param[in]  datasec  Is where the data-section is stored.

    \param[in]  ddsec    Is where the data-descriptor-section is stored.

    \param[in]  datasecl Number of bytes of the data-section.

    \param[in]  ddescl   Number of bytes of the data-descriptor-section.

    \param[out] descr    Array where the data-descriptors are stored 
                         after reading them from the data-descriptor section. 
                         This memory area is allocated by this
                         function and has to be freed by the calling function.

    \param[out] ndescs   Number of data-descriptors in \p descs

    \param[out] vals     Array where the data corresponding to the 
                         data-descriptors is stored.

    \param[out] nvals    Number of values in \p vals

    \return
    1 if both sections were decoded successfuly, 0 on error

    \see bufr_create_msg, bufr_data_to_file

    \todo: write new version that uses bufr_t structure for output 

*/


int bufr_read_msg (void* datasec, void* ddsec, size_t datasecl, size_t ddescl,
                   dd** descr, int* ndescs, varfl** vals, size_t* nvals)


{
    int ok = 0, desch, subsets;
    dd *d;
    bufr_t msg;
    bufrval_t* bufrvals;

    memset (&msg, 0, sizeof (bufr_t));

    msg.sec[3] = ddsec;
    msg.secl[3] = (int) ddescl;
    msg.sec[4] = datasec;
    msg.secl[4] = (int) datasecl;

    /* open bitstreams for section 3 and 4 */

    desch = bufr_open_descsec_r (&msg, &subsets); 
    if (desch < 0) 
        return 0;

    if (bufr_open_datasect_r (&msg) < 0) {
        bufr_close_descsec_r (desch);
        return 0;
    }

    /* calculate number of data descriptors  */
    
    *ndescs = bufr_get_ndescs (&msg);

    /* allocate memory and read data descriptors from bitstream */

    ok = bufr_in_descsec (descr, *ndescs, desch);

    /* Input data from data-section according to the data-descriptors */

    *vals = NULL;
    *nvals = 0;
    d = *descr;

    bufrvals = bufr_open_val_array ();

    if (bufrvals == (bufrval_t*) NULL) {
        ok = 0;
    }

    if (ok) {
        while (subsets--)
        {
            ok = bufr_parse_out (d, 0, *ndescs - 1, bufr_val_to_global, 0);
            if (!ok)
                fprintf (stderr, "Error reading data from data-section !\n");
        }
        *vals = bufrvals->vals;
        *nvals = (size_t) bufrvals->nvals;
        bufrvals->vals = (varfl*) NULL;
        bufr_close_val_array ();
    }

    /* close bitstreams */

    bufr_close_descsec_r (desch);
    bufr_close_datasect_r ();

    return ok;
}

/*===========================================================================*/
/** \ingroup extout
    \brief Read descriptor section of a BUFR message from the bitsream 

    This function reads the descriptor section of a BUFR message 
    from the bitsream which was opened using \ref bufr_open_descsec_r

    \param[in,out] descs Array to hold the data descriptors
    \param[in]     ndescs Number of descriptors
    \param[in]     desch  Handle to the bitstream

    \return 1 on success, 0 on error

    \see bufr_get_ndescs, bufr_open_descsec_r, bufr_out_descsec
*/

int bufr_in_descsec (dd** descs, int ndescs, int desch) {

    int err, i;
    unsigned long l = 0;
    dd* d;

    if (desch < 0) {
        fprintf (stderr, "Descriptor handle not available! \n");
        return 0;
    }


    d = *descs = (dd *) malloc (ndescs * sizeof (dd));
    if (*descs == (dd*) NULL) {
        fprintf (stderr, "Unable to allocate memory for data descriptors !\n");
        return 0;
    }

    for (i = 0; i < ndescs; i ++) {
        err = 0;
        err = err || !bitio_i_input (desch, &l, 2);
        d->f = (unsigned char) l;
        if (!err) err = err || !bitio_i_input (desch, &l, 6);
        d->x = (unsigned char) l;
        if (!err) err = err || !bitio_i_input (desch, &l, 8);
        d->y = (unsigned char) l;
        if (err) {
            fprintf (stderr, 
                     "Number of bits for descriptor-section exceeded !\n");
            free (*descs);
            *descs = (dd*) NULL;
            return 0;
        }
        d ++;
    }
    return 1;
}
/*===========================================================================*/
/** \ingroup extout
    \brief Open bitstream of section 3 for reading 
   
    This function opens a bitstream for reading of section 3. It must be
    closed by \ref bufr_close_descsec_r.

    \param[in] msg The encoded BUFR message

    \return Returns handle to the bitstream or -1 on error

    \see bufr_close_descsec_r, bufr_in_descsec

*/

int bufr_open_descsec_r (bufr_t* msg, int *subsets) {
    
    unsigned long l;
    int desch;

    /* open bitstream */

    desch = bitio_i_open (msg->sec[3], msg->secl[3]);

    if (desch == -1) {
        bitio_i_close (desch);
        return -1;
    }

    /* skip first 7 octets (56 bits) */

    bitio_i_input (desch, &l, 24); /* length of section */
    bitio_i_input (desch, &l, 8);  /* reserved */
    bitio_i_input (desch, &l, 16); /* number of subset */
    if (subsets != NULL)
        *subsets = l;
    bitio_i_input (desch, &l, 8);  /* flags */

    return desch;
}


/*===========================================================================*/
/** \ingroup extout
    \brief close bitstream for section 3 

    This functin closes the input bitstream of section 3 which was opened by
    \ref bufr_open_descsec_r.

    \param[in] desch Handle to the bitstream

    \see bufr_open_descsec_r, bufr_in_descsec
*/

void bufr_close_descsec_r (int desch) {

    if (desch == -1) return;
    bitio_i_close (desch);
}

/*===========================================================================*/
/** \ingroup deprecated_g
    \deprecated use \ref bufr_val_to_array instead.

    This function stores the value V to an array of floats VALS. The memory-
    block for VALS is allocated in this function and has to be freed by the
    calling function.

    \param[in,out] vals The array containing the values
    \param[in]     v    The value to be put into the array
    \param[in,out] nvals Number of values in the array

    \return 1 on success, 0 on error.
*/


int val_to_array (varfl** vals, varfl v, size_t* nvals)

{
  static unsigned int nv;         /* Number of values already read from bitstream */
  static unsigned int memsize;    /* Current size of memory-block holding data-values */
  varfl *d;

/* Allocate memory if not yet done */

  if (*vals == NULL) {
    *vals = (varfl *) malloc (MEMBLOCK * sizeof (varfl));
    if (*vals == NULL) return 0;
		memset (*vals, 0, MEMBLOCK * sizeof (varfl));
    nv = 0;
    memsize = MEMBLOCK;
  }

/* Check if memory block is large anough to hold new data */

  if (memsize == nv) {
    *vals = (varfl *) realloc (*vals, (memsize + MEMBLOCK) * sizeof (varfl));
    if (*vals == NULL) return 0;
		memset ((char *) (*vals + memsize), 0, MEMBLOCK * sizeof (varfl));
    memsize += MEMBLOCK;
    if (memsize - 1 > (~(unsigned int) 0) / sizeof (varfl)) {
      fprintf (stderr, "VAL_TO_ARRAY failed in file %s, line %d\n", __FILE__, __LINE__);
      fprintf (stderr, "Try to define varfl as float in file desc.h \n");
      return 0;
    }
  }

/* Add value to array */

  d = *vals;
  *(d + nv) = v;
  nv ++;
  *nvals = nv;
  return 1;
}

/*===========================================================================*/
/** \ingroup utils_g
    \brief Store a value to an array of floats.

    This function stores the value \p v to an array of floats \p vals. 
    The memory-block for \p vals is allocated in this function and has 
    to be freed by the calling function.
    The number of values is used to calculate the size of the array
    and reallocate memory if necessary.

    \param[in,out] vals The array containing the values
    \param[in]     v    The value to be put into the array
    \param[in,out] nv   Current number of values in the array

    \return 1 on success, 0 on error.
*/


int bufr_val_to_array (varfl** vals, varfl v, int* nv)
{
    /* Allocate memory if not yet done */

    if (*vals == (varfl*) NULL) {
        *vals = (varfl *) malloc (MEMBLOCK * sizeof (varfl));
        if (*vals == (varfl*) NULL) {
            fprintf (stderr, "Could not allocate memory for value array!\n");
            return 0;
        }
		memset (*vals, 0, MEMBLOCK * sizeof (varfl));
        *nv = 0;
    }

    /* Check if memory block is large anough to hold new data */

    if (*nv != 0 && *nv % MEMBLOCK == 0) {
        *vals = (varfl*) realloc (*vals, (*nv + MEMBLOCK) * sizeof (varfl));
        if (*vals == (varfl*) NULL) {
            fprintf (stderr, "Could not allocate memory for value array!\n");
            return 0;
        }
		memset ((varfl*) (*vals + *nv), 0, MEMBLOCK * sizeof (varfl));
    }


    /* Add value to array */

    (*vals)[*nv] = v;
    (*nv)++;
    return 1;
}

/*===========================================================================*/
/** \ingroup utils_g
    \brief Store a descriptor to an array.

    This function stores the descriptor \p d to an array of descriptors
    \p descs. 
    The array descs must be large enough to hold \p ndescs + 1 descriptors.

    \param[in]     descs The array containing the descriptors
    \param[in]     d     The descriptor to be put into the array
    \param[in,out] ndescs   Current number of descriptors in the array

    \return 1 on success, 0 on error.
*/


int bufr_desc_to_array (dd* descs, dd d, int* ndescs)
{

    if (*ndescs >= MAX_DESCS) {
        fprintf (stderr, "Maximum number of descriptors exceeded!\n");
        return 0;
    }


    /* Add descriptor to array */

    descs[(*ndescs)++] = d;
    return 1;
}


/*===========================================================================*/
/** \ingroup extout
    \brief Calculate number of data descriptors in a BUFR message

    This function calculates the number of data descriptors in a BUFR
    message.

    \param[in] msg The complete BUFR message

    \return Returns the number of data descriptors.

    \see bufr_in_descsec

*/

int bufr_get_ndescs (bufr_t* msg) {

    if (msg == (bufr_t*) NULL) {
        fprintf (stderr, "Error in bufr_get_ndescs!\n");
        return -1;
    }
    return (((msg->secl[3] - 7)* 8) / 16);  
}

/*===========================================================================*/
/** \ingroup utils_g
    \brief Recall date/time info of the last BUFR-message created

   This function can be called to recall the data/time-info of the
   last BUFR-message created, if the appropiate data descriptors have
   been used.

   \param[out] year 4 digit year if \ref _bufr_edition is set to 4, 
                    year of century (2 digit) if \ref _bufr_edition is < 4.
   \param[out] mon  Month (1 - 12)
   \param[out] day  (1 - 31)
   \param[out] hour 
   \param[out] min
*/


void bufr_get_date_time (long *year, long *mon, long *day, long *hour,
                         long *min)


{
  *year = year_;
  *mon  = mon_;
  *day  = day_;
  *hour = hour_;
  *min  = min_;
}

/*===========================================================================*/
/* callback functions */
/*===========================================================================*/


/** \ingroup cbin
    \brief Outputs one data-value to the data-bitstream.

    This function outputs one data-value to the data-bitstream
    which has to be opened using \ref bufr_open_datasect_w.

    \param[in] val    Data-value to be output.
    \param[in] ind    Index to the global array 
                      \ref des[] holding the description of
                      known data-descriptors.
    
    \return 1 on success, 0 on a fault.

    \see bufr_open_datasect_w, bufr_close_datasect_w, 
         bufr_val_from_datasect
*/

static int bufr_val_to_datasect (varfl val, int ind)


{
    unsigned long l;
    int ret, wi, scale, ccitt, no_change = 0;

    assert (datah_ >= 0);

    ret = 1;
  
    /* No output for special descriptors and sequence descs*/

    if (ind == _desc_special || des[ind]->id == SEQDESC) return 1;

    /* No data width or scale change for 0 31 y, code tables, flag tables
       and ascii data */

    if (_bufr_edition < 3) {
        
        no_change = (des[ind]->el->d.f == 0 && des[ind]->el->d.x == 31);
    }
    else {
        ccitt = (strcmp (des[ind]->el->unit, "CCITT IA5") == 0 || 
                 ind == ccitt_special);
        no_change = ((des[ind]->el->d.f == 0 && des[ind]->el->d.x == 31) ||
                     ccitt || desc_is_codetable(ind) || desc_is_flagtable(ind) ||
                     ind == add_f_special || ind == cf_special);
    }

    if (no_change) {
        wi = des[ind]->el->dw;
        scale = des[ind]->el->scale;
    }
    else {
        wi = des[ind]->el->dw + dw - 128;
        scale = des[ind]->el->scale + sc - 128;
    }

    /* If this is a missing value set all bits to 1 */
    
    if (val == MISSVAL) {
        l = 0xffffffff;
        if (bitio_o_append (datah_, l, wi) == -1) ret = 0;
    }
    /* Else it is a "normal" value */

    else {
        if (ind == cf_special) 
        {
            if (val < 0 )
                l = (unsigned long) (-val) | 1UL << (wi - 1);
            else
                l = (unsigned long) val;
        } 
        else
            l = (unsigned long) (val * pow (10.0, (varfl) scale) 
                             - des[ind]->el->refval + 0.5);  
        /* + 0.5 to round to integer values */

        if (bitio_o_append (datah_, l, wi) == -1) ret = 0;

        /* check if data width was large enough to hold data to be coded */

        if (l >> wi != 0) {
            fprintf (stderr, 
  "WARNING: Tried to code the value %ld to %d bits (Datadesc.=%2d%3d%4d) !\n", 
                     l, wi, des[ind]->el->d.f, des[ind]->el->d.x, 
                     des[ind]->el->d.y);
            fprintf (stderr, "         Decoding will fail !\n");
        }
    }

    return ret;
}

/*===========================================================================*/
/** \ingroup cbinutl
    \brief Opens bitstream for section 4 writing

    This function opens the data section bitstream for writing and 
    returns its handle.

    \return Returns the handle to the data section bitstream or -1
            on error.

    \see bufr_close_datasect_w, bufr_parse_in
*/

int bufr_open_datasect_w() {
    size_t n;

    if (datah_ >= 0) {
        fprintf (stderr, "Global data handle not available.\n");
        return -1;
    }

    /* open bitstream */

    datah_ = bitio_o_open ();
    if (datah_ == -1) {
        bitio_o_close (datah_, &n);
        return -1;
    }

    /* output default data */

    bitio_o_append (datah_, 0L, 24);  /* Length of section (correct value 
                                         stored by close_datasect_w) */
    bitio_o_append (datah_, 0L, 8);   /* reserved octet, set to 0 */
    return datah_;
}

/*===========================================================================*/
/** \ingroup cboututl
    \brief Opens bitstream for reading section 4

    This function opens the data section bitstream at for reading 
    and returns its handle.

    \param[in] msg The BUFR message containing the data section.

    \return Returns the handle to the data section bitstream or -1 on error.

    \see bufr_close_datasect_r, bufr_parse_out
*/

int bufr_open_datasect_r (bufr_t* msg) {
    
    unsigned long l;
    int i;

    if (datah_ >= 0) {
        fprintf (stderr, "Global data handle not available.\n");
        return -1;
    }

    /* open bitstream */

    datah_ = bitio_i_open (msg->sec[4], (size_t) msg->secl[4]);

    if (datah_ == -1) {
        bitio_i_close (datah_);
        return -1;
    }

    /* skip trailing 4 octets (32 bits) */

    for (i = 0; i < 32; i ++) bitio_i_input (datah_, &l, 1);

    return datah_;
}
/*===========================================================================*/
/** \ingroup cbinutl
    \brief Closes bitstream for section 4 and adds data to BUFR message

    This function closes the data section bitstream
    and appends it to a BUFR message, also stores the length in the
    BUFR message.

    \param[in,out] msg BUFR message where the data has to be stored

    \see bufr_open_datasect_w, bufr_parse_in
*/

/* write length of section 4 and close bitstream */

void bufr_close_datasect_w(bufr_t* msg) {

    int n;
    size_t st;

    if (datah_ == -1 || msg == (bufr_t*) NULL) return;

    /* get current length */

    n = (int) bitio_o_get_size (datah_);

    /* number of bytes must be an even number */

	if (n % 2 != 0) bitio_o_append (datah_, 0L, 8);

    /* write length of section to beginning */
    
    n = (int) bitio_o_get_size (datah_);
    bitio_o_outp (datah_, (long) n, 24, 0L);

    /* close bitstream and return pointer */

    msg->sec[4] = (char *) bitio_o_close (datah_, &st);
    msg->secl[4] = (int) st;
    datah_ = -1;
}

/*===========================================================================*/
/** \ingroup cboututl
    \brief Closes bitstream for section 4

    This function closes the data section bitstream.

    \see bufr_open_datasect_r, bufr_parse_out
*/

void bufr_close_datasect_r () {

    if (datah_ == -1) return;
    bitio_i_close (datah_);
    datah_ = -1;
}
/*===========================================================================*/
/** \ingroup cbin
    \brief Get one value from global array of values.

    This functions gets the next value from the global array of values.

    \param[out] val The received value
    \param[in] ind    Index to the global array 
                      \ref des[] holding the description of
                      known data-descriptors.

    \return 1 on success, 0 on error.

    \see bufr_open_val_array, bufr_close_val_array
*/

int bufr_val_from_global (varfl *val, int ind) {

    assert (val != (varfl*) NULL);
    assert (vals_ != NULL);
    assert (vals_->vals != NULL);

    /* No input for special descriptors and sequence descs*/

    if (ind == _desc_special || des[ind]->id == SEQDESC) return 1;

    *val = *(vals_->vals + vals_->vali++);
    return 1;

}


/*===========================================================================*/
/** \ingroup cbout
    \brief Write one value to global array of values.

    This functions writes one value to the global array of values.

    \param[in] val    The value to store
    \param[in] ind    Index to the global array 
                      \ref des[] holding the description of
                      known data-descriptors.

    \return 1 on success, 0 on error.

    \see bufr_open_val_array, bufr_close_val_array
*/

int bufr_val_to_global (varfl val, int ind) {

    assert (vals_ != (bufrval_t*) NULL);

    /* No output for special descriptors and sequence descs*/

    if (ind == _desc_special || des[ind]->id == SEQDESC) return 1;

    return bufr_val_to_array (&(vals_->vals), val, &(vals_->nvals));
}


/*===========================================================================*/
/** \ingroup cbinutl
    \brief Opens global array of values for read/write

    This function opens the global array of values for use by 
    \ref bufr_val_from_global and \ref bufr_val_to_global and
    returns its pointer.

    \return Pointer to the array of values or NULL on error.

    \see bufr_close_val_array, bufr_val_to_global, #
    bufr_val_from_global
*/

bufrval_t* bufr_open_val_array () {


    if (vals_ != (bufrval_t*)  NULL) {
        fprintf (stderr, "Value array not empty!\n");
        return (bufrval_t*) NULL;
    }
    vals_ = malloc (sizeof (bufrval_t));

    if (vals_ == (bufrval_t*)  NULL) {
        fprintf (stderr, "Error allocating memory for Value array!\n");
        return (bufrval_t*) NULL;
    }
    memset (vals_, 0, sizeof (bufrval_t));
    return vals_;
}
/*===========================================================================*/
/** \ingroup cbinutl
    \brief Closes global array of values and frees all memory

    This function closes the global array of values used by 
    \ref bufr_val_from_global and \ref bufr_val_to_global and
    frees all allocated memory.
    
    \see bufr_open_val_array, bufr_val_to_global, bufr_val_from_global
*/

void bufr_close_val_array () {

    if (vals_ == (bufrval_t*) NULL) return;
        
    if (vals_->vals != (varfl*) NULL) {
        free (vals_->vals);
        vals_->vals = (varfl*) NULL;
    }
    free (vals_);
    vals_ = (bufrval_t*) NULL;
}

/*===========================================================================*/
/** \ingroup cbout
    \brief Reads a single value from the data stream.

    This function outputs one data-value to the data stream which was 
    opened using \ref bufr_open_datasect_r.

    \param[out] val   Data-value read.
    \param[in] ind    Index to the global array 
                      \ref des[] holding the description of
                      known data-descriptors.

    \return 1 on success, 0 on a fault.

    \see bufr_open_datasect_r, bufr_close_datasect_r, 
         bufr_val_to_datasect

*/

static int bufr_val_from_datasect (varfl *val, int ind)


{
    int data_width;
    int scale, no_change = 0, ccitt;
    unsigned long l, mv;

    assert (datah_ >= 0);

    /* No input for special descriptors and sequence descs*/

    if (ind == _desc_special || des[ind]->id == SEQDESC) return 1;

    /* No data width or scale change for 0 31 y, code tables, flag tables
       and ascii data */

    if (_bufr_edition < 3) {
        
        no_change = (des[ind]->el->d.f == 0 && des[ind]->el->d.x == 31);
    }
    else {
        ccitt = (strcmp (des[ind]->el->unit, "CCITT IA5") == 0 || 
                 ind == ccitt_special);
        no_change = ((des[ind]->el->d.f == 0 && des[ind]->el->d.x == 31) ||
                        ccitt || desc_is_codetable(ind) || desc_is_flagtable(ind) ||
                        ind == add_f_special || ind == cf_special);
    }

    if (no_change) {
        data_width = des[ind]->el->dw;
        scale = des[ind]->el->scale;
    }
    else {
        data_width = des[ind]->el->dw + dw - 128;
        scale = des[ind]->el->scale + sc - 128;
    }
    
    if (!bitio_i_input (datah_, &l, data_width)) {
      fprintf (stderr, "Error reading data from bitstream !\n");
      return 0;
    }
  
    /* Check for a missing value. Missval for operator qualifiers is not 
       possible */
    /* no missval for pixel values in bitmaps */
  
    mv = (1UL << data_width) - 1;

    if (l == mv && des[ind]->el->d.x != 31 && ! _opera_mode) /*
        !(des[ind]->el->d.x == 30 && des[ind]->el->d.y <= 4 && _opera_mode) &&
        !(des[ind]->el->d.x == 13 && des[ind]->el->d.y == 11 && _opera_mode) &&
        !(des[ind]->el->d.x == 21 && des[ind]->el->d.y == 14 && _opera_mode)) */ {
            *val = MISSVAL;
    }
    else if (ind == cf_special) 
    {
        *val = l & ((1UL << (data_width - 1)) - 1);
        if (l & (1UL << (data_width - 1)))
            *val = -*val;
    }
    else {
        *val = ((varfl) l + des[ind]->el->refval) / 
            pow (10.0, (varfl) (scale));
    }
    return 1;
}

/*===========================================================================*/
/* local functions */
/*===========================================================================*/



/*===========================================================================*/
/** This function reads from a bufr-message the length of data- and
   data-descriptor-section. Therefore the buffer is opened as a bitstream
   and data is read.

   \param[in] buf   Memory-area containing the BUFR-message.
   \param[in] len   Number of bytes of the complete BUFR-message 
   determined from the length ob the input-file.
   \param[out] secl Array containing the lengths of the BUFR-sections.

   \return 1 on success, 0 on a fault.
*/

static int get_lens (char* buf, long len, int* secl)

{
    int h, co, i, totlen, lens0, ed, opt;
    unsigned long l;
    long sum;

    /* The length of section 0 is constant, but get the length of the
       whole BUFR message */

    h = bitio_i_open (buf, 8);
    bitio_i_input (h, &l, 32);        /* skip that 'BUFR' */
    bitio_i_input (h, &l, 24);        /* length of whole message */
    lens0 = l;
    bitio_i_input (h, &l, 8);         /* BUFR edition */
    ed = l;
    bitio_i_close (h);

    secl[0] = 8;
    co = 8;
    sum = 8;

    /* length of section 1 */

    h = bitio_i_open (buf + co, 20);
    if (h == -1) return 0;
    bitio_i_input (h, &l, 24);
    secl[1] = (int) l;
    co += secl[1];
    bitio_i_input (h, &l, 32);
    if (ed >= 4)
        bitio_i_input (h, &l, 16);
    bitio_i_input (h, &l, 1);
    opt = l;
    bitio_i_close (h);
    sum += secl[1];
    if (sum > len) goto err;

    /* section 2 is optional */

    secl[2] = 0;
    if (opt)
    {
        h = bitio_i_open (buf + co, 20);
        if (h == -1) return 0;
        bitio_i_input (h, &l, 24);
        secl[2] = (int) l;
        bitio_i_close (h);
        co += secl[2];
        sum += l;
        if (sum > len) goto err;
    }

    /* length of section 3 */

    h = bitio_i_open (buf + co, 20);
    if (h == -1) return 0;
    bitio_i_input (h, &l, 24);
    secl[3] = (int) l;
    co += secl[3];
    bitio_i_close (h);
    sum += l;
    if (sum > len) goto err;

    /* length of section 4 */

    h = bitio_i_open (buf + co, 20);
    if (h == -1) return 0;
    bitio_i_input (h, &l, 24);
    secl[4] = (int) l;
    co += secl[4];
    bitio_i_close (h);
    sum += l;
    if (sum > len) goto err;

    /* length of section 5 is constant */

    secl[5] = 4;
    sum += 4;
    if (sum > len) goto err;

    /* Check the total length of the message against the sum of the lengths 
       of the sections. */

    totlen = 0;
    for (i = 0; i < 6; i ++) {
#ifdef VERBOSE
        fprintf (stderr, "section %d length = %d\n", i, secl[i]);
#endif
        totlen += secl[i];
    }
    if (totlen != lens0) {
        fprintf (stderr, 
           "WARNING: Total length of message doesn't match with the lengths\n"
                     "of the individual sections !\n");
    }

    return 1;

    /* Lengths of BUFR-sections not correct */

 err:
    fprintf (stderr, "Lengths of BUFR-sections > size of input-file !\n");
    return 0;
}

/* end of file */

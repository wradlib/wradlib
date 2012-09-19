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

FILE:          APISAMPLE_FLOAT.C
IDENT:         $Id: apisample_float.c,v 1.1 2009/10/08 08:30:38 fuxi Exp $ 

AUTHOR:        Juergen Fuchsberger
               Institute of Broadband Communication, 
               Technical University Graz, Austria

VERSION NUMBER:3.0

DATE CREATED:  4-OCT-2009

STATUS:        DEVELOPMENT FINISHED

AMENDMENT RECORD:

$Log: apisample_float.c,v $
Revision 1.1  2009/10/08 08:30:38  fuxi
Initial revision



--------------------------------------------------------------------------- */

/** \file apisample_float.c
    \brief Sample application for encoding and decoding BUFR using OPERA
           BUFR software as a library.

    This sample application uses the OPERA BUFR software api for encoding and
    decoding a sample radar image to/from a BUFR message.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "bufrlib.h"
#include "apisample_float.h"
#include "bufr_io.h"

/*===========================================================================*/
/* internal function definitons                                              */
/*===========================================================================*/

static void create_source_msg (dd* descs, int* nd, varfl** vals, 
                               radar_data_t* d);
static int our_callback (varfl val, int ind);
static void create_sample_data (radar_data_t* d);

/*===========================================================================*/
/* internal data                                                             */
/*===========================================================================*/

radar_data_t our_data; /* sturcture holding our decoded data */
char *version = "apisample_float V3.1, 5-Oct-2007\n";

/*===========================================================================*/

/** \ingroup samples
    \brief Sample for encoding a BUFR message.

    This function encodes sample data to a BUFR message and saves the
    results to a file apisample.bfr, also returns the encoded message.

    \param[in]  src_data Our source data.
    \param[out] bufr_msg Our encoded BUFR message.

    \see bufr_decoding_sample

*/

void bufr_encoding_sample (radar_data_t* src_data, bufr_t* bufr_msg) {

    sect_1_t s1;          /* structure holding information from section 1 */
    dd descs[MAX_DESCS];  /* array of data descriptors, must be large enough
                             to hold all required descriptors */
    int nd = 0;           /* current number of descriptors in descs */
    varfl* vals = NULL;   /* array of data values */
    int ok;

    long year, mon, day, hour, min;

    memset (&s1, 0, sizeof (sect_1_t));
    
    /* first let's create our source message */

    create_source_msg (descs, &nd, &vals, src_data);

    /* Prepare data for section 1 */

    s1.year = 999;
    s1.mon  = 999;
    s1.day = 999;
    s1.hour = 999;
    s1.min  = 999;
    s1.mtab = 0;                      /* master table used */
    s1.subcent = 255;                 /* originating subcenter */
    s1.gencent = 255;                 /* originating center */
    s1.updsequ = 0;                   /* original BUFR message */
    s1.opsec = 0;                     /* no optional section */
    s1.dcat = 6;                      /* message type */
    s1.dcatst = 0;                    /* message subtype */
    s1.vmtab = 13;                    /* version number of master table used */
    s1.vltab = 6;                     /* version number of local table used */

    /* read supported data descriptors from tables */

    ok = (read_tables (NULL, s1.vmtab, s1.vltab, s1.subcent, s1.gencent) >= 0);

    /* encode our data to a data-descriptor- and data-section */

    if (ok) ok = bufr_encode_sections34 (descs, nd, vals, bufr_msg);

    /* setup date and time if necessary */

    if (ok && s1.year == 999) {
        bufr_get_date_time (&year, &mon, &day, &hour, &min);
        s1.year = (int) year;
        s1.mon = (int) mon;
        s1.day = (int) day;
        s1.hour = (int) hour;
        s1.min = (int) min;
        s1.sec = 0;
    }

    /* encode section 0, 1, 2, 5 */

    if (ok) ok = bufr_encode_sections0125 (&s1, bufr_msg);

    /* Save coded data */

    if (ok) ok = bufr_write_file (bufr_msg, "apisample.bfr");

    if (vals != NULL)
        free (vals);
    free_descs ();

    if (!ok) exit (EXIT_FAILURE);
}

/*===========================================================================*/
/** \ingroup samples
    \brief Sample for decoding a BUFR message.

    This function decodes a BUFR message and stores the values in
    our sample radar data structure. Also saves the result to a file.

    \param[in] msg Our encoded BUFR message.
    \param[out] data Our source data.

    \see bufr_encoding_sample
*/

void bufr_decoding_sample (bufr_t* msg, radar_data_t* data) {

    sect_1_t s1;
    int ok, desch, ndescs, subsets;
    dd* dds = NULL;

    /* initialize variables */

    memset (&s1, 0, sizeof (sect_1_t));

    /* Here we could also read our BUFR message from a file */
    /* bufr_read_file (msg, buffile); */

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

    while (ok && subsets--) 
        ok = bufr_parse_out (dds, 0, ndescs - 1, our_callback, 1);

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
}

/*===========================================================================*/
/* 
Sample for encoding and decoding a BUFR message 

*/

int main (int argc, char* argv[]) {

    bufr_t bufr_msg ;   /* structure holding encoded bufr message */
#ifdef VERBOSE
    int i;
    FILE* fp;
#endif

    /* initialize variables */

    memset (&bufr_msg, 0, sizeof (bufr_t));
    memset (&our_data, 0, sizeof (radar_data_t));

    /* check command line parameters */

    while (argc > 1 && *argv[1] == '-')
    {
        if (*(argv[1] + 1) == 'v')
            fprintf (stderr, "%s", version);
        argc--; argv++;
    }

    /* sample for encoding to BUFR */

    create_sample_data (&our_data);
    bufr_encoding_sample (&our_data, &bufr_msg);

    /* write image from memory for comparison */

#ifdef VERBOSE
    fp = fopen ("apisample_source.asc", "wb");
    if (fp == NULL) {
        fprintf (stderr, "Could not open file %s!\n", "apisample_source.asc");
    }
    for (i = 0; i < our_data.img.nrows * our_data.img.ncols; i++) {
            fprintf (fp, "%.2f ", our_data.img.data_float[i]);
            if ((i + 1) % 10 == 0)
                fprintf (fp, "\n");
    }
    fprintf (fp, "\n");
    fclose(fp);
#endif

    /* free image */

    if (our_data.img.data != (unsigned short*) NULL)
        free (our_data.img.data);
    if (our_data.img.data_float != (float*) NULL)
        free (our_data.img.data_float);
    memset (&our_data, 0, sizeof (radar_data_t));

    /* sample for decoding from BUFR */

    bufr_decoding_sample (&bufr_msg, &our_data);
    bufr_free_data (&bufr_msg);

    /* write images from memory for comparison */

#ifdef VERBOSE
    fp = fopen ("apisample_float.asc", "wb");
    if (fp == NULL) {
        fprintf (stderr, "Could not open file %s!\n", "apisample_float.asc");
    }
    for (i = 0; i < our_data.img.nrows * our_data.img.ncols; i++) {
            fprintf (fp, "%.2f ", our_data.img.data_float[i]);
            if ((i + 1) % 10 == 0)
                fprintf (fp, "\n");
    }
    fprintf (fp, "\n");
    fclose(fp);
    fp = fopen ("apisample.asc", "wb");
    if (fp == NULL) {
        fprintf (stderr, "Could not open file %s!\n", "apisample.asc");
    }
    for (i = 0; i < our_data.img.nrows * our_data.img.ncols; i++) {
            fprintf (fp, "%d ", our_data.img.data[i]);
            if ((i + 1) % 10 == 0)
                fprintf (fp, "\n");
    }
    fprintf (fp, "\n");
    fclose(fp);
#endif

    /* free image */

    if (our_data.img.data != (unsigned short*) NULL)
        free (our_data.img.data);
    if (our_data.img.data_float != (float*) NULL)
        free (our_data.img.data_float);

    exit (EXIT_SUCCESS);
}


/*===========================================================================*/
#define fill_desc(ff,xx,yy) {\
        dd.f=ff; dd.x=xx; dd.y=yy; \
        bufr_desc_to_array (descs, dd, nd);}
#define fill_v(val) bufr_val_to_array (vals, val, &nv);

/**
   create our source BUFR message according to the OPERA BUFR guidelines 
*/
static void create_source_msg (dd* descs, int* nd, varfl** vals, 
                               radar_data_t* d) {

    dd dd;
    int nv = 0;

    fill_desc(3,1,1);           /* WMO block and station number */
    fill_v(d->wmoblock);
    fill_v(d->wmostat);

    fill_desc(3,1,192);         /* Meta information about the product */
    fill_v(d->meta.year);       /* Date */
    fill_v(d->meta.month);
    fill_v(d->meta.day);
    fill_v(d->meta.hour);       /* Time */
    fill_v(d->meta.min);
    fill_v(d->img.nw.lat);      /* Lat. / lon. of NW corner */
    fill_v(d->img.nw.lon);
    fill_v(d->img.ne.lat);      /* Lat. / lon. of NE corner */
    fill_v(d->img.ne.lon);
    fill_v(d->img.se.lat);      /* Lat. / lon. of SE corner */
    fill_v(d->img.se.lon);
    fill_v(d->img.sw.lat);      /* Lat. / lon. of SW corner */
    fill_v(d->img.sw.lon);
    fill_v(d->proj.type);             /* Projection type */
    fill_v(d->meta.radar.lat);        /* Latitude of radar */
    fill_v(d->meta.radar.lon);        /* Longitude of radar */
    fill_v(d->img.psizex);            /* Pixel size along x coordinate */
    fill_v(d->img.psizey);            /* Pixel size along y coordinate */
    fill_v(d->img.nrows);             /* Number of pixels per row */
    fill_v(d->img.ncols);             /* Number of pixels per column */

    fill_desc(3,1,22);          /* Latitude, longitude and height of station */
    fill_v(d->meta.radar.lat);
    fill_v(d->meta.radar.lon);
    fill_v(d->meta.radar_height);

                                /* Projection information (this will be 
                                   a sequence descriptor when using tables 6 */
    fill_desc(0,29,199);        /* Semi-major axis or rotation ellipsoid */
    fill_v(d->proj.majax);
    fill_desc(0,29,200);        /* Semi-minor axis or rotation ellipsoid */
    fill_v(d->proj.minax);
    fill_desc(0,29,193);        /* Longitude Origin */
    fill_v(d->proj.orig.lon);
    fill_desc(0,29,194);        /* Latitude Origin */
    fill_v(d->proj.orig.lat);
    fill_desc(0,29,195);        /* False Easting */
    fill_v(d->proj.xoff);
    fill_desc(0,29,196);        /* False Northing */
    fill_v(d->proj.yoff);
    fill_desc(0,29,197);        /* 1st Standard Parallel */
    fill_v(d->proj.stdpar1);
    fill_desc(0,29,198);        /* 2nd Standard Parallel */
    fill_v(d->proj.stdpar2);

    fill_desc(0,30,31);         /* Image type */
    fill_v(d->img.type);

    fill_desc(0,29,2);          /* Co-ordinate grid */
    fill_v(d->img.grid);

    fill_desc(0,33,3);          /* Quality information */
    fill_v(d->img.qual);

    fill_desc(3,21,200);        /* compressed rain accumulation */

    /* run length encode our bitmap */

    rlenc_from_mem_float (d->img.data_float, d->img.nrows, d->img.ncols, 
            vals, &nv);
}

/*===========================================================================*/

/** Our callback for storing the values in our radar_data_t structure 
    and for run-length decoding the radar image 
*/

static int our_callback (varfl val, int ind) {

    radar_data_t* b = &our_data;   /* our global data structure */
    bufrval_t* v;                  /* array of data values */
    varfl* vv;
    int i = 0, nv, nr, nc;
    dd* d;

    /* do nothing if data modifictaon descriptor or replication descriptor */

    if (ind == _desc_special) return 1;

    /* sequence descriptor */

    if (des[ind]->id == SEQDESC) {

        /* get descriptor */

        d = &(des[ind]->seq->d);

        /* open array for values */

        v = bufr_open_val_array ();
        if (v == (bufrval_t*) NULL) return 0;

        /* WMO block and station number */

        if  (bufr_check_fxy (d, 3,1,1)) {   

            /* decode sequence to global array */

            bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                            bufr_val_to_global, 0);

            /* get our data from the array */

            b->wmoblock = (int) v->vals[i++];
            b->wmostat = (int) v->vals[i];

        }
        /* Meta information */

        else if (bufr_check_fxy (d, 3,1,192)) { 

            bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                            bufr_val_to_global, 0);
            vv = v->vals;
            i = 0;
            b->meta.year = (int) vv[i++];       /* Date */
            b->meta.month = (int) vv[i++];
            b->meta.day = (int) vv[i++];
            b->meta.hour = (int) vv[i++];       /* Time */
            b->meta.min = (int) vv[i++];
            b->img.nw.lat = vv[i++];      /* Lat. / lon. of NW corner */
            b->img.nw.lon = vv[i++];
            b->img.ne.lat = vv[i++];      /* Lat. / lon. of NE corner */
            b->img.ne.lon = vv[i++];
            b->img.se.lat = vv[i++];      /* Lat. / lon. of SE corner */
            b->img.se.lon = vv[i++];
            b->img.sw.lat = vv[i++];      /* Lat. / lon. of SW corner */
            b->img.sw.lon = vv[i++];
            b->proj.type = (int) vv[i++];       /* Projection type */
            b->meta.radar.lat = vv[i++];        /* Latitude of radar */
            b->meta.radar.lon = vv[i++];        /* Longitude of radar */
            b->img.psizex = vv[i++];      /* Pixel size along x coordinate */
            b->img.psizey = vv[i++];      /* Pixel size along y coordinate */
            b->img.nrows = (int) vv[i++];     /* Number of pixels per row */
            b->img.ncols = (int) vv[i++];     /* Number of pixels per column */

        }
        /* Latitude, longitude and height of station */

        else if (bufr_check_fxy (d, 3,1,22)) { 

            bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                            bufr_val_to_global, 0);
            vv = v->vals;
            i = 0;
            b->meta.radar.lat = vv[i++];
            b->meta.radar.lon = vv[i++];
            b->meta.radar_height = vv[i];
        }
        /* Reflectivity scale */

        else if (bufr_check_fxy (d, 3,13,9)) { 
            int j;

            bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                            bufr_val_to_global, 0);
            vv = v->vals;
            i = 0;
            
            b->img.scale.vals[0] = vv[i++];
            b->img.scale.nvals = (int) vv[i++] + 1;  /* number of scale values */ 
            assert(b->img.scale.nvals < 256);
            for (j = 1; j < b->img.scale.nvals; j++) {
                b->img.scale.vals[j] = vv[i++];
            }
        }

        /* our bitmap */

        else if (bufr_check_fxy (d, 3,21,200)) {

            /* read bitmap and run length decode */

            if (!bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                                 bufr_val_to_global, 0)) {
                bufr_close_val_array ();
                return 0;
            }

            if (!rldec_to_mem (v->vals, &(b->img.data), &nv, &nr, 
                        &nc)) { 
                bufr_close_val_array ();
                fprintf (stderr, "Error during runlength-compression.\n");
                return 0;
            }
            if (!rldec_to_mem_float (v->vals, &(b->img.data_float), &nv, &nr, 
                        &nc)) { 
                bufr_close_val_array ();
                fprintf (stderr, "Error during runlength-compression.\n");
                return 0;
            }
        }
     
        else {
            fprintf (stderr,
                     "Unknown sequence descriptor %d %d %d", d->f, d->x, d->y);
        }
        /* close the global value array */

        bufr_close_val_array ();

    }

    /* element descriptor */

    else if (des[ind]->id == ELDESC) {

        d = &(des[ind]->el->d);

        if (bufr_check_fxy (d, 0,29,199))
            /* Semi-major axis or rotation ellipsoid */
            b->proj.majax = val;
        else if (bufr_check_fxy (d, 0,29,200))
            /* Semi-minor axis or rotation ellipsoid */
            b->proj.minax = val;
        else if (bufr_check_fxy (d, 0,29,193))
            /* Longitude Origin */
            b->proj.orig.lon = val;
        else if (bufr_check_fxy (d, 0,29,194))
            /* Latitude Origin */
            b->proj.orig.lat = val;
        else if (bufr_check_fxy (d, 0,29,195))
            /* False Easting */
            b->proj.xoff = (int) val;
        else if (bufr_check_fxy (d, 0,29,196))
            /* False Northing */
            b->proj.yoff = (int) val;
        else if (bufr_check_fxy (d, 0,29,197))
            /* 1st Standard Parallel */
            b->proj.stdpar1 = val;
        else if (bufr_check_fxy (d, 0,29,198))
            /* 2nd Standard Parallel */
            b->proj.stdpar2 = val;
        else if (bufr_check_fxy (d, 0,30,31))
            /* Image type */
            b->img.type = (int) val;
        else if (bufr_check_fxy (d, 0,29,2))
            /* Co-ordinate grid */
            b->img.grid = (int) val;
        else if (bufr_check_fxy (d, 0,33,3))
            /* Quality information */
            b->img.qual = val;
        else if (bufr_check_fxy (d, 0,21,198))
            /* dBZ Value offset */
            b->img.scale.offset = val;
        else if (bufr_check_fxy (d, 0,21,199))
            /* dBZ Value increment */
            b->img.scale.increment = val;
        else {
            fprintf (stderr,
                     "Unknown element descriptor %d %d %d", d->f, d->x, d->y);
            return 0;
        }
    }
    return 1;
}

/*===========================================================================*/
#define NROWS 200   /* Number of rows for our sample radar image */
#define NCOLS 200   /* Number of columns for our sample radar image */

static void create_sample_data (radar_data_t* d) {

    int i;

    /* create a sample radar image */
    
    d->img.data_float = (float*) calloc (NROWS * NCOLS, 
                                            sizeof (float));

    if (d->img.data_float == NULL) {
        fprintf (stderr, "Could not allocate memory for sample image!\n");
        exit (EXIT_FAILURE);
    }

    /* fill image with random data (assuming 8 bit image depth -> max
       value = 254; 255 is missing value) */

#ifdef VERBOSE
    fprintf (stderr, "RAND_MAX = %d\n", RAND_MAX);
#endif

    for (i = 0; i < NROWS * NCOLS; i++) {
        d->img.data_float[i] = (float) rand() / RAND_MAX * 254;
    }
    
    /* create our source data */

    d->wmoblock = 11;
    d->wmostat  = 164;

    d->meta.year = 2007;
    d->meta.month = 12;
    d->meta.day = 5;
    d->meta.hour = 12;
    d->meta.min = 5;
    d->meta.radar.lat = 47.06022;
    d->meta.radar.lon = 15.45772;
    d->meta.radar_height = 355;

    d->img.nw.lat = 50.4371;
    d->img.nw.lon = 8.1938;
    d->img.ne.lat = 50.3750;
    d->img.ne.lon = 19.7773;
    d->img.se.lat = 44.5910;
    d->img.se.lon = 19.1030;
    d->img.sw.lat = 44.6466;
    d->img.sw.lon = 8.7324;
    d->img.psizex = 1000;
    d->img.psizey = 1000;
    d->img.nrows = NROWS;
    d->img.ncols = NCOLS;
    d->img.type = 2;
    d->img.grid = 0;
    d->img.qual = MISSVAL;

    /* create level slicing table */

    d->img.scale.nvals = 255;

    for (i = 0; i < 255; i++) {
        d->img.scale.vals[i] = i * 0.5 - 31.0;
    }
    d->img.scale.offset = -31;
    d->img.scale.increment = 0.5;

    d->proj.type = 2;
    d->proj.majax = 6378137;
    d->proj.minax = 6356752;
    d->proj.orig.lon = 13.333333;
    d->proj.orig.lat = 47.0;
    d->proj.xoff = 458745;
    d->proj.yoff = 364548;
    d->proj.stdpar1 = 46.0;
    d->proj.stdpar2 = 49.0;
}

/** \example apisample_float.c

    This is an example for encoding and decoding a BUFR message.\n
*/

/* end of file */

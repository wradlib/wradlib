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

FILE:          APISAMPLE.H
IDENT:         $Id: apisample.h,v 1.2 2007/12/18 14:40:58 fuxi Exp $

AUTHOR:        Juergen Fuchsberger
               Institute of Broadband Communication, 
               Technical University Graz, Austria

VERSION NUMBER:3.0

DATE CREATED:  4-DEC-2007

STATUS:        DEVELOPMENT FINISHED

AMENDMENT RECORD:

$Log: apisample.h,v $
Revision 1.2  2007/12/18 14:40:58  fuxi
added licence header

Revision 1.1  2007/12/07 08:37:23  fuxi
Initial revision


--------------------------------------------------------------------------- */


/* A coordinate pair */

typedef struct point_s {
    varfl lat;      /* latitude */
    varfl lon;      /* longitude */
} point_t;


/* Meta information about image */

typedef struct meta_s {
    int year;
    int month;
    int day;
    int hour;
    int min;
    point_t radar;  /* Radar position */
    varfl radar_height;
} meta_t;

/* Level slicing table */

typedef struct scale_s {
    /* one method: */
    int nvals;       /* number of values in level slicing table */
    varfl vals[255]; /* scale values */

    /* another method: */
    varfl offset;    /* offset */
    varfl increment; /* increment */
} scale_t;

/* Radar image */

typedef struct img_s {
    int type;       /* Image type */
    varfl qual;     /* quality indicator */
    int grid;       /* Co-ordinate grid type */
    point_t nw;     /* Northwest corner of the image */
    point_t ne;     /* NE corner */
    point_t se;     /* SE corner */
    point_t sw;     /* SW corner */
    int nrows;      /* Number of pixels per row */
    int ncols;      /* Number of pixels per column */
    varfl psizex;   /* Pixel size along x coordinate */
    varfl psizey;   /* Pixel size along y coordinate */
    scale_t scale;  /* Level slicing table */
    unsigned short* data; /* Image data */
} img_t;

/* Projection information */

typedef struct proj_s {
    int type;       /* Projection type */
    varfl majax;    /* Semi-major axis or rotation ellipsoid */
    varfl minax;    /* Semi-minor axis or rotation ellipsoid */
    point_t orig;   /* Projection origin */
    int xoff;       /* False easting */
    int yoff;       /* False northing */
    varfl stdpar1;  /* 1st standard parallel */
    varfl stdpar2;  /* 2nd standard parallel */
} proj_t;


/* This is our internal data structure */

typedef struct radar_data_s {
    int wmoblock;           /* WMO block number */
    int wmostat;            /* WMO station number */
    meta_t meta;            /* Meta information about the product */
    img_t img;              /* Radar reflectivity image */
    proj_t proj;            /* Projection information */
    
} radar_data_t;

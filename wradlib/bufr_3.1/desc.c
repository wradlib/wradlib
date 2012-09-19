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

FILE:          READDESC.C
IDENT:         $Id: desc.c,v 1.18 2010/02/15 11:17:19 helmutp Exp $

AUTHOR:        Konrad Koeck
               Institute of Communication and Wave Propagation, 
               Technical University Graz, Austria

VERSION NUMBER:3.0

DATE CREATED:  18-DEC-2001

STATUS:        DEVELOPMENT FINISHED


FUNCTIONAL DESCRIPTION:
-----------------------
Function(s) for reading the descriptor-file.

AMENDMENT RECORD:

ISSUE       DATE            SCNREF      CHANGE DETAILS
-----       ----            ------      --------------
V2.0        18-DEC-2001     Koeck       Initial Issue

$Log: desc.c,v $
Revision 1.18  2010/02/15 11:17:19  helmutp
added bitmap table functions, getindex optimizations

Revision 1.17  2009/05/18 16:04:24  helmutp
new show_desc_args function which takes command line parameters

Revision 1.16  2009/04/17 13:40:36  helmutp
fixed read tables

Revision 1.15  2009/04/10 11:57:33  helmutp
local table selection now ignores subcenter if no matching
file is found

Revision 1.14  2007/12/18 14:40:13  fuxi
added licence header

Revision 1.13  2007/12/07 08:39:15  fuxi
update to version 3.0

Revision 1.12  2005/04/04 15:31:35  helmutp
update to version 2.3

Revision 1.11  2004/09/28 12:14:00  helmutp
fixed fclose and free

Revision 1.10  2003/09/04 08:07:19  helmutp
add / or \ to directory name

Revision 1.9  2003/06/11 09:33:19  helmutp
changed key calculation

Revision 1.8  2003/06/11 09:02:57  helmutp
remove duplicate entries from desc table (local table overruling)
fixed read_tab_d EOF handling, changed name of master table

Revision 1.7  2003/06/06 11:59:32  helmutp
changed read_tables to support table versions

Revision 1.6  2003/03/27 17:17:39  helmutp
update to version 2.2

Revision 1.5  2003/03/13 17:17:48  helmutp
fixed argc and indentation

Revision 1.4  2003/03/13 17:08:47  helmutp
added search key, use sorted descriptors and bsearch instead of linear search
allow tables to be specified on commndline

Revision 1.3  2003/03/11 10:30:42  helmutp
fixed memory leaks

Revision 1.2  2003/03/06 17:12:32  helmutp
update to version 2.1

Revision 1.1  2003/02/28 13:41:12  helmutp
Initial revision

--------------------------------------------------------------------------- */

/** \file desc.c
    \brief Functions for reading the descriptor tables.
    
    This file contains all functions used for reading the decriptor tables
    and utilites for managing the data descriptors.
*/

#define READDESC_MAIN

#define DESC_SORT
/*#define DESC_USE_INDEX*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include "desc.h"

/*===========================================================================*/
/* internal functions                                                        */
/*===========================================================================*/

static del *decode_tabb_line (char *line);
static char *get_val (char *line, int num);
static dseq *decode_tabd_line (char *line);
static void replace_chars (char *line, char oldc, char newc);
static int key (int typ, dd* d);
static void build_keys();
static void print_desc(int i);
static void free_one_desc(int i);
static char *str_lower(char *str);
static int read_bitmap_tab (char *fn);

/*===========================================================================*/
/* internal variables                                                        */
/*===========================================================================*/

/* \brief Stucture to define the OPERA bitmap descriptors */

typedef struct bm_desc_s 
{
    int f;
    int x;
    int y;
    int dw;  
} bm_desc_t;

#define MAX_BM 100
static bm_desc_t bm_desc[MAX_BM] = {{3,21,192,1},{3,21,193,1},{3,21,194,1},
                                   {3,21,195,1},{3,21,196,1},{3,21,197,1},
                                   {3,21,200,2},{3,21,202,2}};
static int bm_size = 0;

/*===========================================================================*/

/** \ingroup desc_g
    \brief Reads bufr tables from csv-files. 

    This function reads the descriptor tables from csv-files and
    stores the descriptors in a global array \ref des. Memory for the 
    descriptors is allocated by this function and has to be freed using
    \ref free_descs.\n 
    The filenames are
    generated by this function and have the
    form bufrtab{b|d}_Y.csv or loctab{b|d}_X_Y.csv where X is a value
    calculated of the originating center and subcenter. 
    (X = \p subcent * 256 + \p gencent)
    Y is the table version.

    \param[in] dir The directory where to search for tables, if NULL
               the function uses the current directory
    \param[in] vmtab Master table version number
    \param[in] vltab Local table version number.
    \param[in] subcent Originating/generating subcenter
    \param[in] gencent Originating/generating center

    \return Returns 0 on success or -1 on errors.

    \note The local tables are optional

*/
int read_tables (char *dir, int vmtab, int vltab, int subcent, int gencent)
{
    char fn[1024];
#if defined(_WIN32)
    char *sep = "\\";
#else
    char *sep = "/";
#endif
    
    if (dir == NULL)
        dir = "";

    if (strlen(dir) == 0 || dir[strlen(dir) -1] == '/' || 
        dir[strlen(dir) -1] == '\\')
        sep = "";

    /* read master tables, the filename is bufrtab[bd]_x.csv,
       where %d stands for the version number */

    sprintf (fn, "%s%sbufrtabb_%d.csv", dir, sep, vmtab);
    if (!read_tab_b (fn)) 
    {
        fprintf (stderr, "Error: unable to read master BUFR Table B !\n");
        return -1;
    }
    
    sprintf (fn, "%s%sbufrtabd_%d.csv", dir, sep, vmtab);
    if (!read_tab_d (fn)) 
    {
        fprintf (stderr, "Error: unable to read master BUFR Table D !\n");
        return -1;
    }

    /* read local tables, the filename is localtab[bd]_x_y.csv,
       where x is the originating center and y the version number 
       Note: center is a combination of generationg center + subcenter*256,
       if no matching file is found, the subcenter is set to zero 
       TODO: change for bufr edition 4 ? */
    
    if (vltab > 0)
    {    
        sprintf (fn, "%s%slocaltabb_%d_%d.csv", dir, sep, 
                 subcent * 256 + gencent, vltab);

        if (!read_tab_b (fn))
        {
            if (subcent != 0)
            {
                sprintf (fn, "%s%slocaltabb_%d_%d.csv", dir, sep, gencent, vltab);
                if (!read_tab_b (fn))
                    fprintf (stderr, "Warning: unable to read local BUFR Table B !\n");
            }
            else
                fprintf (stderr, "Warning: unable to read local BUFR Table B !\n");
        }

        sprintf (fn, "%s%slocaltabd_%d_%d.csv", dir, sep, 
                 subcent * 256 + gencent, vltab);

        if (!read_tab_d (fn)) 
        {
            if (subcent != 0)
            {
                sprintf (fn, "%s%slocaltabd_%d_%d.csv", dir, sep, gencent, vltab);
                if (!read_tab_d (fn)) 
                    fprintf (stderr, "Warning: unable to read local BUFR Table D !\n");
            }
            else
                fprintf (stderr, "Warning: unable to read local BUFR Table D !\n");
        }
    }

    read_bitmap_tables (dir, vltab, subcent, gencent);
    
    return 0;
}

/** \ingroup desc_g
    \brief Reads list of special bitmap descriptors from csv-files. 

    This function reads a list of descriptors, which are used
    to encode compressed bitmaps or arrays of float values. 
    Each line in the file has 4 parameters (f,x,y,w), where
    f,x,y define the bufr descriptors and w the encoding method.
    The following encoding methods are defined:
    1 - 1 byte pixel value (unsigned)
    2 - 2 byte pixel value (unsigned)
    4 - 4 byte float value
    8 - 8 byte double value
    
    The filenames are generated by this function and have the
    form bmtab_X.csv or bmtab_X_Y.csv where X is a value
    calculated of the originating center and subcenter. 
    (X = \p subcent * 256 + \p gencent) and Y is the table version.

    \param[in] dir The directory where to search for tables, if NULL
               the function uses the current directory
    \param[in] vltab Local table version number.
    \param[in] subcent Originating/generating subcenter
    \param[in] gencent Originating/generating center

    \return Returns 0 on success or -1 on errors.

    \note This table is optional
*/
int read_bitmap_tables (char *dir, int vltab, int subcent, int gencent)
{
    char fn[1024];
    char *name ="bmtab";
#if defined(_WIN32)
    char *sep = "\\";
#else
    char *sep = "/";
#endif

    if (dir == NULL)
        dir = "";

    if (strlen(dir) == 0 || dir[strlen(dir) -1] == '/' || 
        dir[strlen(dir) -1] == '\\')
        sep = "";

    sprintf (fn, "%s%s%s_%d_%d.csv", dir, sep, name, subcent * 256 + gencent, vltab);
    if (read_bitmap_tab (fn) == 0) 
        return 0;
    sprintf (fn, "%s%s%s_%d.csv", dir, sep, name, subcent * 256 + gencent);
    if (read_bitmap_tab (fn) == 0) 
        return 0;
    sprintf (fn, "%s%s%s_%d_%d.csv", dir, sep, name, gencent, vltab);
    if (read_bitmap_tab (fn) == 0) 
        return 0;
    sprintf (fn, "%s%s%s_%d.csv", dir, sep, name, gencent);
    if (read_bitmap_tab (fn) == 0) 
        return 0;
    return -1;
}

/*===========================================================================*/
/* \brief reads a file with special OPERA bitmap descriptor, returns 0
 *  if OK and -1 no file is found
 */

int read_bitmap_tab (char *fn)
{
    FILE *f;
    char line[200];

    if ((f = fopen (fn, "r")) == NULL)
        return -1;

    bm_size = 0;
    while (fgets (line, 200, f) != NULL && bm_size < MAX_BM)
/*            while (! feof (f) && bm_size < 100) */
/*                if (fscanf (f, "%d %d %d %d",  */
        if (sscanf (line, "%d%*[; ]%d%*[; ]%d%*[; ]%d\n",
                    &bm_desc[bm_size].f, &bm_desc[bm_size].x, 
                    &bm_desc[bm_size].y, &bm_desc[bm_size].dw) == 4)
            bm_size++;
    fclose (f);
    return 0;
}

/*===========================================================================*/

/* \brief checks for special OPERA bitmap descriptor and returns
   the type of bitmap encoding, or zero if no bitmap descriptor */

int check_bitmap_desc (dd *d)
{
    int i;
    
    for (i = 0; i < bm_size; i++)
        if (bm_desc[i].f == d->f && bm_desc[i].x == d->x && bm_desc[i].y == d->y)
            return bm_desc[i].dw;
    return 0;
}

/*===========================================================================*/

/** \ingroup desc_g
    \brief Prints the specified descriptor or all if no descriptor specified 

    This function prints all information on the specified descriptor 
    or all descriptors if no descriptor is specified. The command line arguments
    are: [-d tabdir] [-m vmtab] [-l vltab] [-o ocenter] [-s scenter] f x y

    \param[in] argc,argv Command line arguments.
    
*/

void show_desc_args (int argc, char **argv)
{
    int f = 999, x = -1, y = -1;
    int ocent = 255, scent = 255, vmtab = 11, vltab = 4;
    char * table_dir = 0;

    while (argc > 2 && argv[1][0] == '-')
    {
        if (argv[1][1] == 'd')
            table_dir = argv[2];
        else if (argv[1][1] == 'm')
            vmtab = atoi (argv[2]);
        else if (argv[1][1] == 'l')
            vltab = atoi (argv[2]);
        else if (argv[1][1] == 'o')
            ocent = atoi (argv[2]);
        else if (argv[1][1] == 's')
            scent = atoi (argv[2]);
        argc -= 2;
        argv += 2;
    }

    if (argc > 1) f = atoi (argv[1]);
    if (argc > 2) x = atoi (argv[2]);
    if (argc > 3) y = atoi (argv[3]);
    read_tables(table_dir, vmtab, vltab, scent, ocent);
    show_desc (f, x, y);
}

/*===========================================================================*/

/** \ingroup desc_g
    \brief Prints the specified descriptor or all if f = 999 

    This function prints all information on the specified descriptor 
    or all descriptors if f = 999 

    \param[in] f,x,y The descriptor to display.
    
*/

void show_desc (int f, int x, int y)
{
    if (f == 999)
    {
        for (f = 0; f < ndes; f++)
            print_desc (f);
    }
    else if (f >= 0 && x >= 0 && y >= 0)
    {
        int i;
        dd d;
        d.f = f;
        d.x = x;
        d.y = y;
        if ((i = get_index (SEQDESC, &d)) >= 0)
            print_desc (i);
        else if ((i = get_index (ELDESC, &d)) >= 0)
            print_desc (i);
        else
            fprintf (stderr, "Descriptor %d %d %d not found !\n", f, x, y);
    }
}

/*===========================================================================*/

/* Print the descriptor at index i */

static void print_desc(int i)
{
    if (i < 0 || i >= ndes) return;

    if (des[i]->id == ELDESC)
    {
        del *d = des[i]->el;
        printf ("%d %02d %03d %2d %2d %6.2f %s  %s   [%d, %d]\n", d->d.f, d->d.x, d->d.y,
                d->scale, d->dw, d->refval, d->unit, d->elname, i, des[i]->nr);
    }
    else
    {
        int j;
        dseq *d = des[i]->seq;
        printf ("%d %02d %03d  %d %02d %03d   [%d, %d]\n", d->d.f, d->d.x, d->d.y,
                d->del[0].f, d->del[0].x, d->del[0].y, i, des[i]->nr);
        for (j = 1; j < d->nel; j++)
            printf ("          %d %02d %03d\n", d->del[j].f, d->del[j].x, d->del[j].y);
    }
}

/*===========================================================================*/

/* Compare key calculation */

static int key (int typ, dd* d)
{
    return (typ << 16) + (d->f << 14) + (d->x << 8) + d->y;
}

/* Descriptor compare function (for qsort) */ 

#ifdef DESC_SORT
static int dcmp (const void *p1, const void *p2)
{
    desc *d1 = *(desc **) p1;
    desc *d2 = *(desc **) p2;
    
    return d1->key - d2->key;
}
#endif

#ifdef DESC_USE_INDEX
/* index array for fast descriptor lookup */

    int desc_index[1<<17];
#endif

/* Create sort keys and sort the descriptor table, 
   remove duplicate entries (local table overruling) */

static void build_keys()
{
    int i, n;
    if (ndes == 0)
        return;

    for (i = 0; i < ndes; i++)
    {
        if (des[i]->id == ELDESC)
            des[i]->key = key (des[i]->id, &des[i]->el->d);
        if (des[i]->id == SEQDESC)
            des[i]->key = key (des[i]->id, &des[i]->seq->d);
    }

#ifdef DESC_SORT

    /* sort descriptors and remove duplicates */
    /* keep decsriptor with higher serial number */
    
    qsort (des, ndes, sizeof (desc *), dcmp);

    for (i = 1, n = 0; i < ndes; i++)
    {
        if (des[n]->key == des[i]->key)
        {
    	    if (des[i]->nr > des[n]->nr)
    	    {
    	        free_one_desc (n);
                des[n] = des[i];
            }
            else
                free_one_desc (i);
        }
        else
            des[++n] = des[i];
    }
    ndes = n + 1;

#endif

#ifdef DESC_USE_INDEX

    /* build index of descriptors */
    
    for (i = 0; i < (1<<17); i++) 
        desc_index[i] = -1;
    for (i = 0; i < ndes; i++)
        desc_index[des[i]->key] = i;
   
#endif
}

/*===========================================================================*/

/** \ingroup desc_g
    \brief Returns the index for the given descriptor and typ 

    This function returns the index into the global \ref des array 
    of a descriptor given by parameters \p typ and \p descr.

    \param[in] typ The type of descriptor (\ref ELDESC or \ref SEQDESC).
    \param[in] descr The descriptor.

    \return The index of the descriptor in \ref des or -1 on error.
*/

int get_index (int typ, dd* descr)
{
#ifdef DESC_USE_INDEX

    int k = key (typ, descr);
    return desc_index[k];
   
#else

#ifdef DESC_SORT

    int i1 = 0;
    int i2 = ndes -1;
    int k = key (typ, descr);

    while (i2 >= i1)
    {
        int i = (i2 + i1) / 2;
	    int diff = des[i]->key - k;
        if (diff == 0)
            return i;
        if (diff < 0)
            i1 = i + 1;
        else
            i2 = i - 1;
    }
    return -1;

#else

  int i;
  int k = key (typ, descr);
  for (i = 0; i < ndes; i ++) 
  {
      if (des[i]->key == k)
        return i;
  }
  return -1;

#endif
#endif
}

/*===========================================================================*/
/** \ingroup desc_g
    \brief Reads bufr table d from a csv-files. 

    This function reads a sequence descriptor table (d) from  a csv-file and
    stores the descriptors in a global array \ref des. Memory for the 
    descriptors is allocated by this function and has to be freed using
    \ref free_descs.

    \param[in] fname The name of a csv-file.

    \return Returns 1 on success or 0 on error.

    \see read_tables, read_tab_b
*/

int read_tab_d (char *fname)

{
    FILE *fp;
    char line[1000], *l;
    dseq *sdesc;
    int end;

    /* Open input file */

    fp = fopen (fname, "r");
    if (fp == NULL) {
        fprintf (stderr, "unable to open '%s'\n", fname);
        return 0;
    }

/* Run through all lines and decode the ones that contain reasonable data */

    end = 0;
    do {
        if ((l = fgets (line, 1000, fp)) != NULL)
        {
            /* For some reasons the '-' is not correct stored in the csv file */
            replace_chars (l, -105, 45); 
            replace_chars (l, -106, 45);
        }

        sdesc = decode_tabd_line (l);
        if (sdesc != NULL) {
            des[ndes] = malloc (sizeof (desc));
            if (des[ndes] == NULL) {
                fprintf (stderr, "Memory allocation error.\n");
                fclose (fp);
                return 0;
            }
            des[ndes]->id = SEQDESC;
            des[ndes]->nr = ndes;
            des[ndes]->seq = sdesc;
            des[ndes]->el = NULL;
            ndes ++;
            if (ndes >= MAXDESC) {
                fprintf (stderr, "Parameter MAXDESC exceeded.\n");
                fclose (fp);
                return 0;
            }
        }
    } while (l != NULL);

    fclose (fp);

    build_keys();
    return 1;
}

/*===========================================================================*/
static dseq *decode_tabd_line (char *line)

/* Decodes a single Table D Line and returns a pointer to a dseq-structure
         holding the data that has been decoded. The memory area must be
         freed by the calling function
*/

{
/* Get the first 6 strings of the line, each of them separated by a ';'
*/

    char *sf, *sx, *sy, *dx, *dy, *df;
    int isf, isx, isy, idx, idy, idf;
    static dseq *seq = NULL;           /* Holds the current Sequence Descriptor */
    dseq *ret = NULL;
    dd *ddp;
    char tmp[1000];

    if (line == NULL)
    {
        ret = seq;
        seq = NULL;
        return ret;
    }

    strcpy (tmp, line);

    dy = get_val (line, 5);
    dx = get_val (line, 4);
    df = get_val (line, 3);
    sy = get_val (line, 2);
    sx = get_val (line, 1);
    sf = get_val (line, 0);

/* CHeck for valid values */

    if (dy == NULL ||
        dx == NULL ||
        df == NULL ||
        sy == NULL ||
        sx == NULL ||
        sf == NULL) return NULL;

    if (sscanf (sf, "%d", &isf) != 1) isf = 0;
    if (sscanf (sx, "%d", &isx) != 1) isx = 0;
    if (sscanf (sy, "%d", &isy) != 1) isy = 0;
    if (sscanf (df, "%d", &idf) != 1) idf = 0;
    if (sscanf (dx, "%d", &idx) != 1) idx = 0;
    if (sscanf (dy, "%d", &idy) != 1) idy = 0;

/* Check if there is a new seqence descriptor */

    if (isf == 3 || isx != 0 || isy != 0) {
        if (seq != NULL) {
            ret = seq;       /* This is what we return */
        }
        seq = malloc (sizeof (dseq));
        if (seq == NULL) {
            fprintf (stderr, "Memory allocation error !\n");
            return NULL;
        }
        seq->d.f = isf;
        seq->d.x = isx;
        seq->d.y = isy;
        seq->nel = 0;
        seq->del = malloc (sizeof (dd));
        if (seq->del == NULL) {
            fprintf (stderr, "Memory allocation error !\n");
            return NULL;
        }
    }

/* Get the new entry for the sequence */

    if ((idf != 0 || idx != 0 || idy != 0) && seq != NULL) {
        seq->del = realloc (seq->del, (seq->nel + 1) * sizeof (dd));
        if (seq->del == NULL) {
            fprintf (stderr, "Memory allocation error !\n");
            return NULL;
        }
        ddp = seq->del + seq->nel;
        ddp->f = idf;
        ddp->x = idx;
        ddp->y = idy;
        seq->nel += 1;
    }

    return ret;
}

/*===========================================================================*/
/** \ingroup desc_g
    \brief Reads bufr table b from a csv-files. 

    This function reads an element descriptor table (b) from a csv-file and
    stores the descriptors in a global array \ref des. Memory for the 
    descriptors is allocated by this function and has to be freed using
    \ref free_descs.

    \param[in] fname The name of the csv-file.

    \return Returns 1 on success or 0 on error.

    \see read_tables, read_tab_d
*/

int read_tab_b (char *fname)

{
    FILE *fp;
    char line[1000];
    del *descr;

    /* Open input file */

    fp = fopen (fname, "r");
    if (fp == NULL) {
        fprintf (stderr, "unable to open '%s'\n", fname);
        return 0;
    }

    /* Run through all lines and decode the ones that contain reasonable data*/

    while (fgets (line, 1000, fp) != NULL) {
        replace_chars (line, -106, 45); /* For some reasons the '-' is not correct stored in the csv file */
        replace_chars (line, -105, 45); /* For some reasons the '-' is not correct stored in the csv file */
        descr = decode_tabb_line (line);
        if (descr != NULL) {
            des[ndes] = malloc (sizeof (desc));
            if (des[ndes] == NULL) {
                fprintf (stderr, "Memory allocation error.\n");
                fclose (fp);
                return 0;
            }
            des[ndes]->id = ELDESC;
            des[ndes]->nr = ndes;
            des[ndes]->el = descr;
            des[ndes]->seq = NULL;
            ndes ++;
            if (ndes >= MAXDESC) {
                fprintf (stderr, "Parameter MAXDESC exceeded.\n");
                fclose (fp);
                return 0;
            }
        }
    }

    fclose (fp);

    /* Finally we add a dummy descriptor describing a single character 
       in a CCITT IA5 character string */

    if (ccitt_special == 0) {
        ccitt_special = MAXDESC + 1;
        descr = decode_tabb_line ("9999;9999;9999;tmp;value;0;0;8;tmp;0;3");
        if (descr != NULL) {
            des[ccitt_special] = malloc (sizeof (desc));
            if (des[ccitt_special] == NULL) {
                fprintf (stderr, "Memory allocation error.\n");
                return 0;
            }
            des[ccitt_special]->id = ELDESC;
            des[ccitt_special]->nr = ccitt_special;
            des[ccitt_special]->el = descr;
            des[ccitt_special]->seq = NULL;
        }
    }

    /* The same we need for a dummy for saving a change in the 
       reference value */

    if (cf_special == 0) {
        cf_special = MAXDESC + 2;
        descr = decode_tabb_line ("9999;9999;9998;Reference value;value;0;0;8;tmp;0;3");
        if (descr != NULL) {
            des[cf_special] = malloc (sizeof (desc));
            if (des[cf_special] == NULL) {
                fprintf (stderr, "Memory allocation error.\n");
                return 0;
            }
            des[cf_special]->id = ELDESC;
            des[cf_special]->nr = cf_special;
            des[cf_special]->el = descr;
            des[cf_special]->seq = NULL;
        }
    }

    /* dummy descriptor for the associated field */

    if (add_f_special == 0) {
        add_f_special = MAXDESC + 3;
        descr = decode_tabb_line ("0;0;0;Associated Field;value;0;0;0;tmp;0;0");
        if (descr != NULL) {
            des[add_f_special] = malloc (sizeof (desc));
            if (des[add_f_special] == NULL) {
                fprintf (stderr, "Memory allocation error.\n");
                return 0;
            }
            des[add_f_special]->id = ELDESC;
            des[add_f_special]->nr = add_f_special;
            des[add_f_special]->el = descr;
            des[add_f_special]->seq = NULL;
        }
    }

    /* dummy descriptor for no data output */
    
    if (_desc_special == 0) {
        _desc_special = MAXDESC + OPTDESC - 1;
        descr = decode_tabb_line ("0;0;0;Desc;value;0;0;0;tmp;0;0");
        if (descr != NULL) {
            des[_desc_special] = malloc (sizeof (desc));
            if (des[_desc_special] == NULL) {
                fprintf (stderr, "Memory allocation error.\n");
                return 0;
            }
            des[_desc_special]->id = ELDESC;
            des[_desc_special]->nr = _desc_special;
            des[_desc_special]->el = descr;
            des[_desc_special]->seq = NULL;
        }
    }

    build_keys();
    return 1;
}

/*===========================================================================*/
/** \ingroup desc_g
    \brief Frees all memory that has been allocated for data descriptors

    This function frees all memory that has been allocated for data descriptors

    \see read_tables, read_tab_b, read_tab_d
*/

void free_descs (void)


{
    int i;

    for (i = 0; i < ndes; i ++) {
        free_one_desc (i);
    }
    ndes = 0;

    free_one_desc (ccitt_special);
    free_one_desc (cf_special);
    free_one_desc (add_f_special);
    free_one_desc (_desc_special);
    ccitt_special = 0;
    cf_special = 0;
    add_f_special = 0;
    _desc_special = 0;
}

static void free_one_desc (int i)
{
    if (i < 0 || i >= MAXDESC + OPTDESC|| des[i] == NULL) 
       return;

    if (des[i]->id == ELDESC) {
        free (des[i]->el->unit);
        free (des[i]->el->elname);
        free (des[i]->el);
    }
    else if (des[i]->id == SEQDESC) {
        free (des[i]->seq->del);
        free (des[i]->seq);
    }
    free (des[i]);
    des[i] = NULL;
}

/*===========================================================================*/
static del *decode_tabb_line (char *line)

/* Decodes a single Table B Line and returns a pointer to a del-structure
         holding the data that has been decoded. The memory area must be
         freed by the calling function
*/

{
/* Get the first 8 strings of the line, each of them separated by a ';'
*/

    char *data_width, *refval, *scale, *unit, *name, *x, *y, *f;
    del desc, *ret;
    float tmp;
    char tmpline[1000];

    memset (&desc, 0, sizeof (del));
    strcpy (tmpline, line);
 
    data_width = get_val (tmpline, 7);
    refval     = get_val (tmpline, 6);
    scale      = get_val (tmpline, 5);
    unit       = get_val (tmpline, 4);
    name       = get_val (tmpline, 3);
    y          = get_val (tmpline, 2);
    x          = get_val (tmpline, 1);
    f          = get_val (tmpline, 0);

    if (data_width == NULL ||
        refval     == NULL ||
        scale      == NULL ||
        unit       == NULL ||
        name       == NULL ||
        x          == NULL ||
        y          == NULL ||
        f          == NULL) return NULL;


/* A correct line has been found decode data from strings */

    if (sscanf (f,          "%d", &desc.d.f)   != 1) return NULL;
    if (sscanf (x,          "%d", &desc.d.x)   != 1) return NULL;
    if (sscanf (y,          "%d", &desc.d.y)   != 1) return NULL;
    if (sscanf (scale,      "%d", &desc.scale) != 1) return NULL;
    if (sscanf (data_width, "%d", &desc.dw)    != 1) return NULL;
    if (sscanf (refval,     "%f", &tmp)        != 1) return NULL;
    desc.refval = tmp;

    desc.unit = malloc (strlen (unit) + 1);
    if (desc.unit == NULL) {
        fprintf (stderr, "Memory allocation error !\n");
        return NULL;
    }
    strcpy (desc.unit, unit);

    desc.elname = malloc (strlen (name) + 1);
    if (desc.elname == NULL) {
        fprintf (stderr, "Memory allocation error !\n");
        return NULL;
    }
    strcpy (desc.elname, name);

    ret = malloc (sizeof (del));
    if (ret == NULL) {
        fprintf (stderr, "Memory allocation error !\n");
        return NULL;
    }

    memcpy (ret, &desc, sizeof (del));
    return ret;
}
/*===========================================================================*/
/** Checks if a descriptor is a flag-table.
    
    \param[in] ind Index to the global array \ref des[] holding the 
                   description of known data-descriptors.

    \return 1 if descriptor is a flag-table, 0 if not.

    \see desc_is_codetable
*/

int desc_is_flagtable (int ind) {

    char unit[20];

    strncpy (unit, des[ind]->el->unit, 20);
    unit[19] = '\0';

    str_lower (unit);

    return (strcmp (unit, "flag table") == 0 ||
            strcmp (unit, "flag-table") == 0);
}

/*===========================================================================*/
/** Checks if a descriptor is a code-table.
    
    \param[in] ind Index to the global array \ref des[] holding the 
                   description of known data-descriptors.

    \return 1 if descriptor is a code-table, 0 if not.

    \see desc_is_flagtable
*/

int desc_is_codetable (int ind) {

    char unit[20];
    
    strncpy (unit, des[ind]->el->unit, 20);
    unit[19] = '\0';

    str_lower (unit);

    return (strcmp (unit, "code table") == 0 ||
            strcmp (unit, "code-table") == 0);
}

/*===========================================================================*/
static char *get_val (char *line, int num)

/* Gets a single value (character string) from a LINE licated at position
         NUM
*/

{
    char *p;
    int i;

/* seek to the end of the desired value and set it to 0 to identify the
         end of the string. */

    p = line;
    for (i = 0; i < num + 1 && p != NULL; i ++) {
        if (i == 0) p = strchr (p, ';');
        else p = strchr (p + 1, ';');
    }
    if (p != NULL) *p = 0;

/* Now seek to the beginning of the desired value */

    p = line;
    for (i = 0; i < num && p != NULL; i ++) {
        if (i == 0) p = strchr (p, ';');
        else p = strchr (p + 1, ';');
    }
    if (p == NULL) return NULL;
    if (num != 0) p ++;
    return p;
}

/*===========================================================================*/
/** \ingroup utils
    \brief Deletes all terminating blanks in a string.

    This functions deletes all terminating blanks in a string.

    \param[in,out] buf Our string.
*/

void trim (char *buf)

{
  int i, len;

  len = strlen (buf);
  for (i = len - 1; i >= 0; i --) {
    if (*(buf + i) == ' ') *(buf + i) = 0;
    else break;
  }
}

/*===========================================================================*/
/** \ingroup desc_g
    \brief Returns the unit for a given data descriptor

    This function searches the global \ref des array and returns 
    the unit for a data descriptor.

    \param[in] d The descriptor.
    
    \return Pointer to a string containing the unit or NULL if the
            descriptor is not found in the global \ref des array.
*/

char *get_unit (dd* d)

{
  int i;

  for (i = 0; i < ndes; i ++) {
    if (des[i]->id == ELDESC &&
        memcmp (d, &(des[i]->el->d), sizeof (dd)) == 0)
        return des[i]->el->unit;
  }
  return NULL;
}

/*===========================================================================*/
static void replace_chars (char *line, char oldc, char newc)

/* replaces one character of a string against another.
*/

{
    for (;*line != 0; line ++) {
         if (*line == oldc) 
             *line = newc;
    }
}

/*===========================================================================*/
/**
   Converts a given string to lower case characters.
    
   \param[in,out] *str:         pointer to the string
   \return The pointer to the start of the string
*/

static char *str_lower(char *str)
{
    register char *p = str;
    while (*p != '\0') {
        *p = (char) tolower((int) *p);
        p++;
    }
    return str;
}

/*===========================================================================*/

/* end of file */


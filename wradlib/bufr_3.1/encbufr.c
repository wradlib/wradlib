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

FILE:          ENCBUFR.C
IDENT:         $Id: encbufr.c,v 1.15 2010/04/13 12:38:45 helmutp Exp $

AUTHOR:        Juergen Fuchsberger, Helmut Paulitsch, Konrad Koeck
               Institute of Communication and Wave Propagation, 
               Technical University Graz, Austria

VERSION NUMBER:3.1

DATE CREATED:  18-DEC-2001

STATUS:        DEVELOPMENT FINISHED

AMENDMENT RECORD:

$Log: encbufr.c,v $
Revision 1.15  2010/04/13 12:38:45  helmutp
update usage string and version number

Revision 1.14  2009/05/18 16:07:04  helmutp
added cmd line options to use edition 4 and show descriptor info,
update version number

Revision 1.13  2009/04/10 11:16:49  helmutp
added option to change name of section1 file

Revision 1.12  2007/12/18 14:40:13  fuxi
added licence header

Revision 1.11  2007/12/07 08:16:41  fuxi
update to version 3.0

Revision 1.10  2005/06/01 09:47:54  helmutp
update version, use local tables V4

Revision 1.9  2005/04/06 09:08:10  helmutp
local tables v5

Revision 1.8  2005/04/04 15:41:47  helmutp
update to version 2.3
use subcenter and generating center

Revision 1.7  2003/06/11 09:13:13  helmutp
added version string

Revision 1.6  2003/06/06 11:52:34  helmutp
select descriptor tables with different versions

Revision 1.5  2003/03/27 17:17:39  helmutp
update to version 2.2

Revision 1.4  2003/03/13 17:22:24  helmutp
allow tables to be specified on command line

Revision 1.3  2003/03/11 10:30:42  helmutp
fixed memory leaks

Revision 1.2  2003/03/06 17:12:32  helmutp
update to version 2.1

Revision 1.1  2003/02/28 13:41:12  helmutp
Initial revision

--------------------------------------------------------------------------- */

/** \file encbufr.c
    \brief Reads source-data from a textfile and codes it into a BUFR-file.

    This function reads source-data from a textfile and codes is 
    into a BUFR-file. Bitmaps are read from a seperate file.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "bufrlib.h"
#include "bufr_io.h"

#define BUFR_EDITION 3
/*===========================================================================*/
/* internal functions                                                        */
/*===========================================================================*/

static void free_all(bufr_t* bufr);

/*===========================================================================*/
/* internal data                                                             */
/*===========================================================================*/

char *usage = 
"Usage: encbufr [-v] [-d tabdir] [-s1 sect1] [-e4] input_file output_file\n"
"       decbufr -show [-m mtab] [-l ltab] [-o ocent] [-s subcent] [f x y]\n";
char *version = "encbufr V3.1, 12-April-2010\n";

/*===========================================================================*/
int main (int argc, char* argv[])
{
    sect_1_t s1;        /* struct holding information from section 1 */
    bufr_t bufr_dest;   /* struct holding encoded bufr message */
    char srcfile[200], buffile[200]; /* filenames for source and destination */
    char *table_dir = NULL;
    char *sect1_file = "section.1";
    long year, mon, day, hour, min;

    /* initialize variables */

    memset (&bufr_dest, 0, sizeof (bufr_t));
    memset (&s1, 0, sizeof (sect_1_t));
    
    /* set bufr edition */
    
     _bufr_edition = BUFR_EDITION;

    /* check command line parameters */

    while (argc > 1 && *argv[1] == '-')
    {
        if (*(argv[1] + 1) == 'v')
            fprintf (stderr, "%s", version);
        else if (*(argv[1] + 1) == 'd')
        {
            if (argc < 2)
            {
                fprintf (stderr, "Missing parameter for -d\n\n%s", usage);
                exit (EXIT_FAILURE);
            }
            table_dir = argv[2];
            argc--;
            argv++;
        }
        else if (strcmp (argv[1], "-s1") == 0)
        {
            sect1_file = argv[2];
            argc--; 
            argv++;
        }
        else if (strcmp (argv[1], "-show") == 0)
        {
            show_desc_args (argc - 1, argv + 1);
            exit (0);
        }
        else if (strcmp (argv[1], "-e4") == 0)
            _bufr_edition = 4;
        else
        {
            fprintf (stderr, "Invalid parameter %s\n\n%s", argv[1], usage);
            exit (EXIT_FAILURE);
        }
        argc--;
        argv++;
    }

    /* Get input- and output-filenames from the command-line */

    if (argc < 3)
    {
        fprintf (stderr, "%s", usage);
        exit (EXIT_FAILURE);
    }
    strcpy (srcfile, argv[1]);
    strcpy (buffile, argv[2]);

    /* Read section 1 from ASCII input file */

    bufr_sect_1_from_file (&s1, sect1_file);

    /* read supported data descriptors from tables */

    if (read_tables (table_dir, s1.vmtab, s1.vltab, s1.subcent, 
                     s1.gencent) < 0) {
        free_all (&bufr_dest);
        exit (EXIT_FAILURE);
    }

    /* code data in the source-file to a data-descriptor- and data-section */

    if (!bufr_data_from_file (srcfile, &bufr_dest)) {
        free_all (&bufr_dest);
        exit (EXIT_FAILURE);
    }
    
    /* setup date and time if necessary */

    if (s1.year == 999) {
        bufr_get_date_time (&year, &mon, &day, &hour, &min);
        s1.year = (int) year;
        s1.mon = (int) mon;
        s1.day = (int) day;
        s1.hour = (int) hour;
        s1.min = (int) min;
        s1.sec = 0;
    }

    /* encode section 0, 1, 2, 5 */

    if (!bufr_encode_sections0125 (&s1, &bufr_dest)) {
        fprintf (stderr, "Unable to create section 0, 1, 2 and/or 5\n");
        free_all (&bufr_dest);
        exit (EXIT_FAILURE);
    }

    /* Save coded data */

    if (!bufr_write_file (&bufr_dest, buffile)) {
        free_all (&bufr_dest);
        exit (EXIT_FAILURE);
    }

    /* Free data */

    free_all (&bufr_dest);
    exit (EXIT_SUCCESS);
}

/*===========================================================================*/

static void free_all(bufr_t* bufr) {

    free_descs();
    bufr_free_data(bufr);
}


/* end of file */

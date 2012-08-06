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

FILE:          DECBUFR.C
IDENT:         $Id: decbufr.c,v 1.16 2010/04/13 12:39:06 helmutp Exp $

AUTHORS:       Juergen Fuchsberger, Helmut Paulitsch, Konrad Koeck
               Institute of Communication and Wave Propagation, 
               Technical University Graz, Austria

VERSION NUMBER:3.1

DATE CREATED:  18-DEC-2001

STATUS:        DEVELOPMENT FINISHED

AMENDMENT RECORD:

$Log: decbufr.c,v $
Revision 1.16  2010/04/13 12:39:06  helmutp
update version number

Revision 1.15  2010/02/15 11:18:51  helmutp
update usage/version string, removed opera mode

Revision 1.14  2009/05/18 16:08:12  helmutp
moved show_desciptor code to desc.c
update version string

Revision 1.13  2009/04/10 11:14:00  helmutp
added options to show info on a descriptor,
to change the name of the section1 file
OPERA originating center 247

Revision 1.12  2007/12/18 14:40:13  fuxi
added licence header

Revision 1.11  2007/12/07 08:17:14  fuxi
update to version 3.0

Revision 1.10  2005/06/01 09:47:06  helmutp
update version

Revision 1.9  2005/04/04 15:43:06  helmutp
update to version 2.3
use subcenter and generating center

Revision 1.8  2003/06/11 09:13:26  helmutp
added version string

Revision 1.7  2003/06/06 11:57:26  helmutp
support for descriptor tables with different versions

Revision 1.6  2003/03/27 17:17:39  helmutp
update to version 2.2

Revision 1.5  2003/03/24 15:42:44  kon
Added support of multiple CAPPIs

Revision 1.4  2003/03/13 17:22:45  helmutp
allow tables to be specified on command line

Revision 1.3  2003/03/11 10:30:42  helmutp
fixed memory leaks

Revision 1.2  2003/03/06 17:12:32  helmutp
update to version 2.1

Revision 1.1  2003/02/28 13:41:12  helmutp
Initial revision

--------------------------------------------------------------------------- */

/** \file decbufr.c
    \brief Reads a BUFR-file, decodes it and stores decoded data in a
    text-file.

    This function reads a BUFR-file, decodes it and stores decoded data in a
    text-file. Decoded bitmaps are stored in a seperate file.

*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "bufrlib.h"
#include "bufr_io.h"

/*===========================================================================*/
/* internal functions                                                        */
/*===========================================================================*/

#define RIOUTFILE  "img.dec"    /* Name of file for uncompressed radar image */

char *usage = 
"Usage: decbufr [-v] [-d tabdir] [-s1 sect1] input_file output_file [image_file]\n"
"       decbufr -show [-m mtab] [-l ltab] [-o ocent] [-s subcent] [f x y]\n";
char *version = "decbufr V3.1, 12-April-2010\n";

/*===========================================================================*/

int main (int argc, char** argv)

{
    char destfile[200], buffile[200];
    char *table_dir = NULL;
    char *sect1_file = "section.1.out";
    char imgfile[200];  /* filename of uncompressed image */
    sect_1_t s1;
    bufr_t bufr_msg;    /* structure holding encoded bufr message */

    /* initialize variables */

    memset (&bufr_msg, 0, sizeof (bufr_t));
    memset (&s1, 0, sizeof (sect_1_t));

    /* check command line parameter */

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
    strcpy (buffile, argv[1]);
    strcpy (destfile, argv[2]);

    if (argc > 3) 
        strcpy (imgfile, argv[3]);
    else 
        strcpy (imgfile, RIOUTFILE);

    /* read source-file. Therefore allocate memory to hold the complete
       BUFR-message */

    if (!bufr_read_file (&bufr_msg, buffile)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);
    }

    /* decode section 1 */

    if (!bufr_decode_sections01 (&s1, &bufr_msg)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);
    }

    /* Write section 1 to ASCII file */

    if (!bufr_sect_1_to_file (&s1, sect1_file)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);
    }

    /* read descriptor tables */

    if (read_tables (table_dir, s1.vmtab, s1.vltab, s1.subcent, 
                     s1.gencent) < 0) {
        bufr_free_data (&bufr_msg);
        free_descs();
        exit (EXIT_FAILURE);
    }

    /* decode data descriptor- and data-section now */

    if (!bufr_data_to_file (destfile, imgfile, &bufr_msg)) {
        fprintf (stderr, "unable to decode BUFR-message !\n");
        bufr_free_data (&bufr_msg);
        free_descs();
        exit (EXIT_FAILURE);
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
    exit (EXIT_SUCCESS);
}

/* end of file */


/*--------------------------------------------------------------------------
$Id: apisamp.c,v 1.8 2007/12/18 14:40:13 fuxi Exp $
----------------------------------------------------------------------------

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

$Log: apisamp.c,v $
Revision 1.8  2007/12/18 14:40:13  fuxi
added licence header

Revision 1.7  2007/12/10 11:22:23  fuxi
removed depricated functions

Revision 1.6  2007/12/07 08:39:46  fuxi
update to version 3.0

Revision 1.5  2005/06/01 09:49:23  helmutp
update to version 2.3

Revision 1.4  2003/09/03 15:47:08  helmutp
more compiler warnings

Revision 1.3  2003/09/03 15:42:59  helmutp
fixed originating center and compiler warnings

Revision 1.2  2003/06/06 12:32:10  helmutp
read_tables changed

Revision 1.1  2003/02/28 13:41:12  helmutp
Initial revision

--------------------------------------------------------------------------- */


#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "bufrlib.h"

/*===========================================================================*/
/* internal functions                                                        */
/*===========================================================================*/

/*===========================================================================*/
int main ()

{
    bufr_t msg;        /* Our encoded BUFR message */

    dd descr[10];      /* This array must be large enough to hold all 
                          required descriptors */
    varfl v[10];       /* This array must be huge enough to hold all 
                          corresponding data values */
    sect_1_t s1;       /* Here we store section 1 of BUFR message */

    /* Initialize basic data */

    memset (&msg, 0, sizeof(bufr_t));

    /* read supported data descriptors */
    /* parameters are the table-directory 
       (NULL to search in current dirctory), 
       version of mtab, version of ltab, orcenter */

    if (read_tables(NULL, 11, 4, 255, 255) < 0) {
        fprintf (stderr, "Error reading tables.\n");
        exit (EXIT_FAILURE);
    }

    /* Prepare the data we want to encode

    This represents the following information:

    0  1   1      6.00000   0  1   1 WMO block number
    0  1   2    260.00000   0  1   2 WMO station number
    0  2   1      1.00000   0  2   1 Type of station
    
    */
    descr[0].f = 0; descr[0].x = 1; descr[0].y = 1; v[0] = 6;
    descr[1].f = 0; descr[1].x = 1; descr[1].y = 2; v[1] = 260;
    descr[2].f = 0; descr[2].x = 2; descr[2].y = 1; v[2] = 1;

    /* Code the data (section 3 and 4) */

    if (!bufr_encode_sections34 (descr, 3, v, &msg)) {
        fprintf (stderr, "Error creating bufr message.\n");
        exit (EXIT_FAILURE);
    }

    /* Prepare data for Section 1 */

    s1.year = 2003;
    s1.mon  = 1;
    s1.day = 17;
    s1.hour = 18;
    s1.min  = 29;
    s1.mtab = 0;                      /* master table used */
    s1.subcent = 255;                 /* originating subcenter */
    s1.gencent = 255;                 /* originating center */
    s1.updsequ = 0;                   /* original BUFR message */
    s1.opsec = 1;                     /* no optional section */
    s1.dcat = 6;                      /* message type */
    s1.dcatst = 0;                    /* message subtype */
    s1.vmtab = 11;                    /* version number of master table used */
    s1.vltab = 4;                     /* version number of local table used */

    /* Setup section 0, 1, 2, 5 */

    if (!bufr_encode_sections0125 (&s1, &msg)) {
        fprintf (stderr, "Unable to create section 0, 1, 2 and/or 5\n");
        exit (EXIT_FAILURE);
    }

    /* Save coded data */

    if (!bufr_write_file (&msg, "output.buf")) {
        fprintf (stderr, "Error saving sections to file.\n");
        exit (EXIT_FAILURE);
    }

    exit (EXIT_SUCCESS);
}


/* end of file */



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

FILE:          BITIO.H
IDENT:         $Id: bitio.h,v 1.6 2007/12/18 14:40:58 fuxi Exp $

AUTHOR:        Konrad Koeck
               Institute of Communication and Wave Propagation, 
               Technical University Graz, Austria

VERSION NUMBER:3.0

DATE CREATED:  18-DEC-2001

STATUS:        DEVELOPMENT FINISHED


FUNCTIONAL DESCRIPTION:
-----------------------
Includefile for BITIO.C. More details can be found there.

AMENDMENT RECORD:

Revision 1.4  2005/04/04 14:56:09  helmutp
update to version 2.3

Revision 1.3  2003/03/27 17:17:39  helmutp
update to version 2.2

Revision 1.2  2003/03/06 17:12:32  helmutp
update to version 2.1

Revision 1.1  2003/02/28 13:41:12  helmutp
Initial revision

--------------------------------------------------------------------------- */

/** \file bitio.h
    \brief Function definitions for bitstream input and output.

    This file defines all functions for input and output to/from a bitstream.

*/

/*===========================================================================*/
/* function prototypes                                                       */
/*===========================================================================*/

#ifndef BITIO_H_INCLUDED
#define BITIO_H_INCLUDED

int bitio_i_open (void *buf, size_t size);
int bitio_i_input (int handle, unsigned long *val, int nbits);
size_t bitio_o_get_size (int handle);
void bitio_i_close (int handle);
int bitio_o_open ();
long bitio_o_append (int handle, unsigned long val, int nbits);
void bitio_o_outp (int handle, unsigned long val, int nbits, long bitpos);
void *bitio_o_close (int handle, size_t *nbytes);

#endif

/* end of file */


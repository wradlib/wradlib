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

FILE:          LIBBUFR.H
IDENT:         $Id: bufrlib.h,v 1.2 2007/12/18 14:40:58 fuxi Exp $

AUTHOR:        Juergen Fuchsberger
               Institute of Broadband Communication, 
               Technical University Graz, Austria

VERSION NUMBER:3.0

DATE CREATED:  29-NOV-2007

STATUS:        DEVELOPMENT FINISHED

AMENDMENT RECORD:

$Log: bufrlib.h,v $
Revision 1.2  2007/12/18 14:40:58  fuxi
added licence header

Revision 1.1  2007/12/07 08:41:02  fuxi
Initial revision


--------------------------------------------------------------------------- */

/** \file bufrlib.h
    \brief Includes all functions for the OPERA BUFR software library
    
    This file includes all header files used by the OPERA BUFR software
    library.
*/


#include "desc.h"
#include "bufr.h"
#include "bitio.h"
#include "rlenc.h"

/** 
    \defgroup basicin Basic functions for encoding to BUFR
    \defgroup basicout Basic functions for decoding from BUFR
    \defgroup extin   Extended functions for encoding to BUFR
    \defgroup extout  Extended functions for decoding from BUFR
    \defgroup utils_g   BUFR utility functions
    \defgroup desc_g    Functions for data descriptor management
    \defgroup rlenc_g   Functions for run length encoding
    \defgroup rldec_g   Functions for run length decoding
    \defgroup operaio Functions for encoding/decoding from/to OPERA ASCII Files
    \defgroup cbin Callback functions for encoding to BUFR
    \defgroup cbout Callback functions for decoding from BUFR
    \defgroup cbinutl Utilities for encoding callback functions
    \defgroup cboututl Utilities for decoding callback functions
    \defgroup bitio    Functions for input and output to/from a bitstream
    \defgroup deprecated_g Deprecated functions 
    \defgroup samples  API examples
*/

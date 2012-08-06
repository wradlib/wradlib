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

FILE:          BUFR_IO.H
IDENT:         $Id: bufr_io.h,v 1.2 2007/12/18 14:40:58 fuxi Exp $

AUTHOR:        Juergen Fuchsberger
               Institute of Broadband Communication, 
               Technical University Graz, Austria

VERSION NUMBER:3.0

DATE CREATED:  29-NOV-2007

STATUS:        DEVELOPMENT FINISHED

AMENDMENT RECORD:

$Log: bufr_io.h,v $
Revision 1.2  2007/12/18 14:40:58  fuxi
added licence header

Revision 1.1  2007/12/07 08:35:10  fuxi
Initial revision


--------------------------------------------------------------------------- */

/** \file bufr_io.h
    \brief Includes functions for reading/writing to/from OPERA format ASCII
           BUFR files.
    
    This file includes functions  for reading/writing to/from OPERA 
    format ASCII BUFR files.
*/

#ifndef BUFR_IO_H_INCLUDED
#define BUFR_IO_H_INCLUDED

int bufr_data_from_file(char* file, bufr_t* msg);
int bufr_data_to_file (char* file, char* imgfile, bufr_t* msg);

#endif

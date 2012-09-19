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

FILE:          BITIO.C
IDENT:         $Id: bitio.c,v 1.7 2009/04/17 13:43:41 helmutp Exp $

AUTHOR:        Konrad Koeck
               Institute of Communication and Wave Propagation, 
               Technical University Graz, Austria

VERSION NUMBER:3.0

DATE CREATED:  18-DEC-2001

STATUS:        DEVELOPMENT FINISHED

AMENDMENT RECORD:

$Log: bitio.c,v $
Revision 1.7  2009/04/17 13:43:41  helmutp
corrected wrong return value

Revision 1.6  2007/12/18 14:40:13  fuxi
added licence header

Revision 1.5  2007/12/07 08:35:46  fuxi
update to version 3.0

Revision 1.4  2005/04/04 14:58:39  helmutp
update to version 2.3

Revision 1.3  2003/03/27 17:17:39  helmutp
update to version 2.2

Revision 1.2  2003/03/06 17:12:32  helmutp
update to version 2.1

Revision 1.1  2003/02/28 13:41:12  helmutp
Initial revision

--------------------------------------------------------------------------- */

/** \file bitio.c
    \brief functions for input and output to/from a bitstream

The functions in this file can be used for input and output to/from a 
bitstream as needed for BUFR-messages. Data is stored on/read from a bitstream
as follows: For example if you wan to store a 12 bit-value VAL on a bit-stream,
consisting of a character-array C, the bits are assigned (bit 0 is the least
segnificant bit).\n
\n
VAL bit 00 -> C[0] bit 00\n
VAL bit 01 -> C[0] bit 01\n
VAL bit 02 -> C[0] bit 02\n
VAL bit 03 -> C[0] bit 03\n
VAL bit 04 -> C[0] bit 04\n
VAL bit 05 -> C[0] bit 05\n
VAL bit 06 -> C[0] bit 06\n
VAL bit 07 -> C[1] bit 07\n
VAL bit 08 -> C[1] bit 00\n
VAL bit 09 -> C[1] bit 01\n
VAL bit 10 -> C[1] bit 02\n
VAL bit 11 -> C[1] bit 03\n
\n
if you append another 2-bit value VAL1 to the stream:\n
\n
VAL bit 00 -> C[1] bit 04\n
VAL bit 01 -> C[1] bit 05\n
\n
Functions for output of data to a bit-stream are named bitio_o_*, those for 
inputing from a bitstream bitio_i_*.\n
\n
Output to a bit-stream must be as follows:\n
\n
h = \ref bitio_o_open ();       open a bitstrem, handle H is returned to 
                                identify for subsequent calls.\n
\ref bitio_o_append (h, val, nbits); Append VAL to the bitstream.\n
\ref bitio_o_close (h, nbytes);      close bitstream.\n


Input from a bit-stream must be as follows:\n
\n
h = \ref bitio_i_open ();            open a bitstream for input\n
\ref bitio_i_input ();               read a value from the bitstream\n
\ref bitio_i_close ();               close the bitstream\n
\n
More details can be found at the description of the functions. Note that the
buffer holding the bitstream is organized as an array of characters. So the
functions are independent from the computer-architecture (byte-swapping).

*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <memory.h>
#include "desc.h"
#include "bufr.h"
#include "bitio.h"

/*===========================================================================*/
/* functions for bit-io follow:                                   */
/*===========================================================================*/

/* internal data and definitions needed to hold the bitstreams: */

#define MAXIOSTR    10                 /* Max. number of streams that can be 
                                          open simultaneously */
#define INCSIZE     1000               /* for holding a bitstream the open-
                                          function allocates INCSIZE bytes of
                                          memory. When outputing to the 
                                          bitstream and the size of the
                                          memoryblocks exceeds subsequent 
                                          blocks with INCSIZE bytes are
                                          allocated to hold the bitstream. This
                                          is done by a realloc of the buffer. */

typedef struct bitio_stream {          /* structure that defines a bitstrem */
  int used;                            /* identifier if the bitstream is used */
  char *buf;                           /* buffer holding the bitstream */
  long nbits;                          /* currend size of bitstream (counted 
                                          in bits !) */
  size_t size;                         /* current size of allocated memory for
                                          holding the bitstream. */
} bitio_stream;

bitio_stream bios[MAXIOSTR];          /* Data describing MAXIOSTR bitstreams */
int first = 1;                         /* to indicate the first call to one of these functions */

/*===========================================================================*/
/** \ingroup bitio
    \brief This function opens a bitstream for input.

    This function opens a bitstream for input.

    \param[in] buf    Buffer to be used for input
    \param[in] size   Size of buffer.

    \return the function returns a handle by which the bitstream can be 
    identified for all subsequent actions or -1 if the maximum number of 
    opened bitstreams exceeds.

    \see bitio_i_close, bitio_i_input, bitio_o_open
*/

int bitio_i_open (void* buf, size_t size)
{
  int i, handle;

  /* On the first call mark all bitstreams as unused */

  if (first) {
    for (i = 0; i < MAXIOSTR; i ++) bios[i].used = 0;
    first = 0;
  }

  /* search for an unused stream. */

  for (handle = 0; handle < MAXIOSTR; handle ++) {
    if (!bios[handle].used) goto found;
  }
  return -1;

  /* unused bitstream found -> initialize bitstream-data */

found:
  bios[handle].used = 1;
  bios[handle].buf = (char *) buf;
  bios[handle].size = size;
  bios[handle].nbits = 0;                 /* Holds the current bitposition */
  return handle;
}

/*===========================================================================*/
/** \ingroup bitio
    \brief This function reads a value from a bitstream.

    This function reads a value from a bitstream. The bitstream must have 
    been opened by \ref bitio_i_open.

    \param[in] handle   Identifies the bitstream.
    \param[out] val     Is where the input-value is stored.
    \param[in] nbits    Number of bits the value consists of.

    \return Returns 1 on success or 0 on a fault (number of bytes in the
    bitstream exceeded).

    \see bitio_i_open, bitio_i_close, bitio_o_outp
*/

int bitio_i_input (int handle, unsigned long* val, int nbits)
{
  int i, bit;
  size_t byte;
  unsigned long l, bitval;
  char *pc;

  
  l = 0;
  for (i = nbits - 1; i >= 0; i --) {

      /* calculate bit- and byte-number for input and check if bytenumber is
         in a valid range */

      byte = (int) (bios[handle].nbits / 8);
      bit  = (int) (bios[handle].nbits % 8);
      bit = 7 - bit;
      if (byte >= bios[handle].size) return 0;

      /* get bit-value from input-stream */

      pc = bios[handle].buf + byte;
      bitval = (unsigned long) ((*pc >> bit) & 1);

      /* Set a 1-bit in the data value, 0-bits need not to be set, as L has
         been initialized to 0 */

      if (bitval) {
          l |= (bitval << i);
      }
      bios[handle].nbits ++;
  }
  *val = l;
  return 1;
}

/*===========================================================================*/
/** \ingroup bitio
    \brief Closes an bitstream that was opened for input 

    Closes an bitstream that was opened for input 

    \param[in] handle Handle that identifies the bitstream.

    \see bitio_i_open, bitio_i_input
*/

void bitio_i_close (int handle)

{
  bios[handle].used = 0;
}

/*===========================================================================*/
/** \ingroup bitio
    \brief Opens a bitstream for output.

    This function opens a bitstream for output. 

    \return The return-vaule is a handle by which the bit-stream 
    can be identified for all subesquent actions or -1
    if there is no unused bitstream available.
*/

int bitio_o_open ()

{
  int i, handle;

  /* On the first call mark all bitstreams as unused */

  if (first) {
    for (i = 0; i < MAXIOSTR; i ++) bios[i].used = 0;
    first = 0;
  }

  /* search for an unused stream. */

  for (handle = 0; handle < MAXIOSTR; handle ++) {
    if (!bios[handle].used) goto found;
  }
  return -1;

  /* unused bitstream found -> initalize it and allocate memory for it */

found:
  bios[handle].buf = (char *) malloc (INCSIZE);
  if (bios[handle].buf == NULL) return -1;
  memset (bios[handle].buf, 0, INCSIZE);
  bios[handle].used = 1;
  bios[handle].nbits = 0;
  bios[handle].size = INCSIZE;

  return handle;
}

/*===========================================================================*/
/** \ingroup bitio
    \brief This function appends a value to a bitstream.

    This function appends a value to a bitstream which was opened by
    \ref bitio_o_open.

    \param[in] handle  Indicates the bitstream for appending.
    \param[in] val     Value to be output.
    \param[in] nbits   Number of bits of \p val to be output to the stream. 

    \note \p nbits must be less than sizeof (\p long)

    \return The return-value is the bit-position of the value in the 
    bit-stream, or -1 on a fault.

    \see bitio_o_open, bitio_o_close, bitio_o_outp
*/

long bitio_o_append (int handle, unsigned long val, int nbits)

{
    /* Check if bitstream is allready initialized and number of bits does not
       exceed sizeof (unsigned long). */

  assert (bios[handle].used);
  assert (sizeof (unsigned long) * 8 >= nbits);

  /* check if there is enough memory to store the new value. Reallocate
     the memory-block if not */

  if ((bios[handle].nbits + nbits) / 8 + 1 > (long) bios[handle].size) {
    bios[handle].buf = realloc (bios[handle].buf, bios[handle].size + INCSIZE);
    if (bios[handle].buf == NULL) return 0;
	memset (bios[handle].buf + bios[handle].size, 0, INCSIZE);
    bios[handle].size += INCSIZE;
  }

  /* output data to bitstream */

  bitio_o_outp (handle, val, nbits, bios[handle].nbits);
  bios[handle].nbits += nbits;

  return bios[handle].nbits;
}

/*===========================================================================*/
/** \ingroup bitio
    \brief This function outputs a value to a specified position of a bitstream

    This function outputs a value to a specified position of a bitstream.

    \param[in] handle  Indicates the bitstream for output.
    \param[in] val     Value to be output.
    \param[in] nbits   Number of bits of \p val to be output to the stream. 
    \param[in] bitpos  bitposition of the value in the bitstream.

    \note \p nbits must be less then sizeof (\p long)

    \see bitio_o_open, bitio_o_close, bitio_o_append, bitio_i_input

*/

void bitio_o_outp (int handle, unsigned long val, int nbits, long bitpos)

{
  int i, bit, bitval;
  size_t byte;
  char *pc, c;

  /* Check if bitstream is allready initialized and number of bits does not
     exceed sizeof (unsigned long). */

  assert (bios[handle].used);
  assert (sizeof (unsigned long) * 8 >= nbits);

  for (i = nbits - 1; i >= 0; i --) {

      /* Get bit-value */

    bitval = (int) (val >> i) & 1;

    /* calculate bit- and byte-number for output */

    byte = (int) (bitpos / 8);
    bit  = (int) (bitpos % 8);
    bit  = 7 - bit;

    /* set bit-value to output stream */

    pc = bios[handle].buf + byte;
    if (bitval) {
      c = (char) (1 << bit);
      *pc |= c;
    }
    else {
      c = (char) (1 << bit);
      c ^= 0xff;
      *pc &= c;
    }
    bitpos ++;
  }
}

/*===========================================================================*/
/** \ingroup bitio
    \brief Returns the size of an output-bitstream (number of bytes) 

    This function returns the size of an output-bitstream (number of bytes) 
    
    \param[in] handle Identifies the bitstream

    \return Size of the bitstream.

    \see bitio_o_open, bitio_o_outp, bitio_o_append
*/


size_t bitio_o_get_size (int handle)

{
  if (!bios[handle].used) return 0;

  return (size_t) ((bios[handle].nbits - 1) / 8 + 1);
}


/*===========================================================================*/
/** \ingroup bitio
    \brief This function closes an output-bitstream

    This function closes an output-bitstream identified by \p handle and 
    returns a pointer to the memory-area holding the bitstream.

    \param[in] handle   Bit-stream-handle
    \param[out] nbytes  number of bytes in the bitstream.

    \return 
    The funcion returns a pointer to the memory-area holding the bit-stream or
    NULL if an invalid handle was specified. The memory area must be freed by
    the calling function.

    \see bitio_o_open, bitio_o_outp, bitio_o_append, bitio_i_close
*/

void *bitio_o_close (int handle, size_t* nbytes)
{

  if (!bios[handle].used) return NULL;

/* Fill up the last byte with 0-bits */

  while (bios[handle].nbits % 8 != 0) bitio_o_append (handle, 0, 1);

  *nbytes = (size_t) ((bios[handle].nbits - 1) / 8 + 1);
  bios[handle].used = 0;
  return (void *) bios[handle].buf;
}

/* end of file */

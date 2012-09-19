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

FILE:          DESC.H
IDENT:         $Id: desc.h,v 1.11 2010/02/15 11:16:23 helmutp Exp $

AUTHOR:        Konrad Koeck
               Institute of Communication and Wave Propagation, 
               Technical University Graz, Austria

VERSION NUMBER:3.0

DATE CREATED:  18-DEC-2001

STATUS:        DEVELOPMENT FINISHED


FUNCTIONAL DESCRIPTION:
-----------------------
Includefile that defines the data-structures needed to hold the supported
data-descriptors. Function-prototype for READDESC.

AMENDMENT RECORD:

ISSUE       DATE            SCNREF      CHANGE DETAILS
-----       ----            ------      --------------
V2.0        18-DEC-2001     Koeck       Initial Issue

$Log: desc.h,v $
Revision 1.11  2010/02/15 11:16:23  helmutp
added bitmap table functions

Revision 1.10  2009/05/18 16:04:54  helmutp
added show_desc_args

Revision 1.9  2007/12/18 14:40:58  fuxi
added licence header

Revision 1.8  2007/12/07 08:39:24  fuxi
update to version 3.0

Revision 1.7  2005/04/04 14:56:09  helmutp
update to version 2.3

Revision 1.6  2003/06/11 08:43:03  helmutp
added serial number to desc struct (for local table overruling)

Revision 1.5  2003/06/06 11:58:21  helmutp
changed read_tables

Revision 1.4  2003/03/27 17:17:39  helmutp
update to version 2.2

Revision 1.3  2003/03/13 17:03:31  helmutp
define local table names, added search key and optional descriptors

Revision 1.2  2003/03/06 17:12:32  helmutp
update to version 2.1

Revision 1.1  2003/02/28 13:41:12  helmutp
Initial revision

--------------------------------------------------------------------------- */

/** \file desc.h
    \brief Data structures needed for holding the supported data-descriptors.
    
    This file defines the data-structures needed to hold the supported
    data-descriptors. Also defines all functions used for 
    reading the decriptor tables  and utilites for 
    managing the data descriptors.
*/


#ifndef DESC_H_INCLUDED
#define DESC_H_INCLUDED

typedef double varfl;    /**< \brief Defines the internal float-variable type. 

                         Defines the internal float-variable type. 
                            This can
                            be float or double. Float needs less memory than
                            double. Double-floats need not to be converted by
                            your machine before operation (software runs 
                            faster). The default is double.
                            \note The format-string in all scanf-calls
                            must be changed for \p varfl-values !
                         */

/** This is the internal missing value indicator.
    Missing values are indicated as "missing" and if
    we find such a value we set it internally to
    MISSVAL
*/
#define MISSVAL 99999.999999

/*===========================================================================*/
/* definitions of data-structures                                            */
/*===========================================================================*/
/** \brief Holds the information contained in section 1

Holds the information contained in section 1
    \see bufr_sect_1_from_file, bufr_sect_1_to_file, bufr_encode_sections0125,
    bufr_decode_sections01
*/

typedef struct sect_1 {   
  int mtab;      /**<  \brief BUFR master table 

                 BUFR master table 
                 0 for standard WMO BUFR tables  */
  int subcent;   /**<  \brief Originating/generating subcenter */
  int gencent;   /**<  \brief Originating/generating center */
  int updsequ;   /**<  \brief Update sequence number 

                 Update sequence number 
                 zero for original BUFR
                    messages; incremented for updates */
  int opsec;     /**<  \brief optional section 

                 Bit 1 = 0 No optional section
                 = 1 Optional section included
                 Bits 2 - 8 set to zero (reserved) */
  int dcat;      /**<  \brief Data Category type (BUFR Table A) */
  int dcatst;    /**<  \brief Data Category sub-type 

                 Data Category sub-type 
                 defined by local ADP centres */
  int idcatst;   /**<  \brief International Data Category sub-type 

                 International Data Category sub-type 
                 Common Table C-13, used as of BUFR edition 4 */
  int vmtab;     /**<  \brief Version number of master tables used */
  int vltab;     /**<  \brief Version number of local tables used */
  int year;      /**<  \brief Year of century 

                 Year of century 
                 2 digit for BUFR edition < 4, 4 digit year as of BUFR edition
                 4  */
  int mon;       /**< \brief Month */
  int day;       /**< \brief Day */
  int hour;      /**< \brief Hour */
  int min;       /**< \brief Minute */
  int sec;       /**< \brief Second (used as of BUFR edition 4) */
} sect_1_t;

/** \brief Describes one data descriptor */

typedef struct dd {         
    int f; /**< \brief f*/
    int x; /**< \brief x*/
    int y; /**< \brief y*/
} dd;

/** \brief Defines an element descriptor */

typedef struct del {      
  dd d;                      /**< \brief Descriptor ID */
  char *unit;                /**< \brief Unit */
  int scale;                 /**< \brief Scale */
  varfl refval;              /**< \brief Reference Value */
  int dw;                    /**< \brief Data width (number of bits) */
  char *elname;              /**< \brief element name */
} del;

/** \brief Structure that defines a sequence of descriptors */

typedef struct dseq {        
  dd d;                      /**< \brief sequence-descriptor ID */
  int nel;                   /**< \brief Number of elements */
  dd *del;                   /**< \brief list of element descriptors */
} dseq;

/** \brief Structure that defines one descriptor. This can be an 
    element descriptor or a sequence descriptor */

typedef struct _desc {       
  int id;                    /**< \brief Can be \ref SEQDESC or \ref ELDESC */
  del *el;                   /**< \brief Element descriptor */
  dseq *seq;                 /**< \brief Sequence descriptor */
  int key;                   /**< \brief search key */
  int nr;                    /**< \brief serial number (insert position) */
} desc;


#define SEQDESC 0            /**< \brief Identifier for a sequence 
                                descriptor */
#define ELDESC  1            /**< \brief Identifier for an element 
                                descriptor */


/*===========================================================================*/
/* variables needed to hold data descriptors                                 */
/* If READDESC_MAIN is not defined all variables are declared as external.   */
/* So you sould define READDESC_MAIN only in one function. Otherwise you will*/
/* have this symbols multiple defined.                                       */
/*===========================================================================*/

#define MAXDESC   2000       /**< \brief Max. number of descriptors in the 
                                global descriptor-array (\ref des) */
#define OPTDESC   5          /* Number of optional descriptors at end */

#ifdef READDESC_MAIN
  int ndes;                 
  desc *des[MAXDESC+OPTDESC];
  int dw = 128;              
  int sc = 128;             
  int addfields = 0;        
  int ccitt_special = 0;     
  int cf_special = 0;       
  int add_f_special = 0;     
  int _desc_special = 0;     
#else

    /** \brief Total number of descriptors found */

    extern int ndes; 
    
    /** \brief Array holding all data descriptors 
                                        
    Array holding all data descriptors. 
    The descriptors are read from the descriptor table files 
    using \ref read_tables or \ref read_tab_b and read_tab_d

    \see read_tables, read_tab_b, read_tab_d, get_index
    */
    extern desc *des[MAXDESC+OPTDESC];

    /** \brief Current data width modification factor (default: 128)

    Current data width modification factor (default: 128)
    Add dw - 128 to the data-width   
    (dw can be optionally set by 2 01 YYY) 
    */
    extern int dw;

    /** \brief Current scale modification factor (default: 128)

    Current scale modification factor (default: 128).
    Add sc - 128 to the scale-factor
    (sc can be optionally set by 2 02 YYY) 
    */
    extern int sc;

    /** \brief Number of associated fields to be added to 
        any data-item. 
        
        Number of associated fields to be added to any data-item.
        \p addfields can be set by 2 04 YYY and canceled by 
        2 04 000 

    */
    extern int addfields;

    /** \brief Special index for ccitt characters.
        
    This index is used by \ref bufr_parse_new and its derivates
    to indicate that a value is a CCITT character

    \see bufr_parse_new, cbin, cbout
    */
    extern int ccitt_special;

    /* \brief Special index for change of reference field.
       
    \todo implement this
    */
    extern int cf_special;

    /** \brief Special index for associated fields.

    This index is used by \ref bufr_parse_new and its derivates
    to indicate that a value is an associated field.
    
    \see bufr_parse_new, cbin, cbout
    */
    extern int add_f_special;

    /** \brief Special index for descriptors without data.
    
    This index is used by \ref bufr_parse_new and its derivates
    to indicate that we have a descriptor without value for output.
    
    \see bufr_parse_new, cbout
    */
    extern int _desc_special;

#endif

/*===========================================================================*/
/* The following definition will be used to have either                      */
/* function-prototyping in ANSI-C e.g.: void abc (int a, int b);   or        */
/* Kernighan-Ritchie-prototyping link   void abc ();                         */
/*===========================================================================*/

#if defined (NON_ANSI)
#define P0
#define P1(a)                
#define P2(a,b)              
#define P3(a,b,c)            
#define P4(a,b,c,d)          
#define P5(a,b,c,d,e)        
#define P6(a,b,c,d,e,f)      
#define P7(a,b,c,d,e,f,g)    
#define P8(a,b,c,d,e,f,g,h)  
#else
#define P0                   void
#define P1(a)                a
#define P2(a,b)              a,b
#define P3(a,b,c)            a,b,c
#define P4(a,b,c,d)          a,b,c,d
#define P5(a,b,c,d,e)        a,b,c,d,e
#define P6(a,b,c,d,e,f)      a,b,c,d,e,f
#define P7(a,b,c,d,e,f,g)    a,b,c,d,e,f,g
#define P8(a,b,c,d,e,f,g,h)  a,b,c,d,e,f,g,h
#endif

/* for compilers having SEEK_CUR and SEEK_SET not defined: */

#ifndef SEEK_SET
#define SEEK_SET 0
#endif
#ifndef SEEK_END
#define SEEK_END 2
#endif

/*===========================================================================*/
/* function prototype                                                        */
/*===========================================================================*/

int read_tab_b (char *fname);
int read_tab_d (char *fname);
char *get_unit (dd *d);
int get_index (int typ, dd *d);
void free_descs (void);
void trim (char *buf);
int read_tables (char *dir, int vm, int vl, int subcenter, int gencenter);
void show_desc (int f, int x, int y);
void show_desc_args (int argc, char **argv);
int desc_is_codetable (int ind);
int desc_is_flagtable (int ind);
int read_bitmap_tables (char *dir,  int vltab, int gencent, int subcent);
int check_bitmap_desc (dd *d);

#endif

/* end of file */

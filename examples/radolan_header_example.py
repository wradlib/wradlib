#------------------------------------------------------------------------------
# Name:        radolan_header_example.py
# Purpose:     showing header information
#              for radolan composites
#
# Author:      Kai Muehlbauer
#
# Created:     11.02.2015
# Copyright:   (c) Kai Muehlbauer 2015
# Licence:     The MIT License
#------------------------------------------------------------------------------

import wradlib as wrl
import os


def ex_radolan_header():

    # load radolan file
    rx_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-rx_10000-1408102050-dwd---bin.gz'
    ex_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-ex_10000-1408102050-dwd---bin.gz'
    rw_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-rw_10000-1408102050-dwd---bin.gz'
    sf_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-sf_10000-1408102050-dwd---bin.gz'

    rxdata, rxattrs = wrl.io.read_RADOLAN_composite(rx_filename)
    exdata, exattrs = wrl.io.read_RADOLAN_composite(ex_filename)
    rwdata, rwattrs = wrl.io.read_RADOLAN_composite(rw_filename)
    sfdata, sfattrs = wrl.io.read_RADOLAN_composite(sf_filename)

    # print the available attributes
    print("RX Attributes:")
    for key, value in rxattrs.iteritems():
        print(key + ':', value)
    print("----------------------------------------------------------------")
    # print the available attributes
    print("EX Attributes:")
    for key, value in exattrs.iteritems():
        print(key + ':', value)
    print("----------------------------------------------------------------")

    # print the available attributes
    print("RW Attributes:")
    for key, value in rwattrs.iteritems():
        print(key + ':', value)
    print("----------------------------------------------------------------")

    # print the available attributes
    print("SF Attributes:")
    for key, value in sfattrs.iteritems():
        print(key + ':', value)
    print("----------------------------------------------------------------")

# =======================================================
if __name__ == '__main__':
    ex_radolan_header()
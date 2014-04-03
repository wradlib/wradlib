#-------------------------------------------------------------------------------
# Name:        vis_example
# Purpose:
#
# Author:      heistermann
#
# Created:     26.10.2011
# Copyright:   (c) heistermann 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python



def ex_bufr_decode():
    import wradlib.vis as vis
    import wradlib.bufr as bufr
    import os
    # This is our test BUFR file
    buffile = os.path.dirname(__file__) + '/data/test.buf'
    # decode BUFR file
    data, metadata = bufr.decodebufr(buffile)
    # print the BUFR descriptor dictionaries
    print metadata[0]
    print metadata[1]
    # plot the image
    classes = [-32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 70]
    vis.cartesian_plot(data, title='Reflectivity', unit='dBZ', colormap='spectral', classes=classes, extend='max')

#if __name__ == '__main__':
    #ex_bufr_decode()
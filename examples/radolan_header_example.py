# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import wradlib as wrl


def ex_radolan_header():
    # load radolan file
    rx_filename = wrl.util.get_wradlib_data_file('radolan/showcase/raa01-rx_10000-1408102050-dwd---bin.gz')
    ex_filename = wrl.util.get_wradlib_data_file('radolan/showcase/raa01-ex_10000-1408102050-dwd---bin.gz')
    rw_filename = wrl.util.get_wradlib_data_file('radolan/showcase/raa01-rw_10000-1408102050-dwd---bin.gz')
    sf_filename = wrl.util.get_wradlib_data_file('radolan/showcase/raa01-sf_10000-1408102050-dwd---bin.gz')

    rxdata, rxattrs = wrl.io.read_RADOLAN_composite(rx_filename)
    exdata, exattrs = wrl.io.read_RADOLAN_composite(ex_filename)
    rwdata, rwattrs = wrl.io.read_RADOLAN_composite(rw_filename)
    sfdata, sfattrs = wrl.io.read_RADOLAN_composite(sf_filename)

    # print the available attributes
    print("RX Attributes:")
    for key, value in rxattrs.items():
        print(key + ':', value)
    print("----------------------------------------------------------------")
    # print the available attributes
    print("EX Attributes:")
    for key, value in exattrs.items():
        print(key + ':', value)
    print("----------------------------------------------------------------")

    # print the available attributes
    print("RW Attributes:")
    for key, value in rwattrs.items():
        print(key + ':', value)
    print("----------------------------------------------------------------")

    # print the available attributes
    print("SF Attributes:")
    for key, value in sfattrs.items():
        print(key + ':', value)
    print("----------------------------------------------------------------")


# =======================================================
if __name__ == '__main__':
    ex_radolan_header()

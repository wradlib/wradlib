def radar_spec():
    """
    Renders multiple spec with their values.
    """
    # Radar specifications
    nrays = 360   # number of rays
    nbins = 1000  # number of range bins
    bw = 1.0  # half power beam width (deg)
    range_res = 100.0  # range resolution (meters)    # default 100
    # Custom Plots
    height_ylim_curvature = 5000
    height_ylim_terrain = 5000

    return {
        "nrays": nrays,
        "nbins": nbins,
        "bw": bw,
        "range_res": range_res,
        "height_ylim_curvature": height_ylim_curvature,
        "height_ylim_terrain": height_ylim_terrain,
    }

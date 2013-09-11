#-------------------------------------------------------------------------------
# Name:         clutter
# Purpose:
#
# Authors:      Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:      26.10.2011
# Copyright:    (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:      The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python
"""
Clutter Identification
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   filter_gabella
   filter_gabella_a
   filter_gabella_b
   histo_cut

"""
import numpy as np
import scipy.ndimage as ndi

def filter_gabella_a(img, wsize, tr1, cartesian=False, radial=False):
    r"""First part of the Gabella filter looking for large reflectivity
    gradients.

    This function checks for each pixel in `img` how many pixels surrounding
    it in a window of `wsize` are by `tr1` smaller than the central pixel.

    Parameters
    ----------
    img : array_like
        radar image to which the filter is to be applied
    wsize : int
        Size of the window surrounding the central pixel
    tr1 : float
        Threshold value
    cartesian : boolean
        Specify if the input grid is Cartesian or polar
    radial : boolean
        Specify if only radial information should be used 

    Returns
    -------
    output : array_like
        an array with the same shape as `img`, containing the filter's results.

    See Also
    --------
    filter_gabella_b : the second part of the filter
    filter_gabella : the complete filter

    Examples
    --------
    TODO: provide a correct example here

    >>> a=[1,2,3]
    >>> print [x + 3 for x in a]
    :w
[4, 5, 6]
    >>> print "a\n\nb"
    a
    b

    """
    nn = wsize // 2
    count = -np.ones(img.shape,dtype=int)
    range_shift = range(-nn,nn+1)
    azimuth_shift = range(-nn,nn+1)
    if radial:
        azimuth_shift = [0]
    for sa in azimuth_shift:
        ref = np.roll(img,sa,axis=0)
        for sr in range_shift:
            ref = np.roll(ref,sr,axis=1)
            count += ( img - ref < tr1 )
    count[:,0:nn] = wsize**2
    count[:,-nn:] = wsize**2
    if cartesian :
        count[0:nn,:] = wsize**2
        count[-nn:,:] = wsize**2
    return(count)

def filter_gabella_b(img, thrs=0.):
    r"""Second part of the Gabella filter comparing area to circumference of
    contiguous echo regions.



    Parameters
    ----------
    img : array_like
    thrs : float
        Threshold below which the field values will be considered as no rain

    Returns
    -------
    output : array_like
        contains in each pixel the ratio between area and circumference of the
        meteorological echo it is assigned to or 0 for non precipitation pixels.

    See Also
    --------
    filter_gabella_a : the first part of the filter
    filter_gabella : the complete filter

    Examples
    --------
    TODO: provide a correct example here

    >>> a=[1,2,3]
    >>> print [x + 3 for x in a]
    [4, 5, 6]
    >>> print "a\n\nb"
    a
    b

    """
    conn = np.ones((3,3))
    # create binary image of the rainfall field
    binimg = img > thrs
    # label objects (individual rain cells, so to say)
    labelimg, nlabels = ndi.label(binimg, conn)
    # erode the image, thus removing the 'boundary pixels'
    binimg_erode = ndi.binary_erosion(binimg, structure=conn)
    # determine the size of each object
    labelhist, edges = np.histogram(labelimg,
                                    bins=nlabels+1,
                                    range=(-0.5, labelimg.max()+0.5))
    # determine the size of the eroded objects
    erodelabelhist, edges = np.histogram(np.where(binimg_erode, labelimg, 0),
                                         bins=nlabels+1,
                                         range=(-0.5, labelimg.max()+0.5))
    # the boundary is the difference between these two
    boundarypixels = labelhist - erodelabelhist
    # now get the ratio between object size and boundary
    ratio = labelhist.astype(np.float32) / boundarypixels
    # assign it back to the objects
    # first get the indices
    indices = np.digitize(labelimg.ravel(), edges)-1
    # then produce a new field with the ratios in the right place
    result = ratio[indices.ravel()].reshape(img.shape)

    return result

def filter_gabella(img, wsize=5, thrsnorain=0., tr1=6., n_p=6, tr2=1.3, rm_nans=True, radial = False, cartesian = False):
    r"""Clutter identification filter developed by Gabella [Gabella2002]_ .

    This is a two-part identification algorithm using echo continuity and
    minimum echo area to distinguish between meteorological (rain) and non-
    meteorological echos (ground clutter etc.)

    Parameters
    ----------
    img : array_like
    wsize : int
        Size of the window surrounding the central pixel
    thrsnorain : float
    tr1 : float
    n_p : int
    thr2 : float
    rm_nans : boolean
        True replaces nans with Inf
        False takes nans into acount
    radial : boolean
        True to use radial information only in filter_gabella_a.
    cartesian : boolean
        True if cartesian data are used, polar assumed if False.

    Returns
    -------
    output : array
        boolean array with pixels identified as clutter set to True.

    See Also
    --------
    filter_gabella_a : the first part of the filter
    filter_gabella_b : the second part of the filter

    References
    ----------
    .. [Gabella2002] Gabella, M. & Notarpietro, R., 2002.
        Ground clutter characterization and elimination in mountainous terrain.
        In Proceedings of ERAD. Delft: Copernicus GmbH, pp. 305-311.
        Available at: http://www.copernicus.org/erad/online/erad-305.pdf
        [Accessed Oct 27, 2010].

    Examples
    --------
    TODO: provide a correct example here

    >>> a=[1,2,3]
    >>> print [x + 3 for x in a]
    [4, 5, 6]
    >>> print "a\n\nb"
    a
    b

    """
    bad = np.isnan(img)
    if rm_nans:
        img[bad] = np.Inf
    ntr1 = filter_gabella_a(img, wsize, tr1, cartesian, radial)
    if not rm_nans:
        f_good = ndi.filters.uniform_filter((~bad).astype(float),size=wsize)
        f_good[f_good == 0] = 1e-10
        ntr1 = ntr1/f_good
        ntr1[bad] = n_p
    clutter1 = (ntr1 < n_p) 
    ratio = filter_gabella_b(img, thrsnorain)
    clutter2 = ( np.abs(ratio) < tr2 )
    return ( clutter1 | clutter2 ) & ( img > 10 )

def histo_cut(prec_accum):
    r"""Histogram based clutter identification.

    This identification algorithm uses the histogram of temporal accumulated
    rainfall. It iteratively detects classes whose frequency falls below a
    specified percentage (1% by default) of the frequency of the class with the
    biggest frequency and remove the values from the dataset until the changes
    from iteration to iteration falls below a threshold. This algorithm is able
    to detect static clutter as well as shadings. It is suggested to choose a
    representative time periode for the input precipitation accumulation. The
    recommended time period should cover one year.

    Parameters
    ----------
    prec_accum : array_like
        spatial array containing rain accumulation

    Returns
    -------
    output : array
        boolean array with pixels identified as clutter/shadings set to True.

    """

    prec_accum = np.array(prec_accum)

    # initialization of data bounds for clutter and shade definition
    lower_bound = 0
    upper_bound = prec_accum.max()

    # predefinitions for the first iteration
    lower_bound_before = -51
    upper_bound_before = -51

    # iterate as long as the difference between current and last iteration doesn't fall below the stop criterion
    while ((abs(lower_bound - lower_bound_before) > 1) or (abs(upper_bound - upper_bound_before) > 1)):

        # masks for bins with sums over/under the data bounds
        upper_mask = (prec_accum <= upper_bound).astype(int)
        lower_mask = (prec_accum >= lower_bound).astype(int)
        # NaNs in place of masked bins
        prec_accum_masked = np.where((upper_mask * lower_mask) == 0, np.nan,prec_accum) # Kopie der Datenmatrix mit Nans an Stellen, wo der Threshold erreicht wird

        # generate a histogram of the valid bins with 50 classes
        (n, bins) = np.histogram(prec_accum_masked[np.isfinite(prec_accum_masked)].ravel(), bins = 50)
        # get the class with biggest occurence
        index=np.where(n == n.max())
        index= index[0]

        # separeted stop criterion check in case one of the bounds is already robust
        if (abs(lower_bound - lower_bound_before) > 1):
        # get the index of the class which underscores the occurence of the biggest class by 1%,
        #beginning from the class with the biggest occurence to the first class
           for i in range(index, -1, -1):
                if (n[i] < (n[index] * 0.01)): break
        if (abs(upper_bound - upper_bound_before) > 1):
            # get the index of the class which underscores the occurence of the biggest class by 1%,
            #beginning from the class with the biggest occurence to the last class
            for j in range(index, len(n)):
                if (n[j] < (n[index] * 0.01)): break

        lower_bound_before = lower_bound
        upper_bound_before = upper_bound
        # update the new boundaries
        lower_bound = bins[i]
        upper_bound = bins[j + 1]

    return np.isnan(prec_accum_masked)

if __name__ == '__main__':
    print 'wradlib: Calling module <clutter> as main...'

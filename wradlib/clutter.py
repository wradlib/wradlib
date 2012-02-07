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

def _filter_gabella_a(windowdata, tr1=6.):
    r"""helper function to be passed to scipy.ndimage.filters.generic_filter as
    part of filter_gabella_a.

    Parameters
    ----------
    windowdata : array_like
        Data passed in from the generic_filter.
    tr1 : float
        Threshold value

    Returns
    -------
    output : float
        Number of pixels whose difference to the central pixel is smaller
        than tr1.

    See Also
    --------
    filter_gabella_a : the actual function to be called for which this is
        only a helper
    filter_gabella : the complete filter

    """
    window = windowdata.reshape((np.sqrt(windowdata.size),
                                 np.sqrt(windowdata.size)))
    wshape = window.shape
    centerx = (wshape[1]-1)//2
    centery = (wshape[0]-1)//2

    return ((window[centery, centerx] - window) < tr1).sum()-1


def filter_gabella_a(img, wsize, tr1):
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
        TODO check if a 5x5 window would have wsize=2 or wsize=5
    tr1 : float
        Threshold value

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
    [4, 5, 6]
    >>> print "a\n\nb"
    a
    b

    """
    return ndi.filters.generic_filter(img,
                                      _filter_gabella_a,
                                      size=wsize,
                                      mode='wrap',
                                      extra_keywords={'tr1':tr1}
                                      )


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


def filter_gabella(img, wsize=5, thrsnorain=0., tr1=6., n_p=8, tr2=1.3):
    r"""Clutter identification filter developed by Gabella [Gabella2002]_ .

    This is a two-part identification algorithm using echo continuity and
    minimum echo area to distinguish between meteorological (rain) and non-
    meteorological echos (ground clutter etc.)

    Parameters
    ----------
    img : array_like
    wsize : int
    thrsnorain : float
    tr1 : float
    n_p : int
    thr2 : float

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
    assert wsize % 2 == 1

    ntr1 = filter_gabella_a(img, wsize, tr1)
    clutter1 = ntr1 < n_p

    ratio = filter_gabella_b(img, thrsnorain)
    clutter2 = np.abs(ratio) < tr2

    return clutter1 | clutter2

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
        prec_accum_masked = np.where((upper_mask * lower_mask) == 0, np.nan,prec_accum) # Kopie der Datenmatrix mit 0 an Stellen, wo der Threshold erreicht wird

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

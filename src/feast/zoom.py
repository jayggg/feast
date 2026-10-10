"""Contour splitting utilities: given eigenvalues found in a real
interval, propose smaller circular contours isolating each simple
eigenvalue and each cluster of nearby eigenvalues.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["segment_list", "simple_multiple_zoom"]


def segment_list(ic):
    """
    Segment a list "ic", like [2, 3, 4, 9, 11, 12, 13, 17, 18], into
    contiguous parts, like  [[2, 3, 4], [9], [11, 12, 13], [17, 18]].
    """

    if len(ic) > 0:
        dividers = np.zeros(len(ic))
        dividers[0] = 1

        for j in range(len(ic)-1):
            if ic[j+1] == ic[j]+1:
                dividers[j+1] = 0
            else:
                dividers[j+1] = 1

        clust = [[ic[0]]]
        for j in range(1, len(ic)):
            if dividers[j] == 0:
                clust[-1] += [ic[j]]
            else:
                clust += [[ic[j]]]
    else:
        clust = []

    return clust


def simple_multiple_zoom(c1, c2, ew, delta=None):
    """Propose circular contours for isolated values and nearby clusters.

    Parameters
    ----------
    c1, c2 : float
        Endpoints of the enclosing real interval.
    ew : sequence of float
        Nonempty eigenvalue list inside the interval. Supply sorted values;
        unsorted input currently produces inconsistent simple centers.
    delta : float or None, optional
        Neighboring values less than this distance apart are grouped.
        Defaults to one tenth of the interval length divided by the count.

    Returns
    -------
    simple : dict
        ``index`` lists individual indices into ``ew``. ``center`` and
        ``radius`` give the corresponding proposed circular contours.
    multiple : dict
        ``index`` lists arrays of indices for clusters. ``center`` and
        ``radius`` give proposed cluster contours.

    Notes
    -----
    Clustering is based on distance, not a certified multiplicity test.
    Validate that each contour encloses its intended cluster before use.
    """

    # Set default delta
    if delta is None:
        delta = (c2 - c1) / len(ew) / 10

    ind = np.argsort(ew)   # Do not sort/modify the input ew
    e = np.array(ew)       # Copy ew into np.array and sort
    e = e[ind]

    if c1 > e[0] or c2 < e[-1]:
        raise ValueError('List of numbers in ew not contained ' +
                         'in interval (c1, c2).')

    # Make e = [c1, e, c2]
    e = np.insert(e, 0, c1)
    e = np.insert(e, len(e), c2)

    # Make d[i] = e[i+1] - e[i]  for i = 0, 1,..., len(e)-1.
    d = e[1:] - e[:-1]

    if d[0] < delta:
        log.warning('First ew element too close to c1!')
        d[0] = 2*delta
    if d[-1] < delta:
        log.warning('Last ew element too close to c2!')
        d[-1] = 2*delta

    # Identify multiple eigenvalues
    ic = np.where(d < delta)[0]
    clus = segment_list(ic)

    clusters = []
    clusterindices = []
    centers = []
    radii = []

    for cind in clus:

        clusters.append(e[cind[0]:cind[-1]+2])
        clusterindices.append(np.arange(cind[0], cind[-1]+2)-1)

        # center set to mean of ews of this cluster
        centers.append(sum(clusters[-1])/len(clusters[-1]))

        # radius set to minimal distance to next ew or interval endpoints
        radii.append(0.5 * min(e[cind[0]] - e[cind[0]-1],
                               e[cind[-1]+2] - e[cind[-1]+1]))

    # The remaining are simple eigenvalues
    simplewindices = \
        np.setdiff1d(np.arange(len(ew)), np.concatenate(clusterindices)) \
        if len(clusterindices) > 0 else np.arange(len(ew))

    rad = []
    ctr = []
    for sind in simplewindices:
        ctr += [ew[sind]]
        rad += [0.5 * min(e[sind+1] - e[sind],
                          e[sind+2] - e[sind+1])]

    multiplewindices = []
    for c in clusterindices:
        multiplewindices.append(ind[c])

    # Prepare dictionaries for output
    multiple = {}
    multiple['index'] = multiplewindices
    multiple['radius'] = radii
    multiple['center'] = centers

    simple = {}
    simple['index'] = ind[simplewindices].tolist()
    simple['radius'] = rad
    simple['center'] = ctr

    return simple, multiple

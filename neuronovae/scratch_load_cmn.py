from pathlib import Path
from h5py import File
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
matplotlib.interactive(True)
from matplotlib import pyplot as plt
import colorcet as cc
from scipy.sparse import csc_matrix
from typing import Optional
from skimage.measure import find_contours


def extract_events(spikes, idx):
    thr = np.percentile(spikes[idx, :], 95)
    return np.where(spikes[idx, :] >= thr)[0]


def plot_activity(traces, spikes, idx) -> None:
    fig, ax = plt.subplots(1, 1)
    time = np.linspace(0, traces.shape[-1]/30.0, traces.shape[-1])
    trace_max = np.nanmax(traces[idx, :])
    trace = np.multiply(traces[idx, :], 1/trace_max)
    ax.plot(time, trace, lw=2, color="red")
    events = extract_events(spikes, idx)
    for event in events:
        ax.vlines(time[event], -0.1, 0, color="black", lw=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity (AU)")
    ax.set_title(f"Component {idx}")
    ax.set_xlim(-0.1, time[-1] + 0.1)
    ax.set_ylim([-0.15, 1.1])


def get_A(A_):
    data = np.asarray(A_.get("data"))
    indices = np.asarray(A_.get("indices"))
    indptr = np.asarray(A_.get("indptr"))
    shape = np.asarray(A_.get("shape"))
    return csc_matrix((data[:], indices[:], indptr[:]), shape[:])


def com(A, d1: int, d2: int, d3: Optional[int] = None, order: str = 'F') -> np.ndarray:
    """Calculation of the center of mass for spatial components

     Args:
         A: np.ndarray or scipy.sparse array or matrix
              matrix of spatial components (d x K).

         d1, d2, d3: ints
              d1, d2, and (optionally) d3 are the original dimensions of the data.

         order: 'C' or 'F'
              how each column of A should be reshaped to match the given dimensions.

     Returns:
         cm:  np.ndarray
              center of mass for spatial components (K x D)
    """
    if 'csc_matrix' not in str(type(A)):
        A = scipy.sparse.csc_matrix(A)

    dims = [d1, d2]
    if d3 is not None:
        dims.append(d3)

    # make coordinate arrays where coor[d] increases from 0 to npixels[d]-1 along the dth axis
    coors = np.meshgrid(*[range(d) for d in dims], indexing='ij')
    coor = np.stack([c.ravel(order=order) for c in coors])

    # take weighted sum of pixel positions along each coordinate
    cm = (coor @ A / A.sum(axis=0)).T
    return np.array(cm)


def get_contours(A, dims, thr=0.9, thr_method='nrg', swap_dim=False,
                 slice_dim: Optional[int] = None):
    """Gets contour of spatial components and returns their coordinates

     Args:
         A:   np.ndarray or sparse matrix
                   Matrix of Spatial components (d x K)

             dims: tuple of ints
                   Spatial dimensions of movie

             thr: scalar between 0 and 1
                   Energy threshold for computing contours (default 0.9)

             thr_method: string
                  Method of thresholding:
                      'max' sets to zero pixels that have value less than a fraction of the max value
                      'nrg' keeps the pixels that contribute up to a specified fraction of the energy

             swap_dim: bool
                  If False (default), each column of A should be reshaped in F-order to recover the mask;
                  this is correct if the dimensions have not been reordered from (y, x[, z]).
                  If True, each column should be reshaped in C-order; this is correct for dims = ([z, ]x, y).

             slice_dim: int or None
                  Which dimension to slice along if we have 3D data. (i.e., get contours on each plane along this axis).
                  The default (None) is 0 if swap_dim is True, else -1.

     Returns:
         Coor: list of coordinates with center of mass and
                contour plot coordinates (per layer) for each component
    """

    if 'csc_matrix' not in str(type(A)):
        A = csc_matrix(A)
    d, nr = np.shape(A)

    coordinates = []

    # get the center of mass of neurons( patches )
    cm = com(A, *dims, order='C' if swap_dim else 'F')

    # for each patches
    for i in range(nr):
        pars: dict = dict()
        # we compute the cumulative sum of the energy of the Ath component that has been ordered from least to highest
        patch_data = A.data[A.indptr[i]:A.indptr[i + 1]]
        indx = np.argsort(patch_data)[::-1]
        if thr_method == 'nrg':
            cumEn = np.cumsum(patch_data[indx] ** 2)
            if len(cumEn) == 0:
                pars = dict(
                    coordinates=np.array([]),
                    CoM=np.array([np.NaN, np.NaN]),
                    neuron_id=i + 1,
                )
                coordinates.append(pars)
                continue
            else:
                # we work with normalized values
                cumEn /= cumEn[-1]
                Bvec = np.ones(d)
                # we put it in a similar matrix
                Bvec[A.indices[A.indptr[i]:A.indptr[i + 1]][indx]] = cumEn
        else:
            if thr_method != 'max':
                warn("Unknown threshold method. Choosing max")
            Bvec = np.zeros(d)
            Bvec[A.indices[A.indptr[i]:A.indptr[i + 1]]] = patch_data / patch_data.max()

        if swap_dim:
            Bmat = np.reshape(Bvec, dims, order='C')
        else:
            Bmat = np.reshape(Bvec, dims, order='F')

        def get_slice_coords(B: np.ndarray) -> np.ndarray:
            """Get contour coordinates for a 2D slice"""
            d1, d2 = B.shape
            vertices = find_contours(B.T, thr)
            # this fix is necessary for having disjoint figures and borders plotted correctly
            v = np.atleast_2d([np.nan, np.nan])
            for _, vtx in enumerate(vertices):
                num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
                if num_close_coords < 2:
                    if num_close_coords == 0:
                        # case angle
                        newpt = np.round(
                            np.mean(vtx[[0, -1], :], axis=0) / [d2, d1]) * [d2, d1]
                        vtx = np.concatenate(
                            (newpt[np.newaxis, :], vtx, newpt[np.newaxis, :]), axis=0)
                    else:
                        # case one is border
                        vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                v = np.concatenate(
                    (v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)
            return v

        if len(dims) == 2:
            pars['coordinates'] = get_slice_coords(Bmat)
        else:
            # make a list of the contour coordinates for each 2D slice
            pars['coordinates'] = []
            if slice_dim is None:
                slice_dim = 0 if swap_dim else -1
            for s in range(dims[slice_dim]):
                B = Bmat.take(s, axis=slice_dim)
                pars['coordinates'].append(get_slice_coords(B))

        pars['CoM'] = np.squeeze(cm[i, :])
        pars['neuron_id'] = i + 1
        coordinates.append(pars)
    return coordinates


def load_cmn() -> dict:
    CMN_FILE = Path(R"C:\Users\Darik\AppData\Roaming\JetBrains\PyCharm2024.2\scratches\caiman_results.hdf5")
    with File(CMN_FILE, 'r') as file:
        dims = tuple(file.get("dims"))
        estimates = file.get("estimates")
        keys = tuple(estimates.keys())
        rois = tuple(estimates.get("idx_components"))
        drops = tuple(estimates.get("idx_components_bad"))
        coordinates = get_contours(get_A(estimates.get("A")), dims)
        centroids = {idx: coord.pop("CoM") for idx, coord in enumerate(coordinates)}
        coordinates = {idx: coord.get("coordinates") for idx, coord in enumerate(
            coordinates)}
        snr = tuple(estimates.get("SNR_comp"))
        space_correlation = tuple(estimates.get("r_values"))
        cnn_values = tuple(estimates.get("cnn_preds"))
        # ring_model_matrix = np.asarray(estimates.get("W"))
        eccentricity = tuple(estimates.get("ecc")) \
            if len(estimates.get("ecc").shape) != 0 else None
        deconvolved = np.asarray(estimates.get("S"))
        dff = np.asarray(estimates.get("F_dff"))
        tau = tuple(estimates.get("g"))
        pixel_baseline = tuple(estimates.get("b0")) \
            if len(estimates.get("b0").shape) != 0 else None
        pixel_noise = tuple(estimates.get("sn")) \
            if len(estimates.get("sn").shape) != 0 else None
        baseline = tuple(estimates.get("bl"))
        initial = tuple(estimates.get("c1"))
        noise = tuple(estimates.get("neurons_sn"))
        spatial_components = np.asarray(estimates.get("A"))
        spatial_background = np.asarray(estimates.get("b"))
        temporal_components = np.asarray(estimates.get("C"))
        temporal_background = np.asarray(estimates.get("f"))
        residuals = np.asarray(estimates.get("R"))
        denoised_activity = np.asarray(estimates.get("YrA"))
        components = temporal_components.shape[0]
        frames = temporal_components.shape[-1]

        return {
            "dims": dims,
            "rois": rois,
            "drops": drops,
            "coordinates": coordinates,
            "centroids": centroids,
            "snr": snr,
            "space_correlation": space_correlation,
            "cnn_values": cnn_values,
            "eccentricity": eccentricity,
            "deconvolved": deconvolved,
            "dff": dff,
            "tau": tau,
            "pixel_baseline": pixel_baseline,
            "pixel_noise": pixel_noise,
            "baseline": baseline,
            "initial": initial,
            "noise": noise,
            "spatial_components": spatial_components,
            "spatial_background": spatial_background,
            "temporal_components": temporal_components,
            "temporal_background": temporal_background,
            "residuals": residuals,
            "denoised_activity": denoised_activity,
            "components": components,
            "frames": frames
        }

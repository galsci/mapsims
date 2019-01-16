import numpy as np
from . import SO_Noise_Calculator_Public_20180822 as so_noise


def get_bands(telescope):
    """Returns the available bands for a telescope

    Parameters
    ----------
    telescope : {"SA", "LA"}

    Returns
    -------
    bands : ndarray of ints
        Available bands
    """
    bands = getattr(
        so_noise, "Simons_Observatory_V3_{}_bands".format(telescope)
    )().astype(np.int)
    return bands


def get_band_index(telescope, band):

    bands = get_bands(telescope)
    try:
        band_index = bands.tolist().index(band)
    except ValueError:
        print(
            "Band {} not available, available bands for {} are {}".format(
                band, telescope, bands
            )
        )
        raise
    return band_index


def get_beam(telescope, band):
    """Returns the beam in arcminutes for a band

    Parameters
    ----------
    telescope : {"SA", "LA"}
    band : int
        Band center frequency in GHz

    Returns
    -------
    beam : float
        Full width half max in arcmin
    """
    beams = getattr(so_noise, "Simons_Observatory_V3_{}_beams".format(telescope))()
    return beams[get_band_index(telescope, band)]

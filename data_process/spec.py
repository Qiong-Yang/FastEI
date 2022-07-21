
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 10:43:37 2021

@author: qiongyang
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
from matplotlib import pyplot
from collections import defaultdict
from pyteomics.mgf import MGF
import pyteomics.mgf as py_mgf
import sqlite3

def load_database(db_file):
    # db_file = 'data/IN_SILICO_LIBRARY1.db'
    cursor = sqlite3.connect(db_file)
    content = cursor.execute('SELECT * from IN_SILICO_LIBRARY')
    data = [row for row in content]
    return data

class Spikes:
    """
    Adapted from: https://github.com/matchms/matchms
	Copyright (c) 2021, Netherlands eScience Center
	See licenses/MATCHMS_LICENSE
    Stores arrays of intensities and M/z values, with some checks on their internal consistency.
    """
    def __init__(self, mz=None, intensities=None):
        assert isinstance(mz, np.ndarray), "Input argument 'mz' should be a np.array."
        assert isinstance(intensities, np.ndarray), "Input argument 'intensities' should be a np.array."
        assert mz.shape == intensities.shape, "Input arguments 'mz' and 'intensities' should be the same shape."
        assert mz.dtype == "float", "Input argument 'mz' should be an array of type float."
        assert intensities.dtype == "float", "Input argument 'intensities' should be an array of type float."

        self._mz = mz
        self._intensities = intensities

        assert self._is_sorted(), "mz values are out of order."

    def __eq__(self, other):
        return \
            self.mz.shape == other.mz.shape and \
            np.allclose(self.mz, other.mz) and \
            self.intensities.shape == other.intensities.shape and \
            np.allclose(self.intensities, other.intensities)

    def __len__(self):
        return self._mz.size

    def __getitem__(self, item):
        return [self.mz, self.intensities][item]

    def _is_sorted(self):
        return np.all(self.mz[:-1] <= self.mz[1:])

    def clone(self):
        return Spikes(self.mz, self.intensities)

    @property
    def mz(self):
        """getter method for mz private variable"""
        return self._mz.copy()

    @property
    def intensities(self):
        """getter method for intensities private variable"""
        return self._intensities.copy()

    @property
    def to_np(self):
        """getter method to return stacked np array of both peak mz and
        intensities"""
        return np.vstack((self.mz, self.intensities)).T


class Spectrum:

    def __init__(self, mz: np.array, intensities: np.array, metadata: Optional[dict] = None):
        """
        
    	Adapted from: https://github.com/matchms/matchms
	    Copyright (c) 2021, Netherlands eScience Center
	    See licenses/MATCHMS_LICENSE
        
        Parameters
        ----------
        mz
            Array of m/z for the peaks
        intensities
            Array of intensities for the peaks
        metadata
            Dictionary with for example the scan number of precursor m/z.
        """
        self.peaks = Spikes(mz=mz, intensities=intensities)
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

    def __eq__(self, other):
        return \
            self.peaks == other.peaks and \
            self.__metadata_eq(other.metadata)

    def __metadata_eq(self, other_metadata):
        if self.metadata.keys() != other_metadata.keys():
            return False
        for i, value in enumerate(list(self.metadata.values())):
            if isinstance(value, np.ndarray):
                if not np.all(value == list(other_metadata.values())[i]):
                    return False
            elif value != list(other_metadata.values())[i]:
                return False
        return True

    def clone(self):
        """Return a deepcopy of the spectrum instance."""
        clone = Spectrum(mz=self.peaks.mz,
                         intensities=self.peaks.intensities,
                         metadata=self.metadata)
        clone.losses = self.losses
        return clone

    def plot(self, intensity_from=0.0, intensity_to=None, with_histogram=False):
        """To visually inspect a spectrum run ``spectrum.plot()``

        """

        def plot_histogram():
            """Plot the histogram of intensity values as horizontal bars, aligned with the spectrum axes"""

            def calc_bin_edges_intensity():
                """Calculate various properties of the histogram bins, given a range in intensity defined by
                'intensity_from' and 'intensity_to', assuming a number of bins equal to 100."""
                edges = np.linspace(intensity_from, intensity_to, n_bins + 1)
                lefts = edges[:-1]
                rights = edges[1:]
                middles = (lefts + rights) / 2
                widths = rights - lefts
                return edges, middles, widths

            bin_edges, bin_middles, bin_widths = calc_bin_edges_intensity()
            counts, _ = np.histogram(self.peaks.intensities, bins=bin_edges)
            histogram_ax.set_ylim(bottom=intensity_from, top=intensity_to)
            pyplot.barh(bin_middles, counts, height=bin_widths, color="#047495")
            pyplot.title(f"histogram (n_bins={n_bins})")
            pyplot.xlabel("count")

        def plot_spectrum():
            """plot mz v. intensity"""

            def make_stems():
                """calculate where the stems of the spectrum peaks are going to be"""
                x = np.zeros([2, self.peaks.mz.size], dtype="float")
                y = np.zeros(x.shape)
                x[:, :] = np.tile(self.peaks.mz, (2, 1))
                y[1, :] = self.peaks.intensities
                return x, y

            spectrum_ax.set_ylim(bottom=intensity_from, top=intensity_to)
            x, y = make_stems()
            pyplot.plot(x, y, color="#0f0f0f", linewidth=1.0, marker="")
            pyplot.title("Spectrum")
            pyplot.xlabel("M/z")
            pyplot.ylabel("intensity")

        if intensity_to is None:
            intensity_to = self.peaks.intensities.max() * 1.05

        n_bins = 100
        fig = pyplot.figure()

        if with_histogram:
            spectrum_ax = fig.add_axes([0.2, 0.1, 0.5, 0.8])
            plot_spectrum()
            histogram_ax = fig.add_axes([0.72, 0.1, 0.2, 0.8])
            plot_histogram()
            histogram_ax.set_yticklabels([])
        else:
            spectrum_ax = fig.add_axes([0.2, 0.1, 0.7, 0.8])
            plot_spectrum()
            histogram_ax = None

        return fig

    def get(self, key: str, default=None):

        return self._metadata.copy().get(key, default)

    def set(self, key: str, value):

        self._metadata[key] = value
        return self

    @property
    def metadata(self):
        return self._metadata.copy()

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def peaks(self) -> Spikes:
        return self._peaks.clone()

    @peaks.setter
    def peaks(self, value: Spikes):
        self._peaks = value
def load_from_mgf(filename: str) :
    """Load spectrum(s) from mgf file.
    """

    for pyteomics_spectrum in MGF(filename, convert_arrays=1):

        metadata = pyteomics_spectrum.get("params", None)
        mz = pyteomics_spectrum["m/z array"]
        intensities = pyteomics_spectrum["intensity array"]

        # Sort by mz (if not sorted already)
        if not np.all(mz[:-1] <= mz[1:]):
            idx_sorted = np.argsort(mz)
            mz = mz[idx_sorted]
            intensities = intensities[idx_sorted]

        yield Spectrum(mz=mz, intensities=intensities, metadata=metadata)
def save_as_mgf(spectrums, filename):
    """Save spectrum(s) as mgf file.
    Parameters
    ----------
    spectrums:
        Expected input are match.Spectrum.Spectrum() objects.
    filename:
        Provide filename to save spectrum(s).
    """
    if not isinstance(spectrums, list):
        # Assume that input was single Spectrum
        spectrums = [spectrums]

    # Convert matchms.Spectrum() into dictionaries for pyteomics
    for spectrum in spectrums:
        spectrum_dict = {"m/z array": spectrum.peaks.mz,
                         "intensity array": spectrum.peaks.intensities,
                         "params": spectrum.metadata}
        # Append spectrum to file
        py_mgf.write([spectrum_dict], filename)
    
class Document:
    """
    Use this as parent class to build your own document class. An example used for
    mass spectra is SpectrumDocument."""
    def __init__(self, obj):
        """
        Parameters
        ----------
        obj:
            Input object of desired class.
        """
        self._obj = obj
        self._index = 0
        self._make_words()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.words)

    def __next__(self):
        """gensim.models.Word2Vec() wants its corpus elements to be iterable"""
        if self._index < len(self.words):
            word = self.words[self._index]
            self._index += 1
            return word
        self._index = 0
        raise StopIteration

    def __str__(self):
        return self.words.__str__()

    def _make_words(self):
        print("You should override this method in your own subclass.")
        self.words = []
        return self

class SpectrumDocument(Document):
    """Create documents from spectra.

    Every peak (and loss) positions (m/z value) will be converted into a string "word".
    The entire list of all peak words forms a spectrum document. Peak words have
    the form "peak@100.32" (for n_decimals=2), and losses have the format "loss@100.32".
    Peaks with identical resulting strings will not be merged, hence same words can
    exist multiple times in a document (e.g. peaks at 100.31 and 100.29 would lead to
    two words "peak@100.3" when using n_decimals=1).

    """
    def __init__(self, spectrum, n_decimals: int = 1):
        """

        Parameters
        ----------
        spectrum: SpectrumType
            Input spectrum.
        n_decimals
            Peak positions are converted to strings with n_decimal decimals.
            The default is 2, which would convert a peak at 100.387 into the
            word "peak@100.39".
        """
        self.n_decimals = n_decimals
        self.weights = None
        super().__init__(obj=spectrum)
        self._add_weights()

    def _make_words(self):
        """Create word from peaks ."""
        format_string = "{}@{:." + "{}".format(self.n_decimals) + "f}"
        peak_words = [format_string.format("peak", mz) for mz in self._obj.peaks.mz]

        self.words = peak_words
        return self
    
    def _add_weights(self):
        """Add peaks (and loss) intensities as weights."""
        assert self._obj.peaks.intensities.max() <= 1, "peak intensities not normalized"

        peak_intensities = self._obj.peaks.intensities.tolist()
        #peak_mz = self._obj.peaks.mz.tolist()
        #w = np.arange(len(peak_mz )) + 1

        self.weights = peak_intensities
        return self
     
    def get(self, key: str, default=None):
        """Retrieve value from Spectrum metadata dict.

        """
        assert not hasattr(self, key), "Key cannot be attribute of SpectrumDocument class"
        return self._obj.get(key, default)

    @property
    def metadata(self):
        """Return metadata of original spectrum."""
        return self._obj.metadata
    @property
    def peaks(self) -> Spikes:
        """Return peaks of original spectrum."""
        return self._obj.peaks


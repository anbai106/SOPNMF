import abc
import pandas as pd
from .utils import load_data, load_data_apply_mask
import os

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

class WorkFlow:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run(self):
        pass

class Input:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_x(self):
        pass

    @abc.abstractmethod
    def get_x_without_mask(self):
        pass

    @abc.abstractmethod
    def get_x_apply_mask(self):
        pass

class VB_Input(Input):
    """
    Class to read the input image
    """
    def __init__(self, participant_tsv, output_dir, verbose=False):
        self._participant_tsv = participant_tsv
        self._output_dir = output_dir
        self._verbose = verbose
        self._x = None
        self._y = None
        self._images = None

        ## check the participant_tsv & covariate_tsv, the header, the order of the columns, etc
        self._df = pd.read_csv(participant_tsv, sep='\t')
        if ('participant_id' not in list(self._df.columns.values)) or (
                'session_id' not in list(self._df.columns.values)) or \
                ('path' not in list(self._df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id', 'path']")
        self._images = list(self._df['path'])

    def get_x(self):
        """
        Load all images.
        :return:
        """
        self._x, self._orig_shape, self._data_mask = load_data(self._images, verbose=self._verbose, mask=True,)

        return self._x, self._orig_shape, self._data_mask

    def get_x_without_mask(self):
        """
        Load images without mask.
        :return:
        """
        self._x, self._orig_shape, self._data_mask = load_data(self._images, verbose=self._verbose, mask=False)
        return self._x, self._orig_shape, self._data_mask

    def get_x_apply_mask(self):
        """
        Apply the mask to data
        :return:
        """
        predefined_mask_path = os.path.join(self._output_dir, 'NMF', 'data_mask.pickle')
        self._x, self._orig_shape = load_data_apply_mask(self._images, predefined_mask_path)

        return self._x, self._orig_shape






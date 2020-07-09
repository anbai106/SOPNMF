from .base import WorkFlow
from .utils import save_components_as_nifti, reconstruction_error, opnmf_solver, save_loading_coefficient, EarlyStopping
from .base import VB_Input
import os, shutil
import pickle
from multiprocessing.pool import ThreadPool
from tensorboardX import SummaryWriter

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

class VB_OPNMF(WorkFlow):
    """
    Class for running OPNMF with voxel-wise images.
    """

    def __init__(self, output_dir, participant_tsv, num_component_min, num_component_max, num_component_step,
                 init_method='NNDSVD', max_iter=50000, magnitude_tolerance=0, early_stopping_epoch=20, n_threads=8,
                 verbose=False):

        self._output_dir = output_dir
        self._participant_tsv = participant_tsv
        self._num_component_min = num_component_min
        self._num_component_max = num_component_max
        self._num_component_step = num_component_step
        self._init_method = init_method
        self._max_iter = max_iter
        self._magnitude_tolerance = magnitude_tolerance
        self._early_stopping_epoch = early_stopping_epoch
        self._n_threads = n_threads
        self._verbose = verbose

    def run(self):
        VB_data = VB_Input(self._participant_tsv, self._output_dir, self._verbose)
        ## X size is: num_subjects * num_features
        X, orig_shape, data_mask = VB_data.get_x()

        ## define the output structure
        tsv_path = os.path.join(self._output_dir, 'NMF')
        if not os.path.exists(tsv_path):
            os.makedirs(tsv_path)
        ## cp the participant tsv for recording
        shutil.copyfile(self._participant_tsv, os.path.join(tsv_path, 'participant.tsv'))

        ### save data mask for applying the model to unseen data.
        example_dict = {'mask': data_mask}
        pickle_out = open(os.path.join(self._output_dir, 'NMF', "data_mask.pickle"), "wb")
        pickle.dump(example_dict, pickle_out)
        pickle_out.close()

        async_result = {}
        c_list = list(range(self._num_component_min, self._num_component_max + self._num_component_step, self._num_component_step))
        for i in c_list:
            async_result[i] = {}

        log_dir = os.path.join(self._output_dir, "log_dir")
        print("Online monitoring the training, please run tensorboard --logdir LOG_DIR in your terminal : %s" % log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        metric_writer = SummaryWriter(log_dir=log_dir)

        ## apply the model from here with multithreads
        pool = ThreadPool(self._n_threads)
        for num_component in c_list:
            ### check if the model has been trained to be converged.
            if os.path.exists(os.path.join(self._output_dir, 'NMF', 'component_' + str(num_component), "nmf_model.pickle")):
                print("This number of components have been trained and converged: %d" % num_component)
            else:
                if self._verbose:
                    print("Train OPNMF for %d components" % num_component)
                async_result[num_component] = pool.apply_async(opnmf_solver, args=(X.transpose(), self._output_dir,
                                                                num_component, metric_writer, self._init_method,
                                                                self._max_iter, self._magnitude_tolerance,
                                                                self._early_stopping_epoch, self._verbose))
        pool.close()
        pool.join()

class Post_OPNMF(WorkFlow):
    """
    Class for post-analysis after training OPNMF model. Could be applied to:
    i) the training data itself;
    2) also unseen test data
    """

    def __init__(self, participant_tsv, output_dir, component_to_nii=False, extract_reconstruction_error=False, verbose=False):

        self._participant_tsv = participant_tsv
        self._output_dir = output_dir
        self._component_to_nii = component_to_nii
        self._extract_reconstruction_error = extract_reconstruction_error
        self._verbose = verbose

    def run(self):
        ## grab the data mask based on the training data
        data_mask_path = os.path.join(self._output_dir, 'NMF', 'data_mask.pickle')
        file = open(data_mask_path,'rb')
        data_mask = pickle.load(file)['mask']
        file.close()
        VB_data = VB_Input(self._participant_tsv, self._output_dir, self._verbose)
        X_with_mask, orig_shape = VB_data.get_x_apply_mask()
        X_without_mask, _, _ = VB_data.get_x_without_mask()

        if self._verbose:
            print("Data after applying mask: %s" % str(X_with_mask.shape))
            print("Data without masking: %s" % str(X_without_mask.shape))
       
        ## based on the output folder, check out the C ranges
        c_list = []
        for root, dirs, files in os.walk(self._output_dir, topdown=False):
            for c in dirs:
                if c.startswith('component_'):
                    c_list.append(int(c.split('component_')[1]))

        for num_component in c_list:
            if self._verbose:
                print("Apply OPNMF for %s components..." % num_component)

            if self._component_to_nii == True:
                ## convert the coefficient loading matrix back to the original image space and also save the factorization without mask
                save_components_as_nifti(X_without_mask.transpose(), VB_data._images[0], data_mask, orig_shape,
                                     self._output_dir, num_component)
            if self._extract_reconstruction_error == True:
                ## calculate the reconstruction error based on the masked image
                reconstruction_error(X_without_mask.transpose(), self._output_dir, num_component, data_mask)

            ## save the loading coefficient with masking.
            save_loading_coefficient(X_with_mask.transpose(), self._participant_tsv, self._output_dir, num_component)












from .base import WorkFlow
from .utils import save_components_as_nifti, reconstruction_error, train, validate, save_loading_coefficient, MRIDataset, \
    initialization_W, EarlyStopping, folder_not_exist_to_create
import os, shutil
import pickle
from tensorboardX import SummaryWriter
from .base import VB_Input
import numpy as np

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

class VB_OPNMF_mini_batch(WorkFlow):
    """
    Class for running OPNMF with voxel-wise images with mini-batch online learning.
    """

    def __init__(self, output_dir, participant_tsv, participant_tsv_max_memory, num_component_min, num_component_max,
                 num_component_step=1, batch_size=8, init_method='NNDSVD', max_epoch=100, early_stopping_epoch=10, n_threads=8,
                 verbose=False):

        self._output_dir = output_dir
        self._participant_tsv = participant_tsv
        self._participant_tsv_max_memory = participant_tsv_max_memory
        self._num_component_min = num_component_min
        self._num_component_max = num_component_max
        self._num_component_step = num_component_step
        self._batch_size = batch_size
        self._init_method = init_method
        self._max_epoch = max_epoch
        self._early_stopping_epoch = early_stopping_epoch
        self._n_threads = n_threads
        self._verbose = verbose

    def run(self):
        tsv_path = os.path.join(self._output_dir, 'NMF')
        log_dir = os.path.join(self._output_dir, "log_dir")
        folder_not_exist_to_create(tsv_path)
        folder_not_exist_to_create(log_dir)
        ## cp the participant tsv for recording
        shutil.copyfile(self._participant_tsv, os.path.join(tsv_path, 'participant.tsv'))
        VB_data = VB_Input(self._participant_tsv_max_memory, self._output_dir, self._verbose)
        ## X size is: num_subjects * num_features
        X_max, _, data_mask = VB_data.get_x()
        ### save data mask for applying the model to unseen data.
        mask_dict = {'mask': data_mask}
        pickle_out = open(os.path.join(self._output_dir, 'NMF', "data_mask.pickle"), "wb")
        pickle.dump(mask_dict, pickle_out)
        pickle_out.close()

        dataset = MRIDataset(self._participant_tsv, data_mask)
        metric_writer = SummaryWriter(log_dir=log_dir)
        c_list = list(range(self._num_component_min, self._num_component_max + self._num_component_step,
                            self._num_component_step))
        best_loss_valid = np.inf

        for num_component in c_list:
            ### check if the model has been trained to be converged.
            if os.path.exists(os.path.join(self._output_dir, 'NMF', 'component_' + str(num_component), "nmf_model.pickle")):
                print("This number of components have been trained and converged: %d" % num_component)
            else:
                # initialize the early stopping instance
                early_stopping = EarlyStopping('loss', min_delta=0, patience_epoch=self._early_stopping_epoch)
                component_path = os.path.join(self._output_dir, 'NMF', 'component_' + str(num_component))
                print("Train OPNMF for %d components" % num_component)
                ### check if the intermediate model exist, if yes, no need to initialize the W matrix
                if os.path.exists(os.path.join(component_path, "nmf_model_intermediate.pickle")):
                    intermediate_model_path = os.path.join(component_path, "nmf_model_intermediate.pickle")
                    file = open(intermediate_model_path, 'rb')
                    # dump information to that file
                    data = pickle.load(file)
                    W = data['W']
                    file.close()
                else:
                    ## initialization.
                    W = initialization_W(X_max.transpose(), self._init_method, num_component)

                for i in range(self._max_epoch):
                    W, num_iteration = train(W, dataset, self._batch_size, self._n_threads, i, self._output_dir,
                                                   num_component, metric_writer, verbose=self._verbose)

                    validate_loss = validate(W, dataset, self._batch_size, self._n_threads, num_iteration, metric_writer)

                    # save the best model based on the best loss
                    is_best = validate_loss < best_loss_valid
                    if is_best:
                        best_loss_valid = min(validate_loss, best_loss_valid)
                        data_dict = {'iter': num_iteration, 'num_component': num_component, 'W': W}
                        pickle_out = open(os.path.join(self._output_dir, 'NMF', 'component_' + str(num_component),
                                                       "nmf_model_intermediate.pickle"), "wb")
                        pickle.dump(data_dict, pickle_out)
                        pickle_out.close()

                    ## try early stopping criterion
                    if early_stopping.step(validate_loss) or i == self._max_epoch - 1:
                        print("By applying early stopping or at the last epoch defnied by user, the model should be stopped training at %d-th epoch" % i)
                        data_dict = {'iter': num_iteration, 'num_component': num_component, 'W': W}
                        pickle_out = open(os.path.join(self._output_dir, 'NMF', 'component_' + str(num_component), "nmf_model.pickle"), "wb")
                        pickle.dump(data_dict, pickle_out)
                        pickle_out.close()
                        ## remove the intermediate model to save space
                        os.remove(os.path.join(component_path, "nmf_model_intermediate.pickle"))
                        break

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












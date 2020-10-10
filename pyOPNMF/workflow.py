from .base import WorkFlow
from .utils import save_components_as_nifti, reconstruction_error, opnmf_solver, save_loading_coefficient, EarlyStopping, \
    folder_not_exist_to_create, initialization_W, train, validate, MRIDataset, extract_signal
from .base import VB_Input
import os, shutil
import pickle
from multiprocessing.pool import ThreadPool
from tensorboardX import SummaryWriter
import numpy as np

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
        ## define the output structure
        tsv_path = os.path.join(self._output_dir, 'NMF')
        folder_not_exist_to_create(tsv_path)
        ## cp the participant tsv for recording
        shutil.copyfile(self._participant_tsv, os.path.join(tsv_path, 'participant.tsv'))

        VB_data = VB_Input(self._participant_tsv, self._output_dir, self._verbose)
        ## X size is: num_subjects * num_features
        X, orig_shape, data_mask = VB_data.get_x()
        # rank_X = np.linalg.matrix_rank(X)

        ### save data mask for applying the model to unseen data.
        example_dict = {'mask': data_mask}
        pickle_out = open(os.path.join(self._output_dir, 'NMF', "data_mask.pickle"), "wb")
        pickle.dump(example_dict, pickle_out)
        pickle_out.close()

        async_result = {}
        c_list = list(range(self._num_component_min, self._num_component_max + self._num_component_step, self._num_component_step))
        for i in c_list:
            async_result[i] = {}

        ## apply the model from here with multithreads
        pool = ThreadPool(self._n_threads)
        for num_component in c_list:
            ## define log dir
            log_dir = os.path.join(self._output_dir, 'log_dir', 'component_' + str(num_component))
            folder_not_exist_to_create(log_dir)
            metric_writer = SummaryWriter(log_dir=log_dir)

            ## check if the number of components/rank is set correctly
            # if num_component >= rank_X or num_component == 1:
            #     print("The rank of the input matrix is %d" % rank_X)
            #     raise Exception("Number of components should be set correctly, smaller than the rank of the feature maxtrix")

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
        folder_not_exist_to_create(tsv_path)

        ## cp the participant tsv for recording
        shutil.copyfile(self._participant_tsv, os.path.join(tsv_path, 'participant.tsv'))
        VB_data = VB_Input(self._participant_tsv_max_memory, self._output_dir, self._verbose)
        ## X size is: num_subjects * num_features
        X_max, _, data_mask = VB_data.get_x()
        ## check if NAN or INF in X
        if np.isfinite(X_max).all() == False:
            raise Exception("The input matrix contains NAN or INF elements...")
        ### save data mask for applying the model to unseen data.
        mask_dict = {'mask': data_mask}
        pickle_out = open(os.path.join(self._output_dir, 'NMF', "data_mask.pickle"), "wb")
        pickle.dump(mask_dict, pickle_out)
        pickle_out.close()

        dataset = MRIDataset(self._participant_tsv, data_mask)
        c_list = list(range(self._num_component_min, self._num_component_max + self._num_component_step,
                            self._num_component_step))
        best_loss_valid = np.inf

        for num_component in c_list:
            ## define log dir
            log_dir = os.path.join(self._output_dir, 'log_dir', 'component_' + str(num_component))
            folder_not_exist_to_create(log_dir)
            metric_writer = SummaryWriter(log_dir=log_dir)

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

    def __init__(self, participant_tsv, output_dir, num_component, component_to_nii=True, extract_reconstruction_error=False, extract_mean_signal=False, verbose=False):

        self._participant_tsv = participant_tsv
        self._output_dir = output_dir
        self._num_component = num_component
        self._component_to_nii = component_to_nii
        self._extract_reconstruction_error = extract_reconstruction_error
        self._extract_mean_signal=extract_mean_signal
        self._verbose = verbose

    def run(self):
        ## grab the data mask based on the training data
        data_mask_path = os.path.join(self._output_dir, 'NMF', 'data_mask.pickle')
        file = open(data_mask_path, 'rb')
        data_mask = pickle.load(file)['mask']
        file.close()
        VB_data = VB_Input(self._participant_tsv, self._output_dir, self._verbose)
        X_with_mask, orig_shape = VB_data.get_x_apply_mask()
        X_without_mask, _, _ = VB_data.get_x_without_mask()

        if self._verbose:
            print("Data after applying mask: %s" % str(X_with_mask.shape))
            print("Data without masking: %s" % str(X_without_mask.shape))

        if self._verbose:
            print("Apply OPNMF for %s components..." % self._num_component)

        if self._component_to_nii == True:
            ## convert the coefficient loading matrix back to the original image space and also save the factorization without mask
            save_components_as_nifti(X_without_mask.transpose(), VB_data._images[0], data_mask, orig_shape,
                                 self._output_dir, self._num_component)
        if self._extract_reconstruction_error == True:
            ## calculate the reconstruction error based on the masked image
            reconstruction_error(X_without_mask.transpose(), self._output_dir, self._num_component, data_mask)

        ## save the loading coefficient with masking.
        save_loading_coefficient(X_with_mask.transpose(), self._participant_tsv, self._output_dir, self._num_component)
        if self._extract_mean_signal == True:
            ## extract other metrics in the original image space, such as brain volume, shape or texture features
            extract_signal(self._participant_tsv, self._output_dir, self._num_component, 0.3)










from .workflow import VB_OPNMF, VB_OPNMF_mini_batch

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def train(participant_tsv, output_dir, num_component_min, num_component_max, num_component_step=1, max_iter=50000,
          init_method='NNDSVDRandomSVD', magnitude_tolerance=0, early_stopping_epoch=20, n_threads=8, verbose=False):
    """
    Train OPNMF from voxel-wise image
    :param participant_tsv: str, path to the tsv containing the population information, insipred by BIDS convention. The tsv contains
        the following headers: "
                                 "i) the first column is the participant_id;"
                                 "ii) the second column should be the session_id;"
                                 "iii) the third column should be the path, image path for each subject"
    :param output_dir: str, path to the output folder
    :param num_component_min: int, the minimum number of components
    :param num_component_max: int, the maximum number of components
    :param num_component_step: int, default is 1
    :param max_iter: int, maximum number of iterations for convergence
    :param init_method: str, default is NNDSVDRandomSVD ['NNDSVDRandomSVD', 'Random']
    :param magnitude_tolerance: float, the tolerance of loss change magnitude
    :param early_stopping_epoch: int, the tolerance of number of bad epochs for early stopping
    :param n_threads: int, default is 8. The number of threads to run model in parallel.
    :param verbose: Bool, default is False. If the output message is verbose.
    :return:
    """

    ### For voxel approach
    print('Performing OPNMF for voxel-wise features...')
    # ## Here, semi-supervised clustering
    wf = VB_OPNMF(output_dir, participant_tsv, num_component_min, num_component_max, num_component_step, init_method,
                  max_iter, magnitude_tolerance, early_stopping_epoch, n_threads, verbose)

    wf.run()

    print('Finish...')

def train_mini_batch(participant_tsv, participant_tsv_max_memory, output_dir, num_component_min, num_component_max,
                     num_component_step=1, batch_size=200, max_epoch=50000, init_method='NNDSVDRandomSVD', early_stopping_epoch=100,
                     n_threads=8, verbose=False):
    """
    Train OPNMF from voxel-wise image, trained by mini-batch online learning
    :param participant_tsv: str, path to the tsv containing the population information, insipred by BIDS convention. The tsv contains
        the following headers: "
                                 "i) the first column is the participant_id;"
                                 "ii) the second column should be the session_id;"
                                 "iii) the third column should be the path, image path for each subject"
    :param participant_tsv_max_memory: str, path to the tsv containing the maximum number of sub-population for the creation
    of the mask and initialization of W matrix
    :param output_dir: str, path to the output folder
    :param num_component_min: int, the minimum number of components
    :param num_component_max: int, the maximum number of components
    :param num_component_step: int, default is 1
    :param batch_size: int, batch size for the stochastic Lagrangian update rules.
    :param max_epoch: int, maximum number of epochs for convergence
    :param init_method: str, default is NNDSVDRandomSVD ['NNDSVDRandomSVD', 'Random']
    :param early_stopping_epoch: the tolerance for validation loss not decreasing.
    :param n_threads: int, default is 8. The number of threads to run model in parallel.
    :param verbose: Bool, default is False. If the output message is verbose.
    :return:
    """

    ### For voxel approach
    print('Performing OPNMF for voxel-wise features...')
    # ## Here, semi-supervised clustering
    wf = VB_OPNMF_mini_batch(output_dir, participant_tsv, participant_tsv_max_memory, num_component_min, num_component_max,
                             num_component_step, batch_size, init_method, max_epoch, early_stopping_epoch, n_threads, verbose)

    wf.run()

    print('Finish...')
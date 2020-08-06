from .workflow import Post_OPNMF
import os

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"


def apply_to_training(output_dir, num_component, participant_tsv=None, verbose=False):
    """
    This is the function to apply the trained model to the training data itself.
    :param output_dir: str, path to the output folder
    :param verbose: Boolean, default is False

    :return:
    1) loading coefficient matrix;
    2) components to nifti images for visualization;
    3) reconstruction error for each C (number of components);
    """

    ### For voxel approach
    print('Performing postprocessing for OPNMF using voxel-wise features...')
    if participant_tsv == None:
        ## grab the training participant_tsv
        participant_tsv = os.path.join(output_dir, 'NMF', 'participant.tsv')
    wf = Post_OPNMF(participant_tsv, output_dir, num_component, component_to_nii=True, extract_reconstruction_error=True, verbose=verbose)

    wf.run()

    print('Finish...')

def apply_to_test(output_dir, num_component, participant_tsv, verbose=False):
    """
    This is the function to apply the trained model to unseen test data, to extract only loading coefficient matrix in tsv.
    :param participant_tsv: str, path to the participant tsv
    :param output_dir: str, path to the output folder
    :param verbose: Boolean, default is False

    :return:
    1) loading coefficient matrix
    """

    ### For voxel approach
    print('Apply OPNMF to unseen test data...')
    wf = Post_OPNMF(participant_tsv, output_dir, num_component, component_to_nii=False, extract_reconstruction_error=False, verbose=verbose)

    wf.run()

    print('Finish...')



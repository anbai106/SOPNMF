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


def apply_to_training(output_dir, num_component, mask_threshold=100, component_to_nii=True, extract_reconstruction_error=True,
                      verbose=False):
    """
    After the model converges, we extract the componets in original image space for visualization, create the opNMF component-based atlas,
    calculate the loading coefficient matrix and reconstruction error.
    :param output_dir: str, output directory to save the results
    :param num_component: int, number of components to extract
    :param mask_threshold: the threshold used to create the atlas. The threshold is defined to create the population-based tissue map mask,
                            in order to exclusively include voxels in the images.
    :param component_to_nii: save components in nii images and create the atlas
    :param extract_reconstruction_error: calculate the reconstruction errors
    :param verbose: Default is False
    :return:
    """

    ### For voxel approach
    print('Performing postprocessing for OPNMF using voxel-wise features...')
    participant_tsv = os.path.join(output_dir, 'NMF', 'participant.tsv')
    wf = Post_OPNMF(participant_tsv, output_dir, num_component, mask_threshold=mask_threshold, component_to_nii=component_to_nii,
                    extract_reconstruction_error=extract_reconstruction_error, verbose=verbose)

    wf.run()

    print('Finish...')

def apply_to_test(output_dir, num_component, participant_tsv, verbose=False):
    """
    Apply the trained model to external dataset. Both coefficient matrix and the signal based on the opNMF atlas will be extracted.
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



from pyOPNMF.opnmf import opnmf, sopnmf
from pyOPNMF.opnmf_post import apply_to_training

## train the model
# participant_tsv = '/run/user/1000/gvfs/sftp:host=cubic-login,user=wenju/cbica/home/wenju/Dataset/sopNMF_atlas/ISTAGING/tsv/final_population/sopNMF_final_population_train_pyopnmf.tsv'
# participant_tsv_max_memory = '/run/user/1000/gvfs/sftp:host=cubic-login,user=wenju/cbica/home/wenju/Dataset/sopNMF_atlas/ISTAGING/tsv/final_population/sopNMF_final_population_train_pyopnmf_initial_2000_subjecs.tsv'

# output_dir = '/run/user/1000/gvfs/sftp:host=cubic-login,user=wenju/cbica/home/wenju/Project/pyOPNMF/results/Phenom/regularOPNMF'
# output_dir = '/run/user/1000/gvfs/sftp:host=cubic-login,user=wenju/cbica/home/wenju/Project/pyOPNMF/results/Phenom/bs_16'
# output_dir = '/home/hao/test/pyOPNMF'
# sopnmf(participant_tsv, participant_tsv_max_memory, output_dir, 64, 64, early_stopping_epoch=100, batch_size=16, verbose=True)

# participant_tsv = '/run/user/1000/gvfs/sftp:host=cubic-login,user=wenju/cbica/home/wenju/Project/pyOPNMF/data/Phenom/participant_Phenom_GM_ubuntu.tsv'
# opnmf(participant_tsv, output_dir, 20, 20, early_stopping_epoch=100, verbose=True)

# ## apply the model to training data
output_dir = '/run/user/1000/gvfs/sftp:host=cubic-login,user=wenju/cbica/home/wenju/Project/pyOPNMF/results/soPNMF_atlas'
tissue_binary_mask = '/home/hao/Template_atlas/ISTAGING_3.5D/BLSA_SPGR+MPRAGE_averagetemplate_muse_seg_DS222_150.nii.gz'
apply_to_training(output_dir, 32, tissue_binary_mask, verbose=True)

# import pickle
# import numpy as np
# file = open('/home/hao/test/pyOPNMF/stochastic_opnmf/Genomic/NMF/component_128/nmf_model.pickle', 'rb')
# # dump information to that file
# data = pickle.load(file)
# W = data['W']
# # close the file
#
# n = W.size
# for i in range(128):
#     sparsity = np.divide(np.sqrt(n) - np.divide(np.sum(np.absolute(W[:, i])), np.sqrt(np.sum(np.square(W[:, i])))), np.sqrt(n) - 1)
#     print("For componenet : %d, the sparsity is : %f" % (i+1, sparsity))
#
# file.close()

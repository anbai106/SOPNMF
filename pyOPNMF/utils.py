import numpy as np
import os, math
import nibabel as nib
from scipy.sparse import issparse
from scipy.linalg import svd, norm, lu, qr
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment
import pandas as pd
import pickle, warnings
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from multiprocessing.pool import ThreadPool

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def opnmf_solver(X, output_dir, num_component, metric_writer, init_method, max_iter, magnitude_tolerance=0,
                 early_stopping_epoch=20, verbose=False, approach='voxel'):
    """
    This is the orthogonal projective NMF implementation.
    :param X: input data matrix, with size num_features * num_subject
    :param output_dir: path to save the output and intermediate files
    :param num_component: number of components to extract for NMF
    :param metric_writer: tensorboardX instance
    :param init_method: the initialization method,
    :param max_iter: maximum number of iterations
    :param magnitude_tolerance: float, the tolerance of loss change magnitude
    :param early_stopping_epoch: int, the tolerance of number of bad epochs for early stopping
    :param iter_intermediate: initial iteration (used when resuming optimization after
              possible failure - use in combination with saved intermediate results)
    :param approach: default is voxel-wise for high dimensional data, otherwise roi for low dimensional data. Depending on
           different approaches, the multiplicative update formulation could be slightly different for efficiency.
    :return:
            W: the factorizing matrix (D times K)
            H: expansion coefficients
    :reference: Sotiras, Aristeidis, Susan M. Resnick, and Christos Davatzikos. Finding imaging patterns of structural
    covariance via non-negative matrix factorization. NeuroImage 108 (2015): 1-16

    """
    ## define the saving frequency
    save_intermediate_step = 1000
    ## extract the shape of X
    num_features, num_subject = X.shape[0], X.shape[1]
    ## create the output_dir if not exist
    component_path = os.path.join(output_dir, 'NMF', 'component_' + str(num_component))
    if not os.path.exists(component_path):
        os.makedirs(component_path)

    ### check if the intermediate model has been saved
    if os.path.exists(os.path.join(output_dir, 'NMF', 'component_' + str(num_component), "nmf_model_intermediate.pickle")):
        intermediate_model_path = os.path.join(output_dir, 'NMF', 'component_' + str(num_component), "nmf_model_intermediate.pickle")
        file = open(intermediate_model_path, 'rb')
        # dump information to that file
        data = pickle.load(file)
        W = data['W']
        stop_iter = data['iter']

        # close the file
        file.close()
        print('Retrain the model from last stop point: %d ...' % stop_iter)
    else:
        print('Train the model from scratch...')
        W = initialization_W(X, init_method, num_component)

    # initialize the early stopping instance
    early_stopping = EarlyStopping('loss', min_delta=magnitude_tolerance, patience_epoch=early_stopping_epoch)

    ## start optimization
    for i in range(max_iter):
        now = time.time()
        W_old = W
        if approach == 'voxel':
            ## multiplicative update rule, the update rule is slightly modified to account for the high dimensionality of the imaging data
            W = np.divide(np.multiply(W, np.matmul(X, np.matmul(X.transpose(), W))), np.matmul(W, np.matmul(np.matmul(W.transpose(), X), np.matmul(X.transpose(), W))))
        else:
            XX = np.matmul(X, X.transpose())
            W = np.divide(np.multiply(W, np.matmul(XX, W)), np.matmul(W, np.matmul(np.matmul(W.transpose(), XX), W)))

        ## As the iterations were progressing, computational time per iteration was increasing due to operations involving really small values
        W[W < 1e-16] = 1e-16
        W = np.divide(W, norm(W, ord=2))

        ## difference after iteration
        diff_W = norm(W_old - W, 'fro') / norm(W_old, 'fro')
        loss = norm(X - np.matmul(W, np.matmul(W.transpose(), X)), ord='fro')
        ### the sparsity definition is referred from Hoyer 2004.
        n = W.size
        sparsity = np.divide(np.sqrt(n) - np.divide(np.sum(np.absolute(W)), np.sqrt(np.sum(np.square(W)))), np.sqrt(n) - 1)

        ## write to tensorboardX
        metric_writer.add_scalar('diff_W', diff_W, i)
        metric_writer.add_scalar('loss', loss, i)
        metric_writer.add_scalar('sparsity', sparsity, i)

        ## save intermediate results
        if i % save_intermediate_step == 0 and i != 0:
            data_dict = {'iter': i, 'num_component': num_component, 'W': W}
            pickle_out = open(os.path.join(component_path, "nmf_model_intermediate.pickle"), "wb")
            pickle.dump(data_dict, pickle_out)
            pickle_out.close()

        ## try early stopping criterion
        if early_stopping.step(loss) or i == max_iter - 1:
            print(
                "By applying early stopping or at the last epoch defnied by user, the model should be stopped training at %d-th epoch" % i)
            data_dict = {'iter': i, 'num_component': num_component, 'W': W}
            pickle_out = open(os.path.join(component_path, "nmf_model.pickle"), "wb")
            pickle.dump(data_dict, pickle_out)
            pickle_out.close()
            ## remove the intermediate model to save space
            os.remove(os.path.join(component_path, "nmf_model_intermediate.pickle"))
            break

        later = time.time()
        difference = later - now
        if verbose:
            print("Time used in iteration %d is %f \n" % (i, difference))

    ## reorganize the output
    H = np.matmul(W.transpose(), X)
    H_len = np.sqrt(np.sum(np.square(H), 1))

    if np.any(H_len == 0):
        warnings.warn('Having low rank')
        H_len[H_len == 0] = 1

    W_H_len = np.multiply(W, H_len.transpose())
    ## order by W by energy
    index_descend = (-np.sum(np.square(W_H_len), 0)).argsort()
    W_reordered = W[:, index_descend]
    H_reordered = np.matmul(W.transpose(), X)

    return W, H, W_reordered, H_reordered

def opnmf_solver_mini_batch(X, W, output_dir, num_component, num_iteration, metric_writer, verbose=False):
    """
    OPNMF solver for mini batch training
    :param X:
    :param W:
    :param output_dir:
    :param num_component:
    :param num_iteration:
    :param metric_writer:
    :param verbose:
    :return:
    """

    ## create the output_dir if not exist
    component_path = os.path.join(output_dir, 'NMF', 'component_' + str(num_component))
    if not os.path.exists(component_path):
        os.makedirs(component_path)
    ## update the model
    W_old = W

    ## multiplicative update rule, the update rule is slightly modified to account for the high dimensionality of the imaging data
    W = np.divide(np.multiply(W, np.matmul(X, np.matmul(X.transpose(), W))), np.matmul(W, np.matmul(np.matmul(W.transpose(), X), np.matmul(X.transpose(), W))))

    ## As the iterations were progressing, computational time per iteration was increasing due to operations involving really small values
    W[W < 1e-16] = 1e-16
    W = np.divide(W, norm(W, ord=2))
    ## difference after iteration
    diff_W = norm(W_old - W, 'fro') / norm(W_old, 'fro')
    ## sparcity of W
    ### the sparsity definition is referred from Hoyer 2004.
    n = W.size
    sparsity = np.divide(np.sqrt(n) - np.divide(np.sum(np.absolute(W)), np.sqrt(np.sum(np.square(W)))), np.sqrt(n) - 1)
    ## Mini-batch loss
    mini_batch_loss = norm(X - np.matmul(W, np.matmul(W.transpose(), X)), ord='fro')
    ## Display for TensorboardX
    metric_writer.add_scalar('diff_W', diff_W, num_iteration)
    metric_writer.add_scalar('sparsity', sparsity, num_iteration)
    metric_writer.add_scalar('mini_batch_loss', mini_batch_loss, num_iteration)
    if verbose:
        print("Iteration: %d ...\n" % num_iteration)
        print("W difference %f  ...\n" % diff_W)
        print("Mini-batch loss loss: %f  ...\n" % mini_batch_loss)
        print("Sparsity: %f  ...\n" % sparsity)

    return W

def train(W, dataset, batch_size, n_threads, num_epoch, output_dir, num_component, metric_writer, verbose=True):
    """
    Function to train the model in mini-batch mode
    :param W: numpy array
    :param dataset: a dataset instance of pytorch
    :param batch_size: int, batch size
    :param n_threads: int, number of cpus to use
    :param num_epoch: int, number of the current epoch
    :param output_dir: str, output dir
    :param num_component: int, number of C
    :param metric_writer: an instance of Tensorboard SummaryWriter
    :param num_iteration: int, depending on if the model is trained from scratch, or from intermediate step.
    :param verbose: bool.
    :return:
    """
    dataloader_train = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_threads,
                                  drop_last=True)

    t0 = time.time()
    ## apply the model from here with multithreads
    pool = ThreadPool(n_threads)
    for j, batch_data in enumerate(dataloader_train):
        t1 = time.time()
        print("Loading mini-batch data on CPU using time: ", t1 - t0)
        imgs_mini_batch = batch_data['image'].data.numpy()
        ## check if NAN or INF in X
        if np.isfinite(imgs_mini_batch).all() == False:
            raise Exception("The input matrix contains NAN or INF elements...")
        num_iteration_current = num_epoch * len(dataloader_train) + j

        results = pool.apply_async(opnmf_solver_mini_batch, args=(imgs_mini_batch.transpose(), W,
                                                                  output_dir, num_component,
                                                                  num_iteration_current, metric_writer, verbose))
        W = results.get()
    pool.close()
    pool.join()

    return W, num_iteration_current


def validate(W, dataset, batch_size, n_threads, num_iteration, metric_writer):
    """
    FUnction to validate the model with whole dataset
    :param W: numpy array
    :param dataset: a dataset instance of pytorch
    :param batch_size: int, batch size
    :param n_threads: int, number of cpus to use
    :param num_iteration: int, the current iteration at the end of each epoch
    :param metric_writer: an instance of Tensorboard SummaryWriter
    :return:
    """
    ### This is used for evaluate the reconstruction loss
    dataloader_valid = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=n_threads,
                                  drop_last=False)

    validate_loss_square = 0
    ## At the end of each epoch, evaluate the reconsruction based on all data.
    for k, batch_data in enumerate(dataloader_valid):
        imgs_mini_batch = batch_data['image'].data.numpy()
        ## check if NAN or INF in X
        if np.isfinite(imgs_mini_batch).all() == False:
            raise Exception("The input matrix contains NAN or INF elements...")
        mini_batch_loss_sqrt = np.sum(np.square(
            imgs_mini_batch.transpose() - np.matmul(W, np.matmul(W.transpose(), imgs_mini_batch.transpose()))))
        validate_loss_square += mini_batch_loss_sqrt
    validate_loss = np.sqrt(validate_loss_square)
    ## write to tensorboardX
    metric_writer.add_scalar('batch_loss', validate_loss, num_iteration)

    return validate_loss

class EarlyStopping(object):

    """
    This is a class to implement early stopping
    The criterion here:
        i) patience indicates how many epochs that the model could tolerate no loss decreasing
        ii) min_delta gives the amplitude of loss decreasing for each epoch
    """
    def __init__(self, mode='loss', min_delta=0, patience_epoch=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience_epoch
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience_epoch == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'loss'}:
            raise ValueError('mode ' + mode + ' is unknown!')

        if mode == 'loss':
            self.is_better = lambda a, best: a < best - best * min_delta

def non_negative_double_SVD(X, num_component, init_method):
    """
    This function implements the NNDSVD algorithm described in [1] for initialization of Nonnegative Matrix Factorization Algorithms.

    :param X: the input nonnegative num_features * num_subject matrix A
    :param num_component: the rank of the computed factors W,H
    :param init_method: indicates the variant of the NNDSVD Algorithm, NNDSVD, NNDSVDa, NNDSVDar or NNDSVD using random SVD calculation
    :return:
            W   : nonnegative num_features * num_component matrix
            H   : nonnegative num_component matrix x num_subject matrix
            Note, the notation here is slightly different from Aris's Neuroimage paper, where the components matrix (C)
            and loadling coefficient matrix (L) is transposed in the formulation
    :reference:
        [1] C. Boutsidis and E. Gallopoulos, SVD-based initialization: A head
        start for nonnegative matrix factorization, Pattern Recognition,
        Elsevier
    """
    ## check the non-negativity of the input matrix A
    if np.sum(np.isneginf(X)) != 0:
        raise Exception("The input matrix contains negative elements...")

    ## check if NAN or INF in X
    if np.isfinite(X).all() == False:
        raise Exception("The input matrix contains NAN or INF elements...")

    ## extract the size information
    num_features, num_subject = X.shape[0], X.shape[1]

    ## the matrices of the factorization
    W = np.zeros((num_features, num_component)).astype(X.dtype)
    H = np.zeros((num_component, num_subject)).astype(X.dtype)

    ## 1st SVD --> partial SVD rank-k to the input matrix A

    if init_method == 'NNDSVDRandomSVD':
        ## use random svd for efficient computation
        l = max(3 * num_component, 20)
        U, S, V = rand_pca(X, num_component, 8, l, raw=True)
        if np.isnan(U).any() or np.isinf(U).any() or np.isnan(S).any() or np.isinf(S).any() or np.isnan(V).any() or np.isinf(V).any():
            raise Exception("It can't be NAN or INF in initialization, something went wrong!")
    else:
        ## use standard matlab svn implementation
        ## Note, scipy svds implementation is slightly different from matlab svds function, the order of the largest singular values, for instance
        U, S, V = svds(X, k=num_component)
        # reverse S
        S = S[::-1]
        ## number of singular values
        num_singular = S.shape[0]
        S = np.diag(S)
        V = V.transpose()

        # reverse the n first columns of u
        U[:, :num_singular] = U[:, num_singular - 1::-1]
        # reverse the n first rows of vt
        V[:, :num_singular] = V[:, num_singular - 1::-1]


    ## choose the first singular triplet to be nonnegative
    W[:, 0] = np.sqrt(S[0, 0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0, 0]) * np.abs(V[:, 0].transpose())

    ## 2nd SVD for the other factors (see table 1 in our paper)
    for i in range(1, num_component):
        uu = U[:, i]
        vv = V[:, i]
        uup = neg_to_zero(uu)
        uun = pos_to_zero(uu)
        vvp = neg_to_zero(vv)
        vvn = pos_to_zero(vv)
        n_uup = norm(uup, ord=2)
        n_vvp = norm(vvp, ord=2)
        n_uun = norm(uun, ord=2)
        n_vvn = norm(vvn, ord=2)

        termp = n_uup * n_vvp
        termn = n_uun * n_vvn
        if (termp >= termn):
            W[:, i] = np.sqrt(S[i, i] * termp) * uup / n_uup
            H[i, :] = np.sqrt(S[i, i] * termp) * vvp.transpose() / n_vvp
        else:
            W[:, i] = np.sqrt(S[i, i] * termn) * uun / n_uun
            H[i, :] = np.sqrt(S[i, i] * termn) * vvn.transpose() / n_vvn

    ### set small value of elements to be 0
    ## Note, here potential bugs for NNDSVDa & NNDSVDar approach
    W[W < 0.0000000001] = 0.1
    H[H < 0.0000000001] = 0.1
    average = np.mean(X)

    ## NNDSVDa: fill in the zero elements with the average
    if init_method == 'NNDSVDa':
        W[W == 0] = average
        H[H == 0] = average
    ## NNDSVDar: fill in the zero elements with random values in the space [0:average/100]
    elif init_method == 'NNDSVDar':
        if np.sum(W == 0) == 0:
            pass
        else:
            W[W == 0] = average * np.random.rand(np.sum(W == 0), 1) / 100
            H[H == 0] = average * np.random.rand(np.sum(H == 0), 1) / 100
    else:
        pass

    return W, H

def neg_to_zero(A):
    """
    Set negative value to be 0
    :param A:
    :return:
           nonnegative A
    """
    X = A.clip(min=0)

    return X

def pos_to_zero(A):
    """
    Set positive value to be 0, and abs the negative elements
    :param A:
    :return:
           absolute value of A for negative elements
    """
    X = A.copy()
    if np.greater(X, 0).any():
        X[X > 0] = 0
    X = np.abs(X)
    return X

def rand_pca(X, num_component, its, l, raw=True):
    """
   The low-rank approximation USV' comes in the form of a singular
   value decomposition (SVD) -- the columns of U are orthonormal, as
   are the columns of V, the entries of S are all nonnegative, and all
   nonzero entries of S appear in non-increasing order on its diagonal.
   U is num_subject x num_component, V is num_features x num_component,
   and S is num_component x num_component, when X is num_subject x num_features.

   Increasing its or l improves the accuracy of the approximation USV';
   the reference below describes how the accuracy depends on its and l.
   Please note that even its=1 guarantees superb accuracy, whether or
   not there is any gap in the singular values of the matrix X being
   approximated, at least when measuring accuracy as the spectral norm
   ||X-USV'|| of X-USV' (relative to the spectral norm ||X|| of X).


   Note: PCA invokes RAND. To obtain repeatable results, reset the seed
         for the pseudorandom number generator.

   Note: The user may ascertain the accuracy of the approximation USV'
         to X by invoking DIFFSNORM(X,U,S,V), when raw is true.
         The user may ascertain the accuracy of the approximation USV'
         to C(X), where C(X) refers to X after centering its columns,
         by invoking DIFFSNORMC(X,U,S,V), when raw is false.

    :param X: matrix being approximated
    :param num_component: rank of the approximation being constructed;
    :param raw: Boolean, centers A when raw is false but does not when raw is true
    :param its: nonnegative int, number of normalized power iterations to conduct
    :param l: nonnegative int, block size of the normalized power iteration,
                must >= number of components
    :return:
       U -- num_subject x num_components matrix in the rank-k approximation USV' to X or C(X),
        where X is num_subject x num_features, and C(X) refers to X after centering its
        columns; the columns of U are orthonormal
       S -- num_components x num_components matrix in the rank-k approximation USV' to X or C(X),
            where X is num_subject x num_features, and C(X) refers to X after centering its
            columns; the entries of S are all nonnegative, and all nonzero
            entries appear in nonincreasing order on the diagonal
       V -- num_features x num_component matrix in the rank-k approximation USV' to X or C(X),
            where A is num_subject x num_features, and C(X) refers to A after centering its
            columns; the columns of V are orthonormal
    """
    num_features, num_subject = X.shape[0], X.shape[1]

    ## SVD X directly if l >= num_subject / 1.25 or l >= num_features / 1.25.
    if l >= num_subject / 1.25 or l >= num_features / 1.25:
        ## check the sparisity of X
        if not issparse(X):
            ## Note, scipy svd implementation is slightly different here with matlab svd economic approach svd.
            U, S, V = svd(X, full_matrices=False, lapack_driver='gesvd')
        else:
            ## need check if this is correct
            U, S, V = svd(X, lapack_driver='gesvd')

        S = np.diag(S)
        V = V.transpose()

    ## in most case, we have more num_subjects
    else:
        if num_subject < num_features:
            ## Apply X' to a random matrix, obtaining Q.
            if np.isreal(X).all():
                Q = np.matmul(X, 2 * (np.random.rand(num_subject, l) - np.ones((num_subject, l))))
                Q, R = lu(Q, permute_l=True)
            else:
                raise Exception('Neuroimage data should be real number...')

            ## Conduct normalized power iterations.
            for it in range(its):
                Q = np.matmul(Q.transpose(), X).transpose()
                Q, R = lu(Q, permute_l=True)
                Q = np.matmul(X, Q)

                if it < its - 1:
                    Q, R = lu(Q, permute_l=True)
                else:
                    Q, R, E = qr(Q, mode='economic', pivoting=True)

            R, S, V = svd(np.matmul(Q.transpose(), X), full_matrices=False, lapack_driver='gesvd')
            S = np.diag(S)
            V = V.transpose()
            U = np.matmul(Q, R)
        else:
            raise Exception('This formulataion is desinged for high dimensional data, in which case number of subjects is much smaller than the number of features...')


    ## Retain only the leftmost k columns of U, the leftmost k columns of V, and the uppermost leftmost k x k block of S.
    U = U[:, :num_component]
    V = V[:, :num_component]
    S = S[:num_component, :num_component]

    return U, S, V

def load_data(image_list, verbose=False, mask=True):
    """
    Load the image data with/without mask
    Args:
        image_list: a list containing paths to all image
        verbose: if output in verbose mode, default==False
        mask: if mask out the input data, defualt is True.

    Returns:

    """
    data = None
    shape = None
    data_mask = None
    first = True

    for i in range(len(image_list)):

        if verbose:
            print('Loading image: %s with masking \n' % image_list[i])
        subj = nib.load(image_list[i])
        subj_data = np.nan_to_num(subj.get_data(caching='unchanged'))
        shape = subj_data.shape
        ## change dtype to float32 to save memory, in case number of images is huge, consider downsample the image resolution.
        subj_data = subj_data.flatten().astype('float32')
        subj_data[subj_data < 0] = 0

        # Memory allocation for ndarray containing all data to avoid copying the array for each new subject
        if first:
            data = np.ndarray(shape=(len(image_list), subj_data.shape[0]), dtype='float32', order='C')
            first = False

        data[i, :] = subj_data

    if mask:
        data_mask = (np.not_equal(data, np.asarray([0]))).sum(axis=0) != 0
        data = data[:, data_mask]

    return data, shape, data_mask

def load_data_apply_mask(image_list, mask):
    """
    Load the image data with/without mask
    Args:
        image_list: a list containing paths to all image
        mask_path: path to predefined mask

    Returns:

    """
    data = None
    shape = None
    first = True

    for i in range(len(image_list)):
        subj = nib.load(image_list[i])
        subj_data = np.nan_to_num(subj.get_data(caching='unchanged'))
        shape = subj_data.shape
        ## change dtype to float32 to save memory, in case number of images is huge, consider downsample the image resolution.
        subj_data = subj_data.flatten().astype('float32')
        subj_data[subj_data < 0] = 0

        # Memory allocation for ndarray containing all data to avoid copying the array for each new subject
        if first:
            data = np.ndarray(shape=(len(image_list), subj_data.shape[0]), dtype='float32', order='C')
            first = False

        data[i, :] = subj_data
     
    file = open(mask, 'rb')
    mask = pickle.load(file)['mask']
    file.close()
    data = data[:, mask]

    return data, shape

def save_components_as_nifti(X, example_img, data_mask, orig_shape, output_dir, num_component, mask_threshold=100):
    """
    Map the coefficient H of NMF back to image space for spatial visualization, also create the opNMF atlas based on the
    extracted components. Note, the atlas was created based on thresholding each column of the coefficient maxtrix: mask_threshold.
    One should have prior knowledge of the input map's intensity values. For instance, RAVENS GM maps, we exclude the voxels
    whose's intensity values are lower than 100.

    :param X:
    :param example_img:
    :param data_mask:
    :param orig_shape:
    :param output_dir:
    :param num_component:
    :param mask_threshold:
    :return:
    """

    ## grab the saved model and extract the W & H
    model_path = os.path.join(output_dir, 'NMF', 'component_' + str(num_component), "nmf_model.pickle")

    file = open(model_path, 'rb')
    # dump information to that file
    data = pickle.load(file)
    W = data['W']
    # close the file
    file.close()

    W = W.transpose()

    #### convert W back to original image space, by adding background to 0 in the components
    B = np.zeros((orig_shape[0] * orig_shape[1] * orig_shape[2], W.shape[0]))

    for i in range(W.shape[0]):
        output_filename = os.path.join(os.path.join(output_dir, 'NMF', 'component_' + str(num_component)), 'component_' + str(i+1) + '_map.nii.gz')
        data = revert_mask(W[i, :], data_mask, orig_shape)
        B[:, i] = data.flatten().transpose()
        component_to_nifti(data, example_img, output_filename)

    ## save the components masked into one single mask image
    component_to_mask(W, data_mask, example_img, orig_shape, output_dir, num_component, mask_threshold)

    ## new loading coefficient
    C = np.matmul(B.transpose(), X)

    ## save the original B and C
    data_dict = {'B': B,
                 'C': C}
    pickle_out = open(os.path.join(output_dir, 'NMF', 'component_' + str(num_component), "nmf_model_without_masking.pickle"), "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()

def revert_mask(component, mask, shape):
    """
    Revert the created mask
    Args:
        component:
        mask:
        shape:

    Returns:

    """

    z = np.zeros(np.prod(shape))
    z[mask] = component

    new_component = np.reshape(z, shape)

    return new_component

def component_to_nifti(component, image, output_filename):
    """

    Args:
        component:
        image:
        output_filename:

    Returns:

    """
    # Normalize inf-norm
    features = component / abs(component).max()
    img = nib.load(image)
    img_affine = img.affine

    output_image = nib.Nifti1Image(features, img_affine)
    nib.save(output_image, output_filename)

def component_to_mask(W, data_mask, example_img, orig_shape, output_dir, num_component, mask_threshold):
    """
    Convert all component images in original space by converting them to one single mask
    :param C_original: components matrix in original space without
    :param example_img:
    :param orig_shape:
    :return:
    """
    ## turn the small values to be 0
    # W[W < 1e-3] = 0
    final_component = []
    ## convert W's elements per column the max value to its index of the row + 1, the other row values to be 0
    for i in range(W.shape[1]):
        if not np.any(W[:, i]):
            final_component.append(0)
        else:
            ## binarize the column, max to be its row index and the other to be 0.
            max_index = np.argmax(W[:, i]) + 1
            final_component.append(max_index)

    final_component = np.asarray(final_component)
    ## convert the binarized W into original image space
    final_component = revert_mask(final_component, data_mask, orig_shape)
    ## to nifti image
    img = nib.load(example_img)
    img_affine = img.affine
    img_data = img.get_data(caching='unchanged')
    ## created the tissue mask based on the intensity of the example images
    data_mask = np.ma.make_mask(img_data >= mask_threshold)
    final_component[~data_mask] = 0

    output_image = nib.Nifti1Image(final_component, img_affine)
    nib.save(output_image, os.path.join(os.path.join(output_dir, 'NMF', 'component_' + str(num_component)), 'components_mask.nii.gz'))

def reconstruction_error(X, output_dir, num_component, data_mask):
    """
    Compute the reconstruction error.
    :param X: num_feature * num_subject
    :param W: num_feature0 * num_component
    :param H: num_component * num_subj
    :param output_dir:
    :param num_component:
    :return:
    """
    ## grab the saved model and extract the W & H
    model_path = os.path.join(output_dir, 'NMF', 'component_' + str(num_component), "nmf_model_without_masking.pickle")

    file = open(model_path, 'rb')
    # dump information to that file
    data = pickle.load(file)
    B, C = data['B'], data['C']
    # close the file
    file.close()

    ## mask out the image
    X_masked = X[data_mask, :]
    B_masked = B[data_mask, :]

    rec_error = norm(X_masked - np.matmul(B_masked, C), 'fro')

    df = pd.DataFrame({'num_component': [num_component],
                       'reconstruction_error': [rec_error]})
    ## write to tsv file
    df.to_csv(os.path.join(os.path.join(output_dir, 'NMF', 'component_' + str(num_component)), 'reconstruction_error.tsv'), index=False, sep='\t', encoding='utf-8')

def save_loading_coefficient(X, participant_tsv, output_dir, num_component, suffix=None):
    """
    This is a function to save the H transpose matrix into tsv file for each image/participant
    :param imgage_list:
    :param output_dir:
    :param num_component:
    :return:
    """

    df_participant = pd.read_csv(participant_tsv, sep='\t')
    ## grab the saved model and extract the W & H
    model_path = os.path.join(output_dir, 'NMF', 'component_' + str(num_component), "nmf_model.pickle")

    file = open(model_path, 'rb')
    # dump information to that file
    data = pickle.load(file)
    B = data['W']
    # close the file
    file.close()

    ## normalized the B matrix
    B_normalized = np.divide(B, np.sum(B, axis=0))
    C = np.matmul(B_normalized.transpose(), X)

    header_list = ['component_' + str(i) for i in range(1, num_component + 1)]
    df_H = pd.DataFrame(data=C.transpose(), columns=header_list)

    ## concatenate the two dataframe
    df = pd.concat([df_participant, df_H], axis=1)

    ## write to tsv files.
    if suffix == None:
        df.to_csv(os.path.join(os.path.join(output_dir, 'NMF', 'component_' + str(num_component)), 'loading_coefficient.tsv'), index=False, sep='\t', encoding='utf-8')
    else:
        df.to_csv(os.path.join(os.path.join(output_dir, 'NMF', 'component_' + str(num_component)), 'loading_coefficient_' + suffix + '.tsv'), index=False, sep='\t', encoding='utf-8')

def initialization_W(X, init_method, num_component):
    """
    Initialize the W matrix
    :param X: matrix with size: num_features * num_subjects
    :param init_method:
    :param num_component:
    :return: W
    """
    if init_method == 'Random':
        print("random initialization ...")
        W = np.random.rand(X.shape[0], num_component)
    else:
        print('variant NNDSVD initialization ...')
        W, _ = non_negative_double_SVD(X, num_component, init_method)

    return W

class EarlyStopping(object):

    """
    This is a class to implement early stopping
    The criterion here:
        i) patience indicates how many epochs that the model could tolerate no loss decreasing
        ii) min_delta gives the amplitude of loss decreasing for each epoch
    """
    def __init__(self, mode='loss', min_delta=0, patience_epoch=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience_epoch
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience_epoch == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'loss'}:
            raise ValueError('mode ' + mode + ' is unknown!')

        if mode == 'loss':
            self.is_better = lambda a, best: a < best - best * min_delta

def calculate_reproducibility_index(opnmf_output_dir, num_component_min, num_component_max, num_component_step):
    """
    Calculate the reproducibility index after running split-sample strategy. Output the tsv file and the figure
    :param output_dir:
    :param num_component:
    #:return:

    Note, make sure you ran the two splits and put the result into the following structures:
        if not os.path.join(OPNMF_output, 'reproducibility_index', 'split_1'):
            os.makedirs(os.path.join(reproducibility_path, 'split_1'))

        if not os.path.join(OPNMF_output, 'reproducibility_index', 'split_1'):
            os.makedirs(os.path.join(reproducibility_path, 'split_2'))
    """
    for i in range(num_component_min, num_component_max + num_component_step, num_component_step):
        reproducibility_path_1 = os.path.join(opnmf_output_dir, 'reproducibility_index', 'split_1', 'NMF', 'component_' + str(i))
        reproducibility_path_2 = os.path.join(opnmf_output_dir, 'reproducibility_index', 'split_2', 'NMF', 'component_' + str(i))

        ## read the split results B1 and B2
        split_1_model = os.path.join(reproducibility_path_1, 'nmf_model_without_masking.pickle')
        split_2_model = os.path.join(reproducibility_path_2, 'nmf_model_without_masking.pickle')

        file = open(split_1_model, 'rb')
        # dump information to that file
        data = pickle.load(file)
        B_1 = data['B']
        # close the file
        file.close()

        file = open(split_2_model, 'rb')
        # dump information to that file
        data = pickle.load(file)
        B_2 = data['B']
        # close the file
        file.close()

        ## normalize B1 and B2
        wlen_1 = np.sqrt(np.sum(np.square(B_1), axis=0))
        wlen_1 = np.where(wlen_1 == 0, 1, wlen_1)
        wlen_2 = np.sqrt(np.sum(np.square(B_2), axis=0))
        wlen_2 = np.where(wlen_2 == 0, 1, wlen_2)

        W_1 = np.divide(B_1, wlen_1)
        W_2 = np.divide(B_2, wlen_2)

        ## Note, to make sure that H is the correct matrix to use for reproducibility index, not W
        inner_product = np.matmul(W_1.transpose(), W_2)

        ## take a distance
        dist = 2 * (1 - inner_product)

        ## hungarian algorithm
        _, index_hug1 = linear_sum_assignment(dist)

        ## overlap - hungarian
        overlap = np.zeros((wlen_1.shape[0],))
        for i in range(wlen_1.shape[0]):
            overlap[i] = inner_product[i, index_hug1[i]]

        ## calculate the mean and median inner-product to save
        mean_inner_product = np.mean(overlap)
        median_inner_product = np.median(overlap)

        df = pd.DataFrame({'num_component': [i],
                           'mean_index': [mean_inner_product],
                          'median_index': [median_inner_product]})
        ## write to tsv file
        df.to_csv(os.path.join(reproducibility_path_1, 'reproducibility_index.tsv'), index=False, sep='\t', encoding='utf-8')

class MRIDataset(Dataset):
    """Dataset of MRI"""

    def __init__(self, participant_tsv, mask):
        """
        :param participant_tsv: str, path to the participant tsv
        :param mask: predefined mask pickle file based on subpopulation for memory limitation
        """
        self._participant_tsv = participant_tsv
        self._mask = mask
        self._df = pd.read_csv(participant_tsv, sep='\t')
        if ('participant_id' not in list(self._df.columns.values)) or (
                'session_id' not in list(self._df.columns.values)) or \
                ('path' not in list(self._df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id', 'path']")

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        img_name = self._df.loc[idx, 'participant_id']
        sess_name = self._df.loc[idx, 'session_id']
        image_path = self._df.loc[idx, 'path']
        subj = nib.load(image_path)
        image = np.nan_to_num(subj.get_data(caching='unchanged')).flatten().astype('float32')
        image = image[self._mask] ## apply mask to single image, not to multiple image data[:, mask]
        sample = {'image': image, 'participant_id': img_name, 'session_id': sess_name}

        return sample

def folder_not_exist_to_create(path):
    """
    Check if folder exist, if not, create it
    :param path: str
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def extract_atlas_mean_signal(participant_tsv, output_dir, num_component):
    """
    This is a function to extract the sum of values in each ROI in the opNMF-atals.
    """
    df_participant = pd.read_csv(participant_tsv, sep='\t')
    paths = list(df_participant['path'])
    ## read the component image and create the masks
    component_path = os.path.join(output_dir, 'NMF', 'component_' + str(num_component), "components_mask.nii.gz")
    subj = nib.load(component_path)
    subj_data = np.nan_to_num(subj.get_data(caching='unchanged'))
    for i in range(1, num_component + 1):
        values = []
        # create the masks
        data_mask = np.ma.make_mask(subj_data == i)
        # num_voxels_mask = np.sum(data_mask)
        ## read the original image 
        for image in paths:
            data = nib.load(image)
            data = np.nan_to_num(data.get_data(caching='unchanged'))
            data[~data_mask] = 0
            # mean = np.divide(np.sum(data), num_voxels_mask)  ## note that RAVENS maps has been scaled by 1000, thus should be divided by 1000 if input is RAVENS maps
            mean_value = np.sum(data) / np.sum(data_mask)## note that RAVENS maps has been scaled by 1000, thus should be divided by 1000 if input is RAVENS maps
            if math.isnan(mean_value):
                break
            else:
                values.append(mean_value)
        if len(values) == df_participant.shape[0]:
            df_participant['component_' + str(i)] = values
        else:
            print("Component %d vanishes during opnmf-atlas creation..." % i)
    ## write to tsv files.
    df_participant.to_csv(os.path.join(os.path.join(output_dir, 'NMF', 'component_' + str(num_component)), 'atlas_components_signal.tsv'), index=False, sep='\t', encoding='utf-8')

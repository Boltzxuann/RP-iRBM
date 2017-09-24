
# RP-iRBM

A new training strategy to the infinite RBMs.

The key idea of the proposed training strategy is randomly regrouping the hidden units before each gradient descent step. 

Potentially, a mixing of infinite many iRBMs with different permutations of the hidden units can be achieved by this learning method. 

The original iRBM is also modified to be capable of carrying out discriminative training(Discriminative iRBM).

For more details, please see our paper at Arxiv. https://arxiv.org/abs/1709.03239

## Dependencies

Matlab 2016a or higher.


## Usage

Run `initiate.m` to set up the environment.

Run `demo_iRBM_density_estimation_MNIST.m` to train an iRBM using RP on the MNIST.

Run `demo_Dis_iRBM_classification_MNIST.m` to train a Dis-iRBM using RP on the MNIST.

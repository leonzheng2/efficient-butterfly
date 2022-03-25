# BSD 3-Clause License
#
# Copyright (c) 2022, ZHENG Leon
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from utils import partial_prod_butterfly_supports, product_of_factors


class Node:
    def __init__(self, low, high, num_factors):
        self.low = low
        self.high = high
        self.num_factors = num_factors
        self.left = None  # Empty left child
        self.right = None  # Empty right child
        self.support = partial_prod_butterfly_supports(num_factors, self.low, self.high)

    def __str__(self):
        return f"[{self.high}; {self.low}]"

    def is_leaf(self):
        return self.left is None and self.right is None


def generate_balanced_tree(low, high, num_factors):
    root = Node(low, high, num_factors)
    if low < high - 1:
        split_index = (low + high) // 2
        root.left = generate_balanced_tree(split_index, high, num_factors)
        root.right = generate_balanced_tree(low, split_index, num_factors)
    return root


def generate_unbalanced_tree(low, high, num_factors):
    root = Node(low, high, num_factors)
    if low < high - 1:
        split_index = high - 1
        root.left = generate_unbalanced_tree(split_index, high, num_factors)
        root.right = generate_unbalanced_tree(low, split_index, num_factors)
    return root


def project_B_model(matrix, tree_type, solver):
    num_factors = int(np.log2(matrix.shape[1]))
    root = eval(f"generate_{tree_type}_tree")(0, num_factors, num_factors)
    factors = tree_hierarchical_factorization(root, matrix, solver)
    product = product_of_factors(factors)
    return product, factors


def tree_hierarchical_factorization(root, A, solver):
    assert not root.is_leaf()
    X, Y = lifting_two_layers_factorization(root.left.support, root.right.support, A, solver)
    left_factors = [X] if root.left.is_leaf() else tree_hierarchical_factorization(root.left, X, solver)
    right_factors = [Y] if root.right.is_leaf() else tree_hierarchical_factorization(root.right, Y, solver)
    return left_factors + right_factors


def lifting_two_layers_factorization(support1, support2, A, solver):
    assert support1.shape[1] == support2.shape[0]
    dtype = np.complex128 if np.iscomplex(A).any() else np.float64
    X = np.zeros(support1.shape, dtype=dtype)
    Y = np.zeros(support2.shape, dtype=dtype)
    r = support1.shape[1]
    for t in range(r):
        rows = np.where(support1[:, t])[0]
        cols = np.where(support2[t, :])[0]
        subA = A[np.ix_(rows, cols)]
        u, v = best_rank1_approximation(subA, solver)
        X[rows, t] = np.squeeze(u)
        Y[t, cols] = np.squeeze(v)
    return X, Y


def best_rank1_approximation(A, solver):
    assert solver in ["lapack", "propack", "arpack", "lobpcg"]
    if solver == "lapack":
        u, s, vh = scipy.linalg.svd(A)
    elif solver in ["propack", "arpack", "lobpcg"]:
        u, s, vh = scipy.sparse.linalg.svds(A, k=1, solver=solver, maxiter=100)
    else:
        raise NotImplementedError
    sqrt_singular_val = np.sqrt(s[0])
    return sqrt_singular_val * u[:, 0], sqrt_singular_val * vh[0]

    # if svd == "partial":
    #     u, s, vh = scipy.sparse.linalg.svds(A, k=1, solver="propack", maxiter=100)
    #     sqrt_singular_val = np.sqrt(s[0])
    #     return sqrt_singular_val * u[:, 0], sqrt_singular_val * vh[0]
    # assert svd == "complete"
    # # We use propack to compute complete SVD when the matrix has only two rows or two columns
    # if min(A.shape) == 2:
    #     u, s, vh = scipy.sparse.linalg.svds(A, k=2, solver="propack", maxiter=100)
    #     i_max = np.argmax(s)
    #     sqrt_singular_val = np.sqrt(s[i_max])
    #     return sqrt_singular_val * u[:, i_max], sqrt_singular_val * vh[i_max]
    # # Use lapack otherwise
    # u, s, vh = scipy.linalg.svd(A)
    # sqrt_singular_val = np.sqrt(s[0])
    # return sqrt_singular_val * u[:, 0], sqrt_singular_val * vh[0]

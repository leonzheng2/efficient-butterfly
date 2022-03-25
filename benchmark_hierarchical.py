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


import pandas as pd
import numpy as np
import scipy.linalg
import time
from pathlib import Path

from butterfly.factorization import project_B_model
from butterfly.utils import error_cal

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compare unbalanced vs. balanced hierarchical factorization, "
                                     "with different implementation of SVD")
    parser.add_argument("--min_num_factors", type=int, default=2)
    parser.add_argument("--max_num_factors", type=int, default=15,
                        help="Comparison of factorization time of a matrix of size N=2^J, "
                             "for J = min_num_factors, min_num_factors + 1, ..., max_num_factors")
    parser.add_argument("--repeat", type=int, default=3, help="Repeat several factorizations for each setting")
    parser.add_argument("--results_path", type=Path, default="./benchmark_hierarchical_results.csv",
                        help="Save results in .csv at a the given path")
    args = parser.parse_args()

    args.results_path.parent.mkdir(exist_ok=True, parents=True)

    columns = ["tree_type", "solver", "num_factors", "time", "rel_err"]
    results_df = pd.DataFrame(columns=columns)

    for num_factors in range(args.min_num_factors, args.max_num_factors + 1):
        n = 2 ** num_factors
        for _ in range(args.repeat):
            matrix = scipy.linalg.hadamard(n).astype(float) + 0.01 * np.random.randn(n, n)
            for tree_type in ["unbalanced", "balanced"]:
                for solver in ["propack", "arpack", "lobpcg", "lapack"]:
                    begin = time.time()
                    product, factors = project_B_model(matrix, tree_type, solver)
                    end = time.time()
                    running_time = end - begin
                    rel_err = error_cal(factors, matrix)

                    new_row = pd.DataFrame([[tree_type, solver, num_factors, running_time, rel_err]], columns=columns)
                    print(new_row)
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
                    results_df.to_csv(args.results_path)

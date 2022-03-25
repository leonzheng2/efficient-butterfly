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


import scipy.sparse.linalg
import scipy.linalg
import time
import numpy as np
import argparse
from pathlib import Path
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser("SVD benchmark for rectangle (2 x n**2/2) or square matrices (n x n).")
    parser.add_argument("--min_J", type=int, default=2)
    parser.add_argument("--max_J", type=int, default=8)
    parser.add_argument("--matrix_type", type=str, choices=["square", "rectangle"])
    parser.add_argument("--repeat", type=int, default=10, help="Repeat several factorizations for each setting")
    parser.add_argument("--results_path", type=Path, help="Save results in .csv at a the given path")
    args = parser.parse_args()

    args.results_path.parent.mkdir(exist_ok=True, parents=True)

    columns = ["matrix_type", "solver", "n", "time"]
    results_df = pd.DataFrame(columns=columns)

    solver_list = ["propack", "lapack", "arpack", "lobpcg"]

    for J in range(args.min_J, args.max_J + 1):
        n = 2**J
        for _ in range(args.repeat):
            if args.matrix_type == "rectangle":
                B = np.random.randn(2, n**2 // 2)
            elif args.matrix_type == "square":
                B = np.random.randn(n, n)
            else:
                raise NotImplementedError

            for solver in solver_list:
                if solver == "lapack":
                    start = time.time()
                    u2, s2, vh2 = scipy.linalg.svd(B)
                    running_time = time.time() - start
                else:
                    start = time.time()
                    u1, s1, vh1 = scipy.sparse.linalg.svds(B, k=1, solver=solver, maxiter=100)
                    running_time = time.time() - start

                new_row = pd.DataFrame([[args.matrix_type, solver, n, running_time]], columns=columns)
                print(new_row)
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                results_df.to_csv(args.results_path)

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


import argparse
from pathlib import Path
import scipy.linalg
import numpy as np
import pandas as pd
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Measuring the running time of matrix-vector multiplication, "
                                                 "with matrix of size N x N, N = 2**J with min_J <= J <= max_J")
    parser.add_argument("--min_J", type=int, default=2)
    parser.add_argument("--max_J", type=int, default=16)
    parser.add_argument("--repeat_matrix", type=int, default=10, help="Repeat several matrix realization")
    parser.add_argument("--repeat_vector", type=int, default=30, help="Repeat several vector realization")
    parser.add_argument("--results_path", type=Path)
    args = parser.parse_args()

    args.results_path.parent.mkdir(exist_ok=True, parents=True)

    columns = ["matrix_size", "repeat_vector", "time"]
    results_df = pd.DataFrame(columns=columns)

    for J in range(args.min_J, args.max_J + 1):
        n = 2 ** J
        for _ in range(args.repeat_matrix):
            matrix = scipy.linalg.hadamard(n).astype(float) + 0.01 * np.random.randn(n, n)
            time_array = []
            for _ in range(args.repeat_vector):
                vector = np.random.randn(n)
                begin = time.time()
                output = np.matmul(matrix, vector)
                time_array.append(time.time() - begin)

            new_row = pd.DataFrame([[n, args.repeat_vector, np.mean(time_array)]], columns=columns)
            print(new_row)
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            results_df.to_csv(args.results_path)

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.datasketches.vector.decomposition;

import org.apache.datasketches.vector.matrix.Matrix;

/**
 * Abstract class to wrap libraries to compute singular value
 * decompositions and related Matrix operations needed by Frequent
 * Directions.  May return as many singular values as exist, but other
 * operations will limit output to k dimensions.
 */
abstract class MatrixOps {

  // iterations for SISVD
  private static final int DEFAULT_NUM_ITER = 8;

  /**
   * Matrix dimensions
   */
  final int n_; // rows
  final int d_; // columns

  /**
   * Target number of dimensions
   */
  final int k_;

  /**
   * Singular value decomposition method to use
   */
  final SVDAlgo algo_;

  int numSISVDIter_;

  /**
   * Creates an empty MatrixOps object to support Frequent Directions matrix operations
   *
   * @param A Matrix of the required type and correct dimensions
   * @param algo Enum indicating method to use for SVD
   * @param k Target number of dimensions for results
   * @return an empty MatrixOps object
   */
  public static MatrixOps newInstance(final Matrix A, final SVDAlgo algo, final int k) {
    final int n = (int) A.getNumRows();
    final int d = (int) A.getNumColumns();

    final MatrixOps mo;

    switch (A.getMatrixType()) {
      case OJALGO:
        mo = new MatrixOpsImplOjAlgo(n, d, algo, k);
        break;

      default:
        throw new IllegalArgumentException("Unknown MatrixType: " + A.getMatrixType().toString());
    }

    if (algo == SVDAlgo.SISVD) {
      mo.setNumSISVDIter((int) Math.ceil(Math.log(d)));
    }
    return mo;
  }

  MatrixOps(final int n, final int d, final SVDAlgo algo, final int k) {
    // TODO: make these actual checks
    assert n > 0;
    assert d > 0;
    assert n < d;
    assert k > 0;
    assert k < n;

    n_ = n;
    d_ = d;
    algo_ = algo;
    k_ = k;

    numSISVDIter_ = DEFAULT_NUM_ITER;
  }

  /**
   * Computes and returns the singular values, in descending order. May modify the internal state
   * of this object.
   * @param A Matrix to decompose
   * @return Array of singular values
   */
  public double[] getSingularValues(final Matrix A) {
    svd(A, false);
    return getSingularValues();
  }

  /**
   * Returns pre-computed singular values (stored in descending order). Does not perform new computation.
   * @return Singular values from the last computation
   */
  abstract double[] getSingularValues();

  /**
   * Computes and returns the right singular vectors of A. May modify the internal state of this object.
   * @param A Matrix to decompose
   * @return Matrix of size d x k
   */
  public Matrix getVt(final Matrix A) {
    svd(A, true);
    return getVt();
  }

  /**
   * Returns pre-computed right singular vectors (row-wise?). Does not perform new computation.
   *
   * @return Matrix of size d x k
   */
  abstract Matrix getVt();

  /**
   * Performs a Frequent Directions rank reduction with the SVDAlgo used when obtaining the instance.
   * Modifies internal state, with results queried via getVt() and getSingularValues().
   * @return The amount of weight subtracted from the singular values
   */
  abstract double reduceRank(final Matrix A);

  /**
   * Returns Matrix object reconstructed using the provided singular value adjustment. Requires first
   * decomposing the matrix.
   * @param A Matrix to decompose and adjust
   * @param adjustment Amount by which to adjust the singular values
   * @return A new Matrix based on A with singular values adjusted by adjustment
   */
  abstract Matrix applyAdjustment(final Matrix A, final double adjustment);

  /**
   * Computes a singular value decomposition of the provided Matrix.
   *
   * @param A Matrix to decompose. Size must conform, and it may be overwritten on return. Pass a copy to
   *          avoid this.
   * @param computeVectors True to compute Vt, false if only need singular values/
   */
  abstract void svd(final Matrix A, final boolean computeVectors);

  void setNumSISVDIter(final int numSISVDIter) {
    numSISVDIter_ = numSISVDIter;
  }
}

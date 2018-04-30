/* Directly derived from LGPL'd Matrix Toolkit for Java:
 * https://github.com/fommil/matrix-toolkits-java/blob/master/src/main/java/no/uib/cipr/matrix/SVD.java
 */

package com.yahoo.sketches.vector.decomposition;

import com.yahoo.sketches.vector.matrix.Matrix;

/**
 * Computes singular value decompositions and related Matrix operations needed by Frequent Directions. May return as
 * many singular values as exist, but other operations will limit output to k dimensions.
 */
public abstract class MatrixOps {

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

    switch (A.getMatrixType()) {
      case OJALGO:
        //return new MatrixOpsImplOjAlgo(A, algo, k);
        return new MatrixOpsImplOjAlgo(n, d, algo, k);

      case MTJ:
        //return new MatrixOpsImplMTJ((MatrixImplMTJ) A, algo, k);
        return new MatrixOpsImplMTJ(n, d, algo, k);
    }

    throw new IllegalArgumentException("Unknown MatrixType: " + A.getMatrixType().toString());
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
   * Computes and returns the singular values, in descending order.
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
   * Computes and returns the right singular vectors of A.
   * @param A Matrix to decompose
   * @return Matrix of size d x k
   */
  public Matrix getVt(final Matrix A) {
    svd(A, true);
    return Matrix.wrap(getVt());
  }

  /**
   * Returns pre-computed right singular vectors (row-wise?). Does not perform new computation.
   *
   * @return Matrix of size d x k
   */
  abstract Matrix getVt();

  /**
   * Performs a Frequent Directions rank reduction with the SVDAlgo used when obtaining the instance. Modifies internal
   * state, with results queried via getVt() and getSingularValues().
   * @return The amount of weight subtracted from the singular values
   */
  abstract double reduceRank(final Matrix A);

  /**
   * Returns Matrix object reconstructed using the provided singular value adjustment. Requires first decomposing the
   * matrix.
   * @param A Matrix to decompose and adjust
   * @param adjustment Amount by which to adjust the singular values
   * @return A new Matrix based on A with singular values adjusted by adjustment
   */
  abstract Matrix applyAdjustment(final Matrix A, final double adjustment);

  /**
   * Computes a singular value decomposition of the provided Matrix.
   *
   * @param A Matrix to decompose. Size must conform, and it may be overwritten on return. Pass a copy to avoid this.
   * @param computeVectors True to compute Vt, false if only need singular values/
   * @return The current MatrixOps object
   */
  abstract MatrixOps svd(final Matrix A, final boolean computeVectors);

  void setNumSISVDIter(final int numSISVDIter) {
    numSISVDIter_ = numSISVDIter;
  }
}

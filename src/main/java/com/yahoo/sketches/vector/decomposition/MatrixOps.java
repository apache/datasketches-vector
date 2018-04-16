/* Directly derived from LGPL'd Matrix Toolkit for Java:
 * https://github.com/fommil/matrix-toolkits-java/blob/master/src/main/java/no/uib/cipr/matrix/SVD.java
 */
package com.yahoo.sketches.vector.decomposition;

import com.yahoo.sketches.vector.matrix.Matrix;
import com.yahoo.sketches.vector.matrix.MatrixImplMTJ;

/**
 * Computes singular value decompositions and related Matrix operations needed by Frequent Directions. May return as
 * many singular values as exist, but other operations will limit output to k dimensions.
 */
public abstract class MatrixOps {

  /**
   * Creates an empty MatrixOps object to support Frequent Directions matrix operations
   *
   * @param A Matrix of the required type and correct dimensions
   * @param algo Enum indicating method to use for SVD
   * @param k Target number of dimensions for results
   */
  public static MatrixOps newInstance(final Matrix A, final SVDAlgo algo, final int k) {
    switch (A.getMatrixType()) {
      //case OJALGO:
      //  return new MatrixOpsImplOjAlgo(A, algo, k);

      case MTJ:
        return new MatrixOpsImplMTJ((MatrixImplMTJ) A, algo, k);
    }

    throw new IllegalArgumentException("Unknown MatrixType: " + A.getMatrixType().toString());
  }

  /**
   * Returns the singular values (stored in descending order). Does not compute singular vectors.
   * @param sv Target vector to hold singular values
   */
  public abstract void getSingularValues(final double[] sv);

  /**
   * Returns the right singular vectors (row-wise?)
   *
   * @return Matrix of size k x k
   */
  public abstract Matrix getVt();

  /**
   * Performs a Frequent Directions rank reduction with the SVDAlgo used when obtaining the instance. Modifies internal
   * state, with results queried via getVt() and getSingularValues().
   * @return The amount of weight subtracted from the singular values
   */
  public abstract double reduceRank(final Matrix A);

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

}

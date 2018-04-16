/* Directly derived from LGPL'd Matrix Toolkit for Java:
 * https://github.com/fommil/matrix-toolkits-java/blob/master/src/main/java/no/uib/cipr/matrix/SVD.java
 */
package com.yahoo.sketches.vector.decomposition;

import com.github.fommil.netlib.LAPACK;
import com.yahoo.sketches.vector.matrix.Matrix;
import com.yahoo.sketches.vector.matrix.MatrixImplMTJ;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.NotConvergedException;
import org.netlib.util.intW;

/**
 * Computes singular value decompositions
 */
public class MatrixOpsImplMTJ extends MatrixOps {

  /**
   * Matrix dimension
   */
  private final int m, n;

  /**
   * Target number of dimensions
   */
  private final int k_;

  /**
   * The singular values
   */
  private final double[] sv_;

  /**
   * Singular vectors
   */
  private DenseMatrix Vt_;

  /**
   * Singular value decomposition method to use
   */
  private final SVDAlgo algo_;

  /**
   * Work arrays for full SVD
   */
  private double[] work;
  private int[] iwork;


  /**
   * Creates an empty MatrixOps
   *
   * @param A Matrix with target number of dimensions
   * @param algo SVD algorithm to apply
   * @param k Target number of dimensions for any reduction operations
   */
  MatrixOpsImplMTJ(final MatrixImplMTJ A, final SVDAlgo algo, final int k) {
    super();

    m = (int) A.getNumRows();
    n = (int) A.getNumColumns();

    assert m > 0;
    assert n > 0;
    assert k <= Math.min(m, n);

    this.algo_ = algo;

    // Allocate space for the decomposition
    sv_ = new double[Math.min(m, n)];
    Vt_ = null; // lazy allocation
    k_ = k;
  }

  @Override
  MatrixOps svd(final Matrix A, final boolean computeVectors) {
    if (A.getNumRows() != m) {
      throw new IllegalArgumentException("A.numRows() != m");
    } else if (A.getNumColumns() != n) {
      throw new IllegalArgumentException("A.numColumns() != n");
    }

    if (computeVectors && Vt_ == null) {
      Vt_ = new DenseMatrix(n, n);
    }

    if (work == null) {
      allocateWorkspace(computeVectors);
    }

    switch (algo_) {
      case FULL:
        return computeFullSVD(A, computeVectors);

      case SISVD:
        return computeSISVD(A, computeVectors);

      case SYM:
      default:
        throw new RuntimeException("SVDAlgo type not (yet?) supported: " + algo_.toString());
    }
  }

  @Override
  public Matrix getVt() {
    return MatrixImplMTJ.wrap(Vt_);
  }

  @Override
  public void getSingularValues(final Matrix A, final double[] sv) {
    if (sv.length != k_) {
      throw new IllegalArgumentException("Length of vector sv too small. Expected " + k_ + ", found " + sv.length);
    }

    svd(A, false);
    System.arraycopy(sv_, 0, sv, 0, sv_.length);
  }

  private void allocateWorkspace(final boolean wantVectors) {
    switch (algo_) {
      case FULL:
        allocateSpaceFullSVD(wantVectors);
        break;

      case SISVD:
        allocatespaceSISVD();
        break;

      case SYM:
      default:
        throw new RuntimeException("SVDAlgo type not (yet?) supported: " + algo_.toString());
    }
  }

  private void allocateSpaceFullSVD(final boolean vectors) {
    // Find workspace requirements
    iwork = new int[8 * Math.min(m, n)];

    // Query optimal workspace
    final double[] workSize = new double[1];
    final intW info = new intW(0);
    LAPACK.getInstance().dgesdd("O", m, n, new double[0],
            m, new double[0], new double[0], m,
            new double[0], n, workSize, -1, iwork, info);

    // Allocate workspace
    int lwork;
    if (info.val != 0) {
      if (vectors) {
        lwork = 3
                * Math.min(m, n)
                * Math.min(m, n)
                + Math.max(
                Math.max(m, n),
                4 * Math.min(m, n) * Math.min(m, n) + 4
                        * Math.min(m, n));
      } else {
        lwork = 3
                * Math.min(m, n)
                * Math.min(m, n)
                + Math.max(
                Math.max(m, n),
                5 * Math.min(m, n) * Math.min(m, n) + 4
                        * Math.min(m, n));
      }
    } else {
      lwork = (int) workSize[0];
    }

    lwork = Math.max(lwork, 1);
    work = new double[lwork];
  }

  private void allocatespaceSISVD() {

  }

  private MatrixOps computeFullSVD(final Matrix A, final boolean computeVectors) {
    final intW info = new intW(0);
    final String jobType = computeVectors ? "O" : "N";
    final DenseMatrix mtx = (DenseMatrix) A.getRawObject();
    LAPACK.getInstance().dgesdd(jobType, m, n, mtx.getData(),
            m, sv_, new double[0],
            m, computeVectors ? Vt_.getData() : new double[0],
            n, work, work.length, iwork, info);

    if (info.val > 0) {
      throw new RuntimeException("Did not converge after a maximum number of iterations");
    } else if (info.val < 0) {
      throw new IllegalArgumentException();
    }

    return this;
  }



}

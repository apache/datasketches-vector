/* Portions derived from LGPL'd Matrix Toolkit for Java:
 * https://github.com/fommil/matrix-toolkits-java/blob/master/src/main/java/no/uib/cipr/matrix/SVD.java
 */

package com.yahoo.sketches.vector.decomposition;

import java.util.concurrent.ThreadLocalRandom;

import org.netlib.util.intW;

import com.github.fommil.netlib.BLAS;
import com.github.fommil.netlib.LAPACK;
import com.yahoo.sketches.vector.matrix.Matrix;
import com.yahoo.sketches.vector.matrix.MatrixImplMTJ;
import com.yahoo.sketches.vector.matrix.MatrixType;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.NotConvergedException;
import no.uib.cipr.matrix.QR;
import no.uib.cipr.matrix.SVD;
import no.uib.cipr.matrix.SymmDenseEVD;
import no.uib.cipr.matrix.UpperSymmDenseMatrix;
import no.uib.cipr.matrix.sparse.CompDiagMatrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

/**
 * Computes singular value decompositions
 */
class MatrixOpsImplMTJ extends MatrixOps {

  /**
   * The singular values
   */
  private final double[] sv_;

  /**
   * Singular vectors, sparse version of singular value matrix
   */
  private DenseMatrix Vt_;
  private CompDiagMatrix S_;

  /**
   * Work arrays for full SVD
   */
  private double[] work_;
  private int[] iwork_;

  /**
   * Work arrays for SISVD
   */
  private DenseMatrix block_;
  private DenseMatrix T_;

  /**
   * Work objects for SymmEVD
   */
  private SymmDenseEVD evd_;
  private LinkedSparseMatrix rotS_;

  /**
   * Creates an empty MatrixOps
   *
   * @param n Number of rows in matrix
   * @param d Number of columns in matrix
   * @param algo SVD algorithm to apply
   * @param k Target number of dimensions for any reduction operations
   */
  //MatrixOpsImplMTJ(final MatrixImplMTJ A, final SVDAlgo algo, final int k) {
  MatrixOpsImplMTJ(final int n, final int d, final SVDAlgo algo, final int k) {
    super(n, d, algo, k);

    // Allocate space for the decomposition
    sv_ = new double[Math.min(n_, d_)];
    Vt_ = null; // lazy allocation
  }

  @Override
  void svd(final Matrix A, final boolean computeVectors) {
    assert A.getMatrixType() == MatrixType.MTJ;

    if (A.getNumRows() != n_) {
      throw new IllegalArgumentException("A.numRows() != n_");
    } else if (A.getNumColumns() != d_) {
      throw new IllegalArgumentException("A.numColumns() != d_");
    }

    if (computeVectors && Vt_ == null) {
      Vt_ = new DenseMatrix(n_, d_);

      final int[] diag = {0}; // only need the main diagonal
      S_ = new CompDiagMatrix(n_, n_, diag);
    }

    switch (algo_) {
      case FULL:
        // make a copy if not computing vectors to avoid changing the data
        final DenseMatrix mtx = computeVectors ? (DenseMatrix) A.getRawObject()
                : new DenseMatrix((DenseMatrix) A.getRawObject());
        computeFullSVD(mtx, computeVectors);
        return;

      case SISVD:
        computeSISVD((DenseMatrix) A.getRawObject(), computeVectors);
        return;

      case SYM:
        computeSymmEigSVD((DenseMatrix) A.getRawObject(), computeVectors);
        return;

      default:
        throw new RuntimeException("SVDAlgo type not (yet?) supported: " + algo_.toString());
    }
  }

  // Because exact SVD destroys A, need to reconstruct it for MTJ
  @Override
  public double[] getSingularValues(final Matrix A) {
    svd(A, false);
    return getSingularValues();
  }

  @Override
  public double[] getSingularValues() {
    return sv_;
  }

  @Override
  Matrix getVt() {
    return MatrixImplMTJ.wrap(Vt_);
  }

  @Override
  double reduceRank(final Matrix A) {
    svd(A, true);

    double svAdjustment = 0.0;
    S_.zero();

    if (sv_.length >= k_) {
      double medianSVSq = sv_[k_ - 1]; // (l_/2)th item, not yet squared
      medianSVSq *= medianSVSq;
      svAdjustment += medianSVSq; // always track, even if not using compensative mode
      for (int i = 0; i < k_ - 1; ++i) {
        final double val = sv_[i];
        final double adjSqSV = val * val - medianSVSq;
        S_.set(i, i, adjSqSV < 0 ? 0.0 : Math.sqrt(adjSqSV));
      }
      for (int i = k_ - 1; i < S_.numColumns(); ++i) {
        S_.set(i, i, 0.0);
      }
      //nextZeroRow_ = k_;
    } else {
      for (int i = 0; i < sv_.length; ++i) {
        S_.set(i, i, sv_[i]);
      }
      for (int i = sv_.length; i < S_.numColumns(); ++i) {
        S_.set(i, i, 0.0);
      }
      //nextZeroRow_ = sv_.length;
      throw new RuntimeException("Running with d < 2k not yet supported");
    }

    // store the result back in A
    S_.mult(Vt_, (DenseMatrix) A.getRawObject());

    return svAdjustment;
  }

  @Override
  Matrix applyAdjustment(final Matrix A, final double svAdjustment) {
    // copy A before decomposing
    final DenseMatrix result = new DenseMatrix((DenseMatrix) A.getRawObject(), true);
    svd(Matrix.wrap(result), true);

    for (int i = 0; i < k_ - 1; ++i) {
      final double val = sv_[i];
      final double adjSV = Math.sqrt(val * val + svAdjustment);
      S_.set(i, i, adjSV);
    }
    for (int i = k_ - 1; i < S_.numColumns(); ++i) {
      S_.set(i, i, 0.0);
    }

    S_.mult(Vt_, result);

    return Matrix.wrap(result);
  }

  private void allocateSpaceFullSVD(final boolean vectors) {
    // Find workspace requirements
    iwork_ = new int[8 * Math.min(n_, d_)];

    // Query optimal workspace
    final double[] workSize = new double[1];
    final intW info = new intW(0);
    LAPACK.getInstance().dgesdd("S", n_, d_, new double[0],
            n_, new double[0], new double[0], n_,
            new double[0], n_, workSize, -1, iwork_, info);

    // Allocate workspace
    int lwork;
    if (info.val != 0) {
      if (vectors) {
        lwork = 3
                * Math.min(n_, d_)
                * Math.min(n_, d_)
                + Math.max(
                Math.max(n_, d_),
                4 * Math.min(n_, d_) * Math.min(n_, d_) + 4
                        * Math.min(n_, d_));
      } else {
        lwork = 3
                * Math.min(n_, d_)
                * Math.min(n_, d_)
                + Math.max(
                Math.max(n_, d_),
                5 * Math.min(n_, d_) * Math.min(n_, d_) + 4
                        * Math.min(n_, d_));
      }
    } else {
      lwork = (int) workSize[0];
    }

    lwork = Math.max(lwork, 1);
    work_ = new double[lwork];
  }

  private void allocateSpaceSISVD() {
    block_ = new DenseMatrix(d_, k_);
    T_ = new DenseMatrix(n_, k_);
    // TODO: should allocate space for QR and final SVD here?
  }

  private void allocateSpaceSymmEigSVD() {
    T_ = new DenseMatrix(n_, n_);
    rotS_ = new LinkedSparseMatrix(n_, n_); // only need if computing vectors, but only O(n_) size
    evd_ = new SymmDenseEVD(n_, true, true);
  }

  private void computeFullSVD(final DenseMatrix A, final boolean computeVectors) {
    if (work_ == null) {
      allocateSpaceFullSVD(computeVectors);
    }

    final intW info = new intW(0);
    final String jobType = computeVectors ? "S" : "N";
    LAPACK.getInstance().dgesdd(jobType, n_, d_, A.getData(),
            n_, sv_, new double[0],
            n_, computeVectors ? Vt_.getData() : new double[0],
            n_, work_, work_.length, iwork_, info);

    if (info.val > 0) {
      throw new RuntimeException("Did not converge after a maximum number of iterations");
    } else if (info.val < 0) {
      throw new IllegalArgumentException();
    }
  }

  private void computeSISVD(final DenseMatrix A, final boolean computeVectors) {
    if (block_ == null) {
      allocateSpaceSISVD();
    }

    // want block_ filled as ~Normal(0,1))
    final ThreadLocalRandom rand = ThreadLocalRandom.current();
    for (MatrixEntry entry : block_) {
      entry.set(rand.nextGaussian());
    }
    // TODO: in-line QR with direct LAPACK call
    final QR qr = new QR(block_.numRows(), block_.numColumns());
    block_ = qr.factor(block_).getQ(); // important for numeric stability

    for (int i = 0; i < numSISVDIter_; ++i) {
      A.mult(block_, T_);
      A.transAmult(T_, block_);
      block_ = qr.factor(block_).getQ(); // again, for stability
    }

    // Rayleigh-Ritz postprocessing
    A.mult(block_, T_);

    // TODO: use LAPACK directly
    final SVD svd = new SVD(T_.numRows(), T_.numColumns(), computeVectors);
    try {
      svd.factor(T_);
    } catch (final NotConvergedException e) {
      throw new RuntimeException(e.getMessage());
    }
    System.arraycopy(svd.getS(), 0, sv_, 0, svd.getS().length); // sv_ is final

    if (computeVectors) {
      // V^T = (block * V^T)^T = (V^T)^T * block^T
      // using BLAS directly since Vt is (n_ x d_) but result here is only (k_ x d_)
      BLAS.getInstance().dgemm("T", "T", k_, d_, k_,
              1.0, svd.getVt().getData(), k_, block_.getData(), d_,
              0.0, Vt_.getData(), n_);
    }
  }

  private void computeSymmEigSVD(final DenseMatrix A, final boolean computeVectors) {
    if (T_ == null) {
      allocateSpaceSymmEigSVD();
    }

    // want left singular vectors U, aka eigenvectors of AA^T -- so compute that
    A.transBmult(A, T_);
    try {
      // TODO: direct LAPACK call lets us get only the top k values/vectors rather than all
      evd_.factor(new UpperSymmDenseMatrix(T_, false));
    } catch (final NotConvergedException e) {
      throw new RuntimeException(e.getMessage());
    }

    // TODO: can we only use k_ values?
    // EVD gives values low-to-high; SVD does high-to-low and we want that order. Reverse
    // the list when extracting SVs from eigenvalues, and generate a diagonal rotation matrix
    // to save on an extra matrix multiply if we need to compute vectors later.
    final double[] ev = evd_.getEigenvalues();
    for (int i = 0; i < ev.length; ++i) {
      final double val = Math.sqrt(ev[i]);
      sv_[n_ - i - 1] = val;
      if (val > 0) {
        rotS_.set(n_ - i - 1, i, 1 / val);
      }
    }

    if (computeVectors) {
      rotS_.transBmult(evd_.getEigenvectors(), T_);
      T_.mult(A, Vt_);
    }
  }

}

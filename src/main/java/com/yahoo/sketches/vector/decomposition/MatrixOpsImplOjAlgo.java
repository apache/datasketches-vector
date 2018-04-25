package com.yahoo.sketches.vector.decomposition;

import com.yahoo.sketches.vector.matrix.MatrixImplOjAlgo;
import org.ojalgo.matrix.decomposition.QR;
import org.ojalgo.matrix.decomposition.SingularValue;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
import org.ojalgo.matrix.store.SparseStore;
import org.ojalgo.random.Normal;

import com.yahoo.sketches.vector.matrix.Matrix;
import com.yahoo.sketches.vector.matrix.MatrixType;

public class MatrixOpsImplOjAlgo extends MatrixOps {
  //private SingularValue<Double> svd;
  private double[] sv_;
  private PrimitiveDenseStore Vt_;

  // work objects for SISVD
  private PrimitiveDenseStore block_;
  private PrimitiveDenseStore T_;
  private QR<Double> qr_;

  transient private SparseStore<Double> S_; // to hold singular value matrix

  MatrixOpsImplOjAlgo(final int n, final int d, final SVDAlgo algo, final int k) {
    super(n, d, algo, k);

    // Allocate space for the decomposition
    sv_ = new double[Math.min(n_, d_)];
    Vt_ = null; // lazy allocation
  }

  @Override
  MatrixOps svd(final Matrix A, final boolean computeVectors) {
    assert A.getMatrixType() == MatrixType.OJALGO;

    if (A.getNumRows() != n_) {
      throw new IllegalArgumentException("A.numRows() != n_");
    } else if (A.getNumColumns() != d_) {
      throw new IllegalArgumentException("A.numColumns() != d_");
    }

    if (computeVectors && Vt_ == null) {
      //Vt_ = PrimitiveDenseStore.FACTORY.makeZero(k_, d_);
      Vt_ = PrimitiveDenseStore.FACTORY.makeZero(n_, d_);
      S_ = SparseStore.makePrimitive(sv_.length, sv_.length);
    }

    switch (algo_) {
      case FULL:
        return computeFullSVD((PrimitiveDenseStore) A.getRawObject(), computeVectors);

      case SISVD:
        return computeSISVD((PrimitiveDenseStore) A.getRawObject(), computeVectors);

      case SYM:
      default:
        throw new RuntimeException("SVDAlgo type not (yet?) supported: " + algo_.toString());
    }
  }

  @Override
  double[] getSingularValues() {
    return sv_;
  }

  @Override
  Matrix getVt() {
    return MatrixImplOjAlgo.wrap(Vt_);
  }

  @Override
  double reduceRank(final Matrix A) {
    svd(A, true);

    double svAdjustment = 0.0;

    if (sv_.length >= k_) {
      double medianSVSq = sv_[k_ - 1]; // (l_/2)th item, not yet squared
      medianSVSq *= medianSVSq;
      svAdjustment += medianSVSq; // always track, even if not using compensative mode
      for (int i = 0; i < k_ - 1; ++i) {
        final double val = sv_[i];
        final double adjSqSV = val * val - medianSVSq;
        S_.set(i, i, adjSqSV < 0 ? 0.0 : Math.sqrt(adjSqSV));
      }
      for (int i = k_ - 1; i < S_.countColumns(); ++i) {
        S_.set(i, i, 0.0);
      }
      //nextZeroRow_ = k_;
    } else {
      for (int i = 0; i < sv_.length; ++i) {
        S_.set(i, i, sv_[i]);
      }
      for (int i = sv_.length; i < S_.countColumns(); ++i) {
        S_.set(i, i, 0.0);
      }
      //nextZeroRow_ = sv_.length;
      throw new RuntimeException("Running with d < 2k not yet supported");
    }

    // store the result back in A
    S_.multiply(Vt_).supplyTo((PrimitiveDenseStore) A.getRawObject());

    return svAdjustment;
  }

  @Override
  Matrix applyAdjustment(final Matrix A, final double svAdjustment) {
    // copy A before decomposing
    final PrimitiveDenseStore result = PrimitiveDenseStore.FACTORY.copy((PrimitiveDenseStore) A.getRawObject());
    svd(Matrix.wrap(result), true);

    for (int i = 0; i < k_ - 1; ++i) {
      final double val = sv_[i];
      final double adjSV = Math.sqrt(val * val + svAdjustment);
      S_.set(i, i, adjSV);
    }
    for (int i = k_ - 1; i < S_.countColumns(); ++i) {
      S_.set(i, i, 0.0);
    }

    S_.multiply(Vt_).supplyTo(result);

    return Matrix.wrap(result);
  }

  private MatrixOps computeFullSVD(final MatrixStore<Double> A, final boolean computeVectors) {
    final SingularValue<Double> svd = SingularValue.make(A);
    svd.compute(A);

    svd.getSingularValues(sv_);

    if (computeVectors) {
      svd.getQ2().transpose().supplyTo(Vt_);
    }

    return this;
  }

  private MatrixOps computeSISVD(final MatrixStore<Double> A, final boolean computeVectors) {
    // want to iterate on smaller dimension of A (n x d)
    // currently, error in constructor if d < n, so n is always the smaller dimension
    if (block_ == null) {
      block_ = PrimitiveDenseStore.FACTORY.makeFilled(d_, k_, new Normal(0.0, 1.0));
      qr_ = QR.PRIMITIVE.make(block_);
      T_ = PrimitiveDenseStore.FACTORY.makeZero(n_, k_);
    } else {
      block_.fillAll(new Normal(0.0, 1.0));
    }

    // orthogonalize for numeric stability
    qr_.decompose(block_);
    qr_.getQ().supplyTo(block_);

    for (int i = 0; i < DEFAULT_NUM_ITER; ++i) {
      A.multiply(block_).supplyTo(T_);
      A.transpose().multiply(T_).supplyTo(block_);

      // again, just for stability
      qr_.decompose(block_);
      qr_.getQ().supplyTo(block_);
    }

    // Rayleigh-Ritz postprocessing
    A.multiply(block_).supplyTo(T_);

    final SingularValue<Double> svd = SingularValue.make(T_);
    svd.compute(T_);

    svd.getSingularValues(sv_);

    if (computeVectors) {
      block_.multiply(svd.getQ2().transpose()).supplyTo(Vt_);
    }

    return this;
  }
}

package com.yahoo.sketches.vector.decomposition;

import com.yahoo.sketches.vector.matrix.Matrix;
import org.ojalgo.matrix.decomposition.QR;
import org.ojalgo.matrix.decomposition.SingularValue;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
import org.ojalgo.random.Normal;

public class MatrixOpsImplOjAlgo extends MatrixOps {
  private static final int DEFAULT_NUM_ITER = 8;

  private long nIter_;

  //private SingularValue<Double> svd;
  private double[] sv_;
  private MatrixStore<Double> Vt_;

  MatrixOpsImplOjAlgo(final Matrix A, final SVDAlgo algo) {
    super();
  }

  public static MatrixOpsImplOjAlgo(final MatrixStore<Double> A) {
    return make(A, DEFAULT_NUM_ITER);
    //return make(A, Math.min(A.countColumns(), A.countRows()) / 2);
  }

  public static MatrixOpsImplOjAlgo make(final MatrixStore<Double> A, final long numIter) {
    return new MatrixOpsImplOjAlgo(numIter);
  }

  //public MatrixStore<Double> getVt() {
  public Matrix getVt() {
    return Matrix.wrap(Vt_);
  }

  public void getSingularValues(final double[] values) {
    System.arraycopy(sv_, 0, values, 0, sv_.length);
  }

  @SuppressWarnings("unchecked")
  void computeFullSVD(final Matrix A, final int k) {
    final MatrixStore<Double> mtx = (MatrixStore<Double) A.getRawObject();
    final SingularValue<Double> svd = SingularValue.make(mtx);
    svd.compute(mtx);

    svd.getSingularValues(sv_);
    svd.getQ2().transpose().supplyTo(Vt_);
  }

  @SuppressWarnings("unchecked")
  void computeSISVD(final Matrix A, final int k) {
    if (k < 1) {
      throw new IllegalArgumentException("k must be a positive integer, found: " + k);
    }

    // want to iterate on smaller dimension of A (n x d)
    // currently, error in constructor if d < n, so n is always the smaller dimension
    final MatrixStore<Double> mtx = (MatrixStore<Double) A.getRawObject();

    final long d = mtx.countColumns();
    final long n = mtx.countRows();
    final PrimitiveDenseStore block = PrimitiveDenseStore.FACTORY.makeFilled(d, k, new Normal(0.0, 1.0));

    // orthogonalize for numeric stability
    final QR<Double> qr = QR.PRIMITIVE.make(block);
    qr.decompose(block);
    qr.getQ().supplyTo(block);

    final PrimitiveDenseStore T = PrimitiveDenseStore.FACTORY.makeZero(n, k);

    for (int i = 0; i < nIter_; ++i) {
      mtx.multiply(block).supplyTo(T);
      mtx.transpose().multiply(T).supplyTo(block);

      // again, just for stability
      qr.decompose(block);
      qr.getQ().supplyTo(block);
    }

    // Rayleigh-Ritz postprocessing
    mtx.multiply(block).supplyTo(T);

    final SingularValue<Double> svd = SingularValue.make(T);
    svd.compute(T);

    sv_ = new double[k];
    svd.getSingularValues(sv_);

    //block.multiply(svd.getQ2().transpose()).supplyTo(block);
    Vt_ = block.multiply(svd.getQ2()).transpose();
  }

  @SuppressWarnings("unchecked")
  void computeSymmetrizedSVD(final Matrix A, final int k) {

  }

}

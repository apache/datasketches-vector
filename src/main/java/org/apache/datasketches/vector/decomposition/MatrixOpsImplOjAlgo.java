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

import java.util.Optional;

import org.ojalgo.matrix.decomposition.Eigenvalue;
import org.ojalgo.matrix.decomposition.QR;
import org.ojalgo.matrix.decomposition.SingularValue;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.Primitive64Store;
import org.ojalgo.matrix.store.SparseStore;
import org.ojalgo.random.Normal;

import org.apache.datasketches.vector.matrix.Matrix;
import org.apache.datasketches.vector.matrix.MatrixType;

class MatrixOpsImplOjAlgo extends MatrixOps {
  private double[] sv_;
  private Primitive64Store Vt_;

  // work objects for SISVD
  private Primitive64Store block_;
  private Primitive64Store T_; // also used in SymmetricEVD
  private QR<Double> qr_;

  // work objects for Symmetric EVD
  private Eigenvalue<Double> evd_;

  // work object for full SVD
  private SingularValue<Double> svd_;

  transient private SparseStore<Double> S_; // to hold singular value matrix

  MatrixOpsImplOjAlgo(final int n, final int d, final SVDAlgo algo, final int k) {
    super(n, d, algo, k);

    // Allocate space for the decomposition
    sv_ = new double[Math.min(n_, d_)];
    Vt_ = null; // lazy allocation
  }

  @Override
  void svd(final Matrix A, final boolean computeVectors) {
    assert A.getMatrixType() == MatrixType.OJALGO;

    if (A.getNumRows() != n_) {
      throw new IllegalArgumentException("A.numRows() != n_");
    } else if (A.getNumColumns() != d_) {
      throw new IllegalArgumentException("A.numColumns() != d_");
    }

    if (computeVectors && (Vt_ == null)) {
      Vt_ = Primitive64Store.FACTORY.make(n_, d_);
      S_ = SparseStore.makePrimitive(sv_.length, sv_.length);
    }

    switch (algo_) {
      case FULL:
        computeFullSVD((Primitive64Store) A.getRawObject(), computeVectors);
        return;

      case SISVD:
        computeSISVD((Primitive64Store) A.getRawObject(), computeVectors);
        return;

      case SYM:
        computeSymmEigSVD((Primitive64Store) A.getRawObject(), computeVectors);
        return;

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
    return Matrix.wrap(Vt_);
  }

  @Override
  double reduceRank(final Matrix A) {
    svd(A, true);

    double svAdjustment = 0.0;

    if (sv_.length >= k_) {
      double medianSVSq = sv_[k_ - 1]; // (l_/2)th item, not yet squared
      medianSVSq *= medianSVSq;
      svAdjustment += medianSVSq; // always track, even if not using compensative mode
      for (int i = 0; i < (k_ - 1); ++i) {
        final double val = sv_[i];
        final double adjSqSV = (val * val) - medianSVSq;
        S_.set(i, i, adjSqSV < 0 ? 0.0 : Math.sqrt(adjSqSV)); // just to be safe
      }
      for (int i = k_ - 1; i < S_.countColumns(); ++i) {
        S_.set(i, i, 0.0);
      }
    } else {
      throw new RuntimeException("Running with d < 2k not (yet?) supported");
      /*
      for (int i = 0; i < sv_.length; ++i) {
        S_.set(i, i, sv_[i]);
      }
      for (int i = sv_.length; i < S_.countColumns(); ++i) {
        S_.set(i, i, 0.0);
      }
      */
    }

    // store the result back in A
    S_.multiply(Vt_, (Primitive64Store) A.getRawObject());

    return svAdjustment;
  }

  @Override
  Matrix applyAdjustment(final Matrix A, final double svAdjustment) {
    // copy A before decomposing
    final Primitive64Store result
            = Primitive64Store.FACTORY.copy((Primitive64Store) A.getRawObject());
    svd(Matrix.wrap(result), true);

    for (int i = 0; i < (k_ - 1); ++i) {
      final double val = sv_[i];
      final double adjSV = Math.sqrt((val * val) + svAdjustment);
      S_.set(i, i, adjSV);
    }
    for (int i = k_ - 1; i < S_.countColumns(); ++i) {
      S_.set(i, i, 0.0);
    }

    S_.multiply(Vt_, result);

    return Matrix.wrap(result);
  }

  private void computeFullSVD(final MatrixStore<Double> A, final boolean computeVectors) {
    if (svd_ == null) {
      svd_ = SingularValue.PRIMITIVE.make(A);
    }

    if (computeVectors) {
      svd_.decompose(A);
      svd_.getV().transpose().supplyTo(Vt_);
    } else {
      svd_.computeValuesOnly(A);
    }
    svd_.getSingularValues(sv_);
  }

  private void computeSISVD(final MatrixStore<Double> A, final boolean computeVectors) {
    // want to iterate on smaller dimension of A (n x d)
    // currently, error in constructor if d < n, so n is always the smaller dimension
    if (block_ == null) {
      block_ = Primitive64Store.FACTORY.makeFilled(d_, k_, new Normal(0.0, 1.0));
      qr_ = QR.PRIMITIVE.make(block_);
      T_ = Primitive64Store.FACTORY.make(n_, k_);
    } else {
      block_.fillAll(new Normal(0.0, 1.0));
    }

    // orthogonalize for numeric stability
    qr_.decompose(block_);
    qr_.getQ().supplyTo(block_);

    for (int i = 0; i < numSISVDIter_; ++i) {
      A.multiply(block_, T_);

      // again, just for stability
      qr_.decompose(T_.premultiply(A.transpose()));
      qr_.getQ().supplyTo(block_);
    }

    // Rayleigh-Ritz postprocessing

    final SingularValue<Double> svd = SingularValue.PRIMITIVE.make(T_);
    svd.compute(block_.premultiply(A));

    svd.getSingularValues(sv_);

    if (computeVectors) {
      // V = block * Q2^T so V^T = Q2 * block^T
      // and ojAlgo figures out that it only needs to fill the first k_ rows of Vt_
      svd.getV().multiply(block_.transpose()).supplyTo(Vt_);
    }
  }

  private void computeSymmEigSVD(final MatrixStore<Double> A, final boolean computeVectors) {
    if (evd_ == null) {
      evd_ = Eigenvalue.PRIMITIVE.make(n_, true);
    }

    // want left singular vectors U, aka eigenvectors of AA^T -- so compute that
    evd_.decompose(A.transpose().premultiply(A));

    // TODO: can we only use k_ values?
    final double[] ev = new double[n_];
    evd_.getEigenvalues(ev, Optional.empty());
    for (int i = 0; i < ev.length; ++i) {
      final double val = Math.sqrt(ev[i]);
      sv_[i] = val;
      if (computeVectors && (val > 0)) { S_.set(i, i, 1 / val); }
    }

    if (computeVectors) {
      S_.multiply(evd_.getV().transpose()).multiply(A, Vt_);
    }
  }
}

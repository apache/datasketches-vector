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

package org.apache.datasketches.vector.regression;

import org.apache.datasketches.vector.SketchesArgumentException;
import org.ojalgo.array.Array1D;
import org.ojalgo.function.constant.PrimitiveMath;
import org.ojalgo.matrix.decomposition.SingularValue;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.Primitive64Store;

import org.checkerframework.checker.nullness.qual.NonNull;
import org.ojalgo.matrix.store.SparseStore;

public class RobustRidgeRegression {
  private final int d_;
  private final int k_;
  private final int l_; // convenience value so we don't need to compute 2*k frequently

  private Primitive64Store xOffset_;
  private Primitive64Store xScale_;

  private final Primitive64Store B_;          // Sketch of the data
  private final Primitive64Store ATyAccum_;   // Accumulates A^T y - a d-dim vector
  private double intercept_;

  private double gamma_;
  private long n_;
  private int nextZeroRow_;

  private Primitive64Store weights_;

  // transient values for SVD
  private double[] sv_;
  private Primitive64Store Vt_;
  private SparseStore<Double> S_; // to hold singular value matrix
  private SingularValue<Double> svd_;
  //private Eigenvalue<Double> evd_;


  /**
   * Returns an object ready to accept data for Robust Frequent Directions Ridge Regression. The sketch size
   * parameter <tt>k</tt> controls the accuracy/size trade-off, but must be no greater than <tt>2 d</tt>.
   * @param gamma A nonnegative regularization coefficient
   * @param d The number of dimensions in each input vector
   * @param k The sketch size parameter
   */
  RobustRidgeRegression(final double gamma, final int d, final int k) {
    if (gamma < 0.0)
      throw new SketchesArgumentException("Gamma must be nonnegative. Found: " + gamma);
    if (k < 1)
      throw new SketchesArgumentException("k must be at least 1. Found: " + k);
    if (d < 2 * k)
      throw new SketchesArgumentException("d must be at least 2k. Found d=" + d + ", k=" + k);

    gamma_ = gamma;
    d_ = d;
    k_ = k;
    l_ = 2 * k;

    n_ = 0;
    nextZeroRow_ = 0;

    B_ = Primitive64Store.FACTORY.make(l_, d_);
    ATyAccum_ = Primitive64Store.FACTORY.make(d_, 1);
  }

  /**
   * Returns an object ready to accept data for Robust Frequent Directions Ridge Regression. Uses a default
   * sketch size of the maximum supported for a given value of <tt>d</tt>.
   * @param gamma A nonnegative regularization coefficient
   * @param d The number of dimensions in each input vector
   */
  RobustRidgeRegression(final double gamma, final int d) {
    this(gamma, d, d / 2);
  }

  /**
   * Initializes mean/variance normalization based on a VectorNormalizer object. The input normalizer
   * must match the configured dimensionality of the regression object.
   * @param normalizer A VectorNormalizer run on (a sample of) the data to be modeled.
   */
  public void setNormalization(@NonNull final VectorNormalizer normalizer) {
    if (normalizer.getD() != d_)
      throw new SketchesArgumentException("VectorNormalizer dimension must match configured dimensions. "
          + normalizer.getD() + " != " + d_);

    xOffset_ = Primitive64Store.wrap(normalizer.getMean());
    xScale_ = Primitive64Store.wrap(normalizer.getSampleVariance()); // variance, not std. deviation
    xScale_.modifyAll(PrimitiveMath.SQRT);
    intercept_ = normalizer.getIntercept();
  }

  /**
   * Initializes mean/variance normalization using specified arrays. Note that the second argument
   * is an array of <em>variance</em> values, not standard deviations. Both arrays must match the
   * configured dimensionality of the regression object.
   * @param means An array of mean values
   * @param variances An array of variance values
   */
  public void setNormalization(final double[] means, final double[] variances, final double intercept) {
    if (means == null || variances == null)
      throw new SketchesArgumentException("Mean and variance arrays cannot be null.");
    if (means.length != d_ || variances.length != d_)
      throw new SketchesArgumentException("Mean and variance arrays must be of length " + d_);

    xOffset_ = Primitive64Store.wrap(means.clone());
    xScale_ = Primitive64Store.wrap(variances.clone()); // variance, not std. deviation
    xScale_.modifyAll(PrimitiveMath.SQRT);
    intercept_ = intercept;
  }

  // add a single vector to the sketch
  public void update(final double[] data, final double target) {
    if (data == null || data.length != d_)
      throw new SketchesArgumentException("data must be a non-null vector of length " + d_);

    update(Primitive64Store.wrap(data, 1), Primitive64Store.wrap(target));
  }

  // add multiple vectors to the sketch
  public void update(@NonNull final Primitive64Store data, @NonNull final Primitive64Store targets) {
    if (data.countColumns() != d_)
      throw new SketchesArgumentException("data must have " + d_ + " columns. Found: " + data.countColumns());
    if (data.countRows() != targets.count())
      throw new SketchesArgumentException("number of rows in data (" + data.countRows() + ") does not match"
          + " number of targete (" + targets.count() + ")");

    // append rows to B_ until we have l_ of them, then reduce and adjust gamma
    for (int i = 0; i < data.countRows(); ++i) {
      Array1D<Double> row = data.sliceRow(i);

      if (nextZeroRow_ == l_) {
        reduceRank();
      }

      // accumulate values and copy row into B_, applying normalization if supplied
      // TODO: may be able to improve performance by normalizing data first?
      if (xOffset_ != null) {
        for (int j = 0; j < d_; ++j) {
          final double accumUpdate = ATyAccum_.get(j) + row.get(j) * (targets.get(i) - intercept_);
          ATyAccum_.set(j, accumUpdate);
          B_.set(i, j, (row.get(j) - xOffset_.get(j)) / xScale_.get(j));
        }
      } else {
        for (int j = 0; j < d_; ++j) {
          final double accumUpdate = ATyAccum_.get(j) + (row.get(j) * targets.get(i));
          ATyAccum_.set(j, accumUpdate);
          B_.set(i, j, row.get(j));
        }
      }

      ++n_;
      ++nextZeroRow_;
    }
  }

  public void merge(@NonNull final RobustRidgeRegression other) {
    // must match d, k (should be able to merge larger k into smaller?)
  }

  public double[] solve() {
    if (weights_ == null) {
      weights_ = Primitive64Store.FACTORY.make(1, d_);
    }

    // make sure any new data has contributed to V^T and singular values
    reduceRank();

    // TODO: seems there should be a modifyDiagonal() to apply this?
    final SparseStore<Double> invDiag = SparseStore.makePrimitive(d_, d_);
    for (int i = 0; i < sv_.length; ++i) {
      invDiag.set(i, i, 1.0/(sv_[i] * sv_[i] + gamma_));
    }

    MatrixStore<Double> firstTerm = (Vt_.transpose().multiply(invDiag)).multiply(Vt_.multiply(ATyAccum_));
    MatrixStore<Double> secondTerm = ATyAccum_.multiply(1.0 / gamma_);
    MatrixStore<Double> thirdTerm = Vt_.transpose().multiply(1.0 / gamma_).multiply( Vt_.multiply(ATyAccum_) );

    firstTerm.add(secondTerm).subtract(thirdTerm).supplyTo(weights_);

    return weights_.toRawCopy1D();
  }

  private void reduceRank() {
    if (nextZeroRow_ < k_) { return; }

    if (svd_ == null) {
      svd_ = SingularValue.PRIMITIVE.make(B_);
      sv_ = new double[l_];
      S_ = SparseStore.makePrimitive(sv_.length, sv_.length);
    }

    // full SVD
    // computes U and V matrices even though we only use the latter
    svd_.decompose(B_);
    svd_.getV().transpose().supplyTo(Vt_);
    svd_.getSingularValues(sv_);

    // zero-out singular values and update gamma_
    double medianSVSq = sv_[k_]; // (l_/2)th item, not yet squared
    medianSVSq *= medianSVSq;
    gamma_ += 0.5 * medianSVSq;
    for (int i = 0; i < (k_ - 1); ++i) {
      final double val = sv_[i];
      final double adjSqSV = (val * val) - medianSVSq;
      S_.set(i, i, adjSqSV < 0 ? 0.0 : Math.sqrt(adjSqSV)); // just to be safe
    }
    for (int i = k_; i < S_.countColumns(); ++i) {
      S_.set(i, i, 0.0);
    }

    // store the result back in B_
    S_.multiply(Vt_, B_);


    // update bookkeeping now
    nextZeroRow_ = (int) Math.min(k_ - 1, n_);
  }

  public static void main(String[] args) {
    RobustRidgeRegression rr = new RobustRidgeRegression(1.0, 10);
    rr.update(new double[10], 1.0);
    rr.solve();
  }
}

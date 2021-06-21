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

import org.apache.datasketches.vector.decomposition.FrequentDirections;
import org.apache.datasketches.vector.matrix.Matrix;
import org.ojalgo.function.PrimitiveFunction;
import org.ojalgo.function.aggregator.Aggregator;
import org.ojalgo.function.constant.PrimitiveMath;
import org.ojalgo.matrix.decomposition.SingularValue;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.Primitive64Store;
import org.ojalgo.matrix.store.SparseStore;

public class RidgeRegression {

  private final int k_;
  private final double gamma_;
  private final boolean useRobust_;

  private Primitive64Store xOffset_;
  private Primitive64Store xScale_;

  private Primitive64Store weights_;
  private double intercept_;

  private long n_;
  private int d_;


  public RidgeRegression(final int k, final double gamma, final boolean useRobust) {
    k_ = k;
    gamma_ = gamma;
    useRobust_ = useRobust;

    n_ = 0;
    d_ = 0;
  }

  public void fit(Matrix data, double[] targets) {
    fit(data, targets, false);
  }

  /**
   *
   * @param data an n x d data Matrix, with one input vector per row. MODIFIES INPUT DATA
   * @param targets an n-dimensional array of regression target values
   * @param exact if true, computes exact solution, otherwise an approximation
   * @return test error on the data set
   */
  public double fit(Matrix data, double[] targets, boolean exact) {
    n_ = data.getNumRows();
    d_ = (int) data.getNumColumns();

    // preallocate the structures we'll use
    xOffset_ = Primitive64Store.FACTORY.make(1, d_);
    xScale_ = Primitive64Store.FACTORY.make(1, d_);
    weights_ = Primitive64Store.FACTORY.make(1, d_);

    preprocessData(data, targets, true, true);
    solve(data, targets, exact);

    double[] predictions = predict(data, true);
    return getError(predictions, targets);
  }

  public double[] predict(final Matrix data) {
    return predict(data, false);
  }

  private double[] predict(final Matrix data, final boolean preNormalized) {
    if (data.getNumColumns() != d_)
      throw new RuntimeException("Input matrix for prediction must have " + d_ + " columns, found " + data.getNumColumns());

    final Primitive64Store mtx = (Primitive64Store) data.getRawObject();
    final MatrixStore<Double> rawPredictions;

    if (preNormalized) {
      rawPredictions = mtx.multiply(weights_);
    } else {
      Primitive64Store adjustedMtx = mtx.copy();
      adjustedMtx.modifyMatchingInRows(PrimitiveMath.SUBTRACT, xOffset_);
      adjustedMtx.modifyMatchingInRows(PrimitiveMath.DIVIDE, xScale_);
      rawPredictions = adjustedMtx.multiply(weights_);
    }

    rawPredictions.onAll(PrimitiveMath.ADD, intercept_);
    return rawPredictions.toRawCopy1D();
  }

  public double getError(final double[] y_pred, final double[] y_true) {
    if (y_pred.length != y_true.length)
      throw new RuntimeException("Predictions and true value vectors differ in length: "
          + y_pred.length + " != " + y_true.length);

    double cumSqErr = 0.0;
    for (int i = 0; i < y_pred.length; ++i) {
      double val = y_pred[i] - y_true[i];
      cumSqErr += val * val;
    }

    return Math.sqrt(cumSqErr) / Math.sqrt(y_pred.length);
  }

  public double[] getWeights() {
    return weights_.data.clone();
  }

  public double getIntercept() {
    return intercept_;
  }

  private void preprocessData(Matrix data, double[] targets, boolean fitIntercept, boolean normalize) {
    Primitive64Store mtx = (Primitive64Store) data.getRawObject();

    if (fitIntercept) {
      mtx.reduceColumns(Aggregator.AVERAGE).supplyTo(xOffset_);
      intercept_ = Primitive64Store.wrap(targets).aggregateAll(Aggregator.AVERAGE);

      // subtract xOffset from input matrix, yOffset from targets
      mtx.modifyMatchingInRows(PrimitiveMath.SUBTRACT, xOffset_);
      for (int r = 0; r < n_; ++r) {
        targets[r] -= intercept_;
      }

      if (normalize) {
        mtx.reduceColumns(Aggregator.NORM2).supplyTo(xScale_);
        // map any zeros to 1.0 and adjust from norm2 to stdev
        PrimitiveFunction.Unary fixZero = arg -> arg == 0.0 ? 1.0 : Math.sqrt(arg * arg / n_);
        xScale_.modifyAll(fixZero);
        mtx.modifyMatchingInRows(PrimitiveMath.DIVIDE, xScale_);
      } else {
        xScale_.fillAll(1.0);
      }
    } else {
      xOffset_.fillAll(0.0);
      xScale_.fillAll(1.0);
      intercept_ = 0.0;
    }
  }


  private void solve(Matrix data, double[] targets, boolean exact) {
    final Primitive64Store sketchMtx;
    final MatrixStore<Double> Vt;
    final double[] sv;
    final int nDim;
    if (exact) {
      nDim = d_;
      sketchMtx = (Primitive64Store) data.getRawObject();
      final SingularValue<Double> svd = SingularValue.PRIMITIVE.make(sketchMtx);
      svd.decompose(sketchMtx);
      sv = new double[nDim];
      svd.getSingularValues(sv);
      Vt = svd.getV().transpose();
    } else {
      final FrequentDirections fd = FrequentDirections.newInstance(k_, d_);
      for (int r = 0; r < data.getNumRows(); ++r) {
        fd.update(data.getRow(r));
      }
      fd.forceReduceRank();

      sv = fd.getSingularValues(useRobust_);
      Vt = (Primitive64Store) fd.getProjectionMatrix().getRawObject();
      nDim = (int) Vt.countRows();
    }

    final MatrixStore<Double> ATy = ((Primitive64Store) data.getRawObject()).transpose().multiply(Primitive64Store.wrap(targets));
    // TODO: seems there should be a modifyDiagonal() to apply this?
    final SparseStore<Double> invDiag = SparseStore.makePrimitive(nDim, nDim);
    for (int i = 0; i < sv.length; ++i) {
      invDiag.set(i, i, 1.0/(sv[i] * sv[i] + gamma_));
    }

    MatrixStore<Double> firstTerm = (Vt.transpose().multiply(invDiag)).multiply(Vt.multiply(ATy));
    MatrixStore<Double> secondTerm = ATy.multiply(1.0 / gamma_);
    MatrixStore<Double> thirdTerm = Vt.transpose().multiply(1.0 / gamma_).multiply( Vt.multiply(ATy) );

    firstTerm.add(secondTerm).subtract(thirdTerm).supplyTo(weights_);
  }
}

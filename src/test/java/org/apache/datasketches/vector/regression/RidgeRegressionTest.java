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

import static org.testng.Assert.assertEquals;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.apache.datasketches.vector.matrix.Matrix;
import org.testng.annotations.Test;

public class RidgeRegressionTest {

  @Test
  public void normalize() {
    final int nRows = 5;
    final int nCols = 2;
    final Matrix m = Matrix.builder().build(nRows, nCols);
    m.setElement(0, 0, 1);
    m.setElement(1, 0, 2);
    m.setElement(2, 0, 3);
    m.setElement(3, 0, 4);
    m.setElement(4, 0, 5);
    m.setElement(0, 1, 10);
    m.setElement(1, 1, 20);
    m.setElement(2, 1, 30);
    m.setElement(3, 1, 40);
    m.setElement(4, 1, 50);

    final double[] targets = new double[] {-1, 1, 0, -1, 0.5};

    //RidgeRegression rr = new RidgeRegression(5, 1.0, true);
    //rr.fit(m, targets);
  }

  @Test
  public void basicExactRegression() {
    final int nRows = 5;
    final int nCols = 2;
    final Matrix m = Matrix.builder().build(nRows, nCols);
    m.setElement(0, 0, 2);
    m.setElement(1, 0, 3);
    m.setElement(2, 0, 5);
    m.setElement(3, 0, 7);
    m.setElement(4, 0, 9);
    m.setColumn(1, new double[]{0,0,0,0,0});
    final double[] targets = new double[] {4, 5, 7, 10, 15};

    RidgeRegression rr = new RidgeRegression(5, 0.0, false);
    rr.fit(m, targets, true);
    System.out.println("Weights:");
    for (int i = 0; i < nCols; ++i) {
      System.out.println("\t" + i + ":\t" + rr.getWeights()[i]);
    }
    //System.out.println("Slope: " + rr.getWeights()[0]);
    System.out.println("Intercept: " + rr.getIntercept());
  }

  @Test
  public void YearDataTest() {
    final int nTrain = 16000;
    final int nValid = 4000;
    final int nTest = 5000;
    final String path = "/Users/jmalkin/projects/FrequentDirectionsRidgeRegression/notebooks/SongPredictions/";
    //Matrix fullTrain = loadTSVData(path + "years_train.tsv", nTrain);
    //Matrix fullTest = loadTSVData(path + "years_test.tsv", nTest);
    Matrix fullTrain = loadTSVData(path + "years_train.out", nTrain);
    Matrix fullValid = loadTSVData(path + "years_valid.out", nValid);
    Matrix fullTest = loadTSVData(path + "years_test.out", nTest);

    final int d = (int) fullTrain.getNumColumns() - 1;
    assertEquals(d, fullTest.getNumColumns() - 1);
    assertEquals(nTrain, fullTrain.getNumRows());
    assertEquals(nValid, fullValid.getNumRows());
    assertEquals(nTest, fullTest.getNumRows());

    // last column is targets
    double[] yTrain = fullTrain.getColumn(d);
    double[] yValid = fullValid.getColumn(d);
    double[] yTest = fullTest.getColumn(d);

    // grab the rest as training sets
    Matrix xTrain = Matrix.builder().build(nTrain, d);
    Matrix xValid = Matrix.builder().build(nValid, d);
    Matrix xTest = Matrix.builder().build(nTest, d);
    for (int i = 0; i < d; ++i) {
      xTrain.setColumn(i, fullTrain.getColumn(i));
      xValid.setColumn(i, fullValid.getColumn(i));
      xTest.setColumn(i, fullTest.getColumn(i));
    }

    RidgeRegression rr = new RidgeRegression(256, 10000.0, false);
    double error = rr.fit(xTrain, yTrain, true);
    System.out.print("[");
    for (final double w : rr.getWeights()) {
      System.out.print(w + "\t");
    }
    System.out.println("]");
    System.out.println("Intercept: " + rr.getIntercept());

    // (needlessly) computed as part of fit
    System.out.println("Train error: " + error);

    double[] pred = rr.predict(xValid);
    error = rr.getError(pred, yValid);
    System.out.println("Validation error: " + error);

    pred = rr.predict(xTest);
    error = rr.getError(pred, yTest);
    System.out.println("Test error: " + error);
  }

  Matrix loadTSVData(final String inputFile, final int nRows) {
    Matrix data = null;
    int row = 0;
    try (BufferedReader br = new BufferedReader(new FileReader(inputFile))) {
      String line;
      while ((line = br.readLine()) != null) {
        String[] strValues = line.split("\t");
        double[] values = new double[strValues.length];

        for (int d = 0; d < strValues.length; ++d)
          values[d] = Double.parseDouble(strValues[d]);

        if (data == null) {
          data = Matrix.builder().build(nRows, values.length);
        }
        data.setRow(row, values);
        ++row;
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    assertEquals(row, nRows);
    return data;
  }


}

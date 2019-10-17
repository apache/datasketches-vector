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

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.fail;

import org.testng.annotations.Test;

import org.apache.datasketches.vector.matrix.Matrix;
import org.apache.datasketches.vector.matrix.MatrixBuilder;
import org.apache.datasketches.vector.matrix.MatrixType;

@SuppressWarnings("javadoc")
public class MatrixOpsTest {

  @Test
  public void compareSVDAccuracy() {
    final int d = 10;
    final int k = 6;
    final Matrix input = generateIncreasingEye(d, 2 * k);

    final MatrixOps moFull  = MatrixOps.newInstance(input, SVDAlgo.FULL, k);
    final MatrixOps moSym   = MatrixOps.newInstance(input, SVDAlgo.SYM, k);
    final MatrixOps moSISVD = MatrixOps.newInstance(input, SVDAlgo.SISVD, k);
    moSISVD.setNumSISVDIter(50 * k); // intentionally run many extra iterations for tighter convegence

    // just singular values first
    moFull.svd(input, false);
    moSym.svd(input, false);
    moSISVD.svd(input, false);
    final double[] fullSv = moFull.getSingularValues();
    compareSingularValues(fullSv, moSym.getSingularValues(), fullSv.length);
    compareSingularValues(fullSv, moSISVD.getSingularValues(), k); // SISVD only produces k values

    // now with vectors
    moFull.svd(input, true);
    moSym.svd(input, true);
    moSISVD.svd(input, true);
    // TODO: better comparison is vector-wise, testing that sign changes are consistent but that
    // requires non-zero elements
    final Matrix fullVt = moFull.getVt();
    compareMatrixElementMagnitudes(fullVt, moSym.getVt(), (int) fullVt.getNumRows());
    compareMatrixElementMagnitudes(fullVt, moSISVD.getVt(), k); // SISVD only produces k vectors

    // just to be sure
    compareMatrixElementMagnitudes(fullVt, moFull.getVt(input), (int) fullVt.getNumRows());
  }

  @Test
  public void checkInvalidMatrixSize() {
    final int d = 10;
    final int k = 6;
    final Matrix A = generateIncreasingEye(d, 2 * k);
    final MatrixOps mo = MatrixOps.newInstance(A, SVDAlgo.FULL, k);

    Matrix B = generateIncreasingEye(d, (2 * k) + 1);
    try {
      mo.svd(B, true);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }

    B = generateIncreasingEye(d - 1, 2 * k);
    try {
      mo.svd(B, false);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }

  }

  private static void compareSingularValues(final double[] A, final double[] B, final int n) {
    assertEquals(A.length, B.length);

    for (int i = 0; i < n; ++i) {
      assertEquals(A[i], B[i], 1e-6);
    }
  }


  private static void compareMatrixElementMagnitudes(final Matrix A, final Matrix B, final int n) {
    assertEquals(A.getNumColumns(), B.getNumColumns());
    assertEquals(A.getNumRows(), B.getNumRows());

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < A.getNumColumns(); ++j) {
        assertEquals(Math.abs(A.getElement(i, j)), Math.abs(B.getElement(i, j)), 1e-6);
      }
    }
  }

  /**
   * Creates a scaled I matrix, where the diagonal consists of increasing integers,
   * starting with 1.0.
   * @param nRows number of rows
   * @param nCols number of columns
   * @return PrimitiveDenseStore, suitable for direct use or wrapping
   */
  private static Matrix generateIncreasingEye(final int nRows, final int nCols) {
    final Matrix m = new MatrixBuilder().setType(MatrixType.OJALGO).build(nRows, nCols);
    for (int i = 0; (i < nRows) && (i < nCols); ++i) {
      m.setElement(i, i, 1.0 + i);
    }
    return m;
  }

}

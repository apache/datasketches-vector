/*
 * Copyright 2017, Yahoo, Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.yahoo.sketches.vector.matrix;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.fail;

import org.testng.annotations.Test;

import com.yahoo.memory.Memory;
import com.yahoo.memory.WritableMemory;
import no.uib.cipr.matrix.DenseMatrix;

public class MatrixImplMTJTest {
  @Test
  public void checkInstantiation() {
    final int nRows = 10;
    final int nCols = 15;
    final Matrix m = MatrixImplMTJ.newInstance(nRows, nCols);
    assertEquals(m.getNumRows(), nRows);
    assertEquals(m.getNumColumns(), nCols);

    final DenseMatrix pds = (DenseMatrix) m.getRawObject();
    assertEquals(pds.numRows(), nRows);
    assertEquals(pds.numColumns(), nCols);

    final Matrix wrapped = Matrix.wrap(pds);
    MatrixTest.checkMatrixEquality(wrapped, m);
    assertEquals(wrapped.getRawObject(), pds);
  }

  @Test
  public void updateAndQueryValues() {
    final int nRows = 5;
    final int nCols = 5;
    final Matrix m = generateIncreasingEye(nRows, nCols); // tests setElement() in method

    for (int i = 0; i < nRows; ++i) {
      for (int j = 0; j < nCols; ++j) {
        final double val = m.getElement(i, j);
        if (i == j) {
          assertEquals(val, i + 1.0);
        } else {
          assertEquals(val, 0.0);
        }
      }
    }
  }

  @Test
  public void checkStandardSerialization() {
    final int nRows = 3;
    final int nCols = 7;
    final Matrix m = generateIncreasingEye(nRows, nCols);

    final byte[] mtxBytes = m.toByteArray();
    assertEquals(mtxBytes.length, m.getSizeBytes());

    final Memory mem = Memory.wrap(mtxBytes);
    final Matrix tgt = MatrixImplMTJ.heapifyInstance(mem);
    MatrixTest.checkMatrixEquality(tgt, m);
  }

  @Test
  public void checkCompactSerialization() {
    final int nRows = 4;
    final int nCols = 7;
    final Matrix m = generateIncreasingEye(nRows, nCols);

    byte[] mtxBytes = m.toCompactByteArray(nRows - 1, 7);
    assertEquals(mtxBytes.length, m.getCompactSizeBytes(nRows - 1, 7));

    Memory mem = Memory.wrap(mtxBytes);
    Matrix tgt = MatrixImplMTJ.heapifyInstance(mem);
    for (int c = 0; c < nCols; ++c) {
      for (int r = 0; r < (nRows - 1); ++r) {
        assertEquals(tgt.getElement(r, c), m.getElement(r, c)); // equal here
      }
      // assuming nRows - 1 so check only the last row as being 0
      assertEquals(tgt.getElement(nRows - 1, c), 0.0);
    }

    // test without compacting
    mtxBytes = m.toCompactByteArray(nRows, nCols);
    assertEquals(mtxBytes.length, m.getSizeBytes());
    mem = Memory.wrap(mtxBytes);
    tgt = MatrixImplMTJ.heapifyInstance(mem);
    MatrixTest.checkMatrixEquality(tgt, m);
  }

  @Test
  public void matrixRowOperations() {
    final int nRows = 7;
    final int nCols = 5;
    final Matrix m = generateIncreasingEye(nRows, nCols);

    final int tgtCol = 2;
    final double[] v = m.getRow(tgtCol); // diagonal matrix, so this works ok
    for (int i = 0; i < v.length; ++i) {
      assertEquals(v[i], (i == tgtCol ? i + 1.0 : 0.0));
    }

    assertEquals(m.getElement(6, tgtCol), 0.0);
    m.setRow(6, v);
    assertEquals(m.getElement(6, tgtCol), tgtCol + 1.0);
  }

  @Test
  public void matrixColumnOperations() {
    final int nRows = 9;
    final int nCols = 4;
    final Matrix m = generateIncreasingEye(nRows, nCols);

    final int tgtRow = 3;
    final double[] v = m.getColumn(tgtRow); // diagonal matrix, so this works ok
    for (int i = 0; i < v.length; ++i) {
      assertEquals(v[i], (i == tgtRow ? i + 1.0 : 0.0));
    }

    assertEquals(m.getElement(tgtRow, 0), 0.0);
    m.setColumn(0, v);
    assertEquals(m.getElement(tgtRow, 0), tgtRow + 1.0);
  }

  @Test
  public void invalidRowColumnOperations() {
    final int nRows = 9;
    final int nCols = 4;
    final Matrix m = generateIncreasingEye(nRows, nCols);

    final double[] shortRow = new double[nCols - 2];
    try {
      m.setRow(1, shortRow);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }

    final double[] longColumn = new double[nRows + 2];
    try {
      m.setColumn(1, longColumn);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void invalidSerVer() {
    final int nRows = 3;
    final int nCols = 3;
    final Matrix m = generateIncreasingEye(nRows, nCols);
    final byte[] sketchBytes = m.toByteArray();
    final WritableMemory mem = WritableMemory.wrap(sketchBytes);
    MatrixPreambleUtil.insertSerVer(mem.getArray(), mem.getCumulativeOffset(0L), 0);

    try {
      MatrixImplMTJ.heapifyInstance(mem);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void invalidFamily() {
    final int nRows = 3;
    final int nCols = 3;
    final Matrix m = generateIncreasingEye(nRows, nCols);
    final byte[] sketchBytes = m.toByteArray();
    final WritableMemory mem = WritableMemory.wrap(sketchBytes);
    MatrixPreambleUtil.insertFamilyID(mem.getArray(), mem.getCumulativeOffset(0L), 0);

    try {
      MatrixImplMTJ.heapifyInstance(mem);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void insufficientMemoryCapacity() {
    final byte[] bytes = new byte[6];
    final Memory mem = Memory.wrap(bytes);
    try {
      MatrixImplMTJ.heapifyInstance(mem);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
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
    final Matrix m = MatrixImplMTJ.newInstance(nRows, nCols);
    for (int i = 0; (i < nRows) && (i < nCols); ++i) {
      m.setElement(i, i, 1.0 + i);
    }
    return m;
  }
}

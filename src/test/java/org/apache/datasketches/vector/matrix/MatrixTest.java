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

package org.apache.datasketches.vector.matrix;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.fail;

import org.testng.annotations.Test;

import org.apache.datasketches.memory.Memory;
import org.apache.datasketches.memory.WritableMemory;
import org.apache.datasketches.vector.MatrixFamily;

@SuppressWarnings("javadoc")
public class MatrixTest {

  @Test
  public void checkHeapify() {
    final Matrix m = Matrix.builder().setType(MatrixType.OJALGO).build(3, 3);
    final byte[] bytes = m.toByteArray();
    final Memory mem = Memory.wrap(bytes);
    println(MatrixPreambleUtil.preambleToString(mem));

    final Matrix tgt = Matrix.heapify(mem, MatrixType.OJALGO);
    assertTrue(tgt instanceof MatrixImplOjAlgo);
    checkMatrixEquality(m, tgt);
  }

  @Test
  public void checkWrap() {
    assertNull(Matrix.wrap(null));

    final Matrix src = Matrix.builder().setType(MatrixType.OJALGO).build(3, 3);
    final Object obj = src.getRawObject();
    final Matrix tgt = Matrix.wrap(obj);
    assertTrue(tgt instanceof MatrixImplOjAlgo);
    checkMatrixEquality(src, tgt);

    try {
      final Object notAMatrix = 1.0;
      Matrix.wrap(notAMatrix);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }

    assertNotNull(src.toString());
  }

  @Test
  public void checkSize() {
    final int nRow = 7;
    final int nCol = 3;
    final Matrix m = Matrix.builder().build(nRow, nCol);

    int expectedSize = (MatrixFamily.MATRIX.getMinPreLongs() * Long.BYTES)
            + (nRow * nCol * Double.BYTES);
    assertEquals(m.getSizeBytes(), expectedSize);

    // this should redirect call to getSizeBytes()
    assertEquals(m.getCompactSizeBytes(nRow, nCol), expectedSize);

    // degenerate cases
    expectedSize = (MatrixFamily.MATRIX.getMinPreLongs() * Long.BYTES);
    assertEquals(m.getCompactSizeBytes(0, nCol), expectedSize);
    assertEquals(m.getCompactSizeBytes(nRow, 0), expectedSize);

    // matrix subsets
    expectedSize = (MatrixFamily.MATRIX.getMaxPreLongs() * Long.BYTES)
            + ((5 * 3) * Double.BYTES);
    assertEquals(m.getCompactSizeBytes(5, 3), expectedSize);

    expectedSize = (MatrixFamily.MATRIX.getMaxPreLongs() * Long.BYTES)
            + ((7 * 2) * Double.BYTES);
    assertEquals(m.getCompactSizeBytes(7, 2), expectedSize);

    expectedSize = (MatrixFamily.MATRIX.getMaxPreLongs() * Long.BYTES)
            + ((2 * 2) * Double.BYTES);
    assertEquals(m.getCompactSizeBytes(2, 2), expectedSize);
  }

  @Test
  public void invalidSerVer() {
    final int nRows = 50;
    final int nCols = 75;
    final MatrixBuilder mb = new MatrixBuilder();
    final Matrix m = mb.build(nRows, nCols);
    final byte[] sketchBytes = m.toByteArray();
    final WritableMemory mem = WritableMemory.wrap(sketchBytes);
    MatrixPreambleUtil.insertSerVer(mem.getArray(), mem.getCumulativeOffset(0L), 0);

    try {
      MatrixPreambleUtil.preambleToString(mem);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void invalidFamily() {
    final int nRows = 3;
    final int nCols = 3;
    final MatrixBuilder mb = new MatrixBuilder();
    final Matrix m = mb.build(nRows, nCols);
    final byte[] sketchBytes = m.toByteArray();
    final WritableMemory mem = WritableMemory.wrap(sketchBytes);
    MatrixPreambleUtil.insertFamilyID(mem.getArray(), mem.getCumulativeOffset(0L), 0);

    try {
      MatrixPreambleUtil.preambleToString(mem);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void checkInsufficientMemoryCapacity() {
    final byte[] bytes = new byte[6];
    final Memory mem = Memory.wrap(bytes);
    try {
      MatrixPreambleUtil.preambleToString(mem);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void checkCompactPreamble() {
    final int nRows = 4;
    final int nCols = 7;
    final MatrixBuilder mb = new MatrixBuilder();
    final Matrix m = mb.build(nRows, nCols);

    final Memory mem = Memory.wrap(m.toCompactByteArray(nRows - 1, 7));
    MatrixPreambleUtil.preambleToString(mem);
  }

  static void checkMatrixEquality(final Matrix m1, final Matrix m2) {
    assertEquals(m1.numRows_, m2.numRows_);
    assertEquals(m1.numCols_, m2.numCols_);

    for (int i = 0; i < m1.numRows_; ++i) {
      for (int j = 0; j < m1.numCols_; ++j) {
        assertEquals(m1.getElement(i, j), m2.getElement(i, j),
                "Mismatch at (" + i + ", " + j + ")");
      }
    }
  }

  /**
   * println the message
   * @param msg the message
   */
  static void println(final String msg) {
    //System.out.println(msg);
  }
}

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

import static org.apache.datasketches.vector.matrix.MatrixPreambleUtil.COMPACT_FLAG_MASK;
import static org.apache.datasketches.vector.matrix.MatrixPreambleUtil.extractFamilyID;
import static org.apache.datasketches.vector.matrix.MatrixPreambleUtil.extractFlags;
import static org.apache.datasketches.vector.matrix.MatrixPreambleUtil.extractNumColumns;
import static org.apache.datasketches.vector.matrix.MatrixPreambleUtil.extractNumColumnsUsed;
import static org.apache.datasketches.vector.matrix.MatrixPreambleUtil.extractNumRows;
import static org.apache.datasketches.vector.matrix.MatrixPreambleUtil.extractNumRowsUsed;
import static org.apache.datasketches.vector.matrix.MatrixPreambleUtil.extractPreLongs;
import static org.apache.datasketches.vector.matrix.MatrixPreambleUtil.extractSerVer;

import org.ojalgo.matrix.store.Primitive64Store;

import org.apache.datasketches.memory.Memory;
import org.apache.datasketches.memory.WritableMemory;
import org.apache.datasketches.vector.MatrixFamily;

/**
 * Implements the ojAlgo Matrix operations.
 */
public final class MatrixImplOjAlgo extends Matrix {
  private Primitive64Store mtx_;

  private MatrixImplOjAlgo(final int numRows, final int numCols) {
    mtx_ = Primitive64Store.FACTORY.make(numRows, numCols);
    numRows_ = numRows;
    numCols_ = numCols;
  }

  private MatrixImplOjAlgo(final Primitive64Store mtx) {
    mtx_ = mtx;
    numRows_ = (int) mtx.countRows();
    numCols_ = (int) mtx.countColumns();
  }

  static Matrix newInstance(final int numRows, final int numCols) {
    return new MatrixImplOjAlgo(numRows, numCols);
  }

  static Matrix heapifyInstance(final Memory srcMem) {
    final int minBytes = MatrixFamily.MATRIX.getMinPreLongs() * Long.BYTES;
    final long memCapBytes = srcMem.getCapacity();
    if (memCapBytes < minBytes) {
      throw new IllegalArgumentException("Source Memory too small: " + memCapBytes
              + " < " + minBytes);
    }

    final int preLongs = extractPreLongs(srcMem);
    final int serVer = extractSerVer(srcMem);
    final int familyID = extractFamilyID(srcMem);

    if (serVer != 1) {
      throw new IllegalArgumentException("Invalid SerVer reading srcMem. Expected 1, found: "
              + serVer);
    }
    if (familyID != MatrixFamily.MATRIX.getID()) {
      throw new IllegalArgumentException("srcMem does not point to a Matrix");
    }

    final int flags = extractFlags(srcMem);
    final boolean isCompact = (flags & COMPACT_FLAG_MASK) > 0;

    int nRows = extractNumRows(srcMem);
    int nCols = extractNumColumns(srcMem);

    final MatrixImplOjAlgo matrix = new MatrixImplOjAlgo(nRows, nCols);
    if (isCompact) {
      nRows = extractNumRowsUsed(srcMem);
      nCols = extractNumColumnsUsed(srcMem);
    }

    int memOffset = preLongs * Long.BYTES;
    for (int c = 0; c < nCols; ++c) {
      for (int r = 0; r < nRows; ++r) {
        matrix.mtx_.set(r, c, srcMem.getDouble(memOffset));
        memOffset += Double.BYTES;
      }
    }

    return matrix;
  }

  static Matrix wrap(final Primitive64Store mtx) {
    return new MatrixImplOjAlgo(mtx);
  }

  @Override
  public Object getRawObject() {
    return mtx_;
  }

  @Override
  public byte[] toByteArray() {
    final int preLongs = 2;
    final long numElements = mtx_.count();
    assert numElements == (mtx_.countColumns() * mtx_.countRows());

    final int outBytes = (int) (((long)preLongs * Long.BYTES) + (numElements * Double.BYTES));
    final byte[] outByteArr = new byte[outBytes];
    final WritableMemory memOut = WritableMemory.wrap(outByteArr);
    final Object memObj = memOut.getArray();
    final long memAddr = memOut.getCumulativeOffset(0L);

    MatrixPreambleUtil.insertPreLongs(memObj, memAddr, preLongs);
    MatrixPreambleUtil.insertSerVer(memObj, memAddr, MatrixPreambleUtil.SER_VER);
    MatrixPreambleUtil.insertFamilyID(memObj, memAddr, MatrixFamily.MATRIX.getID());
    MatrixPreambleUtil.insertFlags(memObj, memAddr, 0);
    MatrixPreambleUtil.insertNumRows(memObj, memAddr, (int) mtx_.countRows());
    MatrixPreambleUtil.insertNumColumns(memObj, memAddr, (int) mtx_.countColumns());
    memOut.putDoubleArray(preLongs << 3, mtx_.data, 0, (int) numElements);

    return outByteArr;
  }

  @Override
  public byte[] toCompactByteArray(final int numRows, final int numCols) {
    // TODO: row/col limit checks

    final int preLongs = 3;

    // for non-compact we can do an array copy, so save as non-compact if using the entire matrix
    final long numElements = (long) numRows * numCols;
    final boolean isCompact = numElements < mtx_.count();
    if (!isCompact) {
      return toByteArray();
    }

    assert numElements < mtx_.count();

    final int outBytes = (int) (((long)preLongs * Long.BYTES) + (numElements * Double.BYTES));
    final byte[] outByteArr = new byte[outBytes];
    final WritableMemory memOut = WritableMemory.wrap(outByteArr);
    final Object memObj = memOut.getArray();
    final long memAddr = memOut.getCumulativeOffset(0L);

    MatrixPreambleUtil.insertPreLongs(memObj, memAddr, preLongs);
    MatrixPreambleUtil.insertSerVer(memObj, memAddr, MatrixPreambleUtil.SER_VER);
    MatrixPreambleUtil.insertFamilyID(memObj, memAddr, MatrixFamily.MATRIX.getID());
    MatrixPreambleUtil.insertFlags(memObj, memAddr, COMPACT_FLAG_MASK);
    MatrixPreambleUtil.insertNumRows(memObj, memAddr, (int) mtx_.countRows());
    MatrixPreambleUtil.insertNumColumns(memObj, memAddr, (int) mtx_.countColumns());
    MatrixPreambleUtil.insertNumRowsUsed(memObj, memAddr, numRows);
    MatrixPreambleUtil.insertNumColumnsUsed(memObj, memAddr, numCols);

    // write elements in column-major order
    long offsetBytes = (long)preLongs * Long.BYTES;
    for (int c = 0; c < numCols; ++c) {
      for (int r = 0; r < numRows; ++r) {
        memOut.putDouble(offsetBytes, mtx_.doubleValue(r, c));
        offsetBytes += Double.BYTES;
      }
    }

    return outByteArr;
  }

  @Override
  public double getElement(final int row, final int col) {
    return mtx_.doubleValue(row, col);
  }

  @Override
  public double[] getRow(final int row) {
    final int cols = (int) mtx_.countColumns();
    final double[] result = new double[cols];
    for (int c = 0; c < cols; ++c) {
      result[c] = mtx_.doubleValue(row, c);
    }
    return result;
  }

  @Override
  public double[] getColumn(final int col) {
    final int rows = (int) mtx_.countRows();
    final double[] result = new double[rows];
    for (int r = 0; r < rows; ++r) {
      result[r] = mtx_.doubleValue(r, col);
    }
    return result;
  }

  @Override
  public void setElement(final int row, final int col, final double value) {
    mtx_.set(row, col, value);
  }

  @Override
  public void setRow(final int row, final double[] values) {
    if (values.length != mtx_.countColumns()) {
      throw new IllegalArgumentException("Invalid number of elements for row. Expected "
              + mtx_.countColumns() + ", found " + values.length);
    }

    for (int i = 0; i < mtx_.countColumns(); ++i) {
      mtx_.set(row, i, values[i]);
    }
  }

  @Override
  public void setColumn(final int column, final double[] values) {
    if (values.length != mtx_.countRows()) {
      throw new IllegalArgumentException("Invalid number of elements for column. Expected "
              + mtx_.countRows() + ", found " + values.length);
    }

    for (int i = 0; i < mtx_.countRows(); ++i) {
      mtx_.set(i, column, values[i]);
    }
  }

  @Override
  public MatrixType getMatrixType() {
    return MatrixType.OJALGO;
  }

}

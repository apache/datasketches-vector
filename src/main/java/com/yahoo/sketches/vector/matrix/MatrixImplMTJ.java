/*
 * Copyright 2017, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root
 * for terms.
 */

package com.yahoo.sketches.vector.matrix;

import static com.yahoo.sketches.vector.matrix.MatrixPreambleUtil.COMPACT_FLAG_MASK;
import static com.yahoo.sketches.vector.matrix.MatrixPreambleUtil.extractFamilyID;
import static com.yahoo.sketches.vector.matrix.MatrixPreambleUtil.extractFlags;
import static com.yahoo.sketches.vector.matrix.MatrixPreambleUtil.extractNumColumns;
import static com.yahoo.sketches.vector.matrix.MatrixPreambleUtil.extractNumColumnsUsed;
import static com.yahoo.sketches.vector.matrix.MatrixPreambleUtil.extractNumRows;
import static com.yahoo.sketches.vector.matrix.MatrixPreambleUtil.extractNumRowsUsed;
import static com.yahoo.sketches.vector.matrix.MatrixPreambleUtil.extractPreLongs;
import static com.yahoo.sketches.vector.matrix.MatrixPreambleUtil.extractSerVer;

import com.yahoo.memory.Memory;
import com.yahoo.memory.WritableMemory;
import com.yahoo.sketches.vector.MatrixFamily;
import no.uib.cipr.matrix.DenseMatrix;

public final class MatrixImplMTJ extends Matrix {
  private DenseMatrix mtx_;

  private MatrixImplMTJ(final int numRows, final int numCols) {
    mtx_ = new DenseMatrix(numRows, numCols);
    numRows_ = numRows;
    numCols_ = numCols;
  }

  private MatrixImplMTJ(final DenseMatrix mtx) {
    mtx_ = mtx;
    numRows_ = mtx.numRows();
    numCols_ = mtx.numColumns();
  }

  static Matrix newInstance(final int numRows, final int numCols) {
    return new MatrixImplMTJ(numRows, numCols);
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

    final MatrixImplMTJ matrix;

    if (isCompact) {
      matrix = new MatrixImplMTJ(nRows, nCols);

      nRows = extractNumRowsUsed(srcMem);
      nCols = extractNumColumnsUsed(srcMem);

      int memOffset = preLongs * Long.BYTES;
      for (int c = 0; c < nCols; ++c) {
        for (int r = 0; r < nRows; ++r) {
          matrix.mtx_.set(r, c, srcMem.getDouble(memOffset));
          memOffset += Double.BYTES;
        }
      }
    } else {
      final int nElements = nRows * nCols;
      final double[] data = new double[nElements];
      srcMem.getDoubleArray(preLongs * Long.BYTES, data, 0, nElements);

      matrix = new MatrixImplMTJ(new DenseMatrix(nRows, nCols, data, false));
    }

    return matrix;
  }

  static Matrix wrap(final DenseMatrix mtx) {
    return new MatrixImplMTJ(mtx);
  }

  @Override
  public Object getRawObject() {
    return mtx_;
  }

  @Override
  public byte[] toByteArray() {
    final int preLongs = 2;
    final long numElements = numRows_ * numCols_;
    assert numElements == (mtx_.numRows() * mtx_.numColumns());

    final int outBytes = (int) ((preLongs * Long.BYTES) + (numElements * Double.BYTES));
    final byte[] outByteArr = new byte[outBytes];
    final WritableMemory memOut = WritableMemory.wrap(outByteArr);
    final Object memObj = memOut.getArray();
    final long memAddr = memOut.getCumulativeOffset(0L);

    MatrixPreambleUtil.insertPreLongs(memObj, memAddr, preLongs);
    MatrixPreambleUtil.insertSerVer(memObj, memAddr, MatrixPreambleUtil.SER_VER);
    MatrixPreambleUtil.insertFamilyID(memObj, memAddr, MatrixFamily.MATRIX.getID());
    MatrixPreambleUtil.insertFlags(memObj, memAddr, 0);
    MatrixPreambleUtil.insertNumRows(memObj, memAddr, numRows_);
    MatrixPreambleUtil.insertNumColumns(memObj, memAddr, numCols_);
    memOut.putDoubleArray(preLongs << 3, mtx_.getData(), 0, (int) numElements);

    return outByteArr;
  }

  @Override
  public byte[] toCompactByteArray(final int numRows, final int numCols) {
    // TODO: row/col limit checks

    final int preLongs = 3;

    // for non-compact we can do an array copy, so save as non-compact if using the entire matrix
    final long numElements = (long) numRows * numCols;
    final boolean isCompact = numElements < (mtx_.numRows() * mtx_.numColumns());
    if (!isCompact) {
      return toByteArray();
    }

    final int outBytes = (int) ((preLongs * Long.BYTES) + (numElements * Double.BYTES));
    final byte[] outByteArr = new byte[outBytes];
    final WritableMemory memOut = WritableMemory.wrap(outByteArr);
    final Object memObj = memOut.getArray();
    final long memAddr = memOut.getCumulativeOffset(0L);

    MatrixPreambleUtil.insertPreLongs(memObj, memAddr, preLongs);
    MatrixPreambleUtil.insertSerVer(memObj, memAddr, MatrixPreambleUtil.SER_VER);
    MatrixPreambleUtil.insertFamilyID(memObj, memAddr, MatrixFamily.MATRIX.getID());
    MatrixPreambleUtil.insertFlags(memObj, memAddr, COMPACT_FLAG_MASK);
    MatrixPreambleUtil.insertNumRows(memObj, memAddr, mtx_.numRows());
    MatrixPreambleUtil.insertNumColumns(memObj, memAddr, mtx_.numColumns());
    MatrixPreambleUtil.insertNumRowsUsed(memObj, memAddr, numRows);
    MatrixPreambleUtil.insertNumColumnsUsed(memObj, memAddr, numCols);

    // write elements in column-major order
    long offsetBytes = preLongs * Long.BYTES;
    for (int c = 0; c < numCols; ++c) {
      for (int r = 0; r < numRows; ++r) {
        memOut.putDouble(offsetBytes, mtx_.get(r, c));
        offsetBytes += Double.BYTES;
      }
    }

    return outByteArr;
  }

  @Override
  public double getElement(final int row, final int col) {
    return mtx_.get(row, col);
  }

  @Override
  public double[] getRow(final int row) {
    final int cols = mtx_.numColumns();
    final double[] result = new double[cols];
    for (int c = 0; c < cols; ++c) {
      result[c] = mtx_.get(row, c);
    }
    return result;
  }

  @Override
  public double[] getColumn(final int col) {
    final int rows = mtx_.numRows();
    final double[] result = new double[rows];
    for (int r = 0; r < rows; ++r) {
      result[r] = mtx_.get(r, col);
    }
    return result;
  }

  @Override
  public void setElement(final int row, final int col, final double value) {
    mtx_.set(row, col, value);
  }

  @Override
  public void setRow(final int row, final double[] values) {
    if (values.length != mtx_.numColumns()) {
      throw new IllegalArgumentException("Invalid number of elements for row. Expected "
              + mtx_.numColumns() + ", found " + values.length);
    }

    for (int i = 0; i < mtx_.numColumns(); ++i) {
      mtx_.set(row, i, values[i]);
    }
  }

  @Override
  public void setColumn(final int column, final double[] values) {
    if (values.length != mtx_.numRows()) {
      throw new IllegalArgumentException("Invalid number of elements for column. Expected "
              + mtx_.numRows() + ", found " + values.length);
    }

    for (int i = 0; i < mtx_.numRows(); ++i) {
      // TODO: System.arraycopy()?
      mtx_.set(i, column, values[i]);
    }
  }

  @Override
  public MatrixType getMatrixType() {
    return MatrixType.MTJ;
  }
}

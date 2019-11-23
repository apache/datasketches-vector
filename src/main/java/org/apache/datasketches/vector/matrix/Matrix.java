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

import static org.apache.datasketches.vector.matrix.MatrixPreambleUtil.LS;

import org.ojalgo.matrix.store.Primitive64Store;

import org.apache.datasketches.memory.Memory;
import org.apache.datasketches.vector.MatrixFamily;

/**
 * Provides an implementation-agnostic wrapper around Matrix classes.
 *
 * @author Jon Malkin
 */
public abstract class Matrix {
  int numRows_;
  int numCols_;

  /**
   * Loads matrix from srcMem, assuming storage in column-major order to ensure portability.
   * Does not necessarily encode matrix size; do not expect size checks based on passed-in
   * parameters.
   *
   * @param srcMem Memory wrapping the matrix
   * @param type Matrix implementation type to use
   * @return The heapified matrix
   */
  public static Matrix heapify(final Memory srcMem, final MatrixType type) {
    switch (type) {
      case OJALGO:
        return MatrixImplOjAlgo.heapifyInstance(srcMem);
      default:
        return null;
    }
  }

  /**
   * Wraps an object without allocating memory. This method will throw an exception if the mtx
   * Object is not of the same type as the implementing class's native format.
   * @param mtx Matrix object to wrap
   * @return A Matrix object
   */
  public static Matrix wrap(final Object mtx) {
    if (mtx == null) {
      return null;
    } else if (mtx instanceof Primitive64Store) {
      return MatrixImplOjAlgo.wrap((Primitive64Store) mtx);
    }
    else {
      throw new IllegalArgumentException("wrap() does not currently support "
              + mtx.getClass().toString());
    }
  }

  /**
   * Gets a builder to be able to create instances of Matrix objects
   * @return a MatrixBuilder object
   */
  public static MatrixBuilder builder() {
    return new MatrixBuilder();
  }

  /**
   * Returns the raw data object backing this Matrix, as an Object. Must be cast to the
   * appropriate type (assuming knowledge of the implementation) to be used.
   * @return An Object pointing to the raw data backing this Matrix
   */
  public abstract Object getRawObject();

  /**
   * Serializes the Matrix in a custom format as a byte array
   * @return A byte[] conttaining a serialized Matrix
   */
  public abstract byte[] toByteArray();

  /**
   * Serializes a sub-Matrix by storing only the first numRows and numCols rows and columns,
   * respsectively.
   * @param numRows Number of rows to write
   * @param numCols Number of columns to write
   * @return A byte[] containing the serialized sub-Matrix.
   */
  public abstract byte[] toCompactByteArray(int numRows, int numCols);

  /**
   * Returns a single element from the Matrix
   * @param row Row index of target element (0-based)
   * @param col Column index of target elemtn (0-based)
   * @return Matrix value at (row, column)
   */
  public abstract double getElement(int row, int col);

  /**
   * Returns a copy of an entire row of the Matrix
   * @param row Row index to return (0-based)
   * @return A double[] representing the Matrix row
   */
  public abstract double[] getRow(int row);

  /**
   * Returns a copy of an entire column of the Matrix
   * @param col Column index to return (0-based)
   * @return A double[] representing the Matrix column
   */
  public abstract double[] getColumn(int col);

  /**
   * Sets a single element inthe Matrix
   * @param row Row index of target element (0-based)
   * @param col Column index of target element (0-based)
   * @param value The value to insert into the Matrix at (row, column)
   */
  public abstract void setElement(int row, int col, double value);

  /**
   * Sets an entire row of the Matrix, by copying data from the input
   * @param row Target row index (0-based)
   * @param values Array of values to write into the Matrix
   */
  public abstract void setRow(int row, double[] values);

  /**
   * Sets an entire column of the Matrix, by copying data from the input
   * @param column Target column index (0-based)
   * @param values Array of values to write into the Matrix
   */
  public abstract void setColumn(int column, double[] values);

  /**
   * Gets the number of rows in the Matrix
   * @return Configured number of rows in the Matrix
   */
  public long getNumRows() { return numRows_; }

  /**
   * Gets the number of columns in the Matrix
   * @return Configured number of columns in the Matrix
   */
  public long getNumColumns() { return numCols_; }

  /**
   * Gets serialized size of the Matrix, in bytes.
   * @return Number of bytes needed for a serialized Matrix
   */
  public int getSizeBytes() {
    final int preBytes = MatrixFamily.MATRIX.getMinPreLongs() * Long.BYTES;
    final int mtxBytes = (numRows_ * numCols_) * Double.BYTES;
    return preBytes + mtxBytes;
  }

  /**
   * Gets serialized size of the Matrix in compact form, in bytes.
   * @param rows Number of rows to select for writing
   * @param cols Number of columns to select for writing
   * @return Number of bytes needed to serialize the first (rows, cols) of this Matrix
   */
  public int getCompactSizeBytes(final int rows, final int cols) {
    final int nRows = Math.min(rows, numRows_);
    final int nCols = Math.min(cols, numCols_);

    if ((nRows < 1) || (nCols < 1)) {
      return MatrixFamily.MATRIX.getMinPreLongs() * Long.BYTES;
    } else if ((nRows == numRows_) && (nCols == numCols_)) {
      return getSizeBytes();
    }

    final int preBytes = MatrixFamily.MATRIX.getMaxPreLongs() * Long.BYTES;
    final int mtxBytes = (nRows * nCols) * Double.BYTES;
    return preBytes + mtxBytes;
  }

  /**
   * Writes information about this Matrix to a String.
   * @return A human-readable representation of a Matrix
   */
  @Override
  public String toString() {
    final StringBuilder sb = new StringBuilder();

    sb.append("   Matrix data  :").append(LS);
    sb.append(this.getClass().getName());
    sb.append(" < ").append(numRows_).append(" x ").append(numCols_).append(" >");

    // First element
    sb.append("\n{ { ").append(getElement(0, 0));

    // Rest of the first row
    for (int j = 1; j < numCols_; j++) {
      sb.append(",\t").append(getElement(0, j));
    }

    // For each of the remaining rows
    for (int i = 1; i < numRows_; i++) {

      // First column
      sb.append(" },\n{ ").append(getElement(i, 0));

      // Remaining columns
      for (int j = 1; j < numCols_; j++) {
        sb.append(",\t").append(getElement(i, j));
      }
    }

    // Finish
    sb.append(" } }").append(LS);

    return sb.toString();
  }

  /**
   * Gets the matrix type
   * @return the matrix type
   */
  public abstract MatrixType getMatrixType();
}

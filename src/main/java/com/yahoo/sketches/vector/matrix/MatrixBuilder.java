/*
 * Copyright 2017, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root
 * for terms.
 */

package com.yahoo.sketches.vector.matrix;

/**
 * Provides a builder for Matrix objects.
 */
public class MatrixBuilder {

  private MatrixType type_ = MatrixType.OJALGO; // default type

  public MatrixBuilder() {}

  /**
   * Sets the underlying type of object to use with any Matrix objects created.
   * @param type One of the supported types
   * @return This MatrixBuilder object
   */
  public MatrixBuilder setType(final MatrixType type) {
    type_ = type;
    return this;
  }

  /**
   * Returns a value from an enum defining the type of object backing any Matrix objects created.
   * @return An item from the MatrixType enum.
   */
  public MatrixType getBackingType() {
    return type_;
  }

  /**
   * Instantiates a new, empty matrix of the target size
   *
   * @param numRows Number of rows in matrix
   * @param numCols Number of columns in matrix
   * @return An empty matrix of the requested size
   */
  public Matrix build(final int numRows, final int numCols) {
    switch (type_) {
      case OJALGO:
        return MatrixImplOjAlgo.newInstance(numRows, numCols);

      case MTJ:
        return MatrixImplMTJ.newInstance(numRows, numCols);

      default:
        throw new IllegalArgumentException("OJALGO and MTJ are currently the only supported MatrixTypes");
    }
  }
}

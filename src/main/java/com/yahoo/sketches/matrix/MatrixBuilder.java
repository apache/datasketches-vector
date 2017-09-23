/*
 * Copyright 2017, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root
 * for terms.
 */

package com.yahoo.sketches.matrix;

import com.yahoo.sketches.SketchesArgumentException;

/**
 * Provides a builder for Matrix objects.
 */
public class MatrixBuilder {
  public enum Algo {
    OJALGO(1, "ojAlgo"),
    NATIVE(2, "native");

    private int id_;
    private String name_;

    Algo(final int id, final String name) {
      id_ = id;
      name_ = name;
    }

    public int getId() { return id_; }

    public String getName() { return name_; }

    @Override
    public String toString() { return name_; }
  }

  private Algo type_ = Algo.OJALGO; // default type

  MatrixBuilder() {}

  /**
   * Sets the underlying type of object to use with any Matrix objects created.
   * @param type One of the supported types
   * @return This MatrixBuilder object
   */
  public MatrixBuilder setType(final Algo type) {
    type_ = type;
    return this;
  }

  /**
   * Returns a value from an enum definig the type of object backing any Matrix objects created.
   * @return An item from the Algo enum.
   */
  public Algo getFamily() {
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

      case NATIVE:
      default:
        throw new SketchesArgumentException("Only Algo.OJALGO is currently supported Matrix type");
    }
  }
}

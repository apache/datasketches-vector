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

/**
 * Provides a builder for Matrix objects.
 */
public class MatrixBuilder {

  private MatrixType type_ = MatrixType.OJALGO; // default type

  /**
   * Default no-op constructor.
   */
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

      default:
        throw new IllegalArgumentException("OJALGO is currently the only supported MatrixTypes");
    }
  }
}

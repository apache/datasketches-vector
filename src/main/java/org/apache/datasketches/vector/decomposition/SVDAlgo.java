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

/**
 * This class allows a choice of algorithms for Singular Value Decomposition. The options are:
 * <ul>
 *   <li>FULL: The matrix library's default SVD implementation.</li>
 *   <li>SISVD: Simultaneous iteration, an approximate method likely to be more efficient only with sparse
 *   matrices or when <em>k</em> is significantly smaller than the number of rows in the sketch.</li>
 *   <li>SYM: Takes advantage of matrix dimensionality, first computing eigenvalues of AA^T, then computes
 *   intended results. Squaring A alters condition number and may cause numeric stability issues,
 *   but unlikely an issue for Frequent Directions since discarding the smaller singular values/vectors.</li>
 * </ul>
 */
public enum SVDAlgo {

  /**
   * The matrix library's default SVD implementation.
   */
  FULL(1, "Full"),

  /**
   * Simultaneous iteration, an approximate method likely to be more efficient only with sparse
   * matrices or when <em>k</em> is significantly smaller than the number of rows in the sketch.
   */
  SISVD(2, "SISVD"),

  /**
   * Takes advantage of matrix dimensionality, first computing eigenvalues of AA^T, then computes
   *   intended results. Squaring A alters condition number and may cause numeric stability issues,
   *   but unlikely an issue for Frequent Directions since discarding the smaller singular values/vectors.
   */
  SYM(3, "Symmetrized");

  private int id_;
  private String name_;

  SVDAlgo(final int id, final String name) {
    id_ = id;
    name_ = name;
  }

  /**
   * Returns the ID.
   * @return the ID.
   */
  public int getId() { return id_; }

  /**
   * Gets the name
   * @return the name
   */
  public String getName() { return name_; }

  @Override
  public String toString() { return name_; }
}

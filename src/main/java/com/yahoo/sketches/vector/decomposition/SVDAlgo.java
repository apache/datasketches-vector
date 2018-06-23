package com.yahoo.sketches.vector.decomposition;

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
  FULL(1, "Full"),
  SISVD(2, "SISVD"),
  SYM(3, "Symmetrized");

  private int id_;
  private String name_;

  SVDAlgo(final int id, final String name) {
    id_ = id;
    name_ = name;
  }

  public int getId() { return id_; }

  public String getName() { return name_; }

  @Override
  public String toString() { return name_; }
}

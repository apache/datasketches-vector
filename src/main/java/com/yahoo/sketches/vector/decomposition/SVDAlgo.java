package com.yahoo.sketches.vector.decomposition;

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

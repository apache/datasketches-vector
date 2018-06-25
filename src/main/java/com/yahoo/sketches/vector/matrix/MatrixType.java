package com.yahoo.sketches.vector.matrix;

public enum MatrixType {
  OJALGO(1, "ojAlgo");

  private int id_;
  private String name_;

  MatrixType(final int id, final String name) {
    id_ = id;
    name_ = name;
  }

  public int getId() { return id_; }

  public String getName() { return name_; }

  @Override
  public String toString() { return name_; }
}

/*
 * Copyright 2017, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root
 * for terms.
 */

package com.yahoo.sketches.vector.matrix;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;

import org.testng.annotations.Test;

public class MatrixBuilderTest {
  @Test
  public void checkBuild() {
    final MatrixBuilder builder = new MatrixBuilder();
    assertEquals(builder.getBackingType(), MatrixType.OJALGO); // default type

    final Matrix m = builder.build(128, 512);
    assertNotNull(m);
  }

  @Test
  public void checkSetType() {
    final MatrixBuilder builder = new MatrixBuilder();
    final MatrixType type = builder.getBackingType();
    assertEquals(type, MatrixType.OJALGO); // default type
    assertEquals(type.getId(), MatrixType.OJALGO.getId());
    assertEquals(type.getName(), MatrixType.OJALGO.getName());

    builder.setType(MatrixType.NATIVE);
    assertEquals(builder.getBackingType(), MatrixType.NATIVE);
    assertEquals(builder.getBackingType().toString(), "native");

    try {
      builder.build(10, 20);
    } catch (final IllegalArgumentException e) {
      // expected until native is implemented
    }
  }

}

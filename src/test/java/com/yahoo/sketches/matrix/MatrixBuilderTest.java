/*
 * Copyright 2017, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root
 * for terms.
 */

package com.yahoo.sketches.matrix;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;

import org.testng.annotations.Test;

public class MatrixBuilderTest {
  @Test
  public void checkBuild() {
    final MatrixBuilder builder = new MatrixBuilder();
    assertEquals(builder.getBackingType(), MatrixBuilder.Algo.OJALGO); // default type

    Matrix m = builder.build(128, 512);
    assertNotNull(m);
  }

  @Test
  public void checkSetType() {
    final MatrixBuilder builder = new MatrixBuilder();
    MatrixBuilder.Algo type = builder.getBackingType();
    assertEquals(type, MatrixBuilder.Algo.OJALGO); // default type
    assertEquals(type.getId(), MatrixBuilder.Algo.OJALGO.getId());
    assertEquals(type.getName(), MatrixBuilder.Algo.OJALGO.getName());

    builder.setType(MatrixBuilder.Algo.NATIVE);
    assertEquals(builder.getBackingType(), MatrixBuilder.Algo.NATIVE);
    assertEquals(builder.getBackingType().toString(), "native");

    try {
      builder.build(10, 20);
    } catch (final IllegalArgumentException e) {
      // expected until native is implemented
    }
  }

}

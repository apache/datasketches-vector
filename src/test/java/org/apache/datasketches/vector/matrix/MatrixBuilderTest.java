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

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;

import org.testng.annotations.Test;

@SuppressWarnings("javadoc")
public class MatrixBuilderTest {
  @Test
  public void checkBuild() {
    final MatrixBuilder builder = new MatrixBuilder();
    assertEquals(builder.getBackingType(), MatrixType.OJALGO); // default type

    Matrix m = builder.build(128, 512);
    assertNotNull(m);

    m = builder.build(128, 512);
    assertNotNull(m);
  }

  @Test
  public void checkSetType() {
    final MatrixBuilder builder = new MatrixBuilder();
    final MatrixType type = builder.getBackingType();
    assertEquals(type, MatrixType.OJALGO); // default type
    assertEquals(type.getId(), MatrixType.OJALGO.getId());
    assertEquals(type.getName(), MatrixType.OJALGO.getName());
  }

}

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

package org.apache.datasketches.vector.regression;

import static org.apache.datasketches.vector.regression.VectorNormalizer.D_INT;
import static org.apache.datasketches.vector.regression.VectorNormalizer.EMPTY_FLAG_MASK;
import static org.apache.datasketches.vector.regression.VectorNormalizer.FAMILY_BYTE;
import static org.apache.datasketches.vector.regression.VectorNormalizer.FLAGS_BYTE;
import static org.apache.datasketches.vector.regression.VectorNormalizer.N_LONG;
import static org.apache.datasketches.vector.regression.VectorNormalizer.PREAMBLE_LONGS_BYTE;
import static org.apache.datasketches.vector.regression.VectorNormalizer.SER_VER;
import static org.apache.datasketches.vector.regression.VectorNormalizer.SER_VER_BYTE;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.fail;

import java.util.concurrent.ThreadLocalRandom;

import org.apache.datasketches.memory.Memory;
import org.apache.datasketches.memory.WritableMemory;
import org.apache.datasketches.vector.MatrixFamily;
import org.testng.annotations.Test;


import com.google.common.primitives.Longs;

public class VectorNormalizerTest {

  @Test
  public void instantiationTest() {
    final int d = 5;
    final VectorNormalizer vn = new VectorNormalizer(d);
    assertNotNull(vn);
    assertEquals(vn.getD(), d);
    assertEquals(vn.getN(), 0);
    assertTrue(vn.isEmpty());


    final double[] mean = vn.getMean();
    assertNotNull(mean);

    final double[] sampleVar = vn.getSampleVariance();
    assertNotNull(sampleVar);

    final double[] popVar = vn.getPopulationVariance();
    assertNotNull(popVar);

    // no data, so everything should be Double.NaN
    for (int i = 0; i < d; ++i) {
      assertTrue(Double.isNaN(mean[i]));
      assertTrue(Double.isNaN(sampleVar[i]));
      assertTrue(Double.isNaN(popVar[i]));
    }

    // error case
    try {
      new VectorNormalizer(0);
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void singleUpdateTest() {
    final int d = 3;
    final VectorNormalizer vn = new VectorNormalizer(d);

    final double[] input = {-1, 0, 0.5};
    final double target = 0.3;
    vn.update(input, target);
    assertEquals(vn.getN(), 1);
    assertFalse(vn.isEmpty());

    final double[] mean = vn.getMean();
    assertNotNull(mean);

    final double[] sampleVar = vn.getSampleVariance();
    assertNotNull(sampleVar);

    final double[] popVar = vn.getPopulationVariance();
    assertNotNull(popVar);

    // mean should equal input, others should be 0.0
    for (int i = 0; i < d; ++i) {
      assertEquals(mean[i], input[i]);
      assertEquals(sampleVar[i], 0.0);
      assertEquals(popVar[i], 0.0);
    }
    // intercept shoudl equal target
    assertEquals(vn.getIntercept(), target);
  }

  @Test
  public void multipleUpdateTest() {
    final int n = 100000;
    final int d = 3;
    final double tol = 0.01;

    final VectorNormalizer vn = new VectorNormalizer(d);

    final ThreadLocalRandom rand = ThreadLocalRandom.current();
    final double[] input = new double[d];
    for (int i = 0; i < n; ++i) {
      input[0] = rand.nextGaussian();      // mean = 0.0, var = 1.0
      input[1] = rand.nextDouble() * 2.0;  // mean = 1.0, var = (2-0)^2/12 = 1/3
      input[2] = rand.nextDouble() - 0.5;  // mean = 0.0, var = (1-0)^2/12
      double target = rand.nextGaussian() - 1.0; // mean = -1.0
      vn.update(input, target);
    }
    assertFalse(vn.isEmpty());

    final double[] mean = vn.getMean();
    assertNotNull(mean);
    assertEquals(mean[0], 0.0, tol);
    assertEquals(mean[1], 1.0, tol);
    assertEquals(mean[2], 0.0, tol);
    assertEquals(vn.getIntercept(), -1.0, tol);

    // n is large enough that sample vs population variance won't matter for testing
    final double[] sampleVar = vn.getSampleVariance();
    assertNotNull(sampleVar);
    assertEquals(sampleVar[0], 1.0, tol);
    assertEquals(sampleVar[1], 1.0 / 3.0, tol);
    assertEquals(sampleVar[2], 1.0 / 12.0, tol);

    final double[] popVar = vn.getPopulationVariance();
    assertNotNull(popVar);
    assertEquals(popVar[0], 1.0, tol);
    assertEquals(popVar[1], 1.0 / 3.0, tol);
    assertEquals(popVar[2], 1.0 / 12.0, tol);

    // n is small enough that we still expect a difference with doubles
    for (int i = 0; i < d; ++i) {
      assertTrue(popVar[i] > sampleVar[i]);
    }
  }

  @Test
  public void mergeTest() {
    final int n = 1000000;
    final int d = 2;
    final double tol = 0.01;
    final VectorNormalizer vn1 = new VectorNormalizer(d);
    final VectorNormalizer vn2 = new VectorNormalizer(d);

    final ThreadLocalRandom rand = ThreadLocalRandom.current();

    // data expectations:
    // dimension 0: zero-mean, unit-variance Gaussian, even after merging
    // dimension 1: U[0,2] + U[2,4) -> U[0,4), so mean = 2.0 and var = 4^2/12=4/3
    // target: N(-1,1) + N(1,1), so mean = 0.0 and variance unmeasured
    final double[] input = new double[d];
    for (int i = 0; i < n; ++i) {
      input[0] = rand.nextGaussian();
      input[1] = (rand.nextDouble() * 2.0) + 2.0;
      double target = rand.nextGaussian() - 1.0;
      vn1.update(input, target);

      input[0] = rand.nextGaussian();
      input[1] = rand.nextDouble() * 2.0;
      target = rand.nextGaussian() + 1.0;
      vn2.update(input, target);
    }

    vn1.merge(vn2);
    assertEquals(vn1.getN(), 2 * n);

    final double[] mean = vn1.getMean();
    assertEquals(mean[0], 0.0, tol);
    assertEquals(mean[1], 2.0, tol);
    assertEquals(vn1.getIntercept(), 0.0, tol);

    // n is large enough that sample vs population variance won't matter for testing
    final double[] sampleVar = vn1.getSampleVariance();
    assertEquals(sampleVar[0], 1.0, tol);
    assertEquals(sampleVar[1], 4.0 / 3.0, tol);

    final double[] popVar = vn1.getPopulationVariance();
    assertEquals(popVar[0], 1.0, tol);
    assertEquals(popVar[1], 4.0 / 3.0, tol);
  }

  @Test
  public void invalidUpdateSizeTest() {
    final int d = 5;
    final VectorNormalizer vn = new VectorNormalizer(d);

    final double[] input = new double[d];
    for (int i = 0; i < d; ++i) { input[i] = 1.0 * i; }
    vn.update(input, 0.0);
    assertEquals(vn.getN(), 1);

    try {
      final double[] badInput = {1.0};
      vn.update(badInput, 0.0);
      fail();
    } catch (IllegalArgumentException e) {
      // expected
      assertEquals(vn.getN(), 1);
    }
  }

  @Test
  public void invalidMergeSizeTest() {
    final int d = 3;
    final VectorNormalizer vn1 = new VectorNormalizer(d);

    double[] input = new double[d];
    for (int i = 0; i < d; ++i) { input[i] = 1.0 * i; }
    vn1.update(input, 1.0);
    assertEquals(vn1.getN(), 1);

    // update with a non-empty VN with a different value of d
    final int d2 = d + 3;
    final VectorNormalizer vn2 = new VectorNormalizer(d2);
    input = new double[d2];
    for (int i = 0; i < d2; ++i) { input[i] = 1.0 * i; }
    vn2.update(input, 2.0);
    assertEquals(vn2.getN(), 1);

    try {
      vn1.merge(vn2);
      fail();
    } catch (IllegalArgumentException e) {
      // expected
      assertEquals(vn1.getN(), 1);
    }
  }

  @Test
  public void copyConstructorTest() {
    final int d = 5;
    final int n = 100;

    final VectorNormalizer vn = new VectorNormalizer(d);
    final ThreadLocalRandom rand = ThreadLocalRandom.current();
    final double[] input = new double[d];
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < d; ++j) {
        input[j] = rand.nextDouble();
      }
      vn.update(input, rand.nextGaussian());
    }

    final VectorNormalizer vnCopy = new VectorNormalizer(vn);

    // we'll assume serialization works for this test and compare serialized images for equality
    final byte[] origBytes = vn.toByteArray();
    final byte[] copyBytes = vnCopy.toByteArray();
    assertEquals(copyBytes, origBytes);
  }

  @Test
  public void serializationTest() {
    final int d = 7;
    final int n = 10;

    // empty memory should return null
    assertNull(VectorNormalizer.heapify(null));

    final VectorNormalizer vn = new VectorNormalizer(d);

    // check empty size
    byte[] outBytes = vn.toByteArray();
    assertEquals(outBytes.length, MatrixFamily.VECTORNORMALIZER.getMinPreLongs() * Long.BYTES);
    assertEquals(outBytes.length, vn.getSerializedSizeBytes());

    VectorNormalizer rebuilt = VectorNormalizer.heapify(Memory.wrap(outBytes));
    assertTrue(rebuilt.isEmpty());

    // test with data added
    final double[] input = new double[d];
    final ThreadLocalRandom rand = ThreadLocalRandom.current();
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < d; ++j) {
        input[j] = rand.nextGaussian();
      }
      vn.update(input, rand.nextDouble());
    }

    outBytes = vn.toByteArray();
    assertEquals(outBytes.length, vn.getSerializedSizeBytes());

    rebuilt = VectorNormalizer.heapify(Memory.wrap(outBytes));
    assertFalse(rebuilt.isEmpty());
    assertEquals(vn.getD(), rebuilt.getD());
    assertEquals(vn.getN(), rebuilt.getN());
    assertEquals(vn.getIntercept(), rebuilt.getIntercept());

    final double[] originalMean = vn.getMean();
    final double[] rebuiltMean = vn.getMean();
    final double[] originalVar = vn.getSampleVariance();
    final double[] rebuiltVar = vn.getSampleVariance();

    for (int i = 0; i < d; ++i) {
      // expecting identical bits meaning exact equality
      assertEquals(rebuiltMean[i], originalMean[i]);
      assertEquals(rebuiltVar[i], originalVar[i]);
    }
  }

  @Test
  public void corruptPreambleTest() {
    // memory too small
    byte[] bytes = new byte[3];
    try {
      VectorNormalizer.heapify(Memory.wrap(bytes));
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }

    // memory smaller than preLongs
    bytes = new byte[10];
    bytes[PREAMBLE_LONGS_BYTE] = 2;
    try {
      VectorNormalizer.heapify(Memory.wrap(bytes));
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }

    // invalid preLongs
    final int preLongs = MatrixFamily.VECTORNORMALIZER.getMaxPreLongs() + 1;
    bytes = new byte[preLongs * Longs.BYTES];
    bytes[PREAMBLE_LONGS_BYTE] = (byte) preLongs;
    try {
      VectorNormalizer.heapify(Memory.wrap(bytes));
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }

    final int d = 12;
    final VectorNormalizer vn = new VectorNormalizer(d);

    // wrong serialization version
    bytes = vn.toByteArray();
    bytes[SER_VER_BYTE] = ~SER_VER; // any bits that don't match
    try {
      VectorNormalizer.heapify(Memory.wrap(bytes));
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }

    // wrong family id
    bytes = vn.toByteArray();
    bytes[FAMILY_BYTE] = (byte) MatrixFamily.FREQUENTDIRECTIONS.getID();
    try {
      VectorNormalizer.heapify(Memory.wrap(bytes));
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }

    // invalid d
    bytes = vn.toByteArray();
    WritableMemory mem = WritableMemory.wrap(bytes);
    mem.putInt(D_INT, -1);
    try {
      VectorNormalizer.heapify(mem);
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void corruptEmptyHeapifyTest() {
    final int d = 7;
    final VectorNormalizer vn = new VectorNormalizer(d);
    byte[] outBytes = vn.toByteArray();
    WritableMemory mem = WritableMemory.wrap(outBytes);

    // clear empty flag
    mem.putByte(FLAGS_BYTE, (byte) 0);
    try {
      VectorNormalizer.heapify(mem);
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void corruptNonEmptyHeapifyTest() {
    final int d = 1;
    final int n = 100;

    final VectorNormalizer vn = new VectorNormalizer(d);
    final ThreadLocalRandom rand = ThreadLocalRandom.current();
    final double[] input = new double[d];
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < d; ++j) {
        input[j] = rand.nextDouble();
      }
      vn.update(input, rand.nextDouble());
    }
    assertFalse(vn.isEmpty());

    // force-set empty flag
    byte[] bytes = vn.toByteArray();
    WritableMemory mem = WritableMemory.wrap(bytes);
    mem.putByte(FLAGS_BYTE, (byte) EMPTY_FLAG_MASK);
    try {
      VectorNormalizer.heapify(mem);
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }

    // invalid n
    bytes = vn.toByteArray();
    mem = WritableMemory.wrap(bytes);
    mem.putLong(N_LONG, -100);
    try {
      VectorNormalizer.heapify(mem);
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }

    // capacity too small for vectors
    bytes = vn.toByteArray();
    mem = WritableMemory.allocate(bytes.length - 1);
    mem.putByteArray(0, bytes, 0, bytes.length - 1);
    try {
      VectorNormalizer.heapify(mem);
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

}

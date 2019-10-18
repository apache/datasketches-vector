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

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.fail;

import java.util.Arrays;

import org.testng.annotations.Test;

import org.apache.datasketches.memory.Memory;
import org.apache.datasketches.memory.WritableMemory;
import org.apache.datasketches.vector.MatrixFamily;
import org.apache.datasketches.vector.matrix.Matrix;

@SuppressWarnings("javadoc")
public class FrequentDirectionsTest {

  @Test
  public void instantiateFD() {
    final int k = 32;
    final int d = 256;
    final FrequentDirections fd = FrequentDirections.newInstance(k, d);

    assertNotNull(fd);
    assertTrue(fd.isEmpty());
    assertEquals(fd.getK(), k);
    assertEquals(fd.getD(), d);
    assertEquals(fd.getN(), 0);
    assertNull(fd.getResult());

    // error conditions
    // d = 0
    try {
      FrequentDirections.newInstance(k, 0);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }

    // k = -1
    try {
      FrequentDirections.newInstance(-1, d);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }

    // d < 2k (not handled in reduceRank()
    try {
      FrequentDirections.newInstance(d, d);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void checkSymmUpdate() {
    final int k = 4;
    final int d = 16; // should be > 2k
    final FrequentDirections fd = FrequentDirections.newInstance(k, d);
    fd.setSVDAlgo(SVDAlgo.SYM); // default, but force anyway

    runUpdateTest(fd);
  }

  @Test
  public void checkFullSVDUpdate() {
    final int k = 4;
    final int d = 16; // should be > 2k
    final FrequentDirections fd = FrequentDirections.newInstance(k, d);
    fd.setSVDAlgo(SVDAlgo.FULL);

    runUpdateTest(fd);
  }

  private static void runUpdateTest(final FrequentDirections fd) {
    final int k = fd.getK();
    final int d = fd.getD();

    // creates matrix with increasing values along diagonal
    final double[] input = new double[d];
    for (int i = 0; i < (2 * k); ++i) {
      if (i > 0) {
        input[i - 1] = 0.0;
      }
      input[i] = i * 1.0;
      fd.update(input);
    }
    fd.update((double[]) null); // should be a no-op and not impact next lines
    assertEquals(fd.getNumRows(), 2 * k);
    assertEquals(fd.getN(), 2 * k);

    input[(2 * k) - 1] = 0.0;
    input[2 * k] = 2.0 * k;
    fd.update(input); // trigger reduceRank(), then add 1 more row
    assertEquals(fd.getNumRows(), k);
  }


  @Test
  public void updateWithTooFewDimensions() {
    final int k = 4;
    final int d = 16; // should be > 2k
    final FrequentDirections fd = FrequentDirections.newInstance(k, d);

    final double[] input = new double[d - 3];
    try {
      fd.update(input);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void mergeSketches() {
    final int k = 5;
    final int d = 12; // should be > 2k
    final int initialRows = 7;
    final FrequentDirections fd1 = FrequentDirections.newInstance(k, d);
    final FrequentDirections fd2 = FrequentDirections.newInstance(k, d);

    // two diagonal matrices
    final double[] input = new double[d];
    for (int i = 0; i < initialRows; ++i) {
      if (i > 0) {
        input[i - 1] = 0.0;
      }
      //input[i] = (2 * k) - (i * 1.0);
      input[i] = i * 1.0;
      fd1.update(input);

      input[i] = (i * 1.0) - (2 * k);
      fd2.update(input);
    }

    // the next two lines are no-ops
    fd1.update((FrequentDirections) null);
    fd1.update(FrequentDirections.newInstance(k, d));
    assertEquals(fd1.getNumRows(), initialRows);
    assertEquals(fd1.getN(), initialRows);

    assertEquals(fd2.getNumRows(), initialRows);
    assertEquals(fd2.getN(), initialRows);

    fd1.update(fd2);
    final int expectedRows = (((2 * initialRows) % k) + k) - 1; // assumes 2 * initialRows > k
    assertEquals(fd1.getNumRows(), expectedRows);
    assertEquals(fd1.getN(), 2 * initialRows);

    final Matrix result = fd1.getResult(false);
    assertNotNull(result);
    assertEquals(result.getNumRows(), 2 * k);

    println(fd1.toString(true, true, true));
  }

  @Test
  public void checkCompensativeResultSymSVD() {
    final int k = 4;
    final int d = 10; // should be > 2k
    final FrequentDirections fd = FrequentDirections.newInstance(k, d);
    fd.setSVDAlgo(SVDAlgo.SYM);

    runCompensativeResultTest(fd);
  }

  @Test
  public void checkCompensativeResultFullSVD() {
    final int k = 4;
    final int d = 10; // should be > 2k
    final FrequentDirections fd = FrequentDirections.newInstance(k, d);
    fd.setSVDAlgo(SVDAlgo.FULL);

    runCompensativeResultTest(fd);
  }

  private static void runCompensativeResultTest(final FrequentDirections fd) {
    final int d = fd.getD();
    final int k = fd.getK();

    // diagonal matrix for easy checking
    final double[] input = new double[d];
    for (int i = 0; i < (k + 1); ++i) {
      if (i > 0) {
        input[i - 1] = 0.0;
      }
      input[i] = (i + 1) * 1.0;
      fd.update(input);
    }

    Matrix m = fd.getResult();
    for (int i = 0; i < (k + 1); ++i) {
      assertEquals(m.getElement(i,i), 1.0 * (i + 1), 1e-6);
    }

    // without compensation, but force rank reduction and check projection at the same time
    fd.forceReduceRank();
    m = fd.getResult();
    final Matrix p = fd.getProjectionMatrix();
    double[] sv = fd.getSingularValues(false);
    for (int i = k; i > 1; --i) {
      final double val = Math.abs(m.getElement(k - i, i));
      final double expected = Math.sqrt(((i + 1) * (i + 1)) - fd.getSvAdjustment());
      assertEquals(val, expected, 1e-6);
      assertEquals(sv[k - i], expected, 1e-10);
      assertEquals(Math.abs(p.getElement(k - i, i)), 1.0, 1e-6);
    }
    assertEquals(m.getElement(k, 1), 0.0, 0.0); // might return -0.0
    assertEquals(p.getElement(k, 1), 0.0, 0.0); // might return -0.0

    // with compensation
    m = fd.getResult(true);
    sv = fd.getSingularValues(true);
    for (int i = k; i > 1; --i) {
      final double val = Math.abs(m.getElement(k - i, i));
      assertEquals(val, i + 1.0, 1e-6);
      assertEquals(sv[k - i], i + 1.0, 1e-10);
    }
    assertEquals(m.getElement(k, 1), 0.0);
  }

  @Test
  public void mergeIncompatibleSketches() {
    final int k = 5;
    final int d = 12; // should be > 2k
    final FrequentDirections fd1 = FrequentDirections.newInstance(k, d);

    final double[] input = new double[d];
    input[0] = 1.0;
    fd1.update(input);

    // merge in smaller k
    FrequentDirections fd2 = FrequentDirections.newInstance(k - 1, d);
    fd2.update(input);
    try {
      fd1.update(fd2);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }

    // mismatch in d
    fd2 = FrequentDirections.newInstance(k, d - 1);
    fd2.update(new double[d - 1]);
    try {
      fd1.update(fd2);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void checkSerialization() {
    final int k = 7;
    final int d = 20;
    final FrequentDirections fd = FrequentDirections.newInstance(k, d);

    byte[] sketchBytes = fd.toByteArray();
    assertEquals(sketchBytes.length,
            MatrixFamily.FREQUENTDIRECTIONS.getMinPreLongs() * Long.BYTES);
    Memory mem = Memory.wrap(sketchBytes);
    FrequentDirections rebuilt = FrequentDirections.heapify(mem);
    assertTrue(rebuilt.isEmpty());
    assertEquals(rebuilt.getD(), fd.getD());
    assertEquals(rebuilt.getK(), fd.getK());

    // creates matrix with increasing values along diagonal
    // k rows, so shouldn't compress yet
    final double[] input = new double[d];
    for (int i = 0; i < k; ++i) {
      if (i > 0) {
        input[i - 1] = 0.0;
      }
      //input[i] = (2 * k) - (i * 1.0);
      input[i] = i * 1.0;
      fd.update(input);
    }
    sketchBytes = fd.toByteArray();
    mem = Memory.wrap(sketchBytes);
    rebuilt = FrequentDirections.heapify(mem);
    assertEquals(rebuilt.getN(), fd.getN());
    assertEquals(rebuilt.getD(), fd.getD());
    assertEquals(rebuilt.getK(), fd.getK());

    // add another k rows and serialize, compressing this time
    for (int i = k; i < ((2 * k) - 1); ++i) {
      input[i] = i * 1.0;
      fd.update(input);
    }
    assertEquals(fd.getNumRows(), (2 * k) - 1);
    sketchBytes = fd.toByteArray();
    mem = Memory.wrap(sketchBytes);
    rebuilt = FrequentDirections.heapify(mem);
    assertEquals(rebuilt.getN(), fd.getN());
    assertEquals(rebuilt.getNumRows(), fd.getNumRows());

    println(PreambleUtil.preambleToString(mem));
  }

  @Test
  public void checkCorruptedHeapify() {
    final int k = 50;
    final int d = 250;
    final FrequentDirections fd = FrequentDirections.newInstance(k, d);
    byte[] sketchBytes = fd.toByteArray();
    WritableMemory mem = WritableMemory.wrap(sketchBytes);

    final FrequentDirections rebuilt = FrequentDirections.heapify(mem);
    assertTrue(rebuilt.isEmpty());
    println(PreambleUtil.preambleToString(mem));

    // corrupt the serialization version
    mem.putByte(PreambleUtil.SER_VER_BYTE, (byte) 0);
    try {
      FrequentDirections.heapify(mem);
    } catch (final IllegalArgumentException e) {
      // expected
    }

    // corrupt the family ID, after grabbing fresh bytes
    sketchBytes = fd.toByteArray();
    mem = WritableMemory.wrap(sketchBytes);
    mem.putByte(PreambleUtil.FAMILY_BYTE, (byte) 0);
    try {
      FrequentDirections.heapify(mem);
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void checkInsufficientMemory() {
    // no capacity
    byte[] bytes = new byte[0];
    Memory mem = Memory.wrap(bytes);
    try {
      FrequentDirections.heapify(mem);
    } catch (final IllegalArgumentException e) {
      // expected
    }

    // capacity smaller than prelongs size
    final FrequentDirections fd = FrequentDirections.newInstance(10, 50);
    bytes = fd.toByteArray();
    bytes = Arrays.copyOf(bytes, bytes.length - 1);
    mem = Memory.wrap(bytes);
    try {
      FrequentDirections.heapify(mem);
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  /**
   * println the message
   * @param msg the message
   */
  private void println(final String msg) {
    //System.out.println(msg);
  }
}

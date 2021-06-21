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

import static org.apache.datasketches.memory.UnsafeUtil.unsafe;

import org.apache.datasketches.memory.Memory;
import org.apache.datasketches.memory.WritableMemory;
import org.apache.datasketches.vector.MatrixFamily;
import org.checkerframework.checker.nullness.qual.NonNull;

/**
 * Computes mean and variance for each of d dimensions of an input vector using Welford's online algorithm,
 * as described in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 * <p>
 * For serialized images, multi-byte integers (<tt>int</tt> and <tt>long</tt>) are stored in native byte
 * order. All <tt>byte</tt> values are treated as unsigned.</p>
 *
 * <p>An empty object requires 8 bytes. A non-empty sketch requires 16 bytes
 * of preamble.</p>
 *
 * <pre>
 * Long || Start Byte Adr:
 * Adr:
 *      ||       0        |    1   |    2   |    3   |   4   |    5   |    6   |   7   |
 *  0   || Preamble_Longs | SerVer | FamID  | Flags  |---------Vector Dim. (d)---------|
 *
 *      ||       8        |   9    |   10   |   11   |   12  |   13   |   14   |  15   |
 *  1   ||-------------------------Num. Vectors Processed (n)--------------------------|
 *
 *      ||       16       |   17   |   18   |   19   |   20  |   21   |   22   |  23   |
 *  2   ||---------------------------Intercept (target mean)---------------------------|
 *
 *      ||       24       |   25   |   26   |   27   |   28  |   29   |   30   |  31   |
 *  3   ||-----------------------------start of mean array-----------------------------|
 * </pre>
 *
 * @author Jon Malkin

 */
public class VectorNormalizer {
  private final int d_;
  private final double[] mean_;
  private final double[] M2_;
  private double intercept_;
  private long n_;

  // Preamble byte Addresses
  static final int PREAMBLE_LONGS_BYTE   = 0;
  static final int SER_VER_BYTE          = 1;
  static final int FAMILY_BYTE           = 2;
  static final int FLAGS_BYTE            = 3;
  static final int D_INT                 = 4;
  static final int N_LONG                = 8;

  // flag bit masks
  static final int EMPTY_FLAG_MASK        = 4;

  // Other constants
  static final int SER_VER                = 1;


  /**
   * Creates a new, empty VectorNormalizer
   * @param d The number of dimensions the VectorNormalizer holds
   */
  public VectorNormalizer(final int d) {
    if (d < 1)
      throw new IllegalArgumentException("d cannot be < 1. Found: " + d);

    d_ = d;
    n_ = 0;
    intercept_ = 0.0;
    mean_ = new double[d_];
    M2_ = new double[d_];
  }

  /**
   * Copy constructor
   * @param other The VectorNormalizer to copy
   */
  public VectorNormalizer(final VectorNormalizer other) {
    d_ = other.d_;
    n_ = other.n_;
    intercept_ = other.intercept_;
    mean_ = other.mean_.clone();
    M2_ = other.M2_.clone();
  }

  private VectorNormalizer(final int d, final long n, final double[] mean, final double[] M2, final double intercept) {
    d_ = d;
    n_ = n;
    intercept_ = intercept;
    mean_ = mean;
    M2_ = M2;
  }

  /**
   * Instantiates a VectorNormalizer object from a serialized image
   * @param srcMem Memory containing the serialized image of a VectorNormalizer object
   * @return A VectorNormalizer, or null if srcMem is null
   */
  static VectorNormalizer heapify(final Memory srcMem) {
    if (srcMem == null) { return null; }

    final int preLongs = getAndCheckPreLongs(srcMem);
    if (preLongs < MatrixFamily.VECTORNORMALIZER.getMinPreLongs()
        || preLongs > MatrixFamily.VECTORNORMALIZER.getMaxPreLongs()) {
      throw new IllegalArgumentException("Possible corruption: Invalid number of preamble longs: " + preLongs);
    }

    final int serVer = extractSerVer(srcMem);
    if (serVer != SER_VER) {
      throw new IllegalArgumentException("Invalid serialization version: " + serVer);
    }

    final int family = extractFamilyID(srcMem);
    if (family != MatrixFamily.VECTORNORMALIZER.getID()) {
      throw new IllegalArgumentException("Possible corruption: Family id (" + family + ") "
          + "is not a VectorNormalization image");
    }

    final boolean empty = (extractFlags(srcMem) & EMPTY_FLAG_MASK) > 0;
    final int d = extractD(srcMem);
    if (d < 1)
      throw new IllegalArgumentException("Possible corruption: d cannot be < 1. Found: " + d);

    if (empty) {
      if (preLongs != MatrixFamily.VECTORNORMALIZER.getMinPreLongs()) {
        throw new IllegalArgumentException("Possible corruption: Empty flag set but header indicates image has data.");
      }
      return new VectorNormalizer(d);
    }

    if (preLongs == MatrixFamily.VECTORNORMALIZER.getMinPreLongs()) {
      throw new IllegalArgumentException("Possible corruption: Non-empty image too small to contain serialized data");
    }

    final long n = extractN(srcMem);
    if (n <= 0)
      throw new IllegalArgumentException("Possible corruption: n must be positive for a non-empty sketch. Found: " + n);

    long offsetBytes = (long) preLongs * Long.BYTES;

    // check capacity for the rest
    final long bytesNeeded = offsetBytes + (((2L * d) + 1) * Double.BYTES);
    if (srcMem.getCapacity() < bytesNeeded) {
      throw new IllegalArgumentException(
          "Possible Corruption: Size of Memory not large enough: Size: " + srcMem.getCapacity()
              + ", Required: " + bytesNeeded);
    }

    final double intercept = srcMem.getDouble(offsetBytes);
    offsetBytes += Double.BYTES;

    final double[] mean = new double[d];
    srcMem.getDoubleArray(offsetBytes, mean, 0, d);
    offsetBytes += (long) d * Double.BYTES;

    final double[] M2 = new double[d];
    srcMem.getDoubleArray(offsetBytes, M2, 0, d);

    return new VectorNormalizer(d, n, mean, M2, intercept);
  }

  /**
   * Returns an array of bytes with a serialized image of this object.
   * @return A <tt>byte[]</tt> containing the serialized image of this object.
   */
  public byte[] toByteArray() {
    final boolean empty = isEmpty();
    final int familyId = MatrixFamily.VECTORNORMALIZER.getID();

    final int preLongs = empty
        ? MatrixFamily.VECTORNORMALIZER.getMinPreLongs()
        : MatrixFamily.VECTORNORMALIZER.getMaxPreLongs();

    final int outBytes = (preLongs * Long.BYTES) + (empty ? 0 : (1 + 2 * d_) * Double.BYTES);
    final byte[] outArr = new byte[outBytes];
    final WritableMemory memOut = WritableMemory.wrap(outArr);
    final Object memObj = memOut.getArray();
    final long memAddr = memOut.getCumulativeOffset(0L);

    insertPreLongs(memObj, memAddr, preLongs);
    insertSerVer(memObj, memAddr, SER_VER);
    insertFamilyID(memObj, memAddr, familyId);
    insertFlags(memObj, memAddr, (empty ? EMPTY_FLAG_MASK : 0));
    insertD(memObj, memAddr, d_);

    if (!empty) {
      insertN(memObj, memAddr, n_);
      long offset = (long) preLongs * Long.BYTES;
      memOut.putDouble(offset, intercept_);
      offset += Double.BYTES;
      memOut.putDoubleArray(offset, mean_, 0, d_);
      offset += (long) d_ * Double.BYTES;
      memOut.putDoubleArray(offset, M2_, 0, d_);
    }

    return outArr;
  }

  /**
   * Returns true if the object has no data, otherwise false
   * @return True if the object has no data, otherwise false.
   */
  public boolean isEmpty() {
    return n_ == 0;
  }

  /**
   * Returns the number of dimensions configured for this object
   * @return The number of dimensions
   */
  public long getD() {
    return d_;
  }

  /**
   * Returns the number of input vectors processed by this object
   * @return The number of input vectors processed
   */
  public long getN() {
    return n_;
  }

  /**
   * Returns the array of means held by this object
   * @return The array of means
   */
  public double[] getMean() {
    if (n_ == 0) {
      final double[] result = new double[d_];
      for (int i = 0; i < d_; ++i) {
        result[i] = Double.NaN;
      }
      return result;
    } else {
      return mean_.clone();
    }
  }

  /**
   * Returns the mean of the target value, aka the intercept
   * @return Mean of the target value
   */
  public double getIntercept() {
    if (n_ == 0)
      return Double.NaN;
    else
      return intercept_;
  }

  /**
   * Returns the sample variance array represented in this object. Returns an array of NaN if N = 0 and an
   * array of zeros if N = 1.
   * @return The sample variance array represented in this object
   */
  public double[] getSampleVariance() {
    if (n_ == 0) {
      final double[] result = new double[d_];
      for (int i = 0; i < d_; ++i) {
        result[i] = Double.NaN;
      }
      return result;
    } else if (n_ == 1) {
      return new double[d_]; // array of zeros
    } else { // n_ > 1
      double[] result = M2_.clone();
      for (int i = 0; i < d_; ++i) {
        result[i] = M2_[i] / n_;
      }
      return result;
    }
  }

  /**
   * Returns the population variance array represented in this object. Returns an array of NaN if N = 0 and an
   * array of zeros if N = 1.
   * @return The population variance array represented in this object
   */
  public double[] getPopulationVariance() {
    if (n_ == 0) {
      final double[] result = new double[d_];
      for (int i = 0; i < d_; ++i) {
        result[i] = Double.NaN;
      }
      return result;
    } else if (n_ == 1) {
      return new double[d_]; // array of zeros
    } else { // n_ > 1
      double[] result = M2_.clone();
      for (int i = 0; i < d_; ++i) {
        result[i] = M2_[i] / (n_ - 1);
      }
      return result;
    }
  }

  public void update(final double[] x, final double target) {
    if (x == null)
      return;

    if (x.length != d_) {
      throw new IllegalArgumentException("Input vector length must be " + d_ + ". Found: " + x.length );
    }

    ++n_;
    for (int i = 0; i < d_; ++i) {
      double d1 = x[i] - mean_[i];  // x_i - oldMean_i
      mean_[i] += d1 / n_;
      double d2 = x[i] - mean_[i];  // x_i - newMean_i
      M2_[i] += d1 * d2;
    }

    double delta = target - intercept_;
    intercept_ += delta / n_;
  }

  public void merge(@NonNull final VectorNormalizer other) {
    if (other.d_ != d_)
      throw new IllegalArgumentException("Input VectorNormalizer must have d= " + d_ + ". Found: " + other.d_);

    long combinedN = n_ + other.n_;
    double varCountScalar = (n_ * other.n_) / (double) combinedN; // n_A * n_B / (n_A + n_B)
    intercept_ = ((n_ * intercept_) + (other.n_ * other.intercept_)) / combinedN;
    for (int i = 0; i < d_; ++i) {
      double meanDiff = other.mean_[i] - mean_[i];
      mean_[i] = ((n_ * mean_[i]) + (other.n_ * other.mean_[i])) / combinedN;
      M2_[i] += other.M2_[i] + meanDiff * meanDiff * varCountScalar;
    }
    n_ += other.n_;
  }

  public int getSerializedSizeBytes() {
    if (n_ == 0) {
      return MatrixFamily.VECTORNORMALIZER.getMinPreLongs() * Long.BYTES;
    } else {
      return (MatrixFamily.VECTORNORMALIZER.getMaxPreLongs()) * Long.BYTES + ((1 + 2 * d_) * Double.BYTES);
    }
  }

  // Extraction methods
  static int extractPreLongs(final Memory mem) {
    return mem.getInt(PREAMBLE_LONGS_BYTE) & 0xFF;
  }

  static int extractSerVer(final Memory mem) {
    return mem.getInt(SER_VER_BYTE) & 0xFF;
  }

  static int extractFamilyID(final Memory mem) {
    return mem.getByte(FAMILY_BYTE) & 0xFF;
  }

  static int extractFlags(final Memory mem) {
    return mem.getByte(FLAGS_BYTE) & 0xFF;
  }

  static int extractD(final Memory mem) {
    return mem.getInt(D_INT);
  }

  static long extractN(final Memory mem) {
    return mem.getLong(N_LONG);
  }


  // Insertion methods
  private void insertPreLongs(final Object memObj, final long memAddr, final int preLongs) {
    unsafe.putByte(memObj, memAddr + PREAMBLE_LONGS_BYTE, (byte) preLongs);
  }

  private void insertSerVer(final Object memObj, final long memAddr, final int serVer) {
    unsafe.putByte(memObj, memAddr + SER_VER_BYTE, (byte) serVer);
  }

  private void insertFamilyID(final Object memObj, final long memAddr, final int matrixFamId) {
    unsafe.putByte(memObj, memAddr + FAMILY_BYTE, (byte) matrixFamId);
  }

  private void insertFlags(final Object memObj, final long memAddr, final int flags) {
    unsafe.putByte(memObj, memAddr + FLAGS_BYTE, (byte) flags);
  }

  private void insertD(final Object memObj, final long memAddr, final int d) {
    unsafe.putInt(memObj, memAddr + D_INT, d);
  }

  private void insertN(final Object memObj, final long memAddr, final long n) {
    unsafe.putLong(memObj, memAddr + N_LONG, n);
  }

  /**
   * Checks Memory for capacity to hold the preamble and returns the extracted preLongs.
   * @param mem the given Memory
   * @return the extracted prelongs value.
   */
  private static int getAndCheckPreLongs(final Memory mem) {
    final long cap = mem.getCapacity();
    if (cap < Long.BYTES) { throwNotBigEnough(cap, Long.BYTES); }
    final int preLongs = extractPreLongs(mem);
    final int required = Math.max(preLongs << 2, Long.BYTES);
    if (cap < required) { throwNotBigEnough(cap, required); }
    return preLongs;
  }

  private static void throwNotBigEnough(final long cap, final int required) {
    throw new IllegalArgumentException(
        "Possible Corruption: Size of byte array or Memory not large enough: Size: " + cap
            + ", Required: " + required);
  }
}

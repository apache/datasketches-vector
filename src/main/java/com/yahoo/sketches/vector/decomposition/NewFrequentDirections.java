/*
 * Copyright 2017, Yahoo, Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package com.yahoo.sketches.vector.decomposition;

import static com.yahoo.memory.UnsafeUtil.LS;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.EMPTY_FLAG_MASK;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.SER_VER;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.extractFamilyID;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.extractFlags;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.extractK;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.extractN;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.extractNumColumns;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.extractNumRows;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.extractSVAdjustment;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.extractSerVer;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.getAndCheckPreLongs;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.insertFamilyID;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.insertFlags;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.insertK;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.insertN;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.insertNumColumns;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.insertNumRows;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.insertPreLongs;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.insertSVAdjustment;
import static com.yahoo.sketches.vector.decomposition.PreambleUtil.insertSerVer;

import org.ojalgo.matrix.store.PrimitiveDenseStore;

import com.yahoo.memory.Memory;
import com.yahoo.memory.WritableMemory;
import com.yahoo.sketches.vector.MatrixFamily;
import com.yahoo.sketches.vector.matrix.Matrix;
import com.yahoo.sketches.vector.matrix.MatrixBuilder;
import com.yahoo.sketches.vector.matrix.MatrixType;

/**
 * This class implements the Frequent Directions algorithm proposed by Edo Liberty in "Simple and
 * Deterministic Matrix Sketches," KDD 2013. The sketch provides an approximation to the singular
 * value decomposition of a matrix with deterministic error bounds on the error between the
 * approximation and the optimal rank-k matrix decomposition.
 *
 * @author Jon Malkin
 */
public final class NewFrequentDirections {
  private static final MatrixType DEFAULT_MATRIX_TYPE = MatrixType.OJALGO;
  private static final SVDAlgo DEFAULT_SVD_ALGO = SVDAlgo.FULL;

  private final int k_;
  private final int l_;
  private final int d_;
  private long n_;

  private double svAdjustment_;

  private Matrix B_;

  private SVDAlgo algo_ = DEFAULT_SVD_ALGO;

  transient private int nextZeroRow_;
  transient private MatrixOps svd_;                     // avoids re-initializing

  /**
   * Creates a new instance of a Frequent Directions sketch using the default Linear Algebra backing library
   * @param k Number of dimensions (rows) in the sketch output
   * @param d Number of dimensions per input vector (columns)
   * @return An empty Frequent Directions sketch
   */
  public static NewFrequentDirections newInstance(final int k, final int d) {
    return new NewFrequentDirections(k, d, null, DEFAULT_MATRIX_TYPE);
  }

  /**
   * Creates a new instance of a Frequent Directions sketch using a specific MatrixType
   * @param k Number of dimensions (rows) in the sketch output
   * @param d Number of dimensions per input vector (columns)
   * @param type MatrixType to use for backing matrix. Impacts choice of SVD library.
   * @return An empty Frequent Directions sketch
   */
  public static NewFrequentDirections newInstance(final int k, final int d, final MatrixType type) {
    return new NewFrequentDirections(k, d, null, type);
  }

  /**
   * Instantiates a Frequent Directions sketch from a serialized image.
   * @param srcMem Memory containing the serialized image of a Frequent Directions sketch
   * @return A Frequent Directions sketch
   */
  public static NewFrequentDirections heapify(final Memory srcMem) {
    return heapify(srcMem, DEFAULT_MATRIX_TYPE);
  }

  /**
   * Instantiates a Frequent Directions sketch from a serialized image using a specific MatrixType.
   * @param srcMem Memory containing the serialized image of a Frequent Directions sketch
   * @param type The MatrixType to use with this instance
   * @return A Frequent Directions sketch
   */
  public static NewFrequentDirections heapify(final Memory srcMem, final MatrixType type) {
    final int preLongs = getAndCheckPreLongs(srcMem);
    final int serVer = extractSerVer(srcMem);
    if (serVer != SER_VER) {
      throw new IllegalArgumentException("Invalid serialization version: " + serVer);
    }

    final int family = extractFamilyID(srcMem);
    if (family != MatrixFamily.FREQUENTDIRECTIONS.getID()) {
      throw new IllegalArgumentException("Possible corruption: Family id (" + family + ") "
              + "is not a FrequentDirections sketch");
    }

    final int k = extractK(srcMem);
    final int numRows = extractNumRows(srcMem);
    final int d = extractNumColumns(srcMem);
    final boolean empty = (extractFlags(srcMem) & EMPTY_FLAG_MASK) > 0;

    if (empty) {
      return new NewFrequentDirections(k, d);
    }

    final long offsetBytes = preLongs * Long.BYTES;
    final long mtxBytes = srcMem.getCapacity() - offsetBytes;
    final Matrix B = Matrix.heapify(srcMem.region(offsetBytes, mtxBytes), type);
    assert B != null;

    final NewFrequentDirections fd = new NewFrequentDirections(k, d, B, B.getMatrixType());
    fd.n_ = extractN(srcMem);
    fd.nextZeroRow_ = numRows;
    fd.svAdjustment_ = extractSVAdjustment(srcMem);

    return fd;
  }

  private NewFrequentDirections(final int k, final int d) {
    this(k, d, null, DEFAULT_MATRIX_TYPE);
  }

  // uses MatrixType of B, if present, otherwise falls back to type input
  private NewFrequentDirections(final int k, final int d, final Matrix B, final MatrixType type) {
    if (k < 1) {
      throw new IllegalArgumentException("Number of projected dimensions must be at least 1");
    }
    if (d < 1) {
      throw new IllegalArgumentException("Number of feature dimensions must be at least 1");
    }

    k_ = k;
    l_ = 2 * k;
    d_ = d;

    if (d_ < l_) {
      throw new IllegalArgumentException("Running with d < 2k not yet supported");
    }

    svAdjustment_ = 0.0;

    nextZeroRow_ = 0;
    n_ = 0;

    if (B == null) {
      B_ = new MatrixBuilder().setType(type).build(l_, d_);
    } else {
      B_ = B;
    }

    final int svDim = Math.min(l_, d_);
  }

  /**
   * Update sketch with a dense input vector of exactly d dimensions.
   * @param vector A dense input vector representing one row of the input matrix
   */
  public void update(final double[] vector) {
    if (vector == null) {
      return;
    }

    if (vector.length != d_) {
      throw new IllegalArgumentException("Input vector has wrong number of dimensions. Expected "
              + d_ + "; found " + vector.length);
    }

    if (nextZeroRow_ == l_) {
      reduceRank();
    }

    B_.setRow(nextZeroRow_, vector);

    ++n_;
    ++nextZeroRow_;
  }

  /**
   * Merge a Frequent Directions sketch into the current one.
   * @param fd A Frequent Direction sketch to be merged.
   */
  public void update(final NewFrequentDirections fd) {
    if (fd == null || fd.nextZeroRow_ == 0) {
      return;
    }

    if ((fd.d_ != d_) || (fd.k_ < k_)) {
      throw new IllegalArgumentException("Incoming sketch must have same number of dimensions "
              + "and no smaller a value of k");
    }

    for (int m = 0; m < fd.nextZeroRow_; ++m) {
      if (nextZeroRow_ == l_) {
        reduceRank();
      }

      B_.setRow(nextZeroRow_, fd.B_.getRow(m));
      ++nextZeroRow_;
    }

    n_ += fd.n_;
    svAdjustment_ += fd.svAdjustment_;
  }

  /**
   * Checks if the sketch is empty, specifically whether it has processed any input data.
   * @return True if the sketch has not yet processed any input
   */
  public boolean isEmpty() {
    return n_ == 0;
  }

  /**
   * Returns the target number of dimensions, k, for this sketch.
   * @return The sketch's configured k value
   */
  public int getK() { return k_; }

  /**
   * Returns the number of dimensions per input vector, d, for this sketch.
   * @return The sketch's configured number of dimensions per input
   */
  public int getD() { return d_; }

  /**
   * Returns the total number of items this sketch has seen.
   * @return The number of items processed by the sketch.
   */
  public long getN() { return n_; }

  /**
   * Sets the SVD algorithm to use, allowing exact or approximate computation
   * @param algo The SVDAlgo type to use
   */
  public void setSVDAlgo(final SVDAlgo algo) {
    algo_ = algo;
  }

  /**
   * Returns the singular values of the sketch, adjusted for the mass subtracted off during the
   * algorithm.
   * @return An array of singular values.
   */
  public double[] getSingularValues() {
    return getSingularValues(false);
  }

  /**
   * Returns the singular values of the sketch, optionally adjusting for any mass subtracted off
   * during the algorithm.
   * @param compensative If true, adjusts for mass subtracted during the algorithm, otherwise
   *                     uses raw singular values.
   * @return An array of singular values.
   */
  public double[] getSingularValues(final boolean compensative) {
    if (svd_ == null) {
      svd_ = MatrixOps.newInstance(B_, algo_, k_);
    }

    final double[] sv = svd_.getSingularValues(B_);

    double medianSVSq = sv[k_ - 1]; // (l_/2)th item, not yet squared
    medianSVSq *= medianSVSq;
    final double tmpSvAdj = svAdjustment_ + medianSVSq;
    final double[] svList = new double[k_];

    for (int i = 0; i < k_ - 1; ++i) {
      final double val = sv[i];
      double adjSqSV = val * val - medianSVSq;
      if (compensative) { adjSqSV += tmpSvAdj; }
      svList[i] = adjSqSV < 0 ? 0.0 : Math.sqrt(adjSqSV);
    }

    return svList;
  }

  /**
   * Returns an orthonormal projection Matrix that can be used to project input vectors into the
   * k-dimensional space represented by the sketch.
   * @return An orthonormal Matrix object
   */
  public Matrix getProjectionMatrix() {
    if (svd_ == null) {
      svd_ = MatrixOps.newInstance(B_, algo_, k_);
    }

    return svd_.getVt();
  }

  /**
   * Reduces matrix rank to no more than k, regardless of whether the sketch has reached its
   * internal capacity. Has no effect if there are no more than k active rows.
   */
  public void forceReduceRank() {
    if (nextZeroRow_ > k_) {
      reduceRank();
    }
  }

  /**
   * Returns a Matrix with the current state of the sketch. Call <tt>trim()</tt> first to ensure
   * no more than k rows. Equivalent to calling <tt>getResult(false)</tt>.
   * @return A Matrix representing the data in this sketch
   */
  public Matrix getResult() {
    return getResult(false);
  }

  /**
   * Returns a Matrix with the current state of the sketch. Call <tt>trim()</tt> first to ensure
   * no more than k rows. If compensative, uses only the top k singular values. If not applying compensation factor,
   * this method returns the actual data object meaning any changes to the result data will corrupt the sketch.
   * @param compensative If true, returns a copy of the data matrix after applying adjustment to singular values
   *                     based on the cumulative weight subtracted off. If false, returns the actual data matrix.
   * @return A Matrix representing the data in this sketch; the actual
   */
  public Matrix getResult(final boolean compensative) {
    if (isEmpty()) {
      return null;
    }

    final PrimitiveDenseStore result;

    if (compensative) {
      if (svd_ == null) {
        svd_ = MatrixOps.newInstance(B_, algo_, k_);
      }

      return svd_.applyAdjustment(B_, svAdjustment_);
    } else {
      //result = PrimitiveDenseStore.FACTORY.copy(B_);
      return Matrix.wrap(B_);
    }

    //return Matrix.wrap(result);
  }

  /**
   * Resets the sketch to its virgin state.
   */
  public void reset() {
    n_ = 0;
    nextZeroRow_ = 0;
    svAdjustment_ = 0.0;
  }

  /**
   * Returns a serialized representation of the sketch.
   * @return A serialized representation of the sketch.
   */
  public byte[] toByteArray() {
    final boolean empty = isEmpty();
    final int familyId = MatrixFamily.FREQUENTDIRECTIONS.getID();

    final Matrix wrapB = Matrix.wrap(B_);

    final int preLongs = empty
            ? MatrixFamily.FREQUENTDIRECTIONS.getMinPreLongs()
            : MatrixFamily.FREQUENTDIRECTIONS.getMaxPreLongs();

    final int mtxBytes = empty ? 0 : wrapB.getCompactSizeBytes(nextZeroRow_, d_);
    final int outBytes = (preLongs * Long.BYTES) + mtxBytes;

    final byte[] outArr = new byte[outBytes];
    final WritableMemory memOut = WritableMemory.wrap(outArr);
    final Object memObj = memOut.getArray();
    final long memAddr = memOut.getCumulativeOffset(0L);

    insertPreLongs(memObj, memAddr, preLongs);
    insertSerVer(memObj, memAddr, SER_VER);
    insertFamilyID(memObj, memAddr, familyId);
    insertFlags(memObj, memAddr, (empty ? EMPTY_FLAG_MASK : 0));
    insertK(memObj, memAddr, k_);
    insertNumRows(memObj, memAddr, nextZeroRow_);
    insertNumColumns(memObj, memAddr, d_);

    if (empty) {
      return outArr;
    }

    insertN(memObj, memAddr, n_);
    insertSVAdjustment(memObj, memAddr, svAdjustment_);

    memOut.putByteArray(preLongs * Long.BYTES,
            wrapB.toCompactByteArray(nextZeroRow_, d_), 0, mtxBytes);

    return outArr;
  }

  @Override
  public String toString() {
    return toString(false, false, false);
  }

  /**
   * Returns a human-readable summary of the sketch and, optionally, prints the singular values.
   * @param printSingularValues If true, prints sketch's data matrix
   * @return A String representation of the sketch.
   */
  public String toString(final boolean printSingularValues) {
    return toString(printSingularValues, false, false);
  }

  /**
   * Returns a human-readable summary of the sketch, optionally printing either the filled
   * or complete sketch matrix, and also optionally adjusting the singular values based on the
   * total weight subtacted during the algorithm.
   * @param printSingularValues If true, prints the sketch's singular values
   * @param printMatrix If true, prints the sketch's data matrix
   * @param applyCompensation If true, prints adjusted singular values
   * @return A String representation of the sketch.
   */
  public String toString(final boolean printSingularValues,
                         final boolean printMatrix,
                         final boolean applyCompensation) {
    final StringBuilder sb = new StringBuilder();

    final String thisSimpleName = this.getClass().getSimpleName();

    sb.append(LS);
    sb.append("### ").append(thisSimpleName).append(" INFO: ").append(LS);
    if (applyCompensation) {
      sb.append("Applying compensative adjustments to matrix values").append(LS);
    }
    sb.append("   k            : ").append(k_).append(LS);
    sb.append("   d            : ").append(d_).append(LS);
    sb.append("   l            : ").append(l_).append(LS);
    sb.append("   n            : ").append(n_).append(LS);
    sb.append("   numRows      : ").append(nextZeroRow_).append(LS);
    sb.append("   SV adjustment: ").append(svAdjustment_).append(LS);

    if (printSingularValues) {
      sb.append("   Singular Vals: ")
              .append(applyCompensation ? "(adjusted)" : "(unadjusted)").append(LS);
      final double[] sv = getSingularValues(applyCompensation);
      for (int i = 0; i < Math.min(k_, n_); ++i) {
        if (sv[i] > 0.0) {
          double val = sv[i];
          if (val > 0.0 && applyCompensation) {
            val = Math.sqrt(val * val + svAdjustment_);
          }

          sb.append("   \t").append(i).append(":\t").append(val).append(LS);
        }
      }
    }

    if (!printMatrix) {
      return sb.toString();
    }

    final int tmpColDim = (int) B_.getNumColumns();

    sb.append("   Matrix data  :").append(LS);
    sb.append(B_.getClass().getName());
    sb.append(" < ").append(nextZeroRow_).append(" x ").append(tmpColDim).append(" >");

    // First element
    //sb.append("\n{ { ").append(B_.getElement(0, 0));
    sb.append("\n{ { ").append(String.format("%.3f", B_.getElement(0, 0)));

    // Rest of the first row
    for (int j = 1; j < tmpColDim; j++) {
      //sb.append(",\t").append(B_.getElement(0, j));
      sb.append(",\t").append(String.format("%.3f", B_.getElement(0, j)));
    }

    // For each of the remaining rows
    for (int i = 1; i < nextZeroRow_; i++) {

      // First column
      //sb.append(" },\n{ ").append(B_.getElement(i, 0));
      sb.append(" },\n{ ").append(String.format("%.3f", B_.getElement(i, 0)));

      // Remaining columns
      for (int j = 1; j < tmpColDim; j++) {
        //sb.append(",\t").append(B_.getElement(i, j));
        sb.append(",\t").append(String.format("%.3f", B_.getElement(i, j)));
      }
    }

    // Finish
    sb.append(" } }").append(LS);
    sb.append("### END SKETCH SUMMARY").append(LS);

    return sb.toString();
  }

  int getNumRows() { return nextZeroRow_; }

  // exists for testing
  double getSvAdjustment() { return svAdjustment_; }

  private void reduceRank() {
    if (svd_ == null) {
      svd_ = MatrixOps.newInstance(B_, algo_, k_);
    }

    //System.out.println(toString(true, true, false));

    final double newSvAdjustment = svd_.reduceRank(B_);
    svAdjustment_ += newSvAdjustment;
    nextZeroRow_ = k_ - 1; //(int) Math.min(k_, n_); //svd_.getSingularValues().length;

    //System.out.println(toString(true, true, false));
    //System.exit(0);
  }
}

package com.yahoo.sketches.decomposition;

import static com.yahoo.memory.UnsafeUtil.LS;
import static com.yahoo.sketches.decomposition.PreambleUtil.EMPTY_FLAG_MASK;
import static com.yahoo.sketches.decomposition.PreambleUtil.extractFamilyID;
import static com.yahoo.sketches.decomposition.PreambleUtil.extractFlags;
import static com.yahoo.sketches.decomposition.PreambleUtil.extractK;
import static com.yahoo.sketches.decomposition.PreambleUtil.extractN;
import static com.yahoo.sketches.decomposition.PreambleUtil.extractNumColumns;
import static com.yahoo.sketches.decomposition.PreambleUtil.extractNumRows;
import static com.yahoo.sketches.decomposition.PreambleUtil.extractSVAdjustment;
import static com.yahoo.sketches.decomposition.PreambleUtil.extractSerVer;
import static com.yahoo.sketches.decomposition.PreambleUtil.getAndCheckPreLongs;
import static com.yahoo.sketches.decomposition.PreambleUtil.insertFamilyID;
import static com.yahoo.sketches.decomposition.PreambleUtil.insertFlags;
import static com.yahoo.sketches.decomposition.PreambleUtil.insertK;
import static com.yahoo.sketches.decomposition.PreambleUtil.insertN;
import static com.yahoo.sketches.decomposition.PreambleUtil.insertNumColumns;
import static com.yahoo.sketches.decomposition.PreambleUtil.insertNumRows;
import static com.yahoo.sketches.decomposition.PreambleUtil.insertPreLongs;
import static com.yahoo.sketches.decomposition.PreambleUtil.insertSVAdjustment;
import static com.yahoo.sketches.decomposition.PreambleUtil.insertSerVer;

import org.ojalgo.array.Array1D;
import org.ojalgo.matrix.decomposition.SingularValue;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
import org.ojalgo.matrix.store.SparseStore;

import com.yahoo.memory.Memory;
import com.yahoo.memory.WritableMemory;
import com.yahoo.sketches.MatrixFamily;
import com.yahoo.sketches.matrix.Matrix;
import com.yahoo.sketches.matrix.MatrixBuilder;

/**
 * This class implements the Frequent Directions algorithm proposed by Edo Liberty in "Simple and
 * Deterministic Matrix Sketches," KDD 2013. The sketch provides an approximation to the singular
 * value decomposition of a matrix with deterministic error bounds on the error between the
 * approximation and the optimal rank-k matrix decomposition.
 *
 * @author Jon Malkin
 */
public final class FrequentDirections {
  private final int k_;
  private final int l_;
  private final int d_;
  private long n_;

  private double svAdjustment_;

  private PrimitiveDenseStore B_;
  transient private int nextZeroRow_;

  transient private final double[] sv_;           // pre-allocated to fetch singular values
  transient private final SparseStore<Double> S_; // to hold singular value matrix

  /**
   * Creates a new instance of a Frequent Directions sketch.
   * @param k Number of dimensions (rows) in the sketch output
   * @param d Number of dimensions per input vector (columns)
   * @return An empty Frequent Directions sketch
   */
  public static FrequentDirections newInstance(final int k, final int d) {
    return new FrequentDirections(k, d);
  }

  /**
   * Instantiates a Frequent Directions sketch from a serialized image.
   * @param srcMem Memory containing the serialized image of a Frequent Directions sketch
   * @return A Frequent Directions sketch
   */
  public static FrequentDirections heapify(final Memory srcMem) {
    final int preLongs = getAndCheckPreLongs(srcMem);
    final int serVer = extractSerVer(srcMem);
    if (serVer != PreambleUtil.SER_VER) {
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
      return new FrequentDirections(k, d);
    }

    final long offsetBytes = preLongs * Long.BYTES;
    final long mtxBytes = srcMem.getCapacity() - offsetBytes;
    final Matrix B = Matrix.heapify(srcMem.region(offsetBytes, mtxBytes), MatrixBuilder.Algo.OJALGO);
    assert B != null;

    final FrequentDirections fd
            = new FrequentDirections(k, d, (PrimitiveDenseStore) B.getRawObject());
    fd.n_ = extractN(srcMem);
    fd.nextZeroRow_ = numRows;
    fd.svAdjustment_ = extractSVAdjustment(srcMem);

    return fd;
  }

  private FrequentDirections(final int k, final int d) {
    this(k, d, null);
  }

  private FrequentDirections(final int k, final int d, final PrimitiveDenseStore B) {
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
      B_ = PrimitiveDenseStore.FACTORY.makeZero(l_, d_);
    } else {
      B_ = B;
    }

    final int svDim = Math.min(l_, d_);
    sv_ = new double[svDim];
    S_ = SparseStore.makePrimitive(svDim, svDim);
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
      throw new IllegalArgumentException("Input vector has too few dimensions. Expected " + d_
              + "; found " + vector.length);
    }

    if (nextZeroRow_ == l_) {
      reduceRank();
    }

    // dense input so set all values
    for (int i = 0; i < vector.length; ++i) {
      B_.set(nextZeroRow_, i, vector[i]);
    }

    ++n_;
    ++nextZeroRow_;
  }

  /**
   * Merge a Frequent Directions sketch into the current one.
   * @param fd A Frequent Direction sketch to be merged.
   */
  public void update(final FrequentDirections fd) {
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

      final Array1D<Double> rv = fd.B_.sliceRow(m);
      for (int i = 0; i < rv.count(); ++i) {
        B_.set(nextZeroRow_, i, rv.get(i));
      }

      ++nextZeroRow_;
    }

    n_ += fd.n_;
    svAdjustment_ += fd.svAdjustment_;
  }

  /**
   * Checks if the sketch is empty, specifically whether it has processed any input data.
   * @return True if hte sketch has not yet processed any input
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
   * Returns the singular values of the sketch, adjusted for the mass subtracted off during the
   * algorithm.
   * @return An array of singular values.
   */
  public double[] getSingularValues() {
    return getSingularValues(true);
  }

  /**
   * Returns the singular values of the sketch, optionally adjusting for any mass subtracted off
   * during the algorithm.
   * @param compensative If true, adjusts for mass subtracted during the algorithm, otherwise
   *                     uses raw singular values.
   * @return As array of singular values.
   */
  public double[] getSingularValues(final boolean compensative) {
    final SingularValue<Double> svd = SingularValue.make(B_);
    svd.compute(B_);
    svd.getSingularValues(sv_);

    double medianSVSq = sv_[k_ - 1]; // (l_/2)th item, not yet squared
    medianSVSq *= medianSVSq;
    final double tmpSvAdj = svAdjustment_ + medianSVSq;
    final double[] svList = new double[k_];

    for (int i = 0; i < k_ - 1; ++i) {
      final double val = sv_[i];
      double adjSqSV = val * val - medianSVSq;
      if (compensative) { adjSqSV += tmpSvAdj; }
      svList[i] = adjSqSV < 0 ? 0.0 : Math.sqrt(adjSqSV);
    }

    return svList;
  }

  /**
   * Returns an orthonormal projection Matrix that can be use to project input vectors into the
   * k-dimensional space represented by the sketch.
   * @return An orthonormal Matrix object
   */
  public Matrix getProjectionMatrix() {
    final SingularValue<Double> svd = SingularValue.make(B_);
    svd.compute(B_);
    final MatrixStore<Double> m = svd.getQ2().transpose();

    // not super efficient...
    final Matrix result = Matrix.builder().build(k_, d_);
    for (int i = 0; i < k_ - 1; ++i) { // last SV is 0
      result.setRow(i, m.sliceRow(i).toRawCopy1D());
    }

    return result;
  }

  /**
   * Calls <tt>getResult(true, false)</tt>
   * @return A Matrix representing the data in this sketch
   */
  public Matrix getResult() {
    return getResult(true, false);
  }

  /**
   * Returns a Matrix with the sketch's estimate of the SVD of the input data.
   * @param compress If true, force compression down to no more than k vectors
   * @param compensative If true, applies adjustment to singular values based on the cumulative
   *                     weight subtracted off
   * @return A Matrix representing the data in this sketch
   */
  public Matrix getResult(final boolean compress, final boolean compensative) {
    if (isEmpty()) {
      return null;
    }

    if (compress && nextZeroRow_ > k_) {
      reduceRank();
    }

    final PrimitiveDenseStore result;

    if (compensative) {
      // in the event we just called reduceRank(), the high rows are already zeroed out so no need
      // to do so again
      final SingularValue<Double> svd = SingularValue.make(B_);
      svd.compute(B_);
      svd.getSingularValues(sv_);

      for (int i = 0; i < k_ - 1; ++i) {
        final double val = sv_[i];
        final double adjSV = Math.sqrt(val * val + svAdjustment_);
        S_.set(i, i, adjSV);
      }
      for (int i = k_ - 1; i < S_.countColumns(); ++i) {
        S_.set(i, i, 0.0);
      }

      //result = PrimitiveDenseStore.FACTORY.makeZero(l_, d_);
      result = PrimitiveDenseStore.FACTORY.makeZero(nextZeroRow_, d_);
      S_.multiply(svd.getQ2().transpose(), result);
    } else {
      result = PrimitiveDenseStore.FACTORY.makeZero(nextZeroRow_, d_);
      for (int i = 0; i < nextZeroRow_; ++i) {
        int j = 0;
        for (double d : B_.sliceRow(i)) {
          result.set(i, j++, d);
        }
      }
    }

    return Matrix.wrap(result);
  }

  /**
   * Resets the sketch to its virgin state.
   */
  public void reset() {
    n_ = 0;
    nextZeroRow_ = 0;
  }

  /**
   * Returns a serialized representation of the sketch. Equivalent to calling <tt>toByteArray
   * (true)</tt>.
   * <p>Note: May modify sketch state. If the sketch would store more than k rows, applies SVD to
   * compress the sketch to examply k rows.</p>
   * @return A serialized representation of the sketch.
   */
  public byte[] toByteArray() {
    return toByteArray(true);
  }

  /**
   * Returns a serialized representation of the sketch.
   * <p>Note: If compress is true, will modify sketch state if the sketch would store more than k
   * rows by applying SVD to compress the sketch to examply k rows.</p>
   * @param compress If true, compresses teh sketch to no more than k rows.
   * @return A serialized representation of the sketch.
   */
  public byte[] toByteArray(final boolean compress) {
    final boolean empty = isEmpty();
    final int serVer = 1;
    final int familyId = MatrixFamily.FREQUENTDIRECTIONS.getID();

    final Matrix wrapB = Matrix.wrap(B_);

    // project down to k rows to serialize, chasing the 2GB byte[] limit
    if (compress && nextZeroRow_ > k_) {
      reduceRank();
    }

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
    insertSerVer(memObj, memAddr, serVer);
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
   * Returns a human-readable summary of the sketch and, optionally, prints the raw data.
   * @param printMatrix If true, prints sketch's data matrix
   * @return A String representation of the sketch.
   */
  public String toString(final boolean printMatrix) {
    return toString(printMatrix, false, false);
  }

  /**
   * Returns a human-readable summary of the sketch, optionally printing either the filled
   * or complete sketch matrix, and also optionally adjusting the singular values based on the
   * total weight subtacted during the algorithm.
   * @param printMatrix If true, prints the sketch's data matrix
   * @param fullMatrix If true, prints all rows; if false, only non-empty rows
   * @param applyCompensation If true, prints adjusted singular values
   * @return A String representation of the sketch.
   */
  public String toString(final boolean printMatrix, final boolean fullMatrix,
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

    if (!printMatrix) {
      return sb.toString();
    }

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

    final Matrix mtx = Matrix.wrap(B_);

    final int tmpRowDim = fullMatrix ? nextZeroRow_ : Math.min(k_, nextZeroRow_);
    final int tmpColDim = (int) mtx.getNumColumns();

    sb.append("   Matrix data  :").append(LS);
    sb.append(mtx.getClass().getName());
    sb.append(" < ").append(tmpRowDim).append(" x ").append(tmpColDim).append(" >");

    // First element
    sb.append("\n{ { ").append(mtx.getElement(0, 0));

    // Rest of the first row
    for (int j = 1; j < tmpColDim; j++) {
      sb.append(",\t").append(mtx.getElement(0, j));
    }

    // For each of the remaining rows
    for (int i = 1; i < tmpRowDim; i++) {

      // First column
      sb.append(" },\n{ ").append(mtx.getElement(i, 0));

      // Remaining columns
      for (int j = 1; j < tmpColDim; j++) {
        sb.append(",\t").append(mtx.getElement(i, j));
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
    //++numReduce;

    final SingularValue<Double> svd = SingularValue.make(B_);
    svd.compute(B_);
    svd.getSingularValues(sv_);

    if (sv_.length >= k_) {
      double medianSVSq = sv_[k_ - 1]; // (l_/2)th item, not yet squared
      medianSVSq *= medianSVSq;
      svAdjustment_ += medianSVSq; // always track, even if not using compensative mode
      for (int i = 0; i < k_ - 1; ++i) {
        final double val = sv_[i];
        final double adjSqSV = val * val - medianSVSq;
        S_.set(i, i, adjSqSV < 0 ? 0.0 : Math.sqrt(adjSqSV));
      }
      for (int i = k_ - 1; i < S_.countColumns(); ++i) {
        S_.set(i, i, 0.0);
      }
      nextZeroRow_ = k_;
    } else {
      for (int i = 0; i < sv_.length; ++i) {
        S_.set(i, i, sv_[i]);
      }
      for (int i = sv_.length; i < S_.countColumns(); ++i) {
        S_.set(i, i, 0.0);
      }
      nextZeroRow_ = sv_.length;
      throw new RuntimeException("Running with d < 2k not yet supported");
    }

    S_.multiply(svd.getQ2().transpose()).supplyTo(B_);
  }

  /*
  private static double computeFrobNorm(final MatrixStore<Double> M) {
    double sum = 0.0;
    for (double d : M) {
      sum += d * d;
    }
    return Math.sqrt(sum);
  }

  private static double computeFrobNorm(final MatrixStore<Double> M, final int k) {
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
      for (double d : M.sliceRow(i)) {
        sum += d * d;
      }
    }
    return Math.sqrt(sum);
  }

  private static MatrixStore<Double> getKRows(final MatrixStore<Double> M, final int k) {
    PrimitiveDenseStore result = PrimitiveDenseStore.FACTORY.makeZero(k, M.countColumns());
    for (int i = 0; i < k; ++i) {
      int j = 0;
      for (double d : M.sliceRow(i)) {
        result.set(i, j++, d);
      }
    }
    return result;
  }
  */
}

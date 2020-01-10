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

import static org.apache.datasketches.memory.UnsafeUtil.unsafe;

import org.apache.datasketches.memory.Memory;
import org.apache.datasketches.vector.MatrixFamily;

/**
 * This class defines the preamble items structure and provides basic utilities for some of the key fields.
 *
 * <p>
 * The low significance bytes of this <tt>long</tt> items structure are on the right. Multi-byte
 * integers (<tt>int</tt> and <tt>long</tt>) are stored in native byte order. All <tt>byte</tt>
 * values are treated as unsigned.</p>
 *
 * <p>An empty Frequent Directions sketch requires 16 bytes. A non-empty sketch requires 32 bytes
 * of preamble. The matrix is dense and is expected to dominate storage.</p>
 *
 * <pre>
 * Long || Start Byte Adr:
 * Adr:
 *      ||    7   |    6   |    5   |    4   |    3   |    2   |    1   |     0              |
 *  0   ||----------Sketch Size (k)----------|  Flags | FamID  | SerVer |   Preamble_Longs   |
 *
 *      ||   15   |   14   |   13   |   12   |   11   |   10   |    9   |     8              |
 *  1   ||-----------Num. Columns------------|------Current Num Rows-------------------------|
 *
 *      ||   23   |   22   |   21   |   20   |   19   |   18   |   17   |    16              |
 *  2   ||-----------------------------Total Records Seen (n)--------------------------------|
 *
 *      ||   31   |   30   |   29   |   28   |   27   |   26   |   25   |    24              |
 *  3   ||------------------------------Total SV Adjustment----------------------------------|
 *  </pre>
 *
 * @author Jon Malkin
 */
@SuppressWarnings("restriction")
public final class PreambleUtil {

  /**
   * The java line separator character as a String.
   */
  private static final String LS = System.getProperty("line.separator");

  private PreambleUtil() {}

  // ###### DO NOT MESS WITH THIS FROM HERE ...
  // Preamble byte Addresses
  static final int PREAMBLE_LONGS_BYTE   = 0;
  static final int SER_VER_BYTE          = 1;
  static final int FAMILY_BYTE           = 2;
  static final int FLAGS_BYTE            = 3;
  static final int K_INT                 = 4;
  static final int NUM_ROWS_INT          = 8;
  static final int NUM_COLUMNS_INT       = 12;
  static final int N_LONG                = 16;
  static final int SV_ADJUSTMENT_DOUBLE  = 24;

  // flag bit masks
  static final int EMPTY_FLAG_MASK        = 4;

  // Other constants
  static final int SER_VER                = 1;


  /**
   * Returns a human readable string summary of the preamble state of the given Memory.
   * Note: other than making sure that the given Memory size is large
   * enough for just the preamble, this does not do much value checking of the contents of the
   * preamble as this is primarily a tool for debugging the preamble visually.
   *
   * @param mem the given Memory.
   * @return the summary preamble string.
   */
  public static String preambleToString(final Memory mem) {

    final int preLongs = getAndCheckPreLongs(mem);  // make sure we can get the assumed preamble
    final MatrixFamily family = MatrixFamily.idToFamily(extractFamilyID(mem));

    final int serVer = extractSerVer(mem);
    final int flags = extractFlags(mem);
    final String flagsStr = Integer.toBinaryString(flags) + ", " + flags;
    final boolean isEmpty = (flags & EMPTY_FLAG_MASK) > 0;

    final int k = extractK(mem);
    final int d = extractNumColumns(mem);
    final int numRows = extractNumRows(mem);

    long n = 0;
    double svAdjustment = 0.0;
    if (!isEmpty) {
      n = extractN(mem);
      svAdjustment = extractSVAdjustment(mem);
    }

    final StringBuilder sb = new StringBuilder();
    sb.append(LS)
            .append("### START ")
            .append(family.getFamilyName().toUpperCase())
            .append(" PREAMBLE SUMMARY").append(LS)
            .append("Byte  0: Preamble Longs       : ").append(preLongs).append(LS)
            .append("Byte  1: Serialization Version: ").append(serVer).append(LS)
            .append("Byte  2: Family               : ").append(family.toString()).append(LS)
            .append("Byte  3: Flags Field          : ").append(flagsStr).append(LS)
            .append("  EMPTY                       : ").append(isEmpty).append(LS)
            .append("Bytes  4-7 : Sketch Size (k)  : ").append(k).append(LS)
            .append("Bytes  8-11: Num Rows         : ").append(numRows).append(LS)
            .append("Bytes 12-15: Num Dimensions   : ").append(d).append(LS);

    if (!isEmpty) {
      sb.append("Bytes 16-23: Items Seen(n)    : ").append(n).append(LS);
      sb.append("Bytes 24-31: SV Adjustment    : ").append(svAdjustment).append(LS);
    }

    final long numBytes = (long)numRows * d * Double.BYTES;
    sb.append("TOTAL Sketch Bytes            : ").append(mem.getCapacity()).append(LS)
            .append("  Preamble Bytes              : ").append(preLongs << 3).append(LS)
            .append("  Data Bytes                  : ").append(numBytes).append(LS)
            .append("### END ")
            .append(family.getFamilyName().toUpperCase())
            .append(" PREAMBLE SUMMARY").append(LS);
    return sb.toString();
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

  static int extractK(final Memory mem) {
    return mem.getInt(K_INT);
  }

  static int extractNumRows(final Memory mem) {
    return mem.getInt(NUM_ROWS_INT);
  }

  static int extractNumColumns(final Memory mem) {
    return mem.getInt(NUM_COLUMNS_INT);
  }

  static long extractN(final Memory mem) {
    return mem.getLong(N_LONG);
  }

  static double extractSVAdjustment(final Memory mem) {
    return mem.getDouble(SV_ADJUSTMENT_DOUBLE);
  }


  // Insertion methods

  static void insertPreLongs(final Object memObj, final long memAddr, final int preLongs) {
    unsafe.putByte(memObj, memAddr + PREAMBLE_LONGS_BYTE, (byte) preLongs);
  }

  static void insertSerVer(final Object memObj, final long memAddr, final int serVer) {
    unsafe.putByte(memObj, memAddr + SER_VER_BYTE, (byte) serVer);
  }

  static void insertFamilyID(final Object memObj, final long memAddr, final int matrixFamId) {
    unsafe.putByte(memObj, memAddr + FAMILY_BYTE, (byte) matrixFamId);
  }

  static void insertFlags(final Object memObj, final long memAddr, final int flags) {
    unsafe.putByte(memObj, memAddr + FLAGS_BYTE, (byte) flags);
  }

  static void insertK(final Object memObj, final long memAddr, final int k) {
    unsafe.putInt(memObj, memAddr + K_INT, k);
  }

  static void insertNumRows(final Object memObj, final long memAddr, final int numRows) {
    unsafe.putInt(memObj, memAddr + NUM_ROWS_INT, numRows);
  }

  static void insertNumColumns(final Object memObj, final long memAddr, final int numColumns) {
    unsafe.putInt(memObj, memAddr + NUM_COLUMNS_INT, numColumns);
  }

  static void insertN(final Object memObj, final long memAddr, final long n) {
    unsafe.putLong(memObj, memAddr + N_LONG, n);
  }

  static void insertSVAdjustment(final Object memObj, final long memAddr, final double adj) {
    unsafe.putDouble(memObj, memAddr + SV_ADJUSTMENT_DOUBLE, adj);
  }


  /**
   * Checks Memory for capacity to hold the preamble and returns the extracted preLongs.
   * @param mem the given Memory
   * @return the extracted prelongs value.
   */
  static int getAndCheckPreLongs(final Memory mem) {
    final long cap = mem.getCapacity();
    if (cap < Long.BYTES) { throwNotBigEnough(cap, Long.BYTES); }
    final int preLongs = extractPreLongs(mem);
    final int required = Math.max(preLongs << 3, Long.BYTES);
    if (cap < required) { throwNotBigEnough(cap, required); }
    return preLongs;
  }

  private static void throwNotBigEnough(final long cap, final int required) {
    throw new IllegalArgumentException(
            "Possible Corruption: Size of byte array or Memory not large enough: Size: " + cap
                    + ", Required: " + required);
  }
}

/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2016 Matthias Langer (t3l@threelights.de)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package edu.latrobe

import breeze.linalg.DenseMatrix
import org.netlib.util._

/**
  * This class encapsulates low overhead wrappers around native LAPACK functions.
  */
private[latrobe] object _LAPACK {

  final val lapack = com.github.fommil.netlib.LAPACK.getInstance()

  /**
    * SGETRF - computes an LU factorization of a general M-by-N matrix A using
    * partial pivoting with row interchanges.
    *
    * Source:
    * http://www.math.utah.edu/software/lapack/lapack-d/dgetrf.html
    */
  @inline
  final def getrf(a: Array[Float], aOff: Int, aStride: Int, aRows: Int, aCols: Int)
  : Array[Int] = {
    val pivotIndices = new Array[Int](Math.min(aRows, aCols))
    val result       = new intW(0)
    lapack.sgetrf(
      aRows, aCols, a, aOff, Math.max(aStride, 1),
      pivotIndices, 0,
      result
    )
    assume(result.`val` >= 0)
    pivotIndices
  }

  /**
    * SGETRF - computes an LU factorization of a general M-by-N matrix A using
    * partial pivoting with row interchanges.
    *
    * Source:
    * http://www.math.utah.edu/software/lapack/lapack-d/dgetrf.html
    */
  @inline
  final def getrf(a: Array[Double], aOff: Int, aStride: Int, aRows: Int, aCols: Int)
  : Array[Int] = {
    val pivotIndices = new Array[Int](Math.min(aRows, aCols))
    val result       = new intW(0)
    lapack.dgetrf(
      aRows, aCols, a, aOff, Math.max(aStride, 1),
      pivotIndices, 0,
      result
    )
    assume(result.`val` >= 0)
    pivotIndices
  }

  final def getrf(a: DenseMatrix[Real]): Array[Int] = {
    require(!a.isTranspose)
    getrf(a.data, a.offset, a.majorStride, a.rows, a.cols)
  }

  /** DGETRI - compute the inverse of a matrix using the LU fac-
    * torization computed by DGETRF
    *
    * Source:
    * http://www.math.utah.edu/software/lapack/lapack-d/dgetri.html
    */
  final def getri(a: Array[Float], aOff: Int, aStride: Int, aOrder: Int,
                  pi: Array[Int], piOff: Int)
  : Unit = {
    require(pi != null)
    // TODO: This is the minimal size for this buffer. One could make it larger for performance. See documentation.
    val tmp    = new Array[Float](Math.max(aOrder, 1))
    val result = new intW(0)
    lapack.sgetri(
      aOrder, a, aOff, Math.max(aStride, 1),
      pi, piOff,
      tmp, 0, tmp.length,
      result
    )
    assume(result.`val` == 0)
  }

  /** DGETRI - compute the inverse of a matrix using the LU fac-
    * torization computed by DGETRF
    *
    * Source:
    * http://www.math.utah.edu/software/lapack/lapack-d/dgetri.html
    */
  final def getri(a: Array[Double], aOff: Int, aStride: Int, aOrder: Int,
                  pi: Array[Int], piOff: Int)
  : Unit = {
    require(pi != null)
    // TODO: This is the minimal size for this buffer. One could make it larger for performance. See documentation.
    val tmp    = new Array[Double](Math.max(aOrder, 1))
    val result = new intW(0)
    lapack.dgetri(
      aOrder, a, aOff, Math.max(aStride, 1),
      pi, piOff,
      tmp, 0, tmp.length,
      result
    )
    assume(result.`val` == 0)
  }

  final def getri(a: DenseMatrix[Real],
                  pi: Array[Int], piOff: Int): Unit = {
    require(!a.isTranspose && a.rows == a.cols)
    getri(
      a.data, a.offset, a.majorStride, a.rows,
      pi, piOff
    )
  }

  final def inv(a: DenseMatrix[Real]): Unit = {
    val pivotIndices = getrf(a)
    getri(a, pivotIndices, 0)
  }

}

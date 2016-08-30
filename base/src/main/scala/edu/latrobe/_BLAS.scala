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

import breeze.linalg._

/**
 * This class contains low overhead wrappers around native BLAS functions.
 */
private[latrobe] object _BLAS {

  final val blas = com.github.fommil.netlib.BLAS.getInstance()


  // ---------------------------------------------------------------------------
  //    asum: sum(abs(x))
  // ---------------------------------------------------------------------------
  @inline
  final def asum(x: Array[Float])
  : Float = blas.sasum(
    x.length,
    x, 0, 1
  )

  @inline
  final def asum(n: Int,
                 x: Array[Float], xOff: Int, xInc: Int)
  : Float = {
    require(xInc > 0)
    blas.sasum(
      n,
      x, xOff, xInc
    )
  }

  @inline
  final def asum(x: Array[Double])
  : Double = blas.dasum(
    x.length,
    x, 0, 1
  )

  @inline
  final def asum(n: Int,
                 x: Array[Double], xOff: Int, xInc: Int)
  : Double = {
    require(xInc > 0)
    blas.dasum(
      n,
      x, xOff, xInc
    )
  }


  // ---------------------------------------------------------------------------
  //    axpy: y = a * x + y
  // ---------------------------------------------------------------------------
  @inline
  final def axpy(a: Float,
                 x: Array[Float],
                 y: Array[Float])
  : Unit = {
    require(x.length == y.length)
    blas.saxpy(
      x.length,
      a,
      x, 0, 1,
      y, 0, 1
    )
  }

  @inline
  final def axpy(n: Int,
                 a: Float,
                 x: Array[Float], xOff: Int, xInc: Int,
                 y: Array[Float], yOff: Int, yInc: Int)
  : Unit = {
    require(xInc > 0 && yInc > 0)
    blas.saxpy(
      n,
      a,
      x, xOff, xInc,
      y, yOff, yInc
    )
  }

  @inline
  final def axpy(a: Double,
                 x: Array[Double],
                 y: Array[Double])
  : Unit = {
    require(x.length == y.length)
    blas.daxpy(
      x.length,
      a,
      x, 0, 1,
      y, 0, 1
    )
  }

  @inline
  final def axpy(n: Int,
                 a: Double,
                 x: Array[Double], xOff: Int, xInc: Int,
                 y: Array[Double], yOff: Int, yInc: Int)
  : Unit = {
    require(xInc > 0 && yInc > 0)
    blas.daxpy(
      n,
      a,
      x, xOff, xInc,
      y, yOff, yInc
    )
  }


  // ---------------------------------------------------------------------------
  //    copy: y = x
  // ---------------------------------------------------------------------------
  @inline
  final def copy(x: Array[Float],
                 y: Array[Float])
  : Unit = {
    require(x.length == y.length)
    blas.scopy(
      x.length,
      x, 0, 1,
      y, 0, 1
    )
  }

  @inline
  final def copy(n: Int,
                 x: Array[Float], xOff: Int, xInc: Int,
                 y: Array[Float], yOff: Int, yInc: Int)
  : Unit = {
    require(xInc > 0 && yInc > 0)
    blas.scopy(
      n,
      x, xOff, xInc,
      y, yOff, yInc
    )
  }

  @inline
  final def copy(x: Array[Double],
                 y: Array[Double])
  : Unit = {
    require(x.length == y.length)
    blas.dcopy(
      x.length,
      x, 0, 1,
      y, 0, 1
    )
  }

  @inline
  final def copy(n: Int,
                 x: Array[Double], xOff: Int, xInc: Int,
                 y: Array[Double], yOff: Int, yInc: Int)
  : Unit = {
    require(xInc > 0 && yInc > 0)
    blas.dcopy(
      n,
      x, xOff, xInc,
      y, yOff, yInc
    )
  }


  // ---------------------------------------------------------------------------
  //    dot: x' * y
  // ---------------------------------------------------------------------------
  @inline
  final def dot(x: Array[Float],
                y: Array[Float])
  : Float = {
    require(x.length == y.length)
    blas.sdot(
      x.length,
      x, 0, 1,
      y, 0, 1
    )
  }

  @inline
  final def dot(n: Int,
                x: Array[Float], xOff: Int, xInc: Int,
                y: Array[Float], yOff: Int, yInc: Int)
  : Float = {
    require(xInc > 0 && yInc > 0)
    blas.sdot(
      n,
      x, xOff, xInc,
      y, yOff, yInc
    )
  }

  @inline
  final def dot(x: Array[Double],
                y: Array[Double])
  : Double = {
    require(x.length == y.length)
    blas.ddot(
      x.length,
      x, 0, 1,
      y, 0, 1
    )
  }

  @inline
  final def dot(n: Int,
                x: Array[Double], xOff: Int, xInc: Int,
                y: Array[Double], yOff: Int, yInc: Int)
  : Double = {
    require(xInc > 0 && yInc > 0)
    blas.ddot(
      n,
      x, xOff, xInc,
      y, yOff, yInc
    )
  }


  // ---------------------------------------------------------------------------
  //    gbmv: z = alpha * x  * y + beta * z
  //    gbmv: z = alpha * x' * y + beta * z
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  //    gemm: C = alpha * op(A) * op(B) + beta * C
  // ---------------------------------------------------------------------------
  @inline
  final def gemm(alpha: Float,
                 a:     Array[Float], aOff: Int, aStride: Int, aRows: Int, aCols: Int, aTrans:  Boolean,
                 b:     Array[Float], bOff: Int, bStride: Int, bRows: Int, bCols: Int, bTrans:  Boolean,
                 beta:  Float,
                 c:     Array[Float], cOff: Int, cStride: Int, cRows: Int, cCols: Int)
  : Unit = {
    require(
      aStride > 0 &&
      bStride > 0 &&
      cStride > 0 &&
      aRows == cRows &&
      aCols == bRows &&
      bCols == cCols
    )
    blas.sgemm(
      if (aTrans) "T" else "N",
      if (bTrans) "T" else "N",
      aRows,
      bCols,
      aCols,
      alpha,
      a, aOff, aStride,
      b, bOff, bStride,
      beta,
      c, cOff, cStride
    )
  }

  @inline
  final def gemm(alpha: Double,
                 a:     Array[Double], aOff: Int, aStride: Int, aRows: Int, aCols: Int, aTrans: Boolean,
                 b:     Array[Double], bOff: Int, bStride: Int, bRows: Int, bCols: Int, bTrans: Boolean,
                 beta:  Double,
                 c:     Array[Double], cOff: Int, cStride: Int, cRows: Int, cCols: Int)
  : Unit = {
    require(
      aStride > 0 &&
      bStride > 0 &&
      cStride > 0 &&
      aRows == cRows &&
      aCols == bRows &&
      bCols == cCols
    )
    blas.dgemm(
      if (aTrans) "T" else "N",
      if (bTrans) "T" else "N",
      aRows,
      bCols,
      aCols,
      alpha,
      a, aOff, aStride,
      b, bOff, bStride,
      beta,
      c, cOff, cStride
    )
  }

  @inline
  final def gemm(alpha: Real,
                 a:     DenseMatrix[Real],
                 b:     DenseMatrix[Real])
  : DenseMatrix[Float] = {
    val c = DenseMatrix.zeros[Real](a.rows, b.cols)
    gemm(
      alpha,
      a,
      b,
      Real.zero,
      c
    )
    c
  }

  @inline
  final def gemm(alpha: Real,
                 a:     DenseMatrix[Real],
                 b:     DenseMatrix[Real],
                 beta:  Real,
                 c:     DenseMatrix[Real])
  : Unit = gemm(
    alpha,
    a, a.offset, a.rows, a.cols,
    b, b.offset, b.rows, b.cols,
    beta,
    c, c.offset, c.rows, c.cols
  )

  @inline
  final def gemm(alpha: Real,
                 a:     DenseMatrix[Real], aOff: Int, aRows: Int, aCols: Int,
                 b:     DenseMatrix[Real], bOff: Int, bRows: Int, bCols: Int,
                 beta:  Real,
                 c:     DenseMatrix[Real], cOff: Int, cRows: Int, cCols: Int)
  : Unit = {
    require(!c.isTranspose)
    gemm(
      alpha,
      a.data, aOff, a.majorStride, aRows, aCols, a.isTranspose,
      b.data, bOff, b.majorStride, bRows, bCols, b.isTranspose,
      beta,
      c.data, cOff, c.majorStride, cRows, cCols
    )
  }


  // ---------------------------------------------------------------------------
  //    gemv: z = alpha * x  * y + beta * z
  //    gemv: z = alpha * x' * y + beta * z
  // ---------------------------------------------------------------------------
  @inline
  final def gemv(alpha: Float,
                 a:     Array[Float], aOff: Int, aStride: Int, aRows: Int, aCols: Int, aTrans: Boolean,
                 x:     Array[Float], xOff: Int, xInc: Int, xRows: Int,
                 beta:  Float,
                 y:     Array[Float], yOff: Int, yInc: Int, yRows: Int)
  : Unit = {
    require(
      aStride > 0 &&
      xInc > 0 &&
      yInc > 0 &&
      aCols == xRows &&
      aCols == yRows
    )
    blas.sgemv(
      if (aTrans) "T" else "N",
      aRows,
      aCols,
      alpha,
      a, aOff, aStride,
      x, xOff, xInc,
      beta,
      y, yOff, yInc
    )
  }

  @inline
  final def gemv(alpha: Double,
                 a:     Array[Double], aOff: Int, aStride: Int, aRows: Int, aCols: Int, aTrans: Boolean,
                 x:     Array[Double], xOff: Int, xInc: Int, xRows: Int,
                 beta:  Double,
                 y:     Array[Double], yOff: Int, yInc: Int, yRows: Int)
  : Unit = {
    require(
      aStride > 0 &&
      xInc > 0 &&
      yInc > 0 &&
      aCols == xRows &&
      aCols == yRows
    )
    blas.dgemv(
      if (aTrans) "T" else "N",
      aRows,
      aCols,
      alpha,
      a, aOff, aStride,
      x, xOff, xInc,
      beta,
      y, yOff, yInc
    )
  }

  /*
  @inline
  final def gemv(alpha: Real,
                 a:     DenseMatrix[Real],
                 x:     DenseVector[Real])
  : DenseVector[Real] = {
    val y = DenseVector.zeros[Real](x.length)
    gemv(alpha, a, x, Real.zero, y)
    y
  }

  @inline
  final def gemv(alpha: Real,
                 a:     DenseMatrix[Real],
                 x:     DenseVector[Real],
                 beta:  Real,
                 y:     DenseVector[Real])
  : Unit = gemv(
    alpha,
    a, a.offset, a.rows, a.cols,
    x, x.offset, x.length,
    beta,
    y, y.offset, y.length
  )

  @inline
  final def gemv(alpha: Real,
                 a:     DenseMatrix[Real], aOff: Int, aRows: Int, aCols: Int,
                 x:     DenseVector[Real], xOff: Int, xRows: Int,
                 beta:  Real,
                 y:     DenseVector[Real], yOff: Int, yRows: Int)
  : Unit = gemv(
    alpha,
    a.data, aOff, a.majorStride, aRows, aCols, a.isTranspose,
    x.data, xOff, x.stride, xRows,
    beta,
    y.data, yOff, y.stride, yRows
  )
  */


  // ---------------------------------------------------------------------------
  //    ger: z = alpha * x * y' + z
  // ---------------------------------------------------------------------------


  // ---------------------------------------------------------------------------
  //    iamax: index_where(abs(x) = max(abs(x)))
  // ---------------------------------------------------------------------------
  @inline
  final def iamax(x: Array[Float])
  : Int = blas.isamax(
    x.length,
    x, 0, 1
  )

  @inline
  final def iamax(n: Int,
                  x: Array[Float], xOff: Int, xInc: Int)
  : Int = {
    require(xInc > 0)
    blas.isamax(
      n,
      x, xOff, xInc
    )
  }

  @inline
  final def iamax(x: Array[Double])
  : Int = blas.idamax(
    x.length,
    x, 0, 1
  )

  @inline
  final def iamax(n: Int,
                  x: Array[Double], xOff: Int, xInc: Int)
  : Int = {
    require(xInc > 0)
    blas.idamax(
      n,
      x, xOff, xInc
    )
  }


  // ---------------------------------------------------------------------------
  //    nrm2: sqrt(x' * x)
  // ---------------------------------------------------------------------------
  @inline
  final def nrm2(x: Array[Float])
  : Float = {
    blas.snrm2(
      x.length,
      x, 0, 1
    )
  }

  @inline
  final def nrm2(n: Int,
                 x: Array[Float], xOff: Int, xInc: Int)
  : Float = {
    require(xInc > 0)
    blas.snrm2(
      n,
      x, xOff, xInc
    )
  }

  @inline
  final def nrm2(x: Array[Double])
  : Double = {
    blas.dnrm2(
      x.length,
      x, 0, 1
    )
  }

  @inline
  final def nrm2(n: Int,
                 x: Array[Double], xOff: Int, xInc: Int)
  : Double = {
    require(xInc > 0)
    blas.dnrm2(
      n,
      x, xOff, xInc
    )
  }


  // ---------------------------------------------------------------------------
  //    rot:
  //    rotg:
  //    rotm:
  //    rotmg:
  //    dsbmv: z = alpha * x * y + beta * y
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  //    scal: a * x
  // ---------------------------------------------------------------------------
  @inline
  final def scal(a: Float, x: Array[Float])
  : Unit = blas.sscal(
    x.length,
    a,
    x, 0, 1
  )

  @inline
  final def scal(n: Int,
                 a: Float,
                 x: Array[Float], xOff: Int, xInc: Int)
  : Unit = {
    require(xInc > 0)
    blas.sscal(
      n,
      a,
      x, xOff, xInc
    )
  }

  @inline
  final def scal(a: Double, x: Array[Double])
  : Unit = blas.dscal(
    x.length,
    a,
    x, 0, 1
  )

  @inline
  final def scal(n: Int,
                 a: Double,
                 x: Array[Double], xOff: Int, xInc: Int)
  : Unit = {
    require(xInc > 0)
    blas.dscal(
      n,
      a,
      x, xOff, xInc
    )
  }


  // ---------------------------------------------------------------------------
  //    spmv: z = alpha * X * y + beta * z
  //    spr:  Y = alpha * x * x' + Y
  //    spr2: Z = alpha * x * y' + alpha * y * x' + Z
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  //    swap: tmp = x; x = y; y = tmp
  // ---------------------------------------------------------------------------
  @inline
  final def swap(x: Array[Float],
                 y: Array[Float])
  : Unit = {
    require(x.length == y.length)
    blas.sswap(
      x.length,
      x, 0, 1,
      y, 0, 1
    )
  }

  @inline
  final def swap(n: Int,
                 x: Array[Float], xOff: Int, xInc: Int,
                 y: Array[Float], yOff: Int, yInc: Int)
  : Unit = {
    require(xInc > 0 && yInc > 0)
    blas.sswap(
      n,
      x, xOff, xInc,
      y, yOff, yInc
    )
  }

  @inline
  final def swap(x: Array[Double],
                 y: Array[Double])
  : Unit = {
    require(x.length == y.length)
    blas.dswap(
      x.length,
      x, 0, 1,
      y, 0, 1
    )
  }

  @inline
  final def swap(n: Int,
                 x: Array[Double], xOff: Int, xInc: Int,
                 y: Array[Double], yOff: Int, yInc: Int)
  : Unit = {
    require(xInc > 0 && yInc > 0)
    blas.dswap(
      n,
      x, xOff, xInc,
      y, yOff, yInc
    )
  }


  // ---------------------------------------------------------------------------
  //    symm:  Z = alpha * X * Y + beta * Z
  //    symm:  Z = alpha * Y * X + beta * Z
  //    symv:  z = alpha * X * y + beta * z
  //    syr:   Y = alpha * x * x' + Y
  //    syr2:  Z = alpha * x * y' + alpha * y * x' + Z
  //    syr2k: Z = alpha * X * Y' + alpha * Y * X' + beta * Z
  //    syrk:  Y = alpha * X * X' + beta * Y
  //    tbmv:  x = A * x
  //    tbmv:  x = A' * x
  //    tbsv:  A * x = b
  //    tbsv:  A' * x = b
  //    tpmv:  x = A * x
  //    tpmv:  x = A' * x
  //    tpsv:  A * x = b
  //    tpsv:  A' * x = b
  //    trmm:  B = alpha * op(A) * B
  //    trmm:  B = alpha * B * op(A)
  //    trmv:  x = A * x
  //    trmv:  x = A' * x
  //    trsm:  op(A) * X = alpha * B
  //    trsm:  X * op(A) = alpha * B
  //    trsv:  A * x = b
  //    trsv:  A' * x = b
  // ---------------------------------------------------------------------------

}

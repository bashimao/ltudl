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

package edu.latrobe.cublaze

import edu.latrobe._
import edu.latrobe.native._
import org.bytedeco.javacpp.{DoublePointer, FloatPointer}
import org.bytedeco.javacpp.cublas._

private[cublaze] object _CUBLAS {

  lazy val version: Int = {
    val version = new Array[Int](1)
    val result = cublasGetVersion(version)
    check(result)
    version(0)
  }

  @inline
  final private def check(resultCode: Int)
  : Unit = {
    assume(resultCode == CUBLAS_STATUS_SUCCESS)
  }

  @inline
  final def create()
  : cublasContext = {
    val ptr    = new cublasContext
    val result = cublasCreate_v2(ptr)
    check(result)
    ptr
  }

  @inline
  final def destroy(device: LogicalDevice)
  : Unit = {
    val result = cublasDestroy_v2(
      device.blasContextPtr
    )
    check(result)
  }

  @inline
  final def setStream(device: LogicalDevice)
  : Unit = {
    val result = cublasSetStream_v2(
      device.blasContextPtr,
      device.streamPtr
    )
    check(result)
  }


  // ---------------------------------------------------------------------------
  //    asum: sum(abs(x))
  // ---------------------------------------------------------------------------
  @inline
  final def asum(device: LogicalDevice,
                 n:      Int,
                 xPtr:   FloatPointer, xInc: Int)
  : Float = {
    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      val result = cublasSasum_v2(
        device.blasContextPtr,
        n,
        xPtr, xInc,
        tmpPtr
      )
      check(result)
      device.trySynchronize()
      tmpPtr.get()
    })
  }

  @inline
  final def asum(device: LogicalDevice,
                 n:      Int,
                 xPtr:   DoublePointer, xInc: Int)
  : Double = {
    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      val result = cublasDasum_v2(
        device.blasContextPtr,
        n,
        xPtr, xInc,
        tmpPtr
      )
      check(result)
      device.trySynchronize()
      tmpPtr.get()
    })
  }


  // ---------------------------------------------------------------------------
  //    axpy: y = a * x + y
  // ---------------------------------------------------------------------------
  @inline
  final def axpy(device: LogicalDevice,
                 n:      Int,
                 alpha:  NativeFloat,
                 xPtr:   FloatPointer, xInc: Int,
                 yPtr:   FloatPointer, yInc: Int)
  : Unit = {
    val result = cublasSaxpy_v2(
      device.blasContextPtr,
      n,
      alpha.ptr,
      xPtr, xInc,
      yPtr, yInc
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def axpy(device: LogicalDevice,
                 n:      Int,
                 alpha:  NativeDouble,
                 xPtr:   DoublePointer, xInc: Int,
                 yPtr:   DoublePointer, yInc: Int)
  : Unit = {
    val result = cublasDaxpy_v2(
      device.blasContextPtr,
      n,
      alpha.ptr,
      xPtr, xInc,
      yPtr, yInc
    )
    check(result)
    device.trySynchronize()
  }


  // ---------------------------------------------------------------------------
  //    copy: y = x
  // ---------------------------------------------------------------------------
  @inline
  final def copy(device: LogicalDevice,
                 n:      Int,
                 xPtr:   FloatPointer, xInc: Int,
                 yPtr:   FloatPointer, yInc: Int)
  : Unit = {
    val result = cublasScopy_v2(
      device.blasContextPtr,
      n,
      xPtr, xInc,
      yPtr, yInc
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def copy(device: LogicalDevice,
                 n:      Int,
                 xPtr:   DoublePointer, xInc: Int,
                 yPtr:   DoublePointer, yInc: Int)
  : Unit = {
    val result = cublasDcopy_v2(
      device.blasContextPtr,
      n,
      xPtr, xInc,
      yPtr, yInc
    )
    check(result)
    device.trySynchronize()
  }


  // ---------------------------------------------------------------------------
  //    dot: x' * y
  // ---------------------------------------------------------------------------
  @inline
  final def dot(device: LogicalDevice,
                n:      Int,
                xPtr:   FloatPointer, xInc: Int,
                yPtr:   FloatPointer, yInc: Int)
  : Float = {
    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      val result = cublasSdot_v2(
        device.blasContextPtr,
        n,
        xPtr, xInc,
        yPtr, yInc,
        tmpPtr
      )
      check(result)
      tmpPtr.get()
    })
  }

  @inline
  final def dot(device: LogicalDevice,
                n:      Int,
                xPtr:   DoublePointer, xInc: Int,
                yPtr:   DoublePointer, yInc: Int)
  : Double = {
    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      val result = cublasDdot_v2(
        device.blasContextPtr,
        n,
        xPtr, xInc,
        yPtr, yInc,
        tmpPtr
      )
      check(result)
      tmpPtr.get()
    })
  }


  // ---------------------------------------------------------------------------
  //    geam: C = alpha * op(A) + beta * op(B)
  // ---------------------------------------------------------------------------
  @inline
  final def geam(device: LogicalDevice,
                 alpha:  NativeFloat,
                 aPtr:   FloatPointer, aStride: Int, aRows: Int, aCols: Int, aTrans: Boolean,
                 beta:   NativeFloat,
                 bPtr:   FloatPointer, bStride: Int, bRows: Int, bCols: Int, bTrans: Boolean,
                 cPtr:   FloatPointer, cStride: Int, cRows: Int, cCols: Int)
  : Unit = {
    require(
      aStride > 0 &&
      bStride > 0 &&
      cStride > 0 &&
      aRows == bRows && aCols == bCols &&
      aRows == cRows && aCols == cCols
    )
    val result = cublasSgeam(
      device.blasContextPtr,
      if (aTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      if (bTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      aRows,
      aCols,
      alpha.ptr,
      aPtr, aStride,
      beta.ptr,
      bPtr, bStride,
      cPtr, cStride
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def geam(device: LogicalDevice,
                 alpha:  NativeDouble,
                 aPtr:   DoublePointer, aStride: Int, aRows: Int, aCols: Int, aTrans: Boolean,
                 beta:   NativeDouble,
                 bPtr:   DoublePointer, bStride: Int, bRows: Int, bCols: Int, bTrans: Boolean,
                 cPtr:   DoublePointer, cStride: Int, cRows: Int, cCols: Int)
  : Unit = {
    require(
      aStride > 0 &&
      bStride > 0 &&
      cStride > 0 &&
      aRows == bRows && aCols == bCols &&
      aRows == cRows && aCols == cCols
    )
    val result = cublasDgeam(
      device.blasContextPtr,
      if (aTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      if (bTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      aRows,
      aCols,
      alpha.ptr,
      aPtr, aStride,
      beta.ptr,
      bPtr, bStride,
      cPtr, cStride
    )
    check(result)
    device.trySynchronize()
  }


  // ---------------------------------------------------------------------------
  //    gemm: C = alpha * op(A) * op(B) + beta * C
  // ---------------------------------------------------------------------------
  @inline
  final def gemm(device: LogicalDevice,
                 alpha:  Float,
                 a:      FloatPointer, aStride: Int, aRows: Int, aCols: Int, aTrans: Boolean,
                 b:      FloatPointer, bStride: Int, bRows: Int, bCols: Int, bTrans: Boolean,
                 beta:   Float,
                 c:      FloatPointer, cStride: Int, cRows: Int, cCols: Int)
  : Unit = {
    using(NativeFloat(alpha), NativeFloat(beta))(
      gemm(
        device,
        _,
        a, aStride, aRows, aCols, aTrans,
        b, bStride, bRows, bCols, bTrans,
        _,
        c, cStride, cRows, cCols
      )
    )
  }

  @inline
  final def gemm(device: LogicalDevice,
                 alpha:  NativeFloat,
                 aPtr:   FloatPointer, aStride: Int, aRows: Int, aCols: Int, aTrans: Boolean,
                 bPtr:   FloatPointer, bStride: Int, bRows: Int, bCols: Int, bTrans: Boolean,
                 beta:   NativeFloat,
                 cPtr:   FloatPointer, cStride: Int, cRows: Int, cCols: Int)
  : Unit = {
    require(
      aStride > 0 &&
      bStride > 0 &&
      cStride > 0 &&
      aRows == cRows &&
      aCols == bRows &&
      bCols == cCols
    )
    val result = cublasSgemm_v2(
      device.blasContextPtr,
      if (aTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      if (bTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      aRows,
      bCols,
      aCols,
      alpha.ptr,
      aPtr, aStride,
      bPtr, bStride,
      beta.ptr,
      cPtr, cStride
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def gemm(device: LogicalDevice,
                 alpha:  NativeDouble,
                 aPtr:   DoublePointer, aStride: Int, aRows: Int, aCols: Int, aTrans: Boolean,
                 bPtr:   DoublePointer, bStride: Int, bRows: Int, bCols: Int, bTrans: Boolean,
                 beta:   NativeDouble,
                 cPtr:   DoublePointer, cStride: Int, cRows: Int, cCols: Int)
  : Unit = {
    require(
      aStride > 0 &&
      bStride > 0 &&
      cStride > 0 &&
      aRows == cRows &&
      aCols == bRows &&
      bCols == cCols
    )
    val result = cublasDgemm_v2(
      device.blasContextPtr,
      if (aTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      if (bTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      aRows,
      bCols,
      aCols,
      alpha.ptr,
      aPtr, aStride,
      bPtr, bStride,
      beta.ptr,
      cPtr, cStride
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def gemm(device: LogicalDevice,
                 alpha:  NativeFloat,
                 aPtr:   HalfPointer, aStride: Int, aRows: Int, aCols: Int, aTrans: Boolean,
                 bPtr:   HalfPointer, bStride: Int, bRows: Int, bCols: Int, bTrans: Boolean,
                 beta:   NativeFloat,
                 cPtr:   HalfPointer, cStride: Int, cRows: Int, cCols: Int)
  : Unit = {
    require(
      aStride > 0 &&
      bStride > 0 &&
      cStride > 0 &&
      aRows == cRows &&
      aCols == bRows &&
      bCols == cCols
    )
    val result = cublasSgemmEx(
      device.blasContextPtr,
      if (aTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      if (bTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      aRows,
      bCols,
      aCols,
      alpha.ptr,
      aPtr, CUBLAS_DATA_HALF, aStride,
      bPtr, CUBLAS_DATA_HALF, bStride,
      beta.ptr,
      cPtr, CUBLAS_DATA_HALF, cStride
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def gemm(device: LogicalDevice,
                 alpha:  NativeFloat,
                 aPtr:   HalfPointer, aStride: Int, aRows: Int, aCols: Int, aTrans: Boolean,
                 bPtr:   HalfPointer, bStride: Int, bRows: Int, bCols: Int, bTrans: Boolean,
                 beta:   NativeFloat,
                 cPtr:   FloatPointer, cStride: Int, cRows: Int, cCols: Int)
  : Unit = {
    require(
      aStride > 0 &&
      bStride > 0 &&
      cStride > 0 &&
      aRows == cRows &&
      aCols == bRows &&
      bCols == cCols
    )
    val result = cublasSgemmEx(
      device.blasContextPtr,
      if (aTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      if (bTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      aRows,
      bCols,
      aCols,
      alpha.ptr,
      aPtr, CUBLAS_DATA_HALF, aStride,
      bPtr, CUBLAS_DATA_HALF, bStride,
      beta.ptr,
      cPtr, CUBLAS_DATA_FLOAT, cStride
    )
    check(result)
    device.trySynchronize()
  }


  // ---------------------------------------------------------------------------
  //    gemv: z = alpha * x  * y + beta * z
  //    gemv: z = alpha * x' * y + beta * z
  // ---------------------------------------------------------------------------
  @inline
  final def gemv(device: LogicalDevice,
                 alpha:  NativeFloat,
                 aPtr:   FloatPointer, aStride: Int, aRows: Int, aCols: Int, aTrans: Boolean,
                 xPtr:   FloatPointer, xInc: Int, xRows: Int,
                 beta:   NativeFloat,
                 yPtr:   FloatPointer, yInc: Int, yRows: Int)
  : Unit = {
    require(
      aStride > 0 &&
      xInc > 0 &&
      yInc > 0 &&
      aCols == xRows &&
      aCols == yRows
    )
    val result = cublasSgemv_v2(
      device.blasContextPtr,
      if (aTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      aRows,
      aCols,
      alpha.ptr,
      aPtr, aStride,
      xPtr, xInc,
      beta.ptr,
      yPtr, yInc
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def gemv(device: LogicalDevice,
                 alpha:  NativeDouble,
                 aPtr:   DoublePointer, aStride: Int, aRows: Int, aCols: Int, aTrans: Boolean,
                 xPtr:   DoublePointer, xInc: Int, xRows: Int,
                 beta:   NativeDouble,
                 yPtr:   DoublePointer, yInc: Int, yRows: Int)
  : Unit = {
    require(
      aStride > 0 &&
      xInc > 0 &&
      yInc > 0 &&
      aCols == xRows &&
      aCols == yRows
    )
    val result = cublasDgemv_v2(
      device.blasContextPtr,
      if (aTrans) CUBLAS_OP_T else CUBLAS_OP_N,
      aRows,
      aCols,
      alpha.ptr,
      aPtr, aStride,
      xPtr, xInc,
      beta.ptr,
      yPtr, yInc
    )
    check(result)
    device.trySynchronize()
  }


  // ---------------------------------------------------------------------------
  //    gmm: C = diag(X) * A
  //    gmm: C = A * diag(X)
  //
  //    In place allowed if strides match.
  // ---------------------------------------------------------------------------
  @inline
  final def gmm(device: LogicalDevice,
                aPtr:   FloatPointer, aStride: Int, aRows: Int, aCols: Int,
                xPtr:   FloatPointer, xInc: Int, xRows: Int,
                cPtr:   FloatPointer, cStride: Int, cRows: Int, cCols: Int)
  : Unit = {
    require(
      aStride > 0 &&
      xInc > 0 &&
      cStride > 0 &&
      aRows == cRows &&
      aCols == cCols &&
      aCols == xRows
    )
    val result = cublasSdgmm(
      device.blasContextPtr,
      CUBLAS_SIDE_RIGHT,
      aRows,
      aCols,
      aPtr, aStride,
      xPtr, xInc,
      cPtr, cStride
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def gmm(device: LogicalDevice,
                aPtr:   DoublePointer, aStride: Int, aRows: Int, aCols: Int,
                xPtr:   DoublePointer, xInc: Int, xRows: Int,
                cPtr:   DoublePointer, cStride: Int, cRows: Int, cCols: Int)
  : Unit = {
    require(
      aStride > 0 &&
      xInc > 0 &&
      cStride > 0 &&
      aRows == cRows &&
      aCols == cCols &&
      aCols == xRows
    )
    val result = cublasDdgmm(
      device.blasContextPtr,
      CUBLAS_SIDE_RIGHT,
      aRows,
      aCols,
      aPtr, aStride,
      xPtr, xInc,
      cPtr, cStride
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def gmm(device: LogicalDevice,
                xPtr:   FloatPointer, xInc: Int, xRows: Int,
                aPtr:   FloatPointer, aStride: Int, aRows: Int, aCols: Int,
                cPtr:   FloatPointer, cStride: Int, cRows: Int, cCols: Int)
  : Unit = {
    require(
      aStride > 0 &&
      xInc > 0 &&
      cStride > 0 &&
      aRows == cRows &&
      aCols == cCols &&
      aRows == xRows
    )
    val result = cublasSdgmm(
      device.blasContextPtr,
      CUBLAS_SIDE_LEFT,
      aRows,
      aCols,
      aPtr, aStride,
      xPtr, xInc,
      cPtr, cStride
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def gmm(device: LogicalDevice,
                xPtr:   DoublePointer, xInc: Int, xRows: Int,
                aPtr:   DoublePointer, aStride: Int, aRows: Int, aCols: Int,
                cPtr:   DoublePointer, cStride: Int, cRows: Int, cCols: Int)
  : Unit = {
    require(
      aStride > 0 &&
      xInc > 0 &&
      cStride > 0 &&
      aRows == cRows &&
      aCols == cCols &&
      aRows == xRows
    )
    val result = cublasDdgmm(
      device.blasContextPtr,
      CUBLAS_SIDE_LEFT,
      aRows,
      aCols,
      aPtr, aStride,
      xPtr, xInc,
      cPtr, cStride
    )
    check(result)
    device.trySynchronize()
  }


  // ---------------------------------------------------------------------------
  //    iamax: index_where(abs(x) = max(abs(x)))
  // ---------------------------------------------------------------------------
  @inline
  final def iamax(device: LogicalDevice,
                  n:      Int,
                  xPtr:   FloatPointer, xInc: Int)
  : Int = {
    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      val result = cublasIsamax_v2(
        device.blasContextPtr,
        n,
        xPtr, xInc,
        tmpPtr
      )
      check(result)
      device.trySynchronize()
      tmpPtr.get()
    })
  }

  @inline
  final def iamax(device: LogicalDevice,
                  n:      Int,
                  xPtr:   DoublePointer, xInc: Int)
  : Int = {
    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      val result = cublasIdamax_v2(
        device.blasContextPtr,
        n,
        xPtr, xInc,
        tmpPtr
      )
      check(result)
      device.trySynchronize()
      tmpPtr.get()
    })
  }

  @inline
  final def iamin(device: LogicalDevice,
                  n:      Int,
                  xPtr:   FloatPointer, xInc: Int)
  : Int = {
    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      val result = cublasIsamin_v2(
        device.blasContextPtr,
        n,
        xPtr, xInc,
        tmpPtr
      )
      check(result)
      device.trySynchronize()
      tmpPtr.get()
    })
  }

  @inline
  final def iamin(device: LogicalDevice,
                  n:      Int,
                  xPtr:   DoublePointer, xInc: Int)
  : Int = {
    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      val result = cublasIdamin_v2(
        device.blasContextPtr,
        n,
        xPtr, xInc,
        tmpPtr
      )
      check(result)
      device.trySynchronize()
      tmpPtr.get()
    })
  }


  // ---------------------------------------------------------------------------
  //    nrm2: sqrt(x' * x)
  // ---------------------------------------------------------------------------
  @inline
  final def nrm2(device: LogicalDevice,
                 n:      Int,
                 xPtr:   FloatPointer, xInc: Int)
  : Float = {
    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      val result = cublasSnrm2_v2(
        device.blasContextPtr,
        n,
        xPtr, xInc,
        tmpPtr
      )
      check(result)
      device.trySynchronize()
      tmpPtr.get()
    })
  }

  @inline
  final def nrm2(device: LogicalDevice,
                 n:      Int,
                 xPtr:   DoublePointer, xInc: Int)
  : Double = {
    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      val result = cublasDnrm2_v2(
        device.blasContextPtr,
        n,
        xPtr, xInc,
        tmpPtr
      )
      check(result)
      device.trySynchronize()
      tmpPtr.get()
    })
  }


  // ---------------------------------------------------------------------------
  //    scal: a * x
  // ---------------------------------------------------------------------------
  @inline
  final def scal(device: LogicalDevice,
                 n:      Int,
                 alpha:  NativeFloat,
                 xPtr:   FloatPointer, xInc: Int)
  : Unit = {
    val result = cublasSscal_v2(
      device.blasContextPtr,
      n,
      alpha.ptr,
      xPtr, xInc
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def scal(device: LogicalDevice,
                 n:      Int,
                 alpha:  NativeDouble,
                 xPtr:   DoublePointer, xInc: Int)
  : Unit = {
    val result = cublasDscal_v2(
      device.blasContextPtr,
      n,
      alpha.ptr,
      xPtr, xInc
    )
    check(result)
    device.trySynchronize()
  }


  // ---------------------------------------------------------------------------
  //    swap: tmp = x; x = y; y = tmp
  // ---------------------------------------------------------------------------
  @inline
  final def swap(device: LogicalDevice,
                 n:      Int,
                 xPtr:   FloatPointer, xInc: Int,
                 yPtr:   FloatPointer, yInc: Int)
  : Unit = {
    val result = cublasSswap_v2(
      device.blasContextPtr,
      n,
      xPtr, xInc,
      yPtr, yInc
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def swap(device: LogicalDevice,
                 n:      Int,
                 xPtr:   DoublePointer, xInc: Int,
                 yPtr:   DoublePointer, yInc: Int)
  : Unit = {
    val result = cublasDswap_v2(
      device.blasContextPtr,
      n,
      xPtr, xInc,
      yPtr, yInc
    )
    check(result)
    device.trySynchronize()
  }

}

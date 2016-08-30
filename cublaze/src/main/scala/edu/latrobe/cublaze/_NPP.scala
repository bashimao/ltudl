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
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.nppc._
import org.bytedeco.javacpp.nppi._
import org.bytedeco.javacpp.npps._

private[cublaze] object _NPP {

  final val version
  : (Int, Int) = {
    val tmp = nppGetLibVersion()
    (tmp.major(), tmp.minor())
  }

  @inline
  final private def execute(device: LogicalDevice, fn: => Int)
  : Unit = synchronized {
    nppSetStream(device.streamPtr)
    val result = fn
    if (result != NPP_SUCCESS) {
      val str = s"CUDA NPP Error: $result"
      logger.error(str)
      throw new UnknownError(str)
    }
  }

  @inline
  final private def executeEx(device: LogicalDevice, fn: => Int)
  : Unit = {
    execute(device, fn)
    device.trySynchronize()
  }

  final def abs(device: LogicalDevice,
                xPtr:   FloatPointer,
                yPtr:   FloatPointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsAbs_32f(
      xPtr,
      yPtr,
      length
    )
  )

  final def abs(device: LogicalDevice,
                xPtr:   FloatPointer,
                yPtr:   FloatPointer,
                size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiAbs_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def abs(device: LogicalDevice,
                xPtr:   DoublePointer,
                yPtr:   DoublePointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsAbs_64f(
      xPtr,
      yPtr,
      length
    )
  )

  final def abs_I(device: LogicalDevice,
                  yPtr:   FloatPointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsAbs_32f_I(
      yPtr,
      length
    )
  )

  final def abs_I(device: LogicalDevice,
                  yPtr:   FloatPointer,
                  size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiAbs_32f_C1IR(
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def abs_I(device: LogicalDevice,
                  yPtr:   DoublePointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsAbs_64f_I(
      yPtr,
      length
    )
  )

  final def absDiff(device: LogicalDevice,
                    x0Ptr:  FloatPointer,
                    x1Ptr:  FloatPointer,
                    yPtr:   FloatPointer,
                    size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiAbsDiff_32f_C1R(
      x0Ptr, size.linePitch,
      x1Ptr, size.linePitch,
      yPtr,  size.linePitch,
      size.ptr
    )
  )

  final def absDiffC(device: LogicalDevice,
                     xPtr:   FloatPointer,
                     yPtr:   FloatPointer,
                     size:   _SizeStruct,
                     value:  Float)
  : Unit = executeEx(
    device,
    nppiAbsDiffC_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr,
      value
    )
  )

  final def add(device: LogicalDevice,
                x0Ptr:  FloatPointer,
                x1Ptr:  FloatPointer,
                yPtr:   FloatPointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsAdd_32f(
      x0Ptr,
      x1Ptr,
      yPtr,
      length
    )
  )

  final def add(device: LogicalDevice,
                x0Ptr:  FloatPointer,
                x1Ptr:  FloatPointer,
                yPtr:   FloatPointer,
                size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiAdd_32f_C1R(
      x0Ptr, size.linePitch,
      x1Ptr, size.linePitch,
      yPtr,  size.linePitch,
      size.ptr
    )
  )

  final def add(device: LogicalDevice,
                x0Ptr:  DoublePointer,
                x1Ptr:  DoublePointer,
                yPtr:   DoublePointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsAdd_64f(
      x0Ptr,
      x1Ptr,
      yPtr,
      length
    )
  )

  final def add_I(device: LogicalDevice,
                  xPtr:   FloatPointer,
                  yPtr:   FloatPointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsAdd_32f_I(
      xPtr,
      yPtr,
      length
    )
  )

  final def add_I(device: LogicalDevice,
                  xPtr:   FloatPointer,
                  yPtr:   FloatPointer,
                  size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiAdd_32f_C1IR(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def add_I(device: LogicalDevice,
                  xPtr:   DoublePointer,
                  yPtr:   DoublePointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsAdd_64f_I(
      xPtr,
      yPtr,
      length
    )
  )

  final def addC(device: LogicalDevice,
                 x0Ptr:  FloatPointer,
                 x1:     Float,
                 yPtr:   FloatPointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsAddC_32f(
      x0Ptr,
      x1,
      yPtr,
      length
    )
  )

  final def addC(device: LogicalDevice,
                 x0Ptr:  FloatPointer,
                 x1:     Float,
                 yPtr:   FloatPointer,
                 size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiAddC_32f_C1R(
      x0Ptr, size.linePitch,
      x1,
      yPtr,  size.linePitch,
      size.ptr
    )
  )

  final def addC(device: LogicalDevice,
                 x0Ptr:  DoublePointer,
                 x1:     Double,
                 yPtr:   DoublePointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsAddC_64f(
      x0Ptr,
      x1,
      yPtr,
      length
    )
  )

  final def addC_I(device: LogicalDevice,
                   x:      Float,
                   yPtr:   FloatPointer,
                   length: Int)
  : Unit = executeEx(
    device,
    nppsAddC_32f_I(
      x,
      yPtr,
      length
    )
  )

  final def addC_I(device: LogicalDevice,
                   x:      Float,
                   yPtr:   FloatPointer,
                   size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiAddC_32f_C1IR(
      x,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def addC_I(device: LogicalDevice,
                   x:     Double,
                   yPtr:   DoublePointer,
                   length: Int)
  : Unit = executeEx(
    device,
    nppsAddC_64f_I(
      x,
      yPtr,
      length
    )
  )

  final def addProduct(device: LogicalDevice,
                       x0Ptr:  FloatPointer,
                       x1Ptr:  FloatPointer,
                       yPtr:   FloatPointer,
                       length: Int)
  : Unit = executeEx(
    device,
    nppsAddProduct_32f(
      x0Ptr,
      x1Ptr,
      yPtr,
      length
    )
  )

  final def addProduct(device: LogicalDevice,
                       x0Ptr:  FloatPointer,
                       x1Ptr:  FloatPointer,
                       yPtr:   FloatPointer,
                       size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiAddProduct_32f_C1IR(
      x0Ptr, size.linePitch,
      x1Ptr, size.linePitch,
      yPtr,  size.linePitch,
      size.ptr
    )
  )

  final def addProduct(device: LogicalDevice,
                       x0Ptr:  DoublePointer,
                       x1Ptr:  DoublePointer,
                       yPtr:   DoublePointer,
                       length: Int)
  : Unit = executeEx(
    device,
    nppsAddProduct_64f(
      x0Ptr,
      x1Ptr,
      yPtr,
      length
    )
  )

  final def addProductC(device: LogicalDevice,
                        x0Ptr:  FloatPointer,
                        x1:     Float,
                        yPtr:   FloatPointer,
                        length: Int)
  : Unit = executeEx(
    device,
    nppsAddProductC_32f(
      x0Ptr,
      x1,
      yPtr,
      length
    )
  )

  final def addSquare(device: LogicalDevice,
                      xPtr:   FloatPointer,
                      yPtr:   FloatPointer,
                      size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiAddSquare_32f_C1IR(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def addWeighted_I(device: LogicalDevice,
                          xPtr:   FloatPointer,
                          yPtr:   FloatPointer,
                          size:   _SizeStruct,
                          alpha:  Real)
  : Unit = executeEx(
    device,
    nppiAddWeighted_32f_C1IR(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr,
      alpha
    )
  )

  /**
    * The NVIDIA documentation for this is garbage. The original Intel is here:
    * https://software.intel.com/en-us/node/503865
    *
    * x0 = buffer 0
    * a0 = alpha 0
    * x1 = buffer 1
    * a1 = alpha 1
    *
    *                      Color Channels                 Alpha Channels
    * --------------------------------------------------------------------------
    * NPPI_OP_ALPHA_OVER | x0 a0        + x1 (1-a0) a1  | a0        + (1-a0) a1
    *                    |                              |
    * NPPI_OP_ALPHA_IN   | x0 a0 a1                     | a0 a1
    *                    |                              |
    * NPPI_OP_ALPHA_OUT  |                x1 a0 (1-a1)  | a0 (1-a1)
    *                    |                              |
    * NPPI_OP_ALPHA_ATOP | x0 a0 a1     + x1 (1-a0) a1  | a0 a1     + (1-a0) a1
    *                    |                              |
    * NPPI_OP_ALPHA_XOR  | x0 a0 (1-a1) + x1 (1-a0) a1  | a0 (1-a1) + (1-a0) a1
    *                    |                              |
    * NPPI_OP_ALPHA_PLUS | x0 a0        + x1 a1         | a0        + a1
    *                    |                              |
    */
  final def alphaCompC(device: LogicalDevice,
                       x0Ptr:  FloatPointer, alpha0: Float,
                       x1Ptr:  FloatPointer, alpha1: Float,
                       yPtr:   FloatPointer,
                       size:   _SizeStruct,
                       op:     Int = NPPI_OP_ALPHA_PLUS)
  : Unit = executeEx(
    device,
    nppiAlphaCompC_32f_C1R(
      x0Ptr, size.linePitch, alpha0,
      x1Ptr, size.linePitch, alpha1,
      yPtr,  size.linePitch,
      size.ptr,
      op
    )
  )

  final def arctan(device: LogicalDevice,
                   xPtr:   FloatPointer,
                   yPtr:   FloatPointer,
                   length: Int)
  : Unit = executeEx(
    device,
    nppsArctan_32f(
      xPtr,
      yPtr,
      length
    )
  )

  final def arctan(device: LogicalDevice,
                   xPtr:   DoublePointer,
                   yPtr:   DoublePointer,
                   length: Int)
  : Unit = executeEx(
    device,
    nppsArctan_64f(
      xPtr,
      yPtr,
      length
    )
  )

  final def arctan_I(device: LogicalDevice,
                     yPtr:   FloatPointer,
                     length: Int)
  : Unit = executeEx(
    device,
    nppsArctan_32f_I(
      yPtr,
      length
    )
  )

  final def arctan_I(device: LogicalDevice,
                     yPtr:   DoublePointer,
                     length: Int)
  : Unit = executeEx(
    device,
    nppsArctan_64f_I(
      yPtr,
      length
    )
  )

  final def copy(device: LogicalDevice,
                 xPtr:   ShortPointer,
                 yPtr:   ShortPointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsCopy_16s(
      xPtr,
      yPtr,
      length
    )
  )

  final def copy(device: LogicalDevice,
                 xPtr:   FloatPointer,
                 yPtr:   FloatPointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsCopy_32f(
      xPtr,
      yPtr,
      length
    )
  )

  final def copy(device: LogicalDevice,
                 xPtr:   FloatPointer,
                 yPtr:   FloatPointer,
                 size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiCopy_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def cubrt(device: LogicalDevice,
                  xPtr:   FloatPointer,
                  yPtr:   FloatPointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsCubrt_32f(
      xPtr,
      yPtr,
      length
    )
  )

  final def div(device: LogicalDevice,
                x0Ptr:  FloatPointer,
                x1Ptr:  FloatPointer,
                yPtr:   FloatPointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsDiv_32f(
      x0Ptr,
      x1Ptr,
      yPtr,
      length
    )
  )

  final def div(device: LogicalDevice,
                x0Ptr:  FloatPointer,
                x1Ptr:  FloatPointer,
                yPtr:   FloatPointer,
                size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiDiv_32f_C1R(
      x0Ptr, size.linePitch,
      x1Ptr, size.linePitch,
      yPtr,  size.linePitch,
      size.ptr
    )
  )

  final def div(device: LogicalDevice,
                x0Ptr:  DoublePointer,
                x1Ptr:  DoublePointer,
                yPtr:   DoublePointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsDiv_64f(
      x0Ptr,
      x1Ptr,
      yPtr,
      length
    )
  )

  final def div_I(device: LogicalDevice,
                  xPtr:   FloatPointer,
                  yPtr:   FloatPointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsDiv_32f_I(
      xPtr,
      yPtr,
      length
    )
  )

  final def div_I(device: LogicalDevice,
                  xPtr:   FloatPointer,
                  yPtr:   FloatPointer,
                  size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiDiv_32f_C1IR(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def div_I(device: LogicalDevice,
                  xPtr:   DoublePointer,
                  yPtr:   DoublePointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsDiv_64f_I(
      xPtr,
      yPtr,
      length
    )
  )

  final def divC(device: LogicalDevice,
                 x0Ptr:  FloatPointer,
                 x1:     Float,
                 yPtr:   FloatPointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsDivC_32f(
      x0Ptr,
      x1,
      yPtr,
      length
    )
  )

  final def divC(device: LogicalDevice,
                 x0Ptr:  FloatPointer,
                 x1:     Float,
                 yPtr:   FloatPointer,
                 size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiDivC_32f_C1R(
      x0Ptr, size.linePitch,
      x1,
      yPtr,  size.linePitch,
      size.ptr
    )
  )

  final def divC(device: LogicalDevice,
                 x0Ptr:  DoublePointer,
                 x1:     Double,
                 yPtr:   DoublePointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsDivC_64f(
      x0Ptr,
      x1,
      yPtr,
      length
    )
  )

  final def divC_I(device: LogicalDevice,
                   x:      Float,
                   yPtr:   FloatPointer,
                   length: Int)
  : Unit = executeEx(
    device,
    nppsDivC_32f_I(
      x,
      yPtr,
      length
    )
  )

  final def divC_I(device: LogicalDevice,
                   x:      Float,
                   yPtr:   FloatPointer,
                   size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiDivC_32f_C1IR(
      x,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def divC_I(device: LogicalDevice,
                   x:      Double,
                   yPtr:   DoublePointer,
                   length: Int)
  : Unit = executeEx(
    device,
    nppsDivC_64f_I(
      x,
      yPtr,
      length
    )
  )

  final def divCRev(device: LogicalDevice,
                    x0Ptr:  FloatPointer,
                    x1:     Float,
                    yPtr:   FloatPointer,
                    length: Int)
  : Unit = executeEx(
    device,
    nppsDivCRev_32f(
      x0Ptr,
      x1,
      yPtr,
      length
    )
  )

  final def divCRev_I(device: LogicalDevice,
                      x:      Float,
                      yPtr:   FloatPointer,
                      length: Int)
  : Unit = executeEx(
    device,
    nppsDivCRev_32f_I(
      x,
      yPtr,
      length
    )
  )

  final def dotProd(device: LogicalDevice,
                    x0Ptr:  FloatPointer,
                    x1Ptr:  FloatPointer,
                    length: Int)
  : Float = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsDotProdGetBufferSize_32f(length, tmpPtr)
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsDotProd_32f(
          x0Ptr,
          x1Ptr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def dotProd(device: LogicalDevice,
                    x0Ptr:  DoublePointer,
                    x1Ptr:  DoublePointer,
                    length: Int)
  : Double = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsDotProdGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsDotProd_64f(
          x0Ptr,
          x1Ptr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def exp(device: LogicalDevice,
                xPtr:   FloatPointer,
                yPtr:   FloatPointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsExp_32f(
      xPtr,
      yPtr,
      length
    )
  )

  final def exp(device: LogicalDevice,
                xPtr:   FloatPointer,
                yPtr:   FloatPointer,
                size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiExp_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def exp(device: LogicalDevice,
                xPtr:   DoublePointer,
                yPtr:   DoublePointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsExp_64f(
      xPtr,
      yPtr,
      length
    )
  )

  final def exp_I(device: LogicalDevice,
                  yPtr:   FloatPointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsExp_32f_I(
      yPtr,
      length
    )
  )

  final def exp_I(device: LogicalDevice,
                  yPtr:   FloatPointer,
                  size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiExp_32f_C1IR(
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def exp_I(device: LogicalDevice,
                  yPtr:   DoublePointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsExp_64f_I(
      yPtr,
      length
    )
  )

  final def ln(device: LogicalDevice,
               xPtr:   FloatPointer,
               yPtr:   FloatPointer,
               length: Int)
  : Unit = executeEx(
    device,
    nppsLn_32f(
      xPtr,
      yPtr,
      length
    )
  )

  final def ln(device: LogicalDevice,
               xPtr:   FloatPointer,
               yPtr:   FloatPointer,
               size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiLn_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def ln(device: LogicalDevice,
               xPtr:   DoublePointer,
               yPtr:   DoublePointer,
               length: Int)
  : Unit = executeEx(
    device,
    nppsLn_64f(
      xPtr,
      yPtr,
      length
    )
  )

  final def ln_I(device: LogicalDevice,
                 yPtr:   FloatPointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsLn_32f_I(
      yPtr,
      length
    )
  )

  final def ln_I(device: LogicalDevice,
                 yPtr:   FloatPointer,
                 size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiLn_32f_C1IR(
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def ln_I(device: LogicalDevice,
                 yPtr:   DoublePointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsLn_64f_I(
      yPtr,
      length
    )
  )

  final def max(device: LogicalDevice,
                xPtr:   FloatPointer,
                length: Int)
  : Float = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMaxGetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsMax_32f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def max(device: LogicalDevice,
                xPtr:   FloatPointer,
                size:   _SizeStruct)
  : Float = {
    val wsPtr   = device.scratchBuffer.asBytePtr
    val sizePtr = size.ptr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppiMaxGetBufferHostSize_32f_C1R(
          sizePtr,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppiMax_32f_C1R(
          xPtr, size.linePitch,
          sizePtr,
          wsPtr,
          tmpPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def max(device: LogicalDevice,
                xPtr:   DoublePointer,
                length: Int)
  : Double = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMaxGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsMax_64f(
          xPtr, length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def maxEvery_I(device: LogicalDevice,
                       xPtr:   FloatPointer,
                       yPtr:   FloatPointer,
                       length: Int)
  : Unit = executeEx(
    device,
    nppsMaxEvery_32f_I(
      xPtr,
      yPtr,
      length
    )
  )

  final def maxEvery_I(device: LogicalDevice,
                       xPtr:   FloatPointer,
                       yPtr:   FloatPointer,
                       size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiMaxEvery_32f_C1IR(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def maxIndx(device: LogicalDevice,
                    xPtr:   FloatPointer,
                    length: Int)
  : (Float, Int) = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMaxIndxGetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f), NativeInt(0))((tmpMax, tmpMaxIndex) => {
      val tmpMaxPtr      = tmpMax.ptr
      val tmpMaxIndexPtr = tmpMaxIndex.ptr
      executeEx(
        device,
        nppsMaxIndx_32f(
          xPtr,
          length,
          tmpMaxPtr,
          tmpMaxIndexPtr,
          wsPtr
        )
      )
      (tmpMaxPtr.get(), tmpMaxIndexPtr.get())
    })
  }

  final def maxIndx(device: LogicalDevice,
                    xPtr:   DoublePointer,
                    length: Int)
  : (Double, Int) = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMaxIndxGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0), NativeInt(0))((tmpMax, tmpMaxIndex) => {
      val tmpMaxPtr      = tmpMax.ptr
      val tmpMaxIndexPtr = tmpMaxIndex.ptr
      executeEx(
        device,
        nppsMaxIndx_64f(
          xPtr, length,
          tmpMaxPtr, tmpMaxIndexPtr,
          wsPtr
        )
      )
      (tmpMaxPtr.get(), tmpMaxIndexPtr.get())
    })
  }

  final def mean(device: LogicalDevice,
                 xPtr:   FloatPointer,
                 length: Int)
  : Float = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMeanGetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsMean_32f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def mean(device: LogicalDevice,
                 xPtr:   DoublePointer,
                 length: Int)
  : Double = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMeanGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsMean_64f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def meanStdDev(device: LogicalDevice,
                       xPtr:   FloatPointer,
                       length: Int)
  : (Float, Float) = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMeanStdDevGetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f), NativeFloat(0.0f))((tmpMean, tmpStdDev) => {
      val tmpMeanPtr   = tmpMean.ptr
      val tmpStdDevPtr = tmpStdDev.ptr
      executeEx(
        device,
        nppsMeanStdDev_32f(
          xPtr,
          length,
          tmpMeanPtr,
          tmpStdDevPtr,
          wsPtr
        )
      )
      (tmpMeanPtr.get(), tmpStdDevPtr.get())
    })
  }

  final def meanStdDev(device: LogicalDevice,
                       xPtr:   DoublePointer,
                       length: Int)
  : (Double, Double) = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMeanStdDevGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0), NativeDouble(0.0))((tmpMean, tmpStdDev) => {
      val tmpMeanPtr   = tmpMean.ptr
      val tmpStdDevPtr = tmpStdDev.ptr
      executeEx(
        device,
        nppsMeanStdDev_64f(
          xPtr,
          length,
          tmpMeanPtr,
          tmpStdDevPtr,
          wsPtr
        )
      )
      (tmpMeanPtr.get(), tmpStdDevPtr.get())
    })
  }

  final def min(device: LogicalDevice,
                xPtr:   FloatPointer,
                length: Int)
  : Float = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMinGetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsMin_32f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def min(device: LogicalDevice,
                xPtr:   FloatPointer,
                size:   _SizeStruct)
  : Float = {
    val wsPtr   = device.scratchBuffer.asBytePtr
    val sizePtr = size.ptr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppiMinGetBufferHostSize_32f_C1R(
          sizePtr,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppiMin_32f_C1R(
          xPtr, size.linePitch,
          sizePtr,
          wsPtr,
          tmpPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def min(device: LogicalDevice,
                xPtr:   DoublePointer,
                length: Int)
  : Double = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMinGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsMin_64f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def minEvery_I(device: LogicalDevice,
                       xPtr:   FloatPointer,
                       yPtr:   FloatPointer,
                       length: Int)
  : Unit = executeEx(
    device,
    nppsMinEvery_32f_I(
      xPtr,
      yPtr,
      length
    )
  )

  final def minEvery_I(device: LogicalDevice,
                       xPtr:   FloatPointer,
                       yPtr:   FloatPointer,
                       size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiMinEvery_32f_C1IR(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def minEvery_I(device: LogicalDevice,
                       xPtr:   DoublePointer,
                       yPtr:   DoublePointer,
                       length: Int)
  : Unit = executeEx(
    device,
    nppsMinEvery_64f_I(
      xPtr,
      yPtr,
      length
    )
  )

  final def minIndx(device: LogicalDevice,
                    xPtr:   FloatPointer,
                    length: Int)
  : (Float, Int) = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMinIndxGetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f), NativeInt(0))((tmpMin, tmpMinIndex) => {
      val tmpMinPtr      = tmpMin.ptr
      val tmpMinIndexPtr = tmpMinIndex.ptr
      executeEx(
        device,
        nppsMinIndx_32f(
          xPtr,
          length,
          tmpMinPtr,
          tmpMinIndexPtr,
          wsPtr
        )
      )
      (tmpMinPtr.get(), tmpMinIndexPtr.get())
    })
  }

  final def minIndx(device: LogicalDevice,
                    xPtr:   DoublePointer,
                    length: Int)
  : (Double, Int) = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMinIndxGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0), NativeInt(0))((tmpMin, tmpMinIndex) => {
      val tmpMinPtr      = tmpMin.ptr
      val tmpMinIndexPtr = tmpMinIndex.ptr
      executeEx(
        device,
        nppsMinIndx_64f(
          xPtr, length,
          tmpMinPtr,
          tmpMinIndexPtr,
          wsPtr
        )
      )
      (tmpMinPtr.get(), tmpMinIndexPtr.get())
    })
  }

  final def minMax(device: LogicalDevice,
                   xPtr:   FloatPointer,
                   length: Int)
  : (Float, Float) = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMinMaxGetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f), NativeFloat(0.0f))((tmpMin, tmpMax) => {
      val tmpMinPtr = tmpMin.ptr
      val tmpMaxPtr = tmpMax.ptr
      executeEx(
        device,
        nppsMinMax_32f(
          xPtr,
          length,
          tmpMinPtr,
          tmpMaxPtr,
          wsPtr
        )
      )
      (tmpMinPtr.get(), tmpMaxPtr.get())
    })
  }

  final def minMax(device: LogicalDevice,
                   xPtr:   DoublePointer,
                   length: Int)
  : (Double, Double) = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMinMaxGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0), NativeDouble(0.0))((tmpMin, tmpMax) => {
      val tmpMinPtr = tmpMin.ptr
      val tmpMaxPtr = tmpMax.ptr
      executeEx(
        device,
        nppsMinMax_64f(
          xPtr,
          length,
          tmpMinPtr,
          tmpMaxPtr,
          wsPtr
        )
      )
      (tmpMinPtr.get(), tmpMaxPtr.get())
    })
  }

  final def minMaxIndx(device: LogicalDevice,
                       xPtr:   FloatPointer,
                       length: Int)
  : ((Float, Int), (Float, Int)) = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMinMaxIndxGetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(
      NativeFloat(0.0f), NativeInt(0),
      NativeFloat(0.0f), NativeInt(0)
    )((tmpMin, tmpMinIndex, tmpMax, tmpMaxIndex) => {
      val tmpMinPtr      = tmpMin.ptr
      val tmpMinIndexPtr = tmpMinIndex.ptr
      val tmpMaxPtr      = tmpMax.ptr
      val tmpMaxIndexPtr = tmpMaxIndex.ptr
      executeEx(
        device,
        nppsMinMaxIndx_32f(
          xPtr,
          length,
          tmpMinPtr, tmpMinIndexPtr,
          tmpMaxPtr, tmpMaxIndexPtr,
          wsPtr
        )
      )
      Tuple2(
        (tmpMinPtr.get(), tmpMinIndexPtr.get()),
        (tmpMaxPtr.get(), tmpMaxIndexPtr.get())
      )
    })
  }

  final def minMaxIndx(device: LogicalDevice,
                       xPtr:   DoublePointer,
                       length: Int)
  : ((Double, Int), (Double, Int)) = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsMinMaxIndxGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(
      NativeDouble(0.0), NativeInt(0),
      NativeDouble(0.0), NativeInt(0)
    )((tmpMin, tmpMinIndex, tmpMax, tmpMaxIndex) => {
      val tmpMinPtr      = tmpMin.ptr
      val tmpMinIndexPtr = tmpMinIndex.ptr
      val tmpMaxPtr      = tmpMax.ptr
      val tmpMaxIndexPtr = tmpMaxIndex.ptr
      executeEx(
        device,
        nppsMinMaxIndx_64f(
          xPtr,
          length,
          tmpMinPtr, tmpMinIndexPtr,
          tmpMaxPtr, tmpMaxIndexPtr,
          wsPtr
        )
      )
      Tuple2(
        (tmpMinPtr.get(), tmpMinIndexPtr.get()),
        (tmpMaxPtr.get(), tmpMaxIndexPtr.get())
      )
    })
  }

  final def mul(device: LogicalDevice,
                x0Ptr:  FloatPointer,
                x1Ptr:  FloatPointer,
                yPtr:   FloatPointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsMul_32f(
      x0Ptr,
      x1Ptr,
      yPtr,
      length
    )
  )

  final def mul(device: LogicalDevice,
                x0Ptr:  FloatPointer,
                x1Ptr:  FloatPointer,
                yPtr:   FloatPointer,
                size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiMul_32f_C1R(
      x0Ptr, size.linePitch,
      x1Ptr, size.linePitch,
      yPtr,  size.linePitch,
      size.ptr
    )
  )

  final def mul(device: LogicalDevice,
                x0Ptr:  DoublePointer,
                x1Ptr:  DoublePointer,
                yPtr:   DoublePointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsMul_64f(
      x0Ptr,
      x1Ptr,
      yPtr,
      length
    )
  )

  final def mul_I(device: LogicalDevice,
                  xPtr:   FloatPointer,
                  yPtr:   FloatPointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsMul_32f_I(
      xPtr,
      yPtr,
      length
    )
  )

  final def mul_I(device: LogicalDevice,
                  xPtr:   FloatPointer,
                  yPtr:   FloatPointer,
                  size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiMul_32f_C1IR(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def mul_I(device: LogicalDevice,
                  xPtr:   DoublePointer,
                  yPtr:   DoublePointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsMul_64f_I(
      xPtr,
      yPtr,
      length
    )
  )

  final def mulC(device: LogicalDevice,
                 x0Ptr:  FloatPointer,
                 x1:     Float,
                 yPtr:   FloatPointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsMulC_32f(
      x0Ptr,
      x1,
      yPtr,
      length
    )
  )

  final def mulC(device: LogicalDevice,
                 x0Ptr:  FloatPointer,
                 x1:     Float,
                 yPtr:   FloatPointer,
                 size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiMulC_32f_C1R(
      x0Ptr, size.linePitch,
      x1,
      yPtr,  size.linePitch,
      size.ptr
    )
  )

  final def mulC(device: LogicalDevice,
                 x0Ptr:  DoublePointer,
                 x1:     Double,
                 yPtr:   DoublePointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsMulC_64f(
      x0Ptr,
      x1,
      yPtr,
      length
    )
  )

  final def mulC(device: LogicalDevice,
                 x:      Double,
                 yPtr:   DoublePointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsMulC_64f_I(
      x,
      yPtr,
      length
    )
  )

  final def mulC_I(device: LogicalDevice,
                   x:      Float,
                   yPtr:   FloatPointer,
                   length: Int)
  : Unit = executeEx(
    device,
    nppsMulC_32f_I(
      x,
      yPtr,
      length
    )
  )

  final def mulC_I(device: LogicalDevice,
                   x:      Float,
                   yPtr:   FloatPointer,
                   size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiMulC_32f_C1IR(
      x,
      yPtr,
      size.linePitch,
      size.ptr
    )
  )

  final def mulC_I(device: LogicalDevice,
                   x:      Double,
                   yPtr:   DoublePointer,
                   length: Int)
  : Unit = executeEx(
    device,
    nppsMulC_64f_I(
      x,
      yPtr,
      length
    )
  )

  final def normDiffInf(device: LogicalDevice,
                        x0Ptr:  FloatPointer,
                        x1Ptr:  FloatPointer,
                        length: Int)
  : Float = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsNormDiffInfGetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsNormDiff_Inf_32f(
          x0Ptr,
          x1Ptr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def normDiffInf(device: LogicalDevice,
                        x0Ptr:  DoublePointer,
                        x1Ptr:  DoublePointer,
                        length: Int)
  : Double = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsNormDiffInfGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsNormDiff_Inf_64f(
          x0Ptr,
          x1Ptr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      device.trySynchronize()
      tmpPtr.get()
    })
  }

  final def normDiffL1(device: LogicalDevice,
                       x0Ptr:  FloatPointer,
                       x1Ptr:  FloatPointer,
                       length: Int)
  : Float = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsNormDiffL1GetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsNormDiff_L1_32f(
          x0Ptr,
          x1Ptr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def normDiffL1(device: LogicalDevice,
                       x0Ptr:  DoublePointer,
                       x1Ptr:  DoublePointer,
                       length: Int)
  : Double = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsNormDiffL1GetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsNormDiff_L1_64f(
          x0Ptr,
          x1Ptr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def normDiffL2(device: LogicalDevice,
                       x0Ptr:  FloatPointer,
                       x1Ptr:  FloatPointer,
                       length: Int)
  : Float = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsNormDiffL2GetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsNormDiff_L2_32f(
          x0Ptr,
          x1Ptr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def normDiffL2(device: LogicalDevice,
                       x0Ptr:  DoublePointer,
                       x1Ptr:  DoublePointer,
                       length: Int)
  : Double = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsNormDiffL2GetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsNormDiff_L2_64f(
          x0Ptr,
          x1Ptr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def normInf(device: LogicalDevice,
                    xPtr:   FloatPointer,
                    length: Int)
  : Float = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsNormInfGetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsNorm_Inf_32f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def normInf(device: LogicalDevice,
                    xPtr:   DoublePointer,
                    length: Int)
  : Double = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsNormInfGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsNorm_Inf_64f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def normL1(device: LogicalDevice,
                   xPtr:   FloatPointer,
                   length: Int)
  : Float = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsNormL1GetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsNorm_L1_32f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def normL1(device: LogicalDevice,
                   xPtr:   DoublePointer,
                   length: Int)
  : Double = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsNormL1GetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsNorm_L1_64f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def normL2(device: LogicalDevice,
                   xPtr:   FloatPointer,
                   length: Int)
  : Float = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsNormL2GetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsNorm_L2_32f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def normL2(device: LogicalDevice,
                   xPtr:   DoublePointer,
                   length: Int)
  : Double = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsNormL2GetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsNorm_L2_64f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def normalize(device: LogicalDevice,
                      xPtr:   FloatPointer,
                      yPtr:   FloatPointer,
                      length: Int,
                      sub:    Float,
                      div:    Float)
  : Unit = executeEx(
    device,
    nppsNormalize_32f(
      xPtr,
      yPtr,
      length,
      sub,
      div
    )
  )

  final def normalize(device: LogicalDevice,
                      xPtr:   DoublePointer,
                      yPtr:   DoublePointer,
                      length: Int,
                      sub:    Double,
                      div:    Double)
  : Unit = executeEx(
    device,
    nppsNormalize_64f(
      xPtr,
      yPtr,
      length,
      sub,
      div
    )
  )

  final def set(device: LogicalDevice,
                x:      Short,
                yPtr:   ShortPointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsSet_16s(
      x,
      yPtr,
      length
    )
  )

  final def set(device: LogicalDevice,
                x:      Short,
                yPtr:   ShortPointer,
                size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiSet_16s_C1R(
      x,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def set(device: LogicalDevice,
                x:      Float,
                yPtr:   FloatPointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsSet_32f(
      x,
      yPtr,
      length
    )
  )

  final def set(device: LogicalDevice,
                x:      Float,
                yPtr:   FloatPointer,
                size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiSet_32f_C1R(
      x,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def set(device: LogicalDevice,
                x:      Double,
                yPtr:   DoublePointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsSet_64f(
      x,
      yPtr,
      length
    )
  )

  final def sqr(device: LogicalDevice,
                xPtr:   FloatPointer,
                yPtr:   FloatPointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsSqr_32f(
      xPtr,
      yPtr,
      length
    )
  )

  final def sqr(device: LogicalDevice,
                xPtr:   FloatPointer,
                yPtr:   FloatPointer,
                size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiSqr_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def sqr(device: LogicalDevice,
                xPtr:   DoublePointer,
                yPtr:   DoublePointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsSqr_64f(
      xPtr,
      yPtr,
      length
    )
  )

  final def sqr_I(device: LogicalDevice,
                  yPtr:   FloatPointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsSqr_32f_I(
      yPtr,
      length
    )
  )

  final def sqr_I(device: LogicalDevice,
                  yPtr:   FloatPointer,
                  size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiSqr_32f_C1IR(
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def sqr_I(device: LogicalDevice,
                  yPtr:   DoublePointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsSqr_64f_I(
      yPtr,
      length
    )
  )

  final def sqrt(device: LogicalDevice,
                 xPtr:   FloatPointer,
                 yPtr:   FloatPointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsSqrt_32f(
      xPtr,
      yPtr,
      length
    )
  )

  final def sqrt(device: LogicalDevice,
                 xPtr:   FloatPointer,
                 yPtr:   FloatPointer,
                 size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiSqrt_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def sqrt(device: LogicalDevice,
                 xPtr:   DoublePointer,
                 yPtr:   DoublePointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsSqrt_64f(
      xPtr,
      yPtr,
      length
    )
  )

  final def sqrt_I(device: LogicalDevice,
                   yPtr:   FloatPointer,
                   length: Int)
  : Unit = executeEx(
    device,
    nppsSqrt_32f_I(
      yPtr,
      length
    )
  )

  final def sqrt_I(device: LogicalDevice,
                   yPtr:   FloatPointer,
                   size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiSqrt_32f_C1IR(
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def sqrt_I(device: LogicalDevice,
                   yPtr:   DoublePointer,
                   length: Int)
  : Unit = executeEx(
    device,
    nppsSqrt_64f_I(
      yPtr,
      length
    )
  )

  final def stdDev(device: LogicalDevice,
                   xPtr:   FloatPointer,
                   length: Int)
  : Float = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsStdDevGetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsStdDev_32f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def stdDev(device: LogicalDevice,
                   xPtr:   DoublePointer,
                   length: Int)
  : Double = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsStdDevGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsStdDev_64f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def sub(device: LogicalDevice,
                x0Ptr:  FloatPointer,
                x1Ptr:  FloatPointer,
                yPtr:   FloatPointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsSub_32f(
      x0Ptr,
      x1Ptr,
      yPtr,
      length
    )
  )

  final def sub(device: LogicalDevice,
                x0Ptr:  FloatPointer,
                x1Ptr:  FloatPointer,
                yPtr:   FloatPointer,
                size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiSub_32f_C1R(
      x0Ptr, size.linePitch,
      x1Ptr, size.linePitch,
      yPtr,  size.linePitch,
      size.ptr
    )
  )

  final def sub(device: LogicalDevice,
                x0Ptr:  DoublePointer,
                x1Ptr:  DoublePointer,
                yPtr:   DoublePointer,
                length: Int)
  : Unit = executeEx(
    device,
    nppsSub_64f(
      x0Ptr,
      x1Ptr,
      yPtr,
      length
    )
  )

  final def sub_I(device: LogicalDevice,
                  xPtr:   FloatPointer,
                  yPtr:   FloatPointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsSub_32f_I(
      xPtr,
      yPtr,
      length
    )
  )

  final def sub_I(device: LogicalDevice,
                  xPtr:   FloatPointer,
                  yPtr:   FloatPointer,
                  size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiSub_32f_C1IR(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def sub_I(device: LogicalDevice,
                  xPtr:   DoublePointer,
                  yPtr:   DoublePointer,
                  length: Int)
  : Unit = executeEx(
    device,
    nppsSub_64f_I(
      xPtr,
      yPtr,
      length
    )
  )

  final def subC(device: LogicalDevice,
                 x0Ptr:  FloatPointer,
                 x1:     Float,
                 yPtr:   FloatPointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsSubC_32f(
      x0Ptr,
      x1,
      yPtr,
      length
    )
  )

  final def subC(device: LogicalDevice,
                 x0Ptr:  FloatPointer,
                 x1:     Float,
                 yPtr:   FloatPointer,
                 size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiSubC_32f_C1R(
      x0Ptr, size.linePitch,
      x1,
      yPtr,  size.linePitch,
      size.ptr
    )
  )

  final def subC(device: LogicalDevice,
                 x0Ptr:  DoublePointer,
                 x1:     Double,
                 yPtr:   DoublePointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsSubC_64f(
      x0Ptr,
      x1,
      yPtr,
      length
    )
  )

  final def subC_I(device: LogicalDevice,
                   x1:     Float,
                   yPtr:   FloatPointer,
                   length: Int)
  : Unit = executeEx(
    device,
    nppsSubC_32f_I(
      x1,
      yPtr,
      length
    )
  )

  final def subC_I(device: LogicalDevice,
                   x1:     Float,
                   yPtr:   FloatPointer,
                   size:   _SizeStruct)
  : Unit = executeEx(
    device,
    nppiSubC_32f_C1IR(
      x1,
      yPtr, size.linePitch,
      size.ptr
    )
  )

  final def subC_I(device: LogicalDevice,
                   x1:     Double,
                   yPtr:   DoublePointer,
                   length: Int)
  : Unit = executeEx(
    device,
    nppsSubC_64f_I(
      x1,
      yPtr,
      length
    )
  )

  final def subCRev(device: LogicalDevice,
                    x0Ptr:  FloatPointer,
                    x1:     Float,
                    yPtr:   FloatPointer,
                    length: Int)
  : Unit = executeEx(
    device,
    nppsSubCRev_32f(
      x0Ptr,
      x1,
      yPtr,
      length
    )
  )

  final def subCRev(device: LogicalDevice,
                    x0Ptr:  DoublePointer,
                    x1:     Double,
                    yPtr:   DoublePointer,
                    length: Int)
  : Unit = executeEx(
    device,
    nppsSubCRev_64f(
      x0Ptr,
      x1,
      yPtr,
      length
    )
  )

  final def subCRev_I(device: LogicalDevice,
                      x:      Float,
                      yPtr:   FloatPointer,
                      length: Int)
  : Unit = executeEx(
    device,
    nppsSubCRev_32f_I(
      x,
      yPtr,
      length
    )
  )

  final def subCRev_I(device: LogicalDevice,
                      x:      Double,
                      yPtr:   DoublePointer,
                      length: Int)
  : Unit = executeEx(
    device,
    nppsSubCRev_64f_I(
      x,
      yPtr,
      length
    )
  )

  final def sum(device: LogicalDevice,
                xPtr:   FloatPointer,
                length: Int)
  : Float = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsSumGetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsSum_32f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }


  final def sum(device: LogicalDevice,
                xPtr:   FloatPointer,
                size:   _SizeStruct)
  : Double = {
    val wsPtr   = device.scratchBuffer.asBytePtr
    val sizePtr = size.ptr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppiSumGetBufferHostSize_32f_C1R(
          sizePtr,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppiSum_32f_C1R(
          xPtr, size.linePitch,
          sizePtr,
          wsPtr,
          tmpPtr
        )
      )
      tmpPtr.get()
    })
  }


  final def sum(device: LogicalDevice,
                xPtr:   DoublePointer,
                length: Int)
  : Double = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsSumGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsSum_64f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def sumLn(device: LogicalDevice,
                  xPtr:   FloatPointer,
                  length: Int)
  : Float = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsSumLnGetBufferSize_32f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeFloat(0.0f))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsSumLn_32f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def sumLn(device: LogicalDevice,
                  xPtr:   DoublePointer,
                  length: Int)
  : Double = {
    val wsPtr = device.scratchBuffer.asBytePtr

    using(NativeInt(0))(tmp => {
      val tmpPtr = tmp.ptr
      execute(
        device,
        nppsSumLnGetBufferSize_64f(
          length,
          tmpPtr
        )
      )
      assume(tmpPtr.get() <= wsPtr.capacity())
    })

    using(NativeDouble(0.0))(tmp => {
      val tmpPtr = tmp.ptr
      executeEx(
        device,
        nppsSumLn_64f(
          xPtr,
          length,
          tmpPtr,
          wsPtr
        )
      )
      tmpPtr.get()
    })
  }

  final def threshold(device:    LogicalDevice,
                      xPtr:      FloatPointer,
                      yPtr:      FloatPointer,
                      length:    Int,
                      threshold: Float,
                      cmpOp:     Int)
  : Unit = executeEx(
    device,
    nppsThreshold_32f(
      xPtr,
      yPtr,
      length,
      threshold,
      cmpOp
    )
  )

  final def threshold(device:    LogicalDevice,
                      xPtr:      FloatPointer,
                      yPtr:      FloatPointer,
                      size:      _SizeStruct,
                      threshold: Float,
                      cmpOp:     Int)
  : Unit = executeEx(
    device,
    nppiThreshold_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr,
      threshold,
      cmpOp
    )
  )

  final def threshold(device:    LogicalDevice,
                      xPtr:      DoublePointer,
                      yPtr:      DoublePointer,
                      threshold: Int,
                      level:     Double,
                      cmpOp:     Int)
  : Unit = executeEx(
    device,
    nppsThreshold_64f(
      xPtr,
      yPtr,
      threshold,
      level,
      cmpOp
    )
  )

  final def threshold_I(device:    LogicalDevice,
                        yPtr:      FloatPointer,
                        length:    Int,
                        threshold: Float,
                        cmpOp:     Int)
  : Unit = executeEx(
    device,
    nppsThreshold_32f_I(
      yPtr,
      length,
      threshold,
      cmpOp
    )
  )

  final def threshold_I(device:    LogicalDevice,
                        yPtr:      FloatPointer,
                        size:      _SizeStruct,
                        threshold: Float,
                        cmpOp:     Int)
  : Unit = executeEx(
    device,
    nppiThreshold_32f_C1IR(
      yPtr, size.linePitch,
      size.ptr,
      threshold,
      cmpOp
    )
  )

  final def threshold_I(device:    LogicalDevice,
                        yPtr:      DoublePointer,
                        length:    Int,
                        threshold: Float,
                        cmpOp:     Int)
  : Unit = executeEx(
    device,
    nppsThreshold_64f_I(
      yPtr,
      length,
      threshold,
      cmpOp
    )
  )

  final def threshold_GT(device:    LogicalDevice,
                         xPtr:      FloatPointer,
                         yPtr:      FloatPointer,
                         length:    Int,
                         threshold: Float)
  : Unit = executeEx(
    device,
    nppsThreshold_GT_32f(
      xPtr,
      yPtr,
      length,
      threshold
    )
  )

  final def threshold_GT(device:    LogicalDevice,
                         xPtr:      FloatPointer,
                         yPtr:      FloatPointer,
                         size:      _SizeStruct,
                         threshold: Float)
  : Unit = executeEx(
    device,
    nppiThreshold_GT_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr,
      threshold
    )
  )

  final def threshold_GT(device:    LogicalDevice,
                         xPtr:      DoublePointer,
                         yPtr:      DoublePointer,
                         length:    Int,
                         threshold: Double)
  : Unit = executeEx(
    device,
    nppsThreshold_GT_64f(
      xPtr,
      yPtr,
      length,
      threshold
    )
  )

  final def threshold_GT_I(device:    LogicalDevice,
                           yPtr:      FloatPointer,
                           length:    Int,
                           threshold: Float)
  : Unit = executeEx(
    device,
    nppsThreshold_GT_32f_I(
      yPtr,
      length,
      threshold
    )
  )

  final def threshold_GT_I(device:    LogicalDevice,
                           yPtr:      FloatPointer,
                           size:      _SizeStruct,
                           threshold: Float)
  : Unit = executeEx(
    device,
    nppiThreshold_GT_32f_C1IR(
      yPtr, size.linePitch,
      size.ptr,
      threshold
    )
  )

  final def threshold_GT_I(device:    LogicalDevice,
                           yPtr:      DoublePointer,
                           length:    Int,
                           threshold: Double)
  : Unit = executeEx(
    device,
    nppsThreshold_GT_64f_I(
      yPtr,
      length,
      threshold
    )
  )

  final def threshold_GTVal(device:    LogicalDevice,
                            xPtr:      FloatPointer,
                            yPtr:      FloatPointer,
                            length:    Int,
                            threshold: Float,
                            value:     Float)
  : Unit = executeEx(
    device,
    nppsThreshold_GTVal_32f(
      xPtr,
      yPtr,
      length,
      threshold,
      value
    )
  )

  final def threshold_GTVal(device:    LogicalDevice,
                            xPtr:      FloatPointer,
                            yPtr:      FloatPointer,
                            size:      _SizeStruct,
                            threshold: Float,
                            value:     Float)
  : Unit = executeEx(
    device,
    nppiThreshold_GTVal_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr,
      threshold,
      value
    )
  )

  final def threshold_GTVal(device:    LogicalDevice,
                            xPtr:      DoublePointer,
                            yPtr:      DoublePointer,
                            length:    Int,
                            threshold: Double,
                            value:     Double)
  : Unit = executeEx(
    device,
    nppsThreshold_GTVal_64f(
      xPtr,
      yPtr,
      length,
      threshold,
      value
    )
  )

  final def threshold_GTVal_I(device:    LogicalDevice,
                              yPtr:      FloatPointer,
                              length:    Int,
                              threshold: Float,
                              value:     Float)
  : Unit = executeEx(
    device,
    nppsThreshold_GTVal_32f_I(
      yPtr,
      length,
      threshold,
      value
    )
  )

  final def threshold_GTVal_I(device:    LogicalDevice,
                              yPtr:      FloatPointer,
                              size:      _SizeStruct,
                              threshold: Float,
                              value:     Float)
  : Unit = executeEx(
    device,
    nppiThreshold_GTVal_32f_C1IR(
      yPtr, size.linePitch,
      size.ptr,
      threshold,
      value
    )
  )

  final def threshold_GTVal_I(device:    LogicalDevice,
                              yPtr:      DoublePointer,
                              length:    Int,
                              threshold: Double,
                              value:     Double)
  : Unit = executeEx(
    device,
    nppsThreshold_GTVal_64f_I(
      yPtr,
      length,
      threshold,
      value
    )
  )

  final def threshold_LT(device:    LogicalDevice,
                         xPtr:      FloatPointer,
                         yPtr:      FloatPointer,
                         length:    Int,
                         threshold: Float)
  : Unit = executeEx(
    device,
    nppsThreshold_LT_32f(
      xPtr,
      yPtr,
      length,
      threshold
    )
  )

  final def threshold_LT(device:    LogicalDevice,
                         xPtr:      FloatPointer,
                         yPtr:      FloatPointer,
                         size:      _SizeStruct,
                         threshold: Float)
  : Unit = executeEx(
    device,
    nppiThreshold_LT_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr,
      threshold
    )
  )

  final def threshold_LT(device:    LogicalDevice,
                         xPtr:      DoublePointer,
                         yPtr:      DoublePointer,
                         length:    Int,
                         threshold: Double)
  : Unit = executeEx(
    device,
    nppsThreshold_LT_64f(
      xPtr,
      yPtr,
      length,
      threshold
    )
  )

  final def threshold_LT_I(device:    LogicalDevice,
                           yPtr:      FloatPointer,
                           length:    Int,
                           threshold: Float)
  : Unit = executeEx(
    device,
    nppsThreshold_LT_32f_I(
      yPtr,
      length,
      threshold
    )
  )

  final def threshold_LT_I(device:    LogicalDevice,
                           yPtr:      FloatPointer,
                           size:      _SizeStruct,
                           threshold: Float)
  : Unit = executeEx(
    device,
    nppiThreshold_LT_32f_C1IR(
      yPtr, size.linePitch,
      size.ptr,
      threshold
    )
  )

  final def threshold_LT_I(device:    LogicalDevice,
                           yPtr:      DoublePointer,
                           length:    Int,
                           threshold: Double)
  : Unit = executeEx(
    device,
    nppsThreshold_LT_64f_I(
      yPtr,
      length,
      threshold
    )
  )

  final def threshold_LTVal(device:    LogicalDevice,
                            xPtr:      FloatPointer,
                            yPtr:      FloatPointer,
                            length:    Int,
                            threshold: Float,
                            value:     Float)
  : Unit = executeEx(
    device,
    nppsThreshold_LTVal_32f(
      xPtr,
      yPtr,
      length,
      threshold,
      value
    )
  )

  final def threshold_LTVal(device:    LogicalDevice,
                            xPtr:      FloatPointer,
                            yPtr:      FloatPointer,
                            size:      _SizeStruct,
                            threshold: Float,
                            value:     Float)
  : Unit = executeEx(
    device,
    nppiThreshold_LTVal_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr,
      threshold,
      value
    )
  )

  final def threshold_LTVal(device:    LogicalDevice,
                            xPtr:      DoublePointer,
                            yPtr:      DoublePointer,
                            length:    Int,
                            threshold: Double,
                            value:     Double)
  : Unit = executeEx(
    device,
    nppsThreshold_LTVal_64f(
      xPtr,
      yPtr,
      length,
      threshold,
      value
    )
  )

  final def threshold_LTVal_I(device:    LogicalDevice,
                              yPtr:      FloatPointer,
                              length:    Int,
                              threshold: Float,
                              value:     Float)
  : Unit = executeEx(
    device,
    nppsThreshold_LTVal_32f_I(
      yPtr,
      length,
      threshold,
      value
    )
  )

  final def threshold_LTVal_I(device:    LogicalDevice,
                              yPtr:      FloatPointer,
                              size:      _SizeStruct,
                              threshold: Float,
                              value:     Float)
  : Unit = executeEx(
    device,
    nppiThreshold_LTVal_32f_C1IR(
      yPtr, size.linePitch,
      size.ptr,
      threshold,
      value
    )
  )

  final def threshold_LTVal_I(device:    LogicalDevice,
                              yPtr:      DoublePointer,
                              length:    Int,
                              threshold: Double,
                              value:     Double)
  : Unit = executeEx(
    device,
    nppsThreshold_LTVal_64f_I(
      yPtr,
      length,
      threshold,
      value
    )
  )

  final def threshold_LTValGTVal(device:      LogicalDevice,
                                 xPtr:        FloatPointer,
                                 yPtr:        FloatPointer,
                                 size:        _SizeStruct,
                                 thresholdLT: Float,
                                 valueLT:     Float,
                                 thresholdGT: Float,
                                 valueGT:     Float)
  : Unit = executeEx(
    device,
    nppiThreshold_LTValGTVal_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr,
      thresholdLT, valueLT,
      thresholdGT, valueGT
    )
  )

  final def threshold_LTValGTVal_I(device:      LogicalDevice,
                                   yPtr:        FloatPointer,
                                   size:        _SizeStruct,
                                   thresholdLT: Float,
                                   valueLT:     Float,
                                   thresholdGT: Float,
                                   valueGT:     Float)
  : Unit = executeEx(
    device,
    nppiThreshold_LTValGTVal_32f_C1IR(
      yPtr, size.linePitch,
      size.ptr,
      thresholdLT, valueLT,
      thresholdGT, valueGT
    )
  )

  final def threshold_Val(device:    LogicalDevice,
                          xPtr:      FloatPointer,
                          yPtr:      FloatPointer,
                          size:      _SizeStruct,
                          threshold: Float,
                          value:     Float,
                          cmpOp:     Int)
  : Unit = executeEx(
    device,
    nppiThreshold_Val_32f_C1R(
      xPtr, size.linePitch,
      yPtr, size.linePitch,
      size.ptr,
      threshold,
      value,
      cmpOp
    )
  )

  final def threshold_Val_I(device:    LogicalDevice,
                            yPtr:      FloatPointer,
                            size:      _SizeStruct,
                            threshold: Float,
                            value:     Float,
                            cmpOp:     Int)
  : Unit = executeEx(
    device,
    nppiThreshold_Val_32f_C1IR(
      yPtr, size.linePitch,
      size.ptr,
      threshold,
      value,
      cmpOp
    )
  )

  final def zero(device: LogicalDevice,
                 yPtr:   ShortPointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsZero_16s(
      yPtr,
      length
    )
  )

  final def zero(device: LogicalDevice,
                 yPtr:   FloatPointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsZero_32f(
      yPtr,
      length
    )
  )

  final def zero(device: LogicalDevice,
                 yPtr:   DoublePointer,
                 length: Int)
  : Unit = executeEx(
    device,
    nppsZero_64f(
      yPtr,
      length
    )
  )

}

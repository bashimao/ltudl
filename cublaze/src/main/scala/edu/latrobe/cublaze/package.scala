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

import edu.latrobe.native._
import org.bytedeco.javacpp._

package object cublaze {

  final val CUBLAZE_NO_LOGICAL_DEVICES
  : String = Environment.get(
    "CUBLAZE_NO_LOGICAL_DEVICES",
    "1111111111",
    _.length > 0
  )

  final val CUBLAZE_NO_STAGING_BUFFERS
  : Int = Environment.parseInt(
    "CUBLAZE_NO_HOST_STAGING_BUFFERS",
    1,
    _ > 0
  )

  final val CUBLAZE_STAGING_BUFFER_SIZE
  : Long = Environment.parseLong(
    "CUBLAZE_STAGING_BUFFER_SIZE",
    1024L * 1024L * 512L,
    _ > 0L,
    StringEx.render(_, 1024, "%.1f %siB")
  )

  final val CUBLAZE_SCRATCH_BUFFER_SIZE
  : Long = Environment.parseLong(
    "CUBLAZE_SCRATCH_BUFFER_SIZE",
    CUBLAZE_STAGING_BUFFER_SIZE,
    _ >= CUBLAZE_STAGING_BUFFER_SIZE,
    StringEx.render(_, 1024, "%.1f %siB")
  )

  final val CUBLAZE_ASYNCHRONOUS
  : Boolean = Environment.parseBoolean(
    "CUBLAZE_ASYNCHRONOUS",
    default = true
  )


  // ---------------------------------------------------------------------------
  //    REAL SWITCH DOUBLE
  // ---------------------------------------------------------------------------
  /*
  private[cublaze] type _RealDeviceBuffer = _DoubleDeviceBuffer

  final private[cublaze] val _RealDeviceBuffer = _DoubleDeviceBuffer
  */
  // ---------------------------------------------------------------------------
  //    REAL SWITCH FLOAT
  // ---------------------------------------------------------------------------
  ///*
  private[cublaze] type _RealDeviceBuffer = _FloatDeviceBuffer

  final private[cublaze] val _RealDeviceBuffer = _FloatDeviceBuffer
  //*/
  // ---------------------------------------------------------------------------
  //    REAL SWITCH END
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  //    REAL TENSOR SWITCH DOUBLE
  // ---------------------------------------------------------------------------
  /*
  private[cublaze] type _RealTensorReal = Double

  private[cublaze] val _RealTensorReal = DoubleEx

  private[cublaze] type _RealTensorPointer = DoublePointer

  private[cublaze] type _RealTensorNativeReal = NativeDouble

  private[cublaze] val _RealTensorNativeReal = NativeDouble

  private[cublaze] type _RealTensorDeviceBuffer = _DoubleDeviceBuffer

  final private[cublaze] val _RealTensorDeviceBuffer = _DoubleDeviceBuffer
  */
  // ---------------------------------------------------------------------------
  //    REAL TENSOR SWITCH FLOAT
  // ---------------------------------------------------------------------------
  private[cublaze] type _RealTensorReal = Float

  private[cublaze] val _RealTensorReal = FloatEx

  private[cublaze] type _RealTensorPointer = FloatPointer

  private[cublaze] type _RealTensorNativeReal = NativeFloat

  private[cublaze] val _RealTensorNativeReal = NativeFloat

  private[cublaze] type _RealTensorDeviceBuffer = _FloatDeviceBuffer

  final private[cublaze] val _RealTensorDeviceBuffer = _FloatDeviceBuffer
  // ---------------------------------------------------------------------------
  //    REAL TENSOR SWITCH HALF
  // ---------------------------------------------------------------------------
  /*
  private[cublaze] type _RealTensorReal = Half

  private[cublaze] val _RealTensorReal = Half

  private[cublaze] type _RealTensorPointer = HalfPointer

  private[cublaze] type _RealTensorNativeReal = NativeFloat

  private[cublaze] val _RealTensorNativeReal = NativeFloat

  private[cublaze] type _RealTensorDeviceBuffer = _HalfDeviceBuffer

  final private[cublaze] val _RealTensorDeviceBuffer = _HalfDeviceBuffer
  */
  // ---------------------------------------------------------------------------
  //    REAL TENSOR SWITCH END
  // ---------------------------------------------------------------------------

  /*
  final private[cublaze] implicit class RealPtrFunctions(rp: _RealPtr) {

    @inline
    def capacityInBytes
    : Long = rp.capacity() * Real.size

    @inline
    def put(array: Array[Real]): _RealPtr = put(array, 0)

    @inline
    def put(array: Array[Real], offset: Int): _RealPtr = {
      var off = offset
      val end = offset + rp.capacity()
      while (offset < end) {
        val length = Math.min(end - offset, Int.MaxValue).toInt
        rp.put(array, offset, length)
        off += length
      }
      rp
    }

  }

  final private[cublaze] implicit class IntPtrFunctions(rp: NativeIntPtr) {

    def capacityInBytes
    : Long = rp.capacity() * (Integer.SIZE / 8)

  }
  */

  final private[cublaze] implicit class CUBlazeTensorFunctions(ten: Tensor) {

    def asOrToCUDARealTensor(device: LogicalDevice)
    : CUDARealTensor = ten match {
      case ten: CUDARealTensor =>
        if (device eq ten.device) {
          ten
        }
        else {
          toCUDARealTensor(device)
        }
      case _ =>
        toCUDARealTensor(device)
    }

    def toCUDARealTensor(device: LogicalDevice)
    : CUDARealTensor = {
      val res = CUDARealTensor(device, ten.layout.makeIndependent)
      res := ten
      res
    }

  }

}

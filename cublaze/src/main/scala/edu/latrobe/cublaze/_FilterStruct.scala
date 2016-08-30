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
import edu.latrobe.kernels._
import edu.latrobe.native._
import org.bytedeco.javacpp.cudnn._

final class _FilterStruct private(val noMaps:     Int,
                                  val kernel:     Kernel,
                                  val noChannels: Int)
  extends AutoClosingPointerEx[cudnnFilterStruct] {

  override protected val _ptr
  : cudnnFilterStruct = _CUDNN.createFilterDescriptor()

  override protected def doClose()
  : Unit = {
    _CUDNN.destroyFilterDescriptor(this)
    super.doClose()
  }

  @transient
  lazy val noDims
  : Int = _CUDNN.getFilterNdDescriptor(this)._1

  @transient
  lazy val dataType
  : Int = _CUDNN.getFilterNdDescriptor(this)._2

  @transient
  lazy val dims
  : Array[Int] = {
    val noDims = _CUDNN.getFilterNdDescriptor(this)._1
    val dims   = new Array[Int](noDims)
    _CUDNN.getFilterNdDescriptor(this, dims)
    dims
  }

}

private[cublaze] object _FilterStruct {

  final def apply(noMaps:     Int,
                  kernel:     Kernel,
                  noChannels: Int)
  : _FilterStruct = {
    require(noMaps >= 1)
    val result = new _FilterStruct(noMaps, kernel, noChannels)
    kernel match {
      case kernel: Kernel1 =>
        _CUDNN.setFilter4dDescriptor(
          result,
          _RealTensorDeviceBuffer.dataType,
          CUDNN_TENSOR_NCHW,
          noMaps,
          noChannels,
          1,
          kernel.noValues
        )

      case kernel: Kernel2 =>
        _CUDNN.setFilter4dDescriptor(
          result,
          _RealTensorDeviceBuffer.dataType,
          CUDNN_TENSOR_NCHW,
          noMaps,
          noChannels,
          kernel.size._2,
          kernel.size._1
        )

      case kernel: Kernel3 =>
        val dims = Array(
          noMaps,
          noChannels,
          kernel.size._3,
          kernel.size._2,
          kernel.size._1
        )
        _CUDNN.setFilterNdDescriptor(
          result,
          _RealTensorDeviceBuffer.dataType,
          CUDNN_TENSOR_NCHW,
          dims
        )

      case kernel: Kernel4 =>
        val dims = Array(
          noMaps,
          noChannels,
          kernel.size._4,
          kernel.size._3,
          kernel.size._2,
          kernel.size._1
        )
        _CUDNN.setFilterNdDescriptor(
          result,
          _RealTensorDeviceBuffer.dataType,
          CUDNN_TENSOR_NCHW,
          dims
        )

      case _ =>
        // TODO: Implement Nd case.
        throw new MatchError(kernel)
    }
    result
  }

}

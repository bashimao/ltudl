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

final class _PoolingStruct private(val kernel:      Kernel,
                                   val poolingMode: Int)
  extends AutoClosingPointerEx[cudnnPoolingStruct] {

  override protected val _ptr
  : cudnnPoolingStruct = _CUDNN.createPoolingDescriptor()

  override protected def doClose()
  : Unit = {
    _CUDNN.destroyPoolingDescriptor(this)
    super.doClose()
  }

}

private[cublaze] object _PoolingStruct {

  final def apply(kernel: Kernel, poolingMode: Int)
  : _PoolingStruct = {
    val result = new _PoolingStruct(kernel, poolingMode)

    // Populate value fields.
    kernel match {
      case kernel: Kernel1 =>
        require(kernel.padding0 == kernel.padding1)
        _CUDNN.setPooling2dDescriptor(
          result,
          poolingMode,
          CUDNN_NOT_PROPAGATE_NAN,
          1, kernel.size._1,
          0, kernel.padding0._1,
          1, kernel.stride._1
        )

      case kernel: Kernel2 =>
        require(kernel.padding0 == kernel.padding1)
        _CUDNN.setPooling2dDescriptor(
          result,
          poolingMode,
          CUDNN_NOT_PROPAGATE_NAN,
          kernel.size._2,     kernel.size._1,
          kernel.padding0._2, kernel.padding0._1,
          kernel.stride._2,   kernel.stride._1
        )

      case kernel: Kernel3 =>
        require(kernel.padding0 == kernel.padding1)
        val size = Array(
          kernel.size._3,
          kernel.size._2,
          kernel.size._1
        )
        val padding = Array(
          kernel.padding0._3,
          kernel.padding0._2,
          kernel.padding0._1
        )
        val stride = Array(
          kernel.stride._3,
          kernel.stride._2,
          kernel.stride._1
        )
        _CUDNN.setPoolingNdDescriptor(
          result,
          poolingMode,
          CUDNN_NOT_PROPAGATE_NAN,
          size,
          padding,
          stride
        )

      case kernel: Kernel4 =>
        require(kernel.padding0 == kernel.padding1)
        val size = Array(
          kernel.size._4,
          kernel.size._3,
          kernel.size._2,
          kernel.size._1
        )
        val padding = Array(
          kernel.padding0._4,
          kernel.padding0._3,
          kernel.padding0._2,
          kernel.padding0._1
        )
        val stride = Array(
          kernel.stride._4,
          kernel.stride._3,
          kernel.stride._2,
          kernel.stride._1
        )
        _CUDNN.setPoolingNdDescriptor(
          result,
          poolingMode,
          CUDNN_NOT_PROPAGATE_NAN,
          size,
          padding,
          stride
        )

      case _ =>
        // TODO: Implement Nd case.
        throw new MatchError(kernel)
    }

    // Return ptr.
    result
  }

}
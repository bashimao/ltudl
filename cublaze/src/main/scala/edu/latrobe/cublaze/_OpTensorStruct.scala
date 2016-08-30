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

import edu.latrobe.native._
import org.bytedeco.javacpp.cudnn._

final class _OpTensorStruct private
  extends AutoClosingPointerEx[cudnnOpTensorStruct] {

  override protected val _ptr
  : cudnnOpTensorStruct = _CUDNN.createOpTensorDescriptor()

  override protected def doClose()
  : Unit = {
    _CUDNN.destroyOpTensorDescriptor(this)
    super.doClose()
  }

}

private[cublaze] object _OpTensorStruct {

  final def apply(op: Int)
  : _OpTensorStruct = {
    val ptr = new _OpTensorStruct
    _CUDNN.setOpTensorDescriptor(
      ptr,
      op,
      _RealTensorDeviceBuffer.dataType,
      CUDNN_NOT_PROPAGATE_NAN
    )
    ptr
  }

  final val add
  : _OpTensorStruct = apply(CUDNN_OP_TENSOR_ADD)

  final val multiply
  : _OpTensorStruct = apply(CUDNN_OP_TENSOR_MUL)

  final val max
  : _OpTensorStruct = apply(CUDNN_OP_TENSOR_MAX)

  final val min
  : _OpTensorStruct = apply(CUDNN_OP_TENSOR_MIN)

}
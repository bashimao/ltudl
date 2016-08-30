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

final class _ActivationStruct private
  extends AutoClosingPointerEx[cudnnActivationStruct] {

  override protected val _ptr
  : cudnnActivationStruct = _CUDNN.createActivationDescriptor()

  override protected def doClose()
  : Unit = {
    _CUDNN.destroyActivationDescriptor(this)
    super.doClose()
  }

  @transient
  lazy val mode
  : Int = _CUDNN.getActivationDescriptor(this)._1

  @transient
  lazy val reluNaNOpt
  : Int = _CUDNN.getActivationDescriptor(this)._2

  @transient
  lazy val reluThreshold
  : Double = _CUDNN.getActivationDescriptor(this)._3

}

private[cublaze] object _ActivationStruct {

  final def apply(mode: Int)
  : _ActivationStruct = apply(mode, 0.0)

  final def apply(mode: Int, reluThreshold: Double)
  : _ActivationStruct = {
    val result = new _ActivationStruct
    _CUDNN.setActivationDescriptor(
      result, mode, CUDNN_NOT_PROPAGATE_NAN, reluThreshold
    )
    result
  }

}

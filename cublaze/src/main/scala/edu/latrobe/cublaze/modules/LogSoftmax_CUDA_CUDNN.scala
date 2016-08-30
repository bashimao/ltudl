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

package edu.latrobe.cublaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.cublaze._
import edu.latrobe.native._
import org.bytedeco.javacpp.cudnn._

final class LogSoftmax_CUDA_CUDNN(override val builder:        LogSoftmaxBuilder,
                                  override val inputHints:     BuildHints,
                                  override val seed:           InstanceSeed,
                                  override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends LogSoftmax_CUDA {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(inp: CUDARealTensor,
                                   out: CUDARealTensor)
  : Unit = {
    _CUDNN.softmaxForward(
      device,
      CUDNN_SOFTMAX_LOG,
      CUDNN_SOFTMAX_MODE_INSTANCE,
      _RealTensorNativeReal.one,
      inp.desc, inp.data.ptr,
      _RealTensorNativeReal.zero,
      out.desc, out.data.ptr
    )
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(out: CUDARealTensor,
                                            err: CUDARealTensor)
  : Unit = {
    _CUDNN.softmaxBackward(
      device,
      CUDNN_SOFTMAX_LOG,
      CUDNN_SOFTMAX_MODE_INSTANCE,
      _RealTensorNativeReal.one,
      out.desc, out.data.ptr,
      err.desc, err.data.ptr,
      _RealTensorNativeReal.zero,
      err.desc, err.data.ptr
    )
  }

}

object LogSoftmax_CUDA_CUDNN_Description
  extends ModuleVariant_CUDA_Description[LogSoftmaxBuilder] {

  override def build(builder:        LogSoftmaxBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : LogSoftmax_CUDA_CUDNN = new LogSoftmax_CUDA_CUDNN(
    builder, hints, seed, weightsBuilder
  )

}
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
import org.bytedeco.javacpp.cudnn._

final class ReLU_CUDA_CUDNN(override val builder:        ReLUBuilder,
                            override val inputHints:     BuildHints,
                            override val seed:           InstanceSeed,
                            override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ReLU_CUDA {

  private val actDesc
  : _ActivationStruct = _ActivationStruct(CUDNN_ACTIVATION_RELU)

  override protected def doClose()
  : Unit = {
    actDesc.close()
    super.doClose()
  }

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(output: CUDARealTensor)
  : Unit = {
    _CUDNN.activationForward(
      device,
      actDesc,
      _RealTensorNativeReal.one,
      output.desc, output.data.ptr,
      _RealTensorNativeReal.zero,
      output.desc, output.data.ptr
    )
  }

  override protected def doPredictInv(input: CUDARealTensor)
  : Unit = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(input: CUDARealTensor,
                                            error: CUDARealTensor)
  : Unit = {
    _CUDNN.activationBackward(
      device,
      actDesc,
      _RealTensorNativeReal.one,
      input.desc, input.data.ptr,
      error.desc, error.data.ptr,
      input.desc, input.data.ptr,
      _RealTensorNativeReal.zero,
      error.desc, error.data.ptr
    )
  }

}

object ReLU_CUDA_CUDNN_Description
  extends ModuleVariant_CUDA_Description[ReLUBuilder] {

  override def build(builder:        ReLUBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ReLU_CUDA_CUDNN = new ReLU_CUDA_CUDNN(builder, hints, seed, weightsBuilder)

}

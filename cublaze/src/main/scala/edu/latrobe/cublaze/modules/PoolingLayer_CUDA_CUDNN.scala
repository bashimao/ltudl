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

trait PoolingLayer_CUDA_CUDNN[TBuilder <: PoolingLayerExBuilder[_]]
  extends PoolingLayerEx[TBuilder]
    with Layer_CUDA[TBuilder] {

  final override lazy val outputPlatform
  : CUDA.type = CUDA

  /**
   * Best override this with a val.
   */
  def poolingMode
  : Int

  final private val poolDesc
  : _PoolingStruct = _PoolingStruct(kernel, poolingMode)

  final override protected def doClose()
  : Unit = {
    poolDesc.close()
    super.doClose()
  }


  // ---------------------------------------------------------------------------
  //   Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode: Mode, input: Tensor)
  : (Tensor, PredictContext) = {
    // Move data to CUDA device & allocate buffer for results
    val inp       = input.asOrToCUDARealTensor(device)
    val inpLayout = inp.layout
    val inpSize   = inpLayout.size
    val outSize   = kernel.outputSizeFor(inpSize, inpSize.noChannels)
    val outLayout = inpLayout.derive(outSize)
    val out       = CUDARealTensor(device, outLayout)

    // Do pooling.
    _CUDNN.poolingForward(
      device,
      poolDesc,
      _RealTensorNativeReal.one,
      inp.desc, inp.data.ptr,
      _RealTensorNativeReal.zero,
      out.desc, out.data.ptr
    )

    // Release temporary CUDA resources.
    if (inp ne input) {
      inp.close()
    }
    (out, EmptyContext)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  // TODO: What a waste for some types of pooling. Find a way to circumvent this.
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.Required

  // TODO: What a waste for some types of pooling. Find a way to circumvent this.
  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.Required

  final override protected def doDeriveInputError(input:     Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = {
    // Move data to CUDA device & allocate buffer for results
    val inp    = input.asOrToCUDARealTensor(device)
    val out    = output.asOrToCUDARealTensor(device)
    val oldErr = error.asOrToCUDARealTensor(device)
    val newErr = CUDARealTensor(device, inp.layout)

    // Do pooling.
    _CUDNN.poolingBackward(
      device,
      poolDesc,
      _RealTensorNativeReal.one,
      out.desc,    out.data.ptr,
      oldErr.desc, oldErr.data.ptr,
      inp.desc,    inp.data.ptr,
      _RealTensorNativeReal.zero,
      newErr.desc, newErr.data.ptr
    )

    // Release temporary CUDA resources.
    if (oldErr ne error) {
      oldErr.close()
    }
    if (out ne output) {
      out.close()
    }
    if (inp ne input) {
      inp.close()
    }
    newErr
  }

}

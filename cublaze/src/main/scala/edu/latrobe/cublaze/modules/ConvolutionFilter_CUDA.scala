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

abstract class ConvolutionFilter_CUDA
  extends ConvolutionFilter
    with Layer_CUDA[ConvolutionFilterBuilder] {

  final override lazy val outputPlatform
  : CUDA.type = CUDA

  final override val (filter, filterReference) = {
    val ref = builder.filterReference
    val tmp = weightBufferBuilder.get(ref)
    if (tmp.isDefined) {
      val result = tmp.get.asInstanceOf[CUDARealTensor]
      (result, None)
    }
    else {
      val result = CUDARealTensor.zeros(device, filterLayout)
      val newRef = weightBufferBuilder.register(ref, result)
      (result, Some(newRef))
    }
  }

  override protected def doClose()
  : Unit = {
    if (filterReference.isDefined) {
      filter.close()
    }
    super.doClose()
  }


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input: Tensor)
  : (Tensor, PredictContext) = {
    // Move data to CUDA device & allocate buffer for results
    val inp       = input.asOrToCUDARealTensor(device)
    val inpLayout = inp.layout
    val outSize   = kernel.outputSizeFor(inpLayout.size, noMaps)
    val outLayout = inpLayout.derive(outSize)
    val out       = CUDARealTensor(device, outLayout)
    val ctx       = doPredict(inp, out)

    // Cleanup.
    if (inp ne input) {
      inp.close()
    }
    (out, ctx)
  }

  protected def doPredict(input:  CUDARealTensor,
                          output: CUDARealTensor)
  : PredictContext

  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveFilterGradients(input:   Tensor,
                                                       context: PredictContext,
                                                       error:   Tensor,
                                                       sink:    ValueTensor)
  : Unit = {
    require(error.layout.size.noChannels == outputSizeHint.noChannels)
    val inp = input.asOrToCUDARealTensor(device)
    val err = error.asOrToCUDARealTensor(device)
    val dst = sink.asOrToCUDARealTensor(device)

    doDeriveFilterGradients(inp, context, err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      dst.copyTo(sink)
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
    if (inp ne input) {
      inp.close()
    }
  }

  protected def doDeriveFilterGradients(input:   CUDARealTensor,
                                        context: PredictContext,
                                        error:   CUDARealTensor,
                                        sink:    CUDARealTensor)
  : Unit

  final override protected def doDeriveInputError(inputLayout: TensorLayout,
                                                  context:     PredictContext,
                                                  error:       Tensor)
  : CUDARealTensor = {
    require(error.layout.size.noChannels == outputSizeHint.noChannels)

    // Move data to CUDA device & allocate buffer for results
    val oldErr = error.asOrToCUDARealTensor(device)
    val newErr = CUDARealTensor(device, inputLayout.makeIndependent)

    doDeriveInputError(context, oldErr, newErr)

    if (oldErr ne error) {
      oldErr.close()
    }
    newErr
  }

  protected def doDeriveInputError(context:  PredictContext,
                                   oldError: CUDARealTensor,
                                   newError: CUDARealTensor)
  : Unit

}

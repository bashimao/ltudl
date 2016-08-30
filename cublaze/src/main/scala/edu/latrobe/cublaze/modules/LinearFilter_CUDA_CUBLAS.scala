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

final class LinearFilter_CUDA_CUBLAS(override val builder:        LinearFilterBuilder,
                                     override val inputHints:     BuildHints,
                                     override val seed:           InstanceSeed,
                                     override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends LinearFilter_CUDA {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input:  CUDARealTensor,
                                   output: CUDARealTensor)
  : Unit = {
    val inpPtr  = input.data.ptr
    val inpRows = input.layout.size.noValues
    val inpCols = input.layout.noSamples
    val outPtr  = output.data.ptr
    val outRows = output.layout.size.noValues
    val outCols = output.layout.noSamples
    val filPtr  = filter.data.ptr
    val filRows = filter.layout.size.noValues
    val filCols = filter.layout.noSamples
    _CUBLAS.gemm(
      device,
      _RealTensorNativeReal.one,
      filPtr, filRows, filCols, filRows, aTrans = true,
      inpPtr, inpRows, inpRows, inpCols, bTrans = false,
      _RealTensorNativeReal.zero,
      outPtr, outRows, outRows, outCols
    )
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveFilterGradients(input: CUDARealTensor,
                                                 error: CUDARealTensor,
                                                 sink:  CUDARealTensor)
  : Unit = {
    val inpPtr  = input.data.ptr
    val inpRows = input.layout.size.noValues
    val inpCols = input.layout.noSamples
    val errPtr  = error.data.ptr
    val errRows = error.layout.size.noValues
    val errCols = error.layout.noSamples
    val dstPtr  = sink.data.ptr
    val dstRows = sink.layout.size.noValues
    val dstCols = sink.layout.noSamples
    _CUBLAS.gemm(
      device,
      _RealTensorNativeReal.one,
      inpPtr, inpRows, inpRows, inpCols, aTrans = false,
      errPtr, errRows, errCols, errRows, bTrans = true,
      _RealTensorNativeReal.one,
      dstPtr, dstRows, dstRows, dstCols
    )
  }

  override protected def doDeriveInputError(oldError: CUDARealTensor,
                                            newError: CUDARealTensor)
  : Unit = {
    val oldErrPtr  = oldError.data.ptr
    val oldErrRows = oldError.layout.size.noValues
    val oldErrCols = oldError.layout.noSamples
    val newErrPtr  = newError.data.ptr
    val newErrRows = newError.layout.size.noValues
    val newErrCols = newError.layout.noSamples
    val filterPtr  = filter.data.ptr
    val filterRows = filter.layout.size.noValues
    val filterCols = filter.layout.noSamples
    _CUBLAS.gemm(
      device,
      _RealTensorNativeReal.one,
      filterPtr, filterRows, filterRows, filterCols, aTrans = false,
      oldErrPtr, oldErrRows, oldErrRows, oldErrCols, bTrans = false,
      _RealTensorNativeReal.zero,
      newErrPtr, newErrRows, newErrRows, newErrCols
    )
  }

}

object LinearFilter_CUDA_CUBLAS_Description
  extends ModuleVariant_CUDA_Description[LinearFilterBuilder] {

  override def build(builder:       LinearFilterBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : LinearFilter_CUDA_CUBLAS = new LinearFilter_CUDA_CUBLAS(
    builder, hints, seed, weightsBuilder
  )

}

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

final class MultiplyFilter_CUDA_Baseline(override val builder:        MultiplyFilterBuilder,
                                         override val inputHints:     BuildHints,
                                         override val seed:           InstanceSeed,
                                         override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends MultiplyFilter_CUDA {

  private var _ones
  : CUDARealTensor = _

  private lazy val ones
  : CUDARealTensor = {
    if (_ones == null) {
      _ones = CUDARealTensor.fill(device, filterLayout, Real.one)
    }
    _ones
  }

  override protected def doClose()
  : Unit = {
    if (_ones != null) {
      _ones.close()
      _ones = null
    }
    super.doClose()
  }


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredictPerValue(output: CUDARealTensor)
  : Unit = {
    // input:  RRR GGG BBB | RRR GGG BBB
    // filter: RRR GGG BBB | RRR GGG BBB
    output :*= filter
  }

  override protected def doPredictPerUnit(output: CUDARealTensor)
  : Unit = {
    // input:  RRR GGG BBB | RRR GGG BBB
    // filter: RRR GGG BBB
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  output.desc, output.data.ptr,
      _RealTensorNativeReal.one,  filter.desc, filter.data.ptr,
      _RealTensorNativeReal.zero, output.desc, output.data.ptr
    )
  }

  override protected def doPredictPerChannel(output: CUDARealTensor)
  : Unit = {
    // input:  RRR GGG BBB | RRR GGG BBB
    // filter: RGB
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  output.desc, output.data.ptr,
      _RealTensorNativeReal.one,  filter.desc, filter.data.ptr,
      _RealTensorNativeReal.zero, output.desc, output.data.ptr
    )
    /*
    scope match {
      case OperationScope.Channel =>
        CUDNN.opTensor(
          device,
          OpTensorStructPtr.multiply,
          RealPtr.one, output.descPtr, output.dataPtr,
          RealPtr.one, filter.descPtr, filter.dataPtr,
          RealPtr.one, output.descPtr, output.dataPtr
        )
      /*
      // Momentarily reshape the tensor.
      using(
        TensorStructPtr.nhwc(output.layout),
        device.requestScratchBuffer()
      )((nhwcPtr, scratchBuffer) => {
        val tmpPtr = scratchBuffer.ptr
        CUDNN.transformTensor(
          device,
          RealPtr.one,  output.descPtr, output.dataPtr,
          RealPtr.zero, nhwcPtr,        tmpPtr
        )

        val noRows = output.layout.size.noChannels
        val noCols = output.layout.noTuples
        CUBLAS.gmm(
          device,
          filter.dataPtr, 1,      noRows,
          tmpPtr,         noRows, noRows, noCols,
          tmpPtr,         noRows, noRows, noCols
        )

        CUDNN.transformTensor(
          device,
          RealPtr.one,  nhwcPtr,        tmpPtr,
          RealPtr.zero, output.descPtr, output.dataPtr
        )
      })
      */

      case OperationScope.Sample =>
      /*
      val noRows = output.layout.size.noValues
      val noCols = output.layout.noSamples
      CUBLAS.gmm(
        device,
        filter.dataPtr, 1,      noRows,
        output.dataPtr, noRows, noRows, noCols,
        output.dataPtr, noRows, noRows, noCols
      )
      */

      case OperationScope.Batch =>
        val noRows = 1
        val noCols = output.layout.noValues
        CUBLAS.gmm(
          device,
          filter.dataPtr, 1,      noRows,
          output.dataPtr, noRows, noRows, noCols,
          output.dataPtr, noRows, noRows, noCols
        )

      case _ =>
        throw new MatchError(scope)
    }
    */
  }

  override protected def doPredictPerSample(output: CUDARealTensor)
  : Unit = {
    // input:  RRR GGG BBB | RRR GGG BBB
    // filter: R           | R
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  output.desc, output.data.ptr,
      _RealTensorNativeReal.one,  filter.desc, filter.data.ptr,
      _RealTensorNativeReal.zero, output.desc, output.data.ptr
    )
  }

  override protected def doPredictPerBatch(output: CUDARealTensor)
  : Unit = {
    // input:  RRR GGG BBB | RRR GGG BBB
    // filter: R
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  output.desc, output.data.ptr,
      _RealTensorNativeReal.one,  filter.desc, filter.data.ptr,
      _RealTensorNativeReal.zero, output.desc, output.data.ptr
    )
  }

  override protected def doPredictInvPerValue(input: CUDARealTensor)
  : Unit = input :/= filter

  override protected def doPredictInvPerUnit(input: CUDARealTensor)
  : Unit = throw new NotImplementedError

  override protected def doPredictInvPerChannel(input: CUDARealTensor)
  : Unit = throw new NotImplementedError

  override protected def doPredictInvPerSample(input: CUDARealTensor)
  : Unit = throw new NotImplementedError

  override protected def doPredictInvPerBatch(input: CUDARealTensor)
  : Unit = throw new NotImplementedError


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveFilterGradientsPerValue(input: CUDARealTensor,
                                                         error: CUDARealTensor,
                                                         sink:  CUDARealTensor)
  : Unit = {
    // input: RRR GGG BBB | RRR GGG BBB
    // error: RRR GGG BBB | RRR GGG BBB
    // sink:  RRR GGG BBB | RRR GGG BBB
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  input.desc, input.data.ptr,
      _RealTensorNativeReal.one,  error.desc, error.data.ptr,
      _RealTensorNativeReal.zero, sink.desc,  sink.data.ptr
    )
  }

  override protected def doDeriveFilterGradientsPerUnit(input: CUDARealTensor,
                                                        error: CUDARealTensor,
                                                        sink:  CUDARealTensor)
  : Unit = {
    // input: RRR GGG BBB | RRR GGG BBB
    // error: RRR GGG BBB | RRR GGG BBB
    // sink:  RRR GGG BBB
    // TODO: Find a better way to do this!
    val wsPtr = device.scratchBuffer.asRealTensorPtr

    require(error.layout.noValues <= wsPtr.capacity())
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  input.desc, input.data.ptr,
      _RealTensorNativeReal.one,  error.desc, error.data.ptr,
      _RealTensorNativeReal.zero, error.desc, wsPtr
    )
    // TODO: Find something nicer than gemv!
    val noRows = error.layout.size.noValues
    val noCols = error.layout.noSamples
    _CUBLAS.gemv(
      device,
      _RealTensorNativeReal.one,
      wsPtr,         noRows, noRows, noCols, aTrans = false,
      ones.data.ptr, 1,      noRows,
      _RealTensorNativeReal.one,
      sink.data.ptr, 1,      noRows
    )
  }

  override protected def doDeriveFilterGradientsPerChannel(input: CUDARealTensor,
                                                           error: CUDARealTensor,
                                                           sink:  CUDARealTensor)
  : Unit = {
    // input: RRR GGG BBB | RRR GGG BBB
    // error: RRR GGG BBB | RRR GGG BBB
    // sink:  RGB
    // TODO: Find a better way to do this!
    val wsPtr = device.scratchBuffer.asRealTensorPtr

    require(error.layout.noValues <= wsPtr.capacity())
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  input.desc, input.data.ptr,
      _RealTensorNativeReal.one,  error.desc, error.data.ptr,
      _RealTensorNativeReal.zero, error.desc, wsPtr
    )
    _CUDNN.convolutionBackwardBias(
      device,
      _RealTensorNativeReal.one, error.desc, wsPtr,
      _RealTensorNativeReal.one, sink.desc,  sink.data.ptr
    )
  }

  override protected def doDeriveFilterGradientsPerSample(input: CUDARealTensor,
                                                          error: CUDARealTensor,
                                                          sink:  CUDARealTensor)
  : Unit = {
    // input: RRR GGG BBB | RRR GGG BBB
    // error: RRR GGG BBB | RRR GGG BBB
    // sink:  R           | R
    // TODO: Find a better way to do this!
    val wsPtr = device.scratchBuffer.asRealTensorPtr

    require(error.layout.noValues <= wsPtr.capacity())
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  input.desc, input.data.ptr,
      _RealTensorNativeReal.one,  error.desc, error.data.ptr,
      _RealTensorNativeReal.zero, error.desc, wsPtr
    )
    // TODO: Find something nicer than gemv!
    val noRows = error.layout.noSamples
    val noCols = error.layout.size.noValues
    _CUBLAS.gemv(
      device,
      _RealTensorNativeReal.one,
      error.data.ptr, noRows, noRows, noCols, aTrans = true,
      ones.data.ptr,  1,      noRows,
      _RealTensorNativeReal.one,
      sink.data.ptr,  1,      noRows
    )
  }

  override protected def doDeriveFilterGradientsPerBatch(input: CUDARealTensor,
                                                         error: CUDARealTensor,
                                                         sink:  CUDARealTensor)
  : Unit = {
    // input: RRR GGG BBB | RRR GGG BBB
    // error: RRR GGG BBB | RRR GGG BBB
    // sink:  R
    // TODO: Find a better way to do this!
    val wsPtr = device.scratchBuffer.asRealTensorPtr

    require(error.layout.noValues <= wsPtr.capacity())
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  input.desc, input.data.ptr,
      _RealTensorNativeReal.one,  error.desc, error.data.ptr,
      _RealTensorNativeReal.zero, error.desc, wsPtr
    )
    // TODO: Find something nicer than gemv!
    val noRows = 1
    val noCols = error.layout.noValues
    _CUBLAS.gemv(
      device,
      _RealTensorNativeReal.one,
      error.data.ptr, noRows, noRows, noCols, aTrans = false,
      ones.data.ptr,  1,      noRows,
      _RealTensorNativeReal.one,
      sink.data.ptr,  1,      noRows
    )
  }

  override protected def doDeriveInputErrorPerValue(error: CUDARealTensor)
  : Unit = {
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  error.desc,  error.data.ptr,
      _RealTensorNativeReal.one,  filter.desc, filter.data.ptr,
      _RealTensorNativeReal.zero, error.desc,  error.data.ptr
    )
    /*
    scope match {
      case OperationScope.Channel =>
        // Momentarily reshape the tensor.
        using(
          TensorStructPtr.nhwc(error.layout),
          device.requestScratchBuffer()
        )((nhwcPtr, scratchBuffer) => {
          val tmpPtr = scratchBuffer.ptr
          CUDNN.transformTensor(
            device,
            RealPtr.one,  error.descPtr, error.dataPtr,
            RealPtr.zero, nhwcPtr,       tmpPtr
          )

          val noRows = error.layout.size.noChannels
          val noCols = error.layout.noTuples
          CUBLAS.gmm(
            device,
            tmpPtr,         noRows, noRows, noCols,
            filter.dataPtr, 1,      noRows,
            tmpPtr,         noRows, noRows, noCols
          )

          CUDNN.transformTensor(
            device,
            RealPtr.one,  nhwcPtr,       tmpPtr,
            RealPtr.zero, error.descPtr, error.dataPtr
          )
        })

      case OperationScope.Sample =>
        val noRows = error.layout.size.noValues
        val noCols = error.layout.noSamples
        CUBLAS.gmm(
          device,
          error.dataPtr,  noRows, noRows, noCols,
          filter.dataPtr, 1,      noRows,
          error.dataPtr,  noRows, noRows, noCols
        )

      case OperationScope.Batch =>
        val noRows = 1
        val noCols = error.layout.noValues
        CUBLAS.gmm(
          device,
          error.dataPtr,  noRows, noRows, noCols,
          filter.dataPtr, 1,      noRows,
          error.dataPtr,  noRows, noRows, noCols
        )

      case _ =>
        throw new MatchError(scope)
    }
    */
  }

  override protected def doDeriveInputErrorPerUnit(error: CUDARealTensor)
  : Unit = {
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  error.desc,  error.data.ptr,
      _RealTensorNativeReal.one,  filter.desc, filter.data.ptr,
      _RealTensorNativeReal.zero, error.desc,  error.data.ptr
    )
  }

  override protected def doDeriveInputErrorPerChannel(error: CUDARealTensor)
  : Unit = {
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  error.desc,  error.data.ptr,
      _RealTensorNativeReal.one,  filter.desc, filter.data.ptr,
      _RealTensorNativeReal.zero, error.desc,  error.data.ptr
    )
  }

  override protected def doDeriveInputErrorPerSample(error: CUDARealTensor)
  : Unit = {
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  error.desc,  error.data.ptr,
      _RealTensorNativeReal.one,  filter.desc, filter.data.ptr,
      _RealTensorNativeReal.zero, error.desc,  error.data.ptr
    )
  }

  override protected def doDeriveInputErrorPerBatch(error: CUDARealTensor)
  : Unit = {
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  error.desc,  error.data.ptr,
      _RealTensorNativeReal.one,  filter.desc, filter.data.ptr,
      _RealTensorNativeReal.zero, error.desc,  error.data.ptr
    )
  }

}

object ImmediateFilter_CUDA_Baseline_Description
  extends ModuleVariant_CUDA_Description[MultiplyFilterBuilder] {

  override def build(builder:        MultiplyFilterBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : MultiplyFilter_CUDA_Baseline = new MultiplyFilter_CUDA_Baseline(
    builder, hints, seed, weightsBuilder
  )

}

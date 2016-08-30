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

final class AddBias_CUDA_Baseline(override val builder:        AddBiasBuilder,
                                  override val inputHints:     BuildHints,
                                  override val seed:           InstanceSeed,
                                  override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends AddBias_CUDA {

  private var _ones
  : CUDARealTensor = _

  private lazy val ones
  : CUDARealTensor = {
    if (_ones == null) {
      _ones = CUDARealTensor.fill(device, biasLayout, Real.one)
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
    output += bias
  }

  override protected def doPredictPerUnit(output: CUDARealTensor)
  : Unit = {
    // input:  RRR GGG BBB | RRR GGG BBB
    // filter: RRR GGG BBB
    _CUDNN.addTensor(
      device,
      _RealTensorNativeReal.one, bias.desc,   bias.data.ptr,
      _RealTensorNativeReal.one, output.desc, output.data.ptr
    )
  }

  override protected def doPredictPerChannel(output: CUDARealTensor)
  : Unit = {
    // input:  RRR GGG BBB | RRR GGG BBB
    // filter: RGB
    _CUDNN.addTensor(
      device,
      _RealTensorNativeReal.one, bias.desc,   bias.data.ptr,
      _RealTensorNativeReal.one, output.desc, output.data.ptr
    )
    /*
    scope match {
      case OperationScope.Value =>
        CUDNN.addTensor(
          device,
          //CUDNN_ADD_FULL_TENSOR,
          RealPtr.one, bias.descPtr,   bias.dataPtr,
          RealPtr.one, output.descPtr, output.dataPtr
        )

      case OperationScope.Channel =>
        CUDNN.addTensor(
          device,
          //CUDNN_ADD_SAME_C,
          RealPtr.one, bias.descPtr,   bias.dataPtr,
          RealPtr.one, output.descPtr, output.dataPtr
        )

      case OperationScope.Sample =>
        CUDNN.addTensor(
          device,
          //CUDNN_ADD_FEATURE_MAP,
          RealPtr.one, bias.descPtr,   bias.dataPtr,
          RealPtr.one, output.descPtr, output.dataPtr
        )

      case OperationScope.Batch =>
        // TODO: Fix this!
        output += bias.get(0)

      case _ =>
        throw new MatchError(scope)
    }
    */
  }

  override protected def doPredictPerSample(output: CUDARealTensor)
  : Unit = {
    // input:  RRR GGG BBB | RRR GGG BBB
    // filter: R           | R
    _CUDNN.addTensor(
      device,
      _RealTensorNativeReal.one, bias.desc,   bias.data.ptr,
      _RealTensorNativeReal.one, output.desc, output.data.ptr
    )
  }

  override protected def doPredictPerBatch(output: CUDARealTensor)
  : Unit = {
    // input:  RRR GGG BBB | RRR GGG BBB
    // filter: R
    _CUDNN.addTensor(
      device,
      _RealTensorNativeReal.one, bias.desc,   bias.data.ptr,
      _RealTensorNativeReal.one, output.desc, output.data.ptr
    )
  }

  override protected def doPredictInvPerValue(input: CUDARealTensor)
  : Unit = input -= bias

  override protected def doPredictInvPerUnit(input: CUDARealTensor)
  : Unit = {
    _CUDNN.addTensor(
      device,
      _RealTensorNativeReal.minusOne, bias.desc,  bias.data.ptr,
      _RealTensorNativeReal.one,      input.desc, input.data.ptr
    )
  }

  override protected def doPredictInvPerChannel(input: CUDARealTensor)
  : Unit = {
    _CUDNN.addTensor(
      device,
      _RealTensorNativeReal.minusOne, bias.desc,  bias.data.ptr,
      _RealTensorNativeReal.one,      input.desc, input.data.ptr
    )
    /*
    scope match {
      case OperationScope.Channel =>
        CUDNN.addTensor(
          device,
          //CUDNN_ADD_SAME_C,
          RealPtr.minusOne, bias.descPtr,  bias.dataPtr,
          RealPtr.one,      input.descPtr, input.dataPtr
        )

      case OperationScope.Sample =>
        CUDNN.addTensor(
          device,
          //CUDNN_ADD_FEATURE_MAP,
          RealPtr.minusOne, bias.descPtr,  bias.dataPtr,
          RealPtr.one,      input.descPtr, input.dataPtr
        )

      case OperationScope.Batch =>
        // TODO: Fix this!
        input += -bias.get(0)

      case _ =>
        throw new MatchError(scope)
    }
    */
  }

  override protected def doPredictInvPerSample(input: CUDARealTensor)
  : Unit = {
    _CUDNN.addTensor(
      device,
      _RealTensorNativeReal.minusOne, bias.desc,  bias.data.ptr,
      _RealTensorNativeReal.one,      input.desc, input.data.ptr
    )
  }

  override protected def doPredictInvPerBatch(input: CUDARealTensor)
  : Unit = {
    _CUDNN.addTensor(
      device,
      _RealTensorNativeReal.minusOne, bias.desc,  bias.data.ptr,
      _RealTensorNativeReal.one,      input.desc, input.data.ptr
    )
  }



  // ---------------------------------------------------------------------------
  //    Backward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveWeightGradientsPerValue(error: CUDARealTensor,
                                                         sink:  CUDARealTensor)
  : Unit = {
    // error: RRR GGG BBB | RRR GGG BBB
    // sink:  RRR GGG BBB | RRR GGG BBB
    sink += error
  }

  override protected def doDeriveWeightGradientsPerUnit(error: CUDARealTensor,
                                                        sink:  CUDARealTensor)
  : Unit = {
    // error: RRR GGG BBB | RRR GGG BBB
    // sink:  RRR GGG BBB
    // TODO: Find something nicer than gemv!
    val noRows = error.layout.size.noValues
    val noCols = error.layout.noSamples
    _CUBLAS.gemv(
      device,
      _RealTensorNativeReal.one,
      error.data.ptr, noRows, noRows, noCols, aTrans = false,
      ones.data.ptr,  1,      noRows,
      _RealTensorNativeReal.one,
      sink.data.ptr,  1,      noRows
    )
  }

  override protected def doDeriveWeightGradientsPerChannel(error: CUDARealTensor,
                                                           sink:  CUDARealTensor)
  : Unit = {
    // error: RRR GGG BBB | RRR GGG BBB
    // sink:  RGB
    _CUDNN.convolutionBackwardBias(
      device,
      _RealTensorNativeReal.one, error.desc, error.data.ptr,
      _RealTensorNativeReal.one, sink.desc,  sink.data.ptr
    )
  }

  override protected def doDeriveWeightGradientsPerSample(error: CUDARealTensor,
                                                          sink:  CUDARealTensor)
  : Unit = {
    // error: RRR GGG BBB | RRR GGG BBB
    // sink:  R           | R
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

  override protected def doDeriveWeightGradientsPerBatch(error: CUDARealTensor,
                                                         sink:  CUDARealTensor)
  : Unit = {
    // error: RRR GGG BBB | RRR GGG BBB
    // sink:  R
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

}


object AddBias_CUDA_Baseline_Description
  extends ModuleVariant_CUDA_Description[AddBiasBuilder] {

  override def build(builder:       AddBiasBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : AddBias_CUDA_Baseline = new AddBias_CUDA_Baseline(
    builder, hints, seed, weightsBuilder
  )

}

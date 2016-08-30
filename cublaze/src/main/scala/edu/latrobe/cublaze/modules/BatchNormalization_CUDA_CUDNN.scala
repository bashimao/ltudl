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

final class BatchNormalization_CUDA_CUDNN(override val builder:        BatchNormalizationBuilder,
                                          override val inputHints:     BuildHints,
                                          override val seed:           InstanceSeed,
                                          override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends BatchNormalization_CUDA {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredictForTraining(output:       CUDARealTensor,
                                              learningRate: Real)
  : PredictContext = {
    val (meanCache, varianceCache) = _CUDNN.batchNormalizationForwardTraining(
      device,
      CUDNN_BATCHNORM_SPATIAL,
      _RealTensorNativeReal.one,
      output.desc, output.data.ptr,
      _RealTensorNativeReal.zero,
      output.desc, output.data.ptr,
      filter.desc,
      runningMean.data,
      runningVariance.data,
      learningRate,
      filter.data,
      bias.data,
      epsilon
    )

    BatchNormalization_CUDA_CUDNN_Context(meanCache, varianceCache)
  }

  override protected def doPredictForInference(output: CUDARealTensor)
  : Unit = {
    _CUDNN.batchNormalizationForwardInference(
      device,
      CUDNN_BATCHNORM_SPATIAL,
      _RealTensorNativeReal.one,
      output.desc, output.data.ptr,
      _RealTensorNativeReal.zero,
      output.desc, output.data.ptr,
      runningMean.desc,
      runningMean.data,
      runningVariance.data,
      filter.data,
      bias.data,
      epsilon
    )
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveWeightGradients(input:      CUDARealTensor,
                                                 context:    PredictContext,
                                                 error:      CUDARealTensor,
                                                 filterSink: CUDARealTensor,
                                                 biasSink:   CUDARealTensor)
  : Unit = context match {
    case BatchNormalization_CUDA_CUDNN_Context(meanCache, varianceCache) =>
      _CUDNN.batchNormalizationBackward(
        device,
        CUDNN_BATCHNORM_SPATIAL,
        _RealTensorNativeReal.one,
        input.desc, input.data.ptr,
        error.desc, error.data.ptr,
        _RealTensorNativeReal.zero,
        error.desc, error.data.ptr,
        filter.desc,
        _RealTensorNativeReal.one,
        filter.data,
        _RealTensorNativeReal.one,
        filterSink.data,
        biasSink.data,
        epsilon,
        meanCache,
        varianceCache
      )

    case _ =>
      throw new MatchError(context)
  }

}

final case class BatchNormalization_CUDA_CUDNN_Context(meanCache:     _RealTensorDeviceBuffer,
                                                       varianceCache: _RealTensorDeviceBuffer)
  extends PredictContext {

  override protected def doClose()
  : Unit = {
    varianceCache.close()
    meanCache.close()
    super.doClose()
  }

}

object BatchNormalization_CUDA_CUDNN_Description
  extends ModuleVariant_CUDA_Description[BatchNormalizationBuilder] {

  override def build(builder:        BatchNormalizationBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : BatchNormalization_CUDA_CUDNN = new BatchNormalization_CUDA_CUDNN(
    builder, hints, seed, weightsBuilder
  )

}


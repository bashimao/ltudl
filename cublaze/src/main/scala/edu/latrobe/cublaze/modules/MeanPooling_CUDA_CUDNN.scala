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
import org.bytedeco.javacpp.cudnn._

final class MeanPooling_CUDA_CUDNN(override val builder:        MeanPoolingBuilder,
                                   override val inputHints:     BuildHints,
                                   override val seed:           InstanceSeed,
                                   override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends MeanPooling_CUDA
    with PoolingLayer_CUDA_CUDNN[MeanPoolingBuilder] {

  override val poolingMode: Int = {
    if (includePadding) {
      CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
    }
    else {
      CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
    }
  }

}

object MeanPooling_CUDA_CUDNN_Description
  extends ModuleVariant_CUDA_Description[MeanPoolingBuilder] {

  override def build(builder:        MeanPoolingBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : MeanPooling_CUDA_CUDNN = new MeanPooling_CUDA_CUDNN(
    builder, hints, seed, weightsBuilder
  )

}

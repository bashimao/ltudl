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

final class Dropout_CUDA_CUDNN(override val builder:        DropoutBuilder,
                               override val inputHints:     BuildHints,
                               override val seed:           InstanceSeed,
                               override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Dropout_CUDA {

  val stateSize
  : Long = _CUDNN.dropoutGetStateSize(device)


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredictForTraining(output: CUDARealTensor,
                                              rng:    PseudoRNG)
  : PredictContext = {
    // Get next random seed.
    val seed = rng.nextLong()

    // Initialize dropout structure and state.
    val dropDesc = _DropoutStruct()
    val state    = _ByteDeviceBuffer(device, stateSize)
    _CUDNN.setDropoutDescriptor(
      dropDesc,
      device,
      probability,
      state,
      seed
    )

    // FPROP
    val cache = _CUDNN.dropoutForward(
      device,
      dropDesc,
      output.desc, output.data.ptr,
      output.desc, output.data.ptr
    )

    // CUDNN supports only the new algorithm. We have to multiply with (1-p)
    // to get the right values for the traditional algorithm.
    if (useOriginalAlgorithm) {
      output *= probabilityInv
    }

    Dropout_CUDA_CUDNN_Context(dropDesc, state, cache)
  }

  override protected def doPredictForInference(output: CUDARealTensor)
  : Unit = {
    if (useOriginalAlgorithm) {
      output *= probabilityInv
    }
    else {
      // Do nothing.
    }
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(context: PredictContext,
                                            error:   CUDARealTensor)
  : Unit = context match {
    case Dropout_CUDA_CUDNN_Context(dropDesc, state, cache) =>
      val errPtr = error.data.ptr
      _CUDNN.dropoutBackward(
        device,
        dropDesc,
        error.desc, errPtr,
        error.desc, errPtr,
        cache
      )
    case _ =>
      throw new MatchError(context)
  }

  /*
  override protected def doDeriveInputErrorForInference(error: CUDARealTensor)
  : Unit = {
    if (useOriginalAlgorithm) {
      error *= probabilityInv
    }
    else {
      // Do nothing.
    }
  }
  */

}

final case class Dropout_CUDA_CUDNN_Context(dropDesc: _DropoutStruct,
                                            state:    _ByteDeviceBuffer,
                                            cache:    _ByteDeviceBuffer)
  extends PredictContext
    with AutoClosing {

  override protected def doClose()
  : Unit = {
    cache.close()
    state.close()
    dropDesc.close()
    super.doClose()
  }

}

object Dropout_CUDA_CUDNN_Description
  extends ModuleVariant_CUDA_Description[DropoutBuilder] {

  override def build(builder:        DropoutBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Dropout_CUDA_CUDNN = new Dropout_CUDA_CUDNN(
    builder, hints, seed, weightsBuilder
  )

}
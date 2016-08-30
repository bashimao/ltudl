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

abstract class Dropout_CUDA
  extends Dropout
    with MapLayer_CUDA[DropoutBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredictForTraining(inPlaceAllowed: Boolean,
                                                    input:          Tensor,
                                                    rng:            PseudoRNG)
  : (CUDARealTensor, PredictContext) = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToCUDARealTensor(device)
      }
      else {
        input.toCUDARealTensor(device)
      }
    }
    val ctx = doPredictForTraining(out, rng)
    (out, ctx)
  }

  protected def doPredictForTraining(output: CUDARealTensor,
                                     rng:    PseudoRNG)
  : PredictContext

  final override protected def doPredictForInference(inPlaceAllowed: Boolean,
                                                     input:          Tensor)
  : CUDARealTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToCUDARealTensor(device)
      }
      else {
        input.toCUDARealTensor(device)
      }
    }
    doPredictForInference(out)
    out
  }

  protected def doPredictForInference(output: CUDARealTensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveInputError(context: PredictContext,
                                                  error:   Tensor)
  : CUDARealTensor = {
    val newErr = error.asOrToCUDARealTensor(device)
    doDeriveInputError(context, newErr)
    newErr
  }

  protected def doDeriveInputError(context: PredictContext,
                                   error:   CUDARealTensor)
  : Unit

}

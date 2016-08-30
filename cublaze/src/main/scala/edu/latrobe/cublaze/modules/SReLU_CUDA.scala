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
import edu.latrobe.blaze.modules._
import edu.latrobe.cublaze._

abstract class SReLU_CUDA
  extends SReLU
    with MapLayer_CUDA[SReLUBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(inPlaceAllowed: Boolean, input: Tensor)
  : CUDARealTensor = {
    // Move data to CUDA device & allocate buffer for results.
    val out = {
      if (inPlaceAllowed) {
        input.asOrToCUDARealTensor(device)
      }
      else {
        input.toCUDARealTensor(device)
      }
    }
    doPredict(out)
    out
  }

  protected def doPredict(out: CUDARealTensor): Unit

  final protected def doPredictInv(output: Tensor)
  : CUDARealTensor = {
    val inp = output.toCUDARealTensor(device)
    doPredictInv(inp)
    inp
  }

  protected def doPredictInv(inp: CUDARealTensor): Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveInputError(inputOrOutput: Tensor,
                                                  error:         Tensor)
  : CUDARealTensor = {
    // Move data to CUDA device.
    val inp = inputOrOutput.asOrToCUDARealTensor(device)
    val err = error.asOrToCUDARealTensor(device)

    doDeriveInputError(inp, err)

    // Free any temporary memory.
    if (inp ne inputOrOutput) {
      inp.close()
    }
    err
  }

  protected def doDeriveInputError(inp: CUDARealTensor,
                                   err: CUDARealTensor)
  : Unit

}

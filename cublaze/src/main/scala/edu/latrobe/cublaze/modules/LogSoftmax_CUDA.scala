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

abstract class LogSoftmax_CUDA
  extends LogSoftmax
    with MapLayer_CUDA[LogSoftmaxBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(inPlaceAllowed: Boolean, input: Tensor)
  : CUDARealTensor = {
    // Allocate memory and upload data to device.
    val inp = input.asOrToCUDARealTensor(device)
    val out = inp.createSibling()

    // Perform fprop.
    doPredict(inp, out)

    // Free temporary memory on compute device.
    if (inp ne input) {
      inp.close()
    }
    out
  }

  protected def doPredict(inp: CUDARealTensor,
                          out: CUDARealTensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveInputError(output: Tensor, error: Tensor)
  : CUDARealTensor = {
    // Allocate memory and upload data to device (if necessary).
    val out = output.asOrToCUDARealTensor(device)
    val err = error.asOrToCUDARealTensor(device)

    // Perform backprop.
    doDeriveInputError(out, err)

    // Free memory if necessary.
    if (out ne output) {
      out.close()
    }
    err
  }

  protected def doDeriveInputError(out: CUDARealTensor,
                                   err: CUDARealTensor)
  : Unit

}

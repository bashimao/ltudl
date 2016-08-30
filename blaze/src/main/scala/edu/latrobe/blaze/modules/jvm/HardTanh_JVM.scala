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

package edu.latrobe.blaze.modules.jvm

import edu.latrobe._
import edu.latrobe.blaze.modules.{HardTanh, HardTanhBuilder}

abstract class HardTanh_JVM
  extends HardTanh
    with MapLayer_JVM[HardTanhBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(inPlaceAllowed: Boolean, input: Tensor)
  : Tensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredict(out)
    out
  }

  protected def doPredict(output: RealArrayTensor): Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveInputError(inputOrOutput: Tensor,
                                                  error:         Tensor)
  : Tensor = {
    val inp = inputOrOutput.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor
    doDeriveInputError(inp, err)
    err
  }

  protected def doDeriveInputError(input: RealArrayTensor,
                                   error: RealArrayTensor)
  : Unit

}

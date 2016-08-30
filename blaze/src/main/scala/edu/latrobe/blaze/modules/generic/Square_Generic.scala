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

package edu.latrobe.blaze.modules.generic

import edu.latrobe._
import edu.latrobe.blaze.modules.{Square, SquareBuilder}

abstract class Square_Generic
  extends Square
    with MapLayer_Generic[SquareBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(input: Tensor)
  : Tensor = {
    val out = input.copy
    doPredictEx(out)
    out
  }

  protected def doPredictEx(output: Tensor)
  : Unit

  final override protected def doPredictInv(output: Tensor)
  : Tensor = {
    val inp = output.copy
    doPredictInvEx(inp)
    inp
  }

  protected def doPredictInvEx(input: Tensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveInputError(input: Tensor,
                                                  error: Tensor)
  : Tensor = {
    doDeriveInputErrorEx(input, error)
    error
  }

  protected def doDeriveInputErrorEx(input: Tensor,
                                     error: Tensor)
  : Unit

}

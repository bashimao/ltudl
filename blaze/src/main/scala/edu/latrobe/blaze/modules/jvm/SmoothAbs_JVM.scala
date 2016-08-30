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
import edu.latrobe.blaze.modules.{SmoothAbs, SmoothAbsBuilder}

abstract class SmoothAbs_JVM
  extends SmoothAbs
    with MapLayer_JVM[SmoothAbsBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(input: Tensor)
  : RealArrayTensor = {
    val out = input.toRealArrayTensor
    doPredict(out)
    out
  }

  protected def doPredict(output: RealArrayTensor): Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveInputError(input:  Tensor,
                                                  output: Tensor,
                                                  error:  Tensor)
  : RealArrayTensor = {
    val inp = input.asOrToRealArrayTensor
    val out = output.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor
    err.transform(inp, out,
      (err, inp, out) => err * inp / out
    )

    if (inp ne input) {
      inp.close()
    }
    if (out ne output) {
      out.close()
    }
    err
  }

}

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
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._

abstract class Unpool_JVM
  extends Unpool {

  final override lazy val outputPlatform
  : JVM.type = JVM


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(input: Tensor)
  : (RealArrayTensor, PredictContext) = {
    val inp       = input.asOrToRealArrayTensor
    val inpLayout = inp.layout
    val outLayout = inpLayout.derive(outputSize)
    val out       = RealArrayTensor.zeros(outLayout)

    doPredict(inp, out)

    // Deallocate temporaries.
    if (inp ne input) {
      inp.close()
    }
    (out, EmptyContext)
  }

  protected def doPredict(input:  RealArrayTensor,
                          output: RealArrayTensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveInputError(context: PredictContext,
                                                  error:   Tensor)
  : RealArrayTensor = {
    val oldErr       = error.asOrToRealArrayTensor
    val oldErrLayout = oldErr.layout
    val newErrLayout = oldErrLayout.derive(inputSizeHint)
    val newErr       = RealArrayTensor.zeros(newErrLayout)

    doDeriveInputError(oldErr, newErr)

    // Deallocate temporaries.
    if (oldErr ne error) {
      oldErr.close()
    }
    newErr
  }

  protected def doDeriveInputError(oldError: RealArrayTensor,
                                   newError: RealArrayTensor)
  : Unit

}

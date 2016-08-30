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

abstract class Softmax_JVM
  extends Softmax
    with MapLayer_JVM[SoftmaxBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(inPlaceAllowed: Boolean, input: Tensor)
  : RealArrayTensor = {
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
  final override protected def doDeriveInputError(output: Tensor,
                                                  error:  Tensor)
  : RealArrayTensor = {
    val out = output.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor

    doDeriveInputError(out, err)

    if (out ne output) {
      out.close()
    }
    err
  }

  protected def doDeriveInputError(output: RealArrayTensor,
                                   error:  RealArrayTensor)
  : Unit = {
    val out = output.values
    val err = error.values

    error.foreachSample((off, length) => {
      val sum = ArrayEx.dot(
        err, off, 1,
        out, off, 1,
        length
      )
      ArrayEx.transform(
        err, off, 1,
        out, off, 1,
        length
      )((err, out) => (err - sum) * out)
    })
  }

}

final case class SoftMax_JVM_Context(maxValues:    Array[Double],
                                     scaleFactors: Array[Double])
  extends PredictContext {
}

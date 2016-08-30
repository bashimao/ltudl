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
import edu.latrobe.blaze.modules._

abstract class MultiplyValues_JVM
  extends MultiplyValues
    with MapLayer_JVM[MultiplyValuesBuilder] {

  final protected val values
  : Array[Real] = builder.values


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredictPerValue(inPlaceAllowed: Boolean,
                                                 input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictPerValue(out)
    out
  }

  protected def doPredictPerValue(output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerUnit(inPlaceAllowed: Boolean,
                                                input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictPerUnit(out)
    out
  }

  protected def doPredictPerUnit(output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerChannel(inPlaceAllowed: Boolean,
                                                   input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictPerChannel(out)
    out
  }

  protected def doPredictPerChannel(output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerSample(inPlaceAllowed: Boolean,
                                                  input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictPerSample(out)
    out
  }

  protected def doPredictPerSample(output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerBatch(inPlaceAllowed: Boolean,
                                                 input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictPerBatch(out)
    out
  }

  protected def doPredictPerBatch(output: RealArrayTensor)
  : Unit

  final override protected def doPredictInvPerValue(output: Tensor)
  : RealArrayTensor = {
    val inp = output.toRealArrayTensor
    doPredictInvPerValue(inp)
    inp
  }

  protected def doPredictInvPerValue(input: RealArrayTensor)
  : Unit

  final override protected def doPredictInvPerUnit(output: Tensor)
  : RealArrayTensor = {
    val inp = output.toRealArrayTensor
    doPredictInvPerUnit(inp)
    inp
  }

  protected def doPredictInvPerUnit(input: RealArrayTensor)
  : Unit

  final override protected def doPredictInvPerChannel(output: Tensor)
  : RealArrayTensor = {
    val inp = output.toRealArrayTensor
    doPredictInvPerChannel(inp)
    inp
  }

  protected def doPredictInvPerChannel(input: RealArrayTensor)
  : Unit

  final override protected def doPredictInvPerSample(output: Tensor)
  : RealArrayTensor = {
    val inp = output.toRealArrayTensor
    doPredictInvPerSample(inp)
    inp
  }

  protected def doPredictInvPerSample(input: RealArrayTensor)
  : Unit

  final override protected def doPredictInvPerBatch(output: Tensor)
  : RealArrayTensor = {
    val inp = output.toRealArrayTensor
    doPredictInvPerBatch(inp)
    inp
  }

  protected def doPredictInvPerBatch(input: RealArrayTensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveInputErrorPerValue(error: Tensor)
  : RealArrayTensor = {
    val inp = error.toRealArrayTensor
    doDeriveInputErrorPerValue(inp)
    inp
  }

  protected def doDeriveInputErrorPerValue(error: RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorPerUnit(error: Tensor)
  : RealArrayTensor = {
    val inp = error.toRealArrayTensor
    doDeriveInputErrorPerUnit(inp)
    inp
  }

  protected def doDeriveInputErrorPerUnit(error: RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorPerChannel(error: Tensor)
  : RealArrayTensor = {
    val inp = error.toRealArrayTensor
    doDeriveInputErrorPerChannel(inp)
    inp
  }

  protected def doDeriveInputErrorPerChannel(error: RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorPerSample(error: Tensor)
  : RealArrayTensor = {
    val inp = error.toRealArrayTensor
    doDeriveInputErrorPerSample(inp)
    inp
  }

  protected def doDeriveInputErrorPerSample(error: RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorPerBatch(error: Tensor)
  : RealArrayTensor = {
    val inp = error.toRealArrayTensor
    doDeriveInputErrorPerBatch(inp)
    inp
  }

  protected def doDeriveInputErrorPerBatch(error: RealArrayTensor)
  : Unit

}

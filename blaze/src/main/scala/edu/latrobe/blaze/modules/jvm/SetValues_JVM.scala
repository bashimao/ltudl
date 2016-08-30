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

abstract class SetValues_JVM
  extends SetValues
    with MapLayer_JVM[SetValuesBuilder] {

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

  protected def doPredictPerValue(output: RealArrayTensor)
  : Unit

  protected def doPredictPerUnit(output: RealArrayTensor)
  : Unit

  protected def doPredictPerChannel(output: RealArrayTensor)
  : Unit

  protected def doPredictPerSample(output: RealArrayTensor)
  : Unit

  protected def doPredictPerBatch(output: RealArrayTensor)
  : Unit

}

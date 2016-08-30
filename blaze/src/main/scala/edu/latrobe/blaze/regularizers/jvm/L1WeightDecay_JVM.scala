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

package edu.latrobe.blaze.regularizers.jvm

import edu.latrobe._
import edu.latrobe.blaze.regularizers.L1WeightDecay

abstract class L1WeightDecay_JVM
  extends L1WeightDecay {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doEvaluate(weights: ValueTensor)
  : Real = {
    val w      = weights.asOrToRealArrayTensor
    val result = doEvaluate(w)
    if (w ne weights) {
      w.close()
    }
    result
  }

  protected def doEvaluate(weights: RealArrayTensor)
  : Real


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveWeightGradients(sink:        ValueTensor,
                                                       scaleFactor: Real,
                                                       weights:     ValueTensor)
  : Unit = {
    // Move to RAM.
    val w   = weights.asOrToRealArrayTensor
    val dst = sink.asOrToRealArrayTensor

    // Compute.
    doDeriveWeightGradients(
      dst,
      scaleFactor,
      w
    )

    // Cleanup.
    if (w ne weights) {
      w.close()
    }
    if (dst ne sink) {
      sink := dst
      dst.close()
    }
  }

  protected def doDeriveWeightGradients(sink:        RealArrayTensor,
                                        scaleFactor: Real,
                                        weights:     RealArrayTensor)
  : Unit

}

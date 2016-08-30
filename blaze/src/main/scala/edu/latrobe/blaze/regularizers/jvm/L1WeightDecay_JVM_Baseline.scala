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
import edu.latrobe.blaze._
import edu.latrobe.blaze.regularizers._

final class L1WeightDecay_JVM_Baseline(override val builder:      L1WeightDecayBuilder,
                                       override val platformHint: Option[Platform],
                                       override val seed:         InstanceSeed)
  extends L1WeightDecay_JVM {
  require(builder != null && platformHint != null && seed != null)

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doEvaluate(weights: RealArrayTensor)
  : Real = weights.l1Norm(DoubleEx(epsilon))


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveWeightGradients(sink:        RealArrayTensor,
                                                 scaleFactor: Real,
                                                 weights:     RealArrayTensor)
  : Unit = {
    require(scaleFactor == Real.one)
    if (epsilon == 0.0) {
      sink.transform(weights,
        (s, w) => s + (if (w < 0) -Real.one else Real.one)
      )
    }
    else {
      sink.transform(weights,
        (s, w) => Real(s + w / Math.sqrt(w * w + epsilon))
      )
    }
  }

}

object L1Regularizer_JVM_Baseline_Description
  extends RegularizerVariantDescription[L1WeightDecayBuilder] {

  override def build(builder:      L1WeightDecayBuilder,
                     platformHint: Option[Platform],
                     seed:         InstanceSeed)
  : L1WeightDecay_JVM_Baseline = new L1WeightDecay_JVM_Baseline(
    builder, platformHint, seed
  )

}

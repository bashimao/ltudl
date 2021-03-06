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

package edu.latrobe.blaze.regularizers.generic

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.regularizers._

final class L1WeightDecay_Generic_Baseline(override val builder:      L1WeightDecayBuilder,
                                           override val platformHint: Option[Platform],
                                           override val seed:         InstanceSeed)
  extends L1WeightDecay {
  require(builder != null && platformHint != null && seed != null)

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doEvaluate(weights: ValueTensor)
  : Real = weights.l1Norm(DoubleEx(epsilon))

  override protected def doDeriveWeightGradients(sink:        ValueTensor,
                                                 scaleFactor: Real,
                                                 weights:     ValueTensor)
  : Unit = {
    using(weights.copy)(tmp => {
      if (epsilon == Real.zero) {
        tmp.abs()
      }
      else {
        tmp.sqr()
        tmp += epsilon
        tmp.sqrt()
      }
      tmp.divideR(Real.one)
      sink.add(weights, tmp)
    })
  }

}

object L1WeightDecay_Generic_Baseline_Description
  extends RegularizerVariantDescription[L1WeightDecayBuilder] {

  final override def build(builder:      L1WeightDecayBuilder,
                           platformHint: Option[Platform],
                           seed:         InstanceSeed)
  : L1WeightDecay_Generic_Baseline = new L1WeightDecay_Generic_Baseline(
    builder, platformHint, seed
  )

}

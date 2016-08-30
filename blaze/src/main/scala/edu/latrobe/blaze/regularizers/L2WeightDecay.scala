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

package edu.latrobe.blaze.regularizers

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.regularizers.generic._

/**
  * Very simple static regularizer, if that is what you want:
  *
  *            1    2
  * J(w_a) = c - w_a
  *            2
  *
  *        ---
  *        \
  * J(w) = /   J(w_i)
  *        ---
  *         i
  *
  * d J(w_a)
  * -------- = c w_a
  *  d w_a
  *
  *   d J(w_a)
  * ----------- = 0
  * d w_b, a!=b
  *
  *            ---
  * D J(w_a)   \   d J(w_a)
  * -------- = /   -------- di = c w_a da
  *  D w_a     ---  d w_i
  *             i
  *
  */
abstract class L2WeightDecay
  extends WeightDecay[L2WeightDecayBuilder] {
}

final class L2WeightDecayBuilder
  extends WeightDecayBuilder[L2WeightDecayBuilder] {

  override def repr
  : L2WeightDecayBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[L2WeightDecayBuilder]

  override protected def doCopy()
  : L2WeightDecayBuilder = L2WeightDecayBuilder()


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  override def build(platformHint: Option[Platform],
                     seed:         InstanceSeed)
  : Regularizer = L2WeightDecayBuilder.lookupAndBuild(this, platformHint, seed)

}

object L2WeightDecayBuilder
  extends RegularizerVariantTable[L2WeightDecayBuilder] {

  register(64, L2WeightDecay_Generic_Baseline_Description)

  final def apply()
  : L2WeightDecayBuilder = new L2WeightDecayBuilder

  final def apply(scaleCoefficient: ParameterBuilder)
  : L2WeightDecayBuilder = apply().setScaleCoefficient(scaleCoefficient)

  final def apply(scaleCoefficient: ParameterBuilder,
                  baseScope:        NullBuffer)
  : L2WeightDecayBuilder = apply(scaleCoefficient).setBaseScope(baseScope)

}

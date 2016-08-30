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

/**
  * A regularizer that does nothing. Use as template for new regularizers.
  */
final class IdleRegularizer(override val builder:      IdleRegularizerBuilder,
                            override val platformHint: Option[Platform],
                            override val seed:         InstanceSeed)
  extends SimpleRegularizer[IdleRegularizerBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override def evaluate(phaseNo:   Long,
                        weights:   ValueTensorBuffer,
                        input:     Tensor,
                        reference: Tensor,
                        output:    Tensor)
  : Real = Real.zero


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override def deriveGradients(phaseNo:   Long,
                               weights:   ValueTensorBuffer,
                               input:     Tensor,
                               reference: Tensor,
                               output:    Tensor,
                               sink:      ValueTensorBuffer)
  : Unit = {}

}

final class IdleRegularizerBuilder
  extends SimpleRegularizerBuilder[IdleRegularizerBuilder] {

  override def repr
  : IdleRegularizerBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[IdleRegularizerBuilder]

  override protected def doCopy()
  : IdleRegularizerBuilder = IdleRegularizerBuilder()


  // ---------------------------------------------------------------------------
  //    Object building related.
  // ---------------------------------------------------------------------------
  override def build(platformHint: Option[Platform],
                     seed:         InstanceSeed)
  : IdleRegularizer = new IdleRegularizer(this, platformHint, seed)

}

object IdleRegularizerBuilder {

  final def apply()
  : IdleRegularizerBuilder = new IdleRegularizerBuilder

}

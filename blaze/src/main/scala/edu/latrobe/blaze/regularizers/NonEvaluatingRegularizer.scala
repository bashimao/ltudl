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
import scala.collection._

/**
  * Use this to prevent the evaluation of a regularizer during forward pass.
  */
final class NonEvaluatingRegularizer(override val builder:      NonEvaluatingRegularizerBuilder,
                                     override val platformHint: Option[Platform],
                                     override val seed:         InstanceSeed)
  extends ComplexRegularizer[NonEvaluatingRegularizerBuilder] {

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
  : Unit = {
    children.foreach(
      _.deriveGradients(
        phaseNo,
        weights,
        input,
        reference,
        output,
        sink
      )
    )
  }

}

final class NonEvaluatingRegularizerBuilder
  extends ComplexRegularizerBuilder[NonEvaluatingRegularizerBuilder] {

  override def repr
  : NonEvaluatingRegularizerBuilder = this

  override val children
  : mutable.Buffer[RegularizerBuilder] = mutable.Buffer.empty

  def +=(regularizer: RegularizerBuilder)
  : NonEvaluatingRegularizerBuilder = {
    children += regularizer
    repr
  }

  def ++=(regularizers: TraversableOnce[RegularizerBuilder])
  : NonEvaluatingRegularizerBuilder = {
    children ++= regularizers
    repr
  }

  override protected def doToString()
  : List[Any] = children.length :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[NonEvaluatingRegularizerBuilder]

  override protected def doCopy()
  : NonEvaluatingRegularizerBuilder = NonEvaluatingRegularizerBuilder()

  override def build(platformHint: Option[Platform],
                     seed:         InstanceSeed)
  : NonEvaluatingRegularizer = new NonEvaluatingRegularizer(
    this, platformHint, seed
  )

}

object NonEvaluatingRegularizerBuilder {

  final def apply()
  : NonEvaluatingRegularizerBuilder = new NonEvaluatingRegularizerBuilder

  final def apply(child0: RegularizerBuilder)
  : NonEvaluatingRegularizerBuilder = apply() += child0

  final def apply(child0: RegularizerBuilder,
                  childN: RegularizerBuilder*)
  : NonEvaluatingRegularizerBuilder = apply(child0) ++= childN

  final def apply(childN: TraversableOnce[RegularizerBuilder])
  : NonEvaluatingRegularizerBuilder = apply() ++= childN

}
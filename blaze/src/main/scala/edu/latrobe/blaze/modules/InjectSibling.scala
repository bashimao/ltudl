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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

final class InjectSibling(override val builder:             InjectSiblingBuilder,
                          override val inputHints:          BuildHints,
                          override val seed:                InstanceSeed,
                          override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends NonTrainableMapLayer[InjectSiblingBuilder]
    with NonPenalizing {

  override lazy val outputPlatform
  : Platform = inputHints.platform

  val zeroValues
  : Boolean = builder.zeroValues


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor)
  : (Tensor, PredictContext) = (input.createSibling(), EmptyContext)

  override protected def doPredictInv(output:  Tensor,
                                      context: PredictContext)
  : Tensor = output


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  override protected def doDeriveInputError(input:     Tensor,
                                            reference: Tensor,
                                            output:    Tensor,
                                            context:   PredictContext,
                                            error:     Tensor)
  : Tensor = {
    if (zeroValues) {
      error.clear()
    }
    else {
      logger.warn("Backpropagating though InjectSibling without initialization")
    }
    error
  }


}

final class InjectSiblingBuilder
  extends NonTrainableMapLayerBuilder[InjectSiblingBuilder] {

  override def repr
  : InjectSiblingBuilder = this

  var zeroValues
  : Boolean = false

  def setZeroValues(value: Boolean)
  : InjectSiblingBuilder = {
    zeroValues_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = zeroValues :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), zeroValues.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[CopyBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: InjectSiblingBuilder =>
      zeroValues == other.zeroValues
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: InjectSiblingBuilder =>
        other.zeroValues = zeroValues
      case _ =>
    }
  }

  override protected def doCopy()
  : InjectSiblingBuilder = InjectSiblingBuilder()


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = hints.platform

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : InjectSibling = new InjectSibling(this, hints, seed, weightsBuilder)

}

object InjectSiblingBuilder {

  final def apply()
  : InjectSiblingBuilder = new InjectSiblingBuilder

  final def apply(zeroValues: Boolean)
  : InjectSiblingBuilder = apply().setZeroValues(zeroValues)

}

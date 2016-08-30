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
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.blaze.TensorDependency._
import scala.util.hashing._

/**
  *                 { if x > 0  => x
  * predict: f(x) = {
  *                 { else      => alpha * (exp(x) - 1)
  *
  *
  *                 -1   { if x > 0  => x
  * predictInv: f(x)   = {
  *                      { else      => log(x / alpha + 1)
  *
  *                   { if x > 0  => 1
  * gradient: f'(x) = {
  *                   { else      => alpha * exp(x) = f(x) + alpha
  */
abstract class ELU
  extends NonTrainableMapLayer[ELUBuilder]
    with NonPenalizing {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredict(inPlaceAllowed, input)
    (out, EmptyContext)
  }

  protected def doPredict(inPlaceAllowed: Boolean,
                          input:          Tensor)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = doPredictInv(output)

  protected def doPredictInv(output: Tensor): Tensor


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.Required

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = doDeriveInputError(output, error)

  protected def doDeriveInputError(output: Tensor,
                                   error:  Tensor)
  : Tensor

}

final class ELUBuilder
  extends NonTrainableMapLayerBuilder[ELUBuilder] {

  override def repr
  : ELUBuilder = this

  var alpha
  : Real = 0.01f

  def setAlpha(value: Real)
  : ELUBuilder = {
    alpha_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"$alpha%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), alpha.hashCode())

  override def canEqual(that: Any): Boolean = that.isInstanceOf[ELUBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ELUBuilder =>
      alpha == other.alpha
    case _ =>
      false
  })

  override protected def doCopy()
  : ELUBuilder = ELUBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ELUBuilder =>
        other.alpha = alpha
      case _ =>
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = ELUBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = ELUBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object ELUBuilder
  extends ModuleVariantTable[ELUBuilder] {

  register(2, ELU_JVM_Baseline_Description)

  final def apply(): ELUBuilder = new ELUBuilder

  final def apply(alpha: Real): ELUBuilder = apply().setAlpha(alpha)

}

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
import edu.latrobe.blaze.modules.generic._
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.blaze.TensorDependency._
import scala.util.hashing._

/**
  *          { x > t => x
  * f(x_a) = {
  *          { else  => t
  *
  * d f(x_a)   { x > t => 1
  * -------- = {
  *  d x_a     { else  => 0
  *
  */
abstract class SReLU
  extends NonTrainableMapLayer[SReLUBuilder]
    with NonPenalizing{

  final val threshold = builder.threshold


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

  protected def doPredict(inPlaceAllowed: Boolean, input: Tensor)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.RequiresEither

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.RequiresEither

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = {
    if (input != null) {
      doDeriveInputError(input, error)
    }
    else {
      doDeriveInputError(output, error)
    }
  }

  protected def doDeriveInputError(inputOrOutput: Tensor, error: Tensor)
  : Tensor

}

final class SReLUBuilder
  extends NonTrainableMapLayerBuilder[SReLUBuilder] {

  override def repr
  : SReLUBuilder = this

  var threshold
  : Real = Real.zero

  def setThreshold(value: Real)
  : SReLUBuilder = {
    threshold = value
    this
  }

  override protected def doToString()
  : List[Any] = f"$threshold%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), threshold.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SReLUBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: SReLUBuilder =>
      threshold == other.threshold
    case _ =>
      false
  })

  override protected def doCopy()
  : SReLUBuilder = SReLUBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: SReLUBuilder =>
        other.threshold = threshold
      case _ =>
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = SReLUBuilder.outputPlatformFor(this, hints)

  // Lookup variant and create object.
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = SReLUBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object SReLUBuilder
  extends ModuleVariantTable[SReLUBuilder] {

  register( 2, SReLU_JVM_Baseline_Description)
  register(64, SReLU_Generic_Baseline_Description)

  final def apply()
  : SReLUBuilder = new SReLUBuilder

  final def apply(threshold: Real)
  : SReLUBuilder = apply().setThreshold(threshold)

}
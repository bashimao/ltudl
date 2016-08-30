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
import scala.util.hashing._

/**
  * Leaky Rectified linear function. (not properly differentiable)
  */
abstract class LeakyReLU
  extends NonTrainableMapLayer[LeakyReLUBuilder]
    with NonPenalizing {

  val alpha
  : Real = builder.alpha


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    val output = doPredict(inPlaceAllowed, input)
    (output, EmptyContext)
  }

  protected def doPredict(inPlaceAllowed: Boolean, input: Tensor)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = doPredictInv(output)

  protected def doPredictInv(output: Tensor)
  : Tensor


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

  protected def doDeriveInputError(inputOrOutput: Tensor, error: Tensor): Tensor

}

final class LeakyReLUBuilder
  extends NonTrainableMapLayerBuilder[LeakyReLUBuilder] {

  override def repr: LeakyReLUBuilder = this

  var alpha
  : Real = 0.01f

  def setAlpha(value: Real)
  : LeakyReLUBuilder = {
    alpha_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"$alpha%.4g" :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LeakyReLUBuilder]

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), alpha.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: LeakyReLUBuilder =>
      alpha == other.alpha
    case _ =>
      false
  })

  override protected def doCopy()
  : LeakyReLUBuilder = LeakyReLUBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: LeakyReLUBuilder =>
        other.alpha = alpha
      case _ =>
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = LeakyReLUBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = LeakyReLUBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object LeakyReLUBuilder
  extends ModuleVariantTable[LeakyReLUBuilder] {

  register(2, LeakyReLU_JVM_Baseline_Description)

  final def apply()
  : LeakyReLUBuilder = new LeakyReLUBuilder

  final def apply(alpha: Real)
  : LeakyReLUBuilder = apply().setAlpha(alpha)

}
/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2015 Matthias Langer (t3l@threelights.de)
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
  * Epsilon has two functions:
  *
  * 1. It can prevent from quantum effects around zero
  * 2. Depending on x, the epsilon can make this a smooth function.
  *
  *                    /-----
  * Predict: f(x) = \/ x + e
  *
  *              -1    2
  * Inverse: f(x)   = x  - e
  *
  *           d f     1
  * Gradient: --- = ------
  *           d x   2 f(x)
  *
  */
abstract class SquareRoot
  extends NonTrainableMapLayer[SquareRootBuilder]
    with NonPenalizing {

  final val epsilon
  : Real = builder.epsilon


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
  : Tensor = doPredictInv(output)

  protected def doPredictInv(output: Tensor)
  : Tensor


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

final class SquareRootBuilder
  extends NonTrainableMapLayerBuilder[SquareRootBuilder] {

  override def repr
  : SquareRootBuilder = this

  private var _epsilon
  : Real = Real.zero

  def epsilon
  : Real = _epsilon

  def epsilon_=(value: Real)
  : Unit = {
    require(value >= Real.zero)
    _epsilon = value
  }
  
  def setEpsilon(value: Real)
  : SquareRootBuilder = {
    epsilon_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"${_epsilon}%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _epsilon.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SquareRootBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: SquareRootBuilder =>
      _epsilon == other._epsilon
    case _ =>
      false
  })

  override protected def doCopy()
  : SquareRootBuilder = SquareRootBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: SquareRootBuilder =>
        other._epsilon = _epsilon
      case _ =>
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = SquareRootBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = SquareRootBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object SquareRootBuilder
  extends ModuleVariantTable[SquareRootBuilder] {

  register(2, SquareRoot_JVM_ApacheCommons_Description)
  register(4, SquareRoot_JVM_Baseline_Description)

  final def apply()
  : SquareRootBuilder = new SquareRootBuilder

  final def apply(epsilon: Real)
  : SquareRootBuilder = apply().setEpsilon(epsilon)

}
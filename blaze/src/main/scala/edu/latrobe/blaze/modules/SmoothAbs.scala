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
import edu.latrobe.blaze.modules.generic._
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.blaze.TensorDependency._

import scala.util.hashing._

/**
 * Differentiable approximation of the absolute function.
 */
abstract class SmoothAbs
  extends NonTrainableMapLayer[SmoothAbsBuilder]
    with NonPenalizing {

  val epsilon
  : Real = builder.epsilon


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredict(input)
    (out, EmptyContext)
  }

  protected def doPredict(input: Tensor)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.Required

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.Required

  /**
   *     s
   * ----------
   *    -------
   *   /  2
   * \/  s + e   === i.e. About either +1 or -1!
   */
  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = doDeriveInputError(input, output, error)

  protected def doDeriveInputError(input: Tensor, output: Tensor, error: Tensor)
  : Tensor

}

final class SmoothAbsBuilder
  extends NonTrainableMapLayerBuilder[SmoothAbsBuilder] {

  override def repr
  : SmoothAbsBuilder = this

  private var _epsilon
  : Real = 1.00000003e-5f

  def epsilon
  : Real = _epsilon

  def epsilon_=(value: Real)
  : Unit = {
    require(value >= Real.zero)
    _epsilon = value
  }

  def setEpsilon(value: Real)
  : SmoothAbsBuilder = {
    epsilon_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"${_epsilon}%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _epsilon.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SmoothAbsBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: SmoothAbsBuilder =>
      _epsilon == other._epsilon
    case _ =>
      false
  })

  override protected def doCopy()
  : SmoothAbsBuilder = SmoothAbsBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: SmoothAbsBuilder =>
        other._epsilon = _epsilon
      case _ =>
    }
  }

  override def outputPlatformFor(hints: BuildHints)
  : Platform = SmoothAbsBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = SmoothAbsBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object SmoothAbsBuilder
  extends ModuleVariantTable[SmoothAbsBuilder] {

  register( 2, SmoothAbs_JVM_Baseline_Description)
  register(64, SmoothAbs_Generic_Baseline_Description)

  final def apply()
  : SmoothAbsBuilder = new SmoothAbsBuilder

}

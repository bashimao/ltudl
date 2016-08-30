/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.blaze.TensorDependency._

import scala.util.hashing._

abstract class ZCAWhitening
  extends Whitening[ZCAWhiteningBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(mode:           Mode,
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

  override protected def doPredictInv(output:  Tensor,
                                      context: PredictContext)
  : Tensor =  throw new NotImplementedError


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = throw new NotImplementedError

}

final class ZCAWhiteningBuilder
  extends WhiteningBuilder[ZCAWhiteningBuilder] {

  override def repr
  : ZCAWhiteningBuilder = this

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
  : ZCAWhiteningBuilder = {
    epsilon_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"${_epsilon}%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _epsilon.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ZCAWhiteningBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ZCAWhiteningBuilder =>
      _epsilon == other._epsilon
    case _ =>
      false
  })

  override protected def doCopy()
  : ZCAWhiteningBuilder = ZCAWhiteningBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ZCAWhiteningBuilder =>
        other._epsilon = _epsilon
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = ZCAWhiteningBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = ZCAWhiteningBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )

}

object ZCAWhiteningBuilder
  extends ModuleVariantTable[ZCAWhiteningBuilder] {

  register(2, ZCAWhitening_JVM_Breeze_Description)

  final def apply()
  : ZCAWhiteningBuilder = new ZCAWhiteningBuilder

  final def apply(epsilon: Real)
  : ZCAWhiteningBuilder = apply().setEpsilon(epsilon)

}

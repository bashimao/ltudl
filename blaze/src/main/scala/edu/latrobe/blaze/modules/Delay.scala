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
import edu.latrobe.time.TimeSpan
import scala.util.hashing._

/**
  * A dummy layer to cause delay for testing "what-if computation time would
  * be slower?"
  */
final class Delay(override val builder:        DelayBuilder,
                  override val inputHints:     BuildHints,
                  override val seed:           InstanceSeed,
                  override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends NonTrainableMapLayer[DelayBuilder]
    with MapLayer_Generic[DelayBuilder]
    with NonPenalizing {

  val predictDelay
  : TimeSpan = builder.predictDelay

  val predictInvDelay
  : TimeSpan = builder.predictInvDelay

  val deriveInputErrorDelay
  : TimeSpan = builder.deriveInputErrorDelay



  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor)
  : (Tensor, PredictContext) = {
    Thread.sleep(predictDelay.getMillis)
    (input, EmptyContext)
  }

  override protected def doPredictInv(output:  Tensor,
                                      context: PredictContext)
  : Tensor = {
    Thread.sleep(predictInvDelay.getMillis)
    output
  }


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
    Thread.sleep(deriveInputErrorDelay.getMillis)
    error
  }

}

final class DelayBuilder
  extends NonTrainableMapLayerBuilder[DelayBuilder]
    with MapLayer_Generic_Builder[DelayBuilder] {

  override def repr
  : DelayBuilder = this

  private var _predictDelay
  : TimeSpan = TimeSpan.zero

  def predictDelay
  : TimeSpan = _predictDelay

  def predictDelay_=(value: TimeSpan)
  : Unit = {
    require(value != null)
    _predictDelay = value
  }

  def setPredictDelay(value: TimeSpan)
  : DelayBuilder = {
    predictDelay_=(value)
    this
  }

  private var _predictInvDelay
  : TimeSpan = TimeSpan.zero

  def predictInvDelay
  : TimeSpan = _predictInvDelay

  def predictInvDelay_=(value: TimeSpan)
  : Unit = {
    require(value != null)
    _predictInvDelay = value
  }

  def setPredictInvDelay(value: TimeSpan)
  : DelayBuilder = {
    predictInvDelay_=(value)
    this
  }

  private var _deriveInputErrorDelay
  : TimeSpan = TimeSpan.zero

  def deriveInputErrorDelay
  : TimeSpan = _deriveInputErrorDelay

  def deriveInputErrorDelay_=(value: TimeSpan)
  : Unit = {
    require(value != null)
    _deriveInputErrorDelay = value
  }

  def setDeriveInputErrorDelay(value: TimeSpan)
  : DelayBuilder = {
    deriveInputErrorDelay_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    _predictDelay :: _predictInvDelay :: _deriveInputErrorDelay :: super.doToString()
  }

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _predictDelay.hashCode())
    tmp = MurmurHash3.mix(tmp, _predictInvDelay.hashCode())
    tmp = MurmurHash3.mix(tmp, _deriveInputErrorDelay.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[DelayBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: DelayBuilder =>
      _predictDelay          == other._predictDelay    &&
      _predictInvDelay       == other._predictInvDelay &&
      _deriveInputErrorDelay == other._deriveInputErrorDelay
    case _ =>
      false
  })

  override protected def doCopy()
  : DelayBuilder = DelayBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: DelayBuilder =>
        other._predictDelay          = _predictDelay
        other._predictInvDelay       = _predictInvDelay
        other._deriveInputErrorDelay = _deriveInputErrorDelay
      case _ =>
    }
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = new Delay(this, hints, seed, weightsBuilder)

}

object DelayBuilder {

  final def apply()
  : DelayBuilder = new DelayBuilder

}

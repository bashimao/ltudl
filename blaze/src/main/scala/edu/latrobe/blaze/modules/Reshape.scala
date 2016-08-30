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
import edu.latrobe.sizes._
import scala.util.hashing._

final class Reshape(override val builder:        ReshapeBuilder,
                    override val inputHints:     BuildHints,
                    override val seed:           InstanceSeed,
                    override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Layer[ReshapeBuilder]
    with NonTrainableLayer[ReshapeBuilder]
    with NonPenalizing {

  val callback = builder.callback

  override val outputHints
  : BuildHints = builder.outputHintsFor(inputHints)


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor)
  : (Tensor, PredictContext) = {
    val inpSize = input.layout.size
    val outSize = callback(inpSize)
    (input.reshape(outSize), SizeContext(inpSize))
  }

  override protected def doPredictInv(output:  Tensor,
                                      context: PredictContext)
  : Tensor = context match {
    case SizeContext(size) =>
      output.reshape(size)
    case _ =>
      throw new MatchError(context)
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
  : Tensor = context match {
    case SizeContext(size) =>
      error.reshape(size)
    case _ =>
      throw new MatchError(context)
  }

}

final class ReshapeBuilder
  extends LayerBuilder[ReshapeBuilder]
    with NonTrainableLayerBuilder[ReshapeBuilder] {

  override def repr: ReshapeBuilder = this

  private var _callback: Size => Size = _

  def callback
  : Size => Size = _callback

  def callback_=(value: Size => Size)
  : Unit = {
    require(value != null)
    _callback = value
  }

  def setCallback(value: Size => Size)
  : ReshapeBuilder = {
    callback_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _callback :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _callback.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ReshapeBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ReshapeBuilder =>
      _callback == other._callback
    case _ =>
      false
  })

  override protected def doCopy()
  : ReshapeBuilder = ReshapeBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ReshapeBuilder =>
        other._callback = _callback
      case _ =>
    }
  }

  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  def outputSizeFor(sizeHint: Size)
  : Size = _callback(sizeHint)

  def outputLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = layoutHint.derive(outputSizeFor(layoutHint.size))

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = hints.derive(outputLayoutFor(hints.layout))

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Reshape = new Reshape(this, hints, seed, weightsBuilder)

}

object ReshapeBuilder {

  final def apply()
  : ReshapeBuilder = new ReshapeBuilder

  final def apply(outputSize: Size)
  : ReshapeBuilder = apply(inputSize => outputSize)

  final def apply(callback: Size => Size)
  : ReshapeBuilder = apply().setCallback(callback)

  final def collapseDimensions()
  : ReshapeBuilder = apply(x => Size1(1, x.noValues))

}

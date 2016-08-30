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
import edu.latrobe.blaze.TensorDependency._
import scala.util.hashing._

/**
  * This one will simply replace the input with another tensor. You should
  * provide descent cleanup code in order to make this not spilling memory when
  * using special tensors. Actually this is super tricky to use in some
  * situations. In-depth understanding of the tensor dependency engine is highly
  * recommended.
  */
@deprecated
final class InjectTensor(override val builder:        InjectTensorBuilder,
                         override val inputHints:     BuildHints,
                         override val seed:           InstanceSeed,
                         override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Layer[InjectTensorBuilder]
    with NonTrainableLayer[InjectTensorBuilder]
    with NonPenalizing {
  require(
    builder != null && inputHints != null && seed != null && weightBufferBuilder != null
  )

  private val hintsFn = builder.hintsFn

  private val injectFn = builder.injectFn

  override val outputHints: BuildHints = hintsFn(inputHints)


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor)
  : (Tensor, PredictContext) = (injectFn(input), EmptyContext)

  override protected def doPredictInv(output: Tensor, context: PredictContext)
  : Tensor = throw new UnsupportedOperationException


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
  : Tensor = throw new UnsupportedOperationException

}

@deprecated
final class InjectTensorBuilder
  extends LayerBuilder[InjectTensorBuilder]
    with NonTrainableLayerBuilder[InjectTensorBuilder] {

  override def repr
  : InjectTensorBuilder = this

  private var _hintsFn
  : BuildHints => BuildHints = _

  def hintsFn
  : BuildHints => BuildHints = _hintsFn

  def hintsFn_=(value: BuildHints => BuildHints)
  : Unit = {
    require(value != null)
    _hintsFn = value
  }

  def setHintsFn(value: BuildHints => BuildHints)
  : InjectTensorBuilder = {
    hintsFn_=(value)
    repr
  }

  private var _injectFn
  : Tensor => Tensor = _

  def injectFn
  : Tensor => Tensor = _injectFn

  def injectFn_=(value: Tensor => Tensor)
  : Unit = {
    require(value != null)
    _injectFn = value
  }

  def setInjectFn(value: Tensor => Tensor)
  : InjectTensorBuilder = {
    injectFn_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _injectFn :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _injectFn.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[InjectTensorBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: InjectTensorBuilder =>
      _injectFn == other._injectFn
    case _ =>
      false
  })

  override protected def doCopy()
  : InjectTensorBuilder = InjectTensorBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: InjectTensorBuilder =>
        other._injectFn = _injectFn
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = _hintsFn(hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : InjectTensor = new InjectTensor(this, hints, seed, weightsBuilder)

}

@deprecated
object InjectTensorBuilder {

  final def apply()
  : InjectTensorBuilder = new InjectTensorBuilder

  final def apply(hintsFn:  BuildHints => BuildHints,
                  injectFn: Tensor => Tensor)
  : InjectTensorBuilder = apply().setHintsFn(hintsFn).setInjectFn(injectFn)

}
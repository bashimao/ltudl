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
import edu.latrobe.blaze.TensorDependency._
import edu.latrobe.sizes._

import scala.util.hashing._

abstract class Unpool
  extends PoolingLayer[UnpoolBuilder] {

  /**
    * Implement as lazy val.
    */
  def outputPlatform
  : IndependentPlatform

  final val outputSize
  : Size = builder.outputSize

  final val outputLayoutHint
  : IndependentTensorLayout = inputLayoutHint.derive(outputSize)

  final override val outputHints
  : BuildHints = inputHints.derive(outputPlatform, outputLayoutHint)


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode: Mode, input: Tensor)
  : (Tensor, PredictContext) = {
    require(input.layout.size == inputSizeHint)
    doPredict(input)
  }

  protected def doPredict(input: Tensor): (Tensor, PredictContext)


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  final override protected def doDeriveInputError(input:     Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = {
    require(error.layout.size == outputSize)
    doDeriveInputError(context, error)
  }

  protected def doDeriveInputError(context: PredictContext,
                                   error:   Tensor)
  : Tensor

}

final class UnpoolBuilder
  extends PoolingLayerBuilder[UnpoolBuilder] {

  override def repr
  : UnpoolBuilder = this

  private var _outputSize
  : Size = Size1.one

  def outputSize
  : Size = _outputSize

  def outputSize_=(value: Size)
  : Unit = {
    require(value != null)
    _outputSize = value
  }

  def setOutputSize(value: Size)
  : UnpoolBuilder = {
    outputSize_=(value)
    this
  }

  def inputSize
  : Size = kernel.outputSizeFor(_outputSize, _outputSize.noChannels)

  override protected def doToString()
  : List[Any] = _outputSize :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _outputSize.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[UnpoolBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: UnpoolBuilder =>
      _outputSize == other._outputSize
    case _ =>
      false
  })

  override protected def doCopy()
  : UnpoolBuilder = UnpoolBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: UnpoolBuilder =>
        other._outputSize = _outputSize
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  def outputPlatformFor(hints: BuildHints)
  : Platform = UnpoolBuilder.outputPlatformFor(this, hints)

  def outputSizeFor(sizeHint: Size)
  : Size = {
    require(sizeHint == inputSize)
    _outputSize
  }

  def outputLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = layoutHint.derive(outputSizeFor(layoutHint.size))

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = {
    val platform = outputPlatformFor(hints)
    val layout   = outputLayoutFor(hints.layout)
    hints.derive(platform, layout)
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = UnpoolBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )

}

object UnpoolBuilder
  extends ModuleVariantTable[UnpoolBuilder] {

  register(2, Unpool_JVM_Baseline_Description)

  final def apply()
  : UnpoolBuilder = new UnpoolBuilder

  final def apply(kernel: Kernel)
  : UnpoolBuilder = apply().setKernel(kernel)

  final def apply(kernel: Kernel, outputSize: Size)
  : UnpoolBuilder = apply(kernel).setOutputSize(outputSize)

}

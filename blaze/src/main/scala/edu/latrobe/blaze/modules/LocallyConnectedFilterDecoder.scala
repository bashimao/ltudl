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
import edu.latrobe.sizes._
import scala.util.hashing._

abstract class LocallyConnectedFilterDecoder
  extends LocallyConnectedFilterLike[LocallyConnectedFilterDecoderBuilder] {

  final override lazy val outputSize
  : Size = builder.outputSize


  // ---------------------------------------------------------------------------
  //    Weights related.
  // ---------------------------------------------------------------------------
  final override def reset(initializer: Initializer)
  : Unit = {
    val inputFanSize  = kernel.noValues * outputSize.noChannels
    val outputFanSize = kernel.noValues * inputSizeHint.noChannels
    filterReference.foreach(
      initializer(this, _, filter, inputFanSize, outputFanSize)
    )
  }

  final override def extractWeightsFor(neuronNo: Int)
  : Array[Real] = filter.valuesMatrix(neuronNo, ::).inner.toArray

}

final class LocallyConnectedFilterDecoderBuilder
  extends LocallyConnectedFilterLikeBuilder[LocallyConnectedFilterDecoderBuilder] {

  override def repr: LocallyConnectedFilterDecoderBuilder = this

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
  : LocallyConnectedFilterDecoderBuilder = {
    outputSize_=(value)
    this
  }

  def inputSize: Size = kernel.outputSizeFor(_outputSize, noMaps)

  override protected def doToString()
  : List[Any] = _outputSize :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _outputSize.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LocallyConnectedFilterDecoderBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: LocallyConnectedFilterDecoderBuilder =>
      _outputSize == other._outputSize
    case _ =>
      false
  })

  override protected def doCopy()
  : LocallyConnectedFilterDecoderBuilder = LocallyConnectedFilterDecoderBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: LocallyConnectedFilterDecoderBuilder =>
        other._outputSize = _outputSize
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  def filterSizeFor(sizeHint: Size)
  : Size = {
    require(inputSize == sizeHint)
    kernel.inputSizeFor(_outputSize.noChannels)
  }

  override def filterLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = IndependentTensorLayout(
    filterSizeFor(layoutHint.size),
    layoutHint.size.noValues
  )


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  def outputLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = layoutHint.derive(_outputSize)

  override def outputHintsFor(hints: BuildHints): BuildHints = hints.derive(
    LocallyConnectedFilterDecoderBuilder.outputPlatformFor(this, hints),
    outputLayoutFor(hints.layout)
  )

  // TODO: Check for unsupported combinations of settings.
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = LocallyConnectedFilterDecoderBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )

  /**
    * Shorthand to derive a decoder that is compatible with this layer.
    *
    * @return A functional encoder.
    */
  def deriveCompatibleEncoder()
  : LocallyConnectedFilterBuilder = {
    val res = LocallyConnectedFilterBuilder(kernel, noMaps)
    res.setFilterReference(filterReference)
    res
  }

}

object LocallyConnectedFilterDecoderBuilder
  extends ModuleVariantTable[LocallyConnectedFilterDecoderBuilder] {

  register(2, LocallyConnectedFilterDecoder_JVM_Breeze_SparseMM_Description)
  register(4, LocallyConnectedFilterDecoder_JVM_Breeze_MM_Description)

  final def apply()
  : LocallyConnectedFilterDecoderBuilder = {
    new LocallyConnectedFilterDecoderBuilder
  }

  final def apply(kernel: Kernel, noMaps: Int, outputSize: Size)
  : LocallyConnectedFilterDecoderBuilder = apply().setKernel(
    kernel
  ).setNoMaps(noMaps).setOutputSize(outputSize)

}

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

abstract class ConvolutionFilterDecoder
  extends ConvolutionFilterLike[ConvolutionFilterDecoderBuilder] {

  final lazy val inputSize
  : Size = builder.inputSize

  final override lazy val outputSizeHint
  : Size = builder.outputSize


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  final override lazy val noNeurons
  : Long = noMaps


  // ---------------------------------------------------------------------------
  //    Weights related
  // ---------------------------------------------------------------------------
  final override def reset(initializer: Initializer)
  : Unit = {
    val inputFanSize  = kernel.noValues * outputSizeHint.noChannels
    val outputFanSize = kernel.noValues * inputSizeHint.noChannels
    filterReference.foreach(
      initializer(this, _, filter, inputFanSize, outputFanSize)
    )
  }

  final override def extractWeightsFor(neuronNo: Int)
  : Array[Real] = filter.valuesMatrix(neuronNo, ::).inner.toArray

  /*
  final override def regularizationScaleFactorFor(outputSize: Size)
  : Real = Real(outputSize.noValues) / Real(noMaps)
  */

}

final class ConvolutionFilterDecoderBuilder
  extends ConvolutionFilterLikeBuilder[ConvolutionFilterDecoderBuilder] {

  override def repr: ConvolutionFilterDecoderBuilder = this

  private var _outputSize
  : Size = Size1.one

  def outputSize: Size = _outputSize

  def outputSize_=(value: Size): Unit = {
    require(value != null)
    _outputSize = value
  }

  def setOutputSize(value: Size): ConvolutionFilterDecoderBuilder = {
    outputSize_=(value)
    this
  }

  def inputSize: Size = kernel.outputSizeFor(_outputSize, noMaps)

  override protected def doToString()
  : List[Any] = _outputSize :: super.doToString()


  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _outputSize.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ConvolutionFilterDecoderBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ConvolutionFilterDecoderBuilder =>
      _outputSize == other._outputSize
    case _ =>
      false
  })

  override protected def doCopy()
  : ConvolutionFilterDecoderBuilder = ConvolutionFilterDecoderBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ConvolutionFilterDecoderBuilder =>
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

  override def filterLayoutFor(layoutHints: TensorLayout)
  : IndependentTensorLayout = IndependentTensorLayout(
    filterSizeFor(layoutHints.size),
    noMaps
  )


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  def outputPlatformFor(hints: BuildHints)
  : Platform = ConvolutionFilterDecoderBuilder.outputPlatformFor(this, hints)

  def outputLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = {
    require(layoutHint.size == inputSize)
    layoutHint.derive(_outputSize)
  }

  override def outputHintsFor(hints: BuildHints): BuildHints = {
    val platform = outputPlatformFor(hints)
    val layout   = outputLayoutFor(hints.layout)
    hints.derive(platform, layout)
  }

  // TODO: Check for unsupported combinations of settings.
  // Lookup variant and create object.
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = ConvolutionFilterDecoderBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )

  /**
    * Shorthand to derive a decoder that is compatible with this layer.
    *
    * @return A functional encoder.
    */
  def deriveCompatibleEncoder()
  : ConvolutionFilterBuilder = {
    val res = ConvolutionFilterBuilder(kernel, noMaps)
    res.setFilterReference(filterReference)
    res
  }

}

object ConvolutionFilterDecoderBuilder
  extends ModuleVariantTable[ConvolutionFilterDecoderBuilder] {

  register(2, ConvolutionFilterDecoder_JVM_Breeze_SparseMM_Description)
  register(4, ConvolutionFilterDecoder_JVM_Breeze_MM_Description)

  final def apply()
  : ConvolutionFilterDecoderBuilder = new ConvolutionFilterDecoderBuilder

  final def apply(kernel: Kernel, noMaps: Int, outputSize: Size)
  : ConvolutionFilterDecoderBuilder = {
    apply().setKernel(kernel).setNoMaps(noMaps).setOutputSize(outputSize)
  }

}

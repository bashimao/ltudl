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

abstract class ConvolutionFilter
  extends ConvolutionFilterLike[ConvolutionFilterBuilder] {

  final override lazy val outputSizeHint
  : Size = kernel.outputSizeFor(inputSizeHint, noMaps)


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  final override lazy val noNeurons: Long = noMaps


  // ---------------------------------------------------------------------------
  //    Weights related
  final override def reset(initializer: Initializer)
  : Unit = {
    val inputFanSize  = kernel.noValues * inputSizeHint.noChannels
    val outputFanSize = kernel.noValues * outputSizeHint.noChannels
    filterReference.foreach(
      initializer(this, _, filter, inputFanSize, outputFanSize)
    )
  }

  final override def extractWeightsFor(neuronNo: Int)
  : Array[Real] = filter.valuesMatrix(::, neuronNo).toArray

  // ---------------------------------------------------------------------------
  //   Cost and gradient related.
  // ---------------------------------------------------------------------------
  /*
  final override def regularizationScaleFactorFor(outputSize: Size)
  : Real = Real(outputSize.noValues) / Real(noMaps)
  */

}

final class ConvolutionFilterBuilder
  extends ConvolutionFilterLikeBuilder[ConvolutionFilterBuilder] {

  override def repr
  : ConvolutionFilterBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ConvolutionFilterBuilder]

  override protected def doCopy()
  : ConvolutionFilterBuilder = ConvolutionFilterBuilder()


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  def filterSizeFor(sizeHint: Size)
  : Size = kernel.inputSizeFor(sizeHint.noChannels)

  override def filterLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = IndependentTensorLayout(
    filterSizeFor(layoutHint.size),
    noMaps
  )


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  def outputPlatformFor(hints: BuildHints)
  : Platform = ConvolutionFilterBuilder.outputPlatformFor(this, hints)

  def outputSizeFor(sizeHint: Size)
  : Size = kernel.outputSizeFor(sizeHint, noMaps)

  def outputLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = layoutHint.derive(outputSizeFor(layoutHint.size))

  override def outputHintsFor(hints: BuildHints): BuildHints = {
    val platform = outputPlatformFor(hints)
    val layout   = outputLayoutFor(hints.layout)
    hints.derive(platform, layout)
  }

  // Lookup variant and create object.
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = ConvolutionFilterBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )

  /**
   * @param outputSize The reconstruction size of the decoder. Should be
   *                   equivalent to the inputSizeHint given to this instance.
   */
  def deriveCompatibleDecoder(outputSize: Size)
  : ConvolutionFilterDecoderBuilder = {
    val res = ConvolutionFilterDecoderBuilder(kernel, noMaps, outputSize)
    res.setFilterReference(filterReference)
    res
  }

}

object ConvolutionFilterBuilder
  extends ModuleVariantTable[ConvolutionFilterBuilder] {

  register( 2, ConvolutionFilter_JVM_Breeze_SparseMM_Description)
  register( 4, ConvolutionFilter_JVM_Breeze_MM_Description)
  register( 8, ConvolutionFilter_JVM_BLAS_MM_Description)
  register(16, ConvolutionFilter_JVM_BLAS_ImplicitMM_Description)

  final def apply()
  : ConvolutionFilterBuilder = new ConvolutionFilterBuilder

  final def apply(kernel: Kernel, noMaps: Int)
  : ConvolutionFilterBuilder = apply().setKernel(kernel).setNoMaps(noMaps)

}

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

/**
 * Neurons with local connectivity inspired by nature. noMaps neurons observe
 * the same partition of the input space.
 *
 * Memory layout of weights interleaved:
 *  ______         ______
 * | abc |        | abc |
 * | i=0 |   =>   | i=1 |
 * ------         ------
 * <- j ->        <- j ->
 *
 * group    group
 *   0        1
 * -----    -----
 * i j m    i j m
 * 0.0.a    1.0.a
 * 0.1.a    1.1.a
 * 0.0.b    1.0.b
 * 0.1.b    1.1.b
 * 0.0.c    1.0.c
 * 0.1.c    1.1.c
 *
 */
abstract class LocallyConnectedFilter
  extends LocallyConnectedFilterLike[LocallyConnectedFilterBuilder] {

  final override lazy val outputSize
  : Size = kernel.outputSizeFor(inputSizeHint, noMaps)


  // ---------------------------------------------------------------------------
  //    Weights related.
  // ---------------------------------------------------------------------------
  final override def reset(initializer: Initializer)
  : Unit = {
    val inputFanSize  = kernel.noValues * inputSizeHint.noChannels
    val outputFanSize = kernel.noValues * outputSize.noChannels
    filterReference.foreach(
      initializer(this, _, filter, inputFanSize, outputFanSize)
    )
  }

  final override def extractWeightsFor(neuronNo: Int)
  : Array[Real] = filter.valuesMatrix(::, neuronNo).toArray

}

final class LocallyConnectedFilterBuilder
  extends LocallyConnectedFilterLikeBuilder[LocallyConnectedFilterBuilder] {

  override def repr
  : LocallyConnectedFilterBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LocallyConnectedFilterBuilder]

  override protected def doCopy()
  : LocallyConnectedFilterBuilder = LocallyConnectedFilterBuilder()


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  def filterSizeFor(sizeHint: Size)
  : Size = kernel.inputSizeFor(sizeHint.noChannels)

  override def filterLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = IndependentTensorLayout(
    filterSizeFor(layoutHint.size),
    outputSizeFor(layoutHint.size).noValues
  )


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  def outputPlatformFor(hints: BuildHints)
  : Platform = LocallyConnectedFilterBuilder.outputPlatformFor(this, hints)

  def outputSizeFor(sizeHint: Size)
  : Size = kernel.outputSizeFor(sizeHint, noMaps)

  def outputLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = layoutHint.derive(outputSizeFor(layoutHint.size))

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = {
    val platform = outputPlatformFor(hints)
    val layout   = outputLayoutFor(hints.layout)
    hints.derive(platform, layout)
  }

  // Lookup variant and create object.
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = LocallyConnectedFilterBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )

  /**
   * @param outputSize The reconstruction size of the decoder. Should be
   *                   equivalent to the inputSizeHint given to this instance.
   */
  def deriveCompatibleDecoder(outputSize: Size)
  : LocallyConnectedFilterDecoderBuilder = {
    val res = LocallyConnectedFilterDecoderBuilder(kernel, noMaps, outputSize)
    res.setFilterReference(filterReference)
    res
  }

}

object LocallyConnectedFilterBuilder
  extends ModuleVariantTable[LocallyConnectedFilterBuilder] {

  register(2, LocallyConnectedFilter_JVM_Breeze_SparseMM_Description)
  register(4, LocallyConnectedFilter_JVM_Breeze_MM_Description)

  final def apply()
  : LocallyConnectedFilterBuilder = new LocallyConnectedFilterBuilder

  final def apply(kernel: Kernel, noMaps: Int)
  : LocallyConnectedFilterBuilder = apply().setKernel(kernel).setNoMaps(noMaps)

}

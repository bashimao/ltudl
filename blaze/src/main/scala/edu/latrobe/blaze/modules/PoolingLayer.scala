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
import edu.latrobe.kernels._
import scala.util.hashing._

abstract class PoolingLayer[TBuilder <: PoolingLayerBuilder[_]]
  extends Layer[TBuilder]
    with NonTrainableLayer[TBuilder]
    with NonPenalizing {

  final val kernel
  : Kernel = builder.kernel


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = doPredict(mode, input)

  protected def doPredict(mode: Mode, input: Tensor)
  : (Tensor, PredictContext)

  override protected def doPredictInv(output: Tensor, context: PredictContext)
  : Tensor = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = doDeriveInputError(
    input,
    output,
    context,
    error
  )

  protected def doDeriveInputError(input:   Tensor,
                                   output:  Tensor,
                                   context: PredictContext,
                                   error:   Tensor)
  : Tensor

}

abstract class PoolingLayerBuilder[TThis <: PoolingLayerBuilder[_]]
  extends LayerBuilder[TThis]
    with NonTrainableLayerBuilder[TThis] {

  final private var _kernel
  : Kernel = Kernel1.one

  final def kernel
  : Kernel = _kernel

  final def kernel_=(value: Kernel)
  : Unit = {
    require(value != null)
    _kernel = value
  }

  final def setKernel(value: Kernel)
  : TThis = {
    kernel_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _kernel :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _kernel.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: PoolingLayerBuilder[TThis] =>
      _kernel == other._kernel
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: PoolingLayerBuilder[TThis] =>
        other._kernel = _kernel
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  final override def weightLayoutFor(hints:   BuildHints,
                                     builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

}

/**
 * Additions for pooling layers.
 */
abstract class PoolingLayerEx[TBuilder <: PoolingLayerExBuilder[_]]
  extends PoolingLayer[TBuilder] {

  /**
    * Implement as lazy val.
    */
  def outputPlatform
  : IndependentPlatform

  final val outputSizeHint
  : Size = kernel.outputSizeFor(inputSizeHint, inputSizeHint.noChannels)

  final val outputLayoutHint
  : IndependentTensorLayout = inputLayoutHint.derive(outputSizeHint)

  final override val outputHints
  : BuildHints = inputHints.derive(outputPlatform, outputLayoutHint)

}

abstract class PoolingLayerExBuilder[TThis <: PoolingLayerExBuilder[_]]
  extends PoolingLayerBuilder[TThis] {

  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  def outputPlatformFor(hints: BuildHints)
  : Platform

  final def outputSizeFor(sizeHint: Size)
  : Size = kernel.outputSizeFor(sizeHint, sizeHint.noChannels)

  final def outputLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = layoutHint.derive(outputSizeFor(layoutHint.size))

  final override def outputHintsFor(hints: BuildHints)
  : BuildHints = {
    val platform = outputPlatformFor(hints)
    val layout   = outputLayoutFor(hints.layout)
    hints.derive(platform, layout)
  }

}

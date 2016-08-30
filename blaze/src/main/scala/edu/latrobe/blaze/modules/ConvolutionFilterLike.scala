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
import edu.latrobe.io.graph.Vertex

import scala.util.hashing._

abstract class ConvolutionFilterLike[TBuilder <: ConvolutionFilterLikeBuilder[_]]
  extends Layer[TBuilder]
    with FilterLayerLike[TBuilder] {

  final lazy val kernel
  : Kernel = builder.kernel

  final lazy val noMaps
  : Int = builder.noMaps

  /**
    * Must override with lazy val.
    */
  def outputPlatform
  : IndependentPlatform

  /**
    * Must override with lazy val.
    */
  def outputSizeHint
  : Size

  final lazy val outputLayoutHint
  : IndependentTensorLayout = inputLayoutHint.derive(outputSizeHint)

  final override lazy val outputHints
  : BuildHints = inputHints.derive(outputPlatform, outputLayoutHint)


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    require(input.layout.size.noChannels == inputSizeHint.noChannels)
    doPredict(input)
  }

  protected def doPredict(input: Tensor): (Tensor, PredictContext)

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = throw new UnsupportedOperationException

}

abstract class ConvolutionFilterLikeBuilder[TThis <: ConvolutionFilterLikeBuilder[_]]
  extends LayerBuilder[TThis]
    with FilterLayerLikeBuilder[TThis] {

  final private var _kernel
  : Kernel = _

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

  final private var _noMaps
  : Int = 1

  final def noMaps
  : Int = _noMaps

  final def noMaps_=(value: Int): Unit = {
    require(value > 0)
    _noMaps = value
  }

  final def setNoMaps(value: Int): TThis = {
    noMaps_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = s"${_kernel} x ${_noMaps}" :: super.doToString()

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _kernel.hashCode())
    tmp = MurmurHash3.mix(tmp, _noMaps.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ConvolutionFilterLikeBuilder[_] =>
      _kernel == other._kernel &&
      _noMaps == other._noMaps
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ConvolutionFilterLikeBuilder[TThis] =>
        other._kernel = _kernel
        other._noMaps = _noMaps
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  final override def weightLayoutFor(hints:   BuildHints,
                                     builder: TensorLayoutBufferBuilder)
  : BuildHints = {
    if (filterReference.segmentNo == 0 || !builder.contains(filterReference)) {
      val layout = filterLayoutFor(hints.layout)
      builder.register(filterReference, layout)
    }
    outputHintsFor(hints)
  }

}

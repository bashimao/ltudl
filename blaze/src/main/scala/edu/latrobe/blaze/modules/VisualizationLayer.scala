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
import edu.latrobe.blaze.sinks._
import edu.latrobe.io.vega._
import scala.util.hashing._

abstract class VisualizationLayer[TBuilder <: VisualizationLayerBuilder[_], TChartType <: Chart]
  extends ReportingLayer[TBuilder] {

  final val frameWidth
  : Int = builder.frameWidth

  final val frameHeight
  : Int = builder.frameHeight

  final val sink
  : Sink = builder.sink.build(seed)

  protected def doInitializeChart()
  : TChartType

  final val chart
  : TChartType = doInitializeChart()

  if (frameWidth >= 0) {
    chart.width = frameWidth
  }

  if (frameHeight >= 0) {
    chart.height = frameHeight
  }

}

abstract class VisualizationLayerBuilder[TThis <: VisualizationLayerBuilder[_]]
  extends ReportingLayerBuilder[TThis] {

  final private var _frameWidth
  : Int = -1

  final def frameWidth
  : Int = _frameWidth

  final def frameWidth_=(value: Int)
  : Unit = {
    require(value >= -1)
    _frameWidth = value
  }

  final def setFrameWidth(value: Int)
  : TThis = {
    frameWidth_=(value)
    repr
  }

  final private var _frameHeight
  : Int = -1

  final def frameHeight
  : Int = _frameHeight

  final def frameHeight_=(value: Int)
  : Unit = {
    require(value >= -1)
    _frameHeight = value
  }

  final def setFrameHeight(value: Int)
  : TThis = {
    frameHeight_=(value)
    repr
  }

  final private var _sink
  : SinkBuilder = ShowoffSinkBuilder(defaultFrameTitle()).setFrameFormat("vega")

  final def sink
  : SinkBuilder = _sink

  final def sink_=(value: SinkBuilder)
  : Unit = {
    require(value != null)
    _sink = value
  }

  final def setSink(value: SinkBuilder)
  : TThis = {
    sink_=(value)
    repr
  }

  def defaultFrameTitle()
  : String

  override protected def doToString()
  : List[Any] = _frameWidth :: _frameHeight :: _sink :: super.doToString()

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _frameWidth.hashCode())
    tmp = MurmurHash3.mix(tmp, _frameHeight.hashCode())
    tmp = MurmurHash3.mix(tmp, _sink.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: VisualizationLayerBuilder[TThis] =>
      _frameWidth  == other._frameWidth  &&
      _frameHeight == other._frameHeight &&
      _sink        == other._sink
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: VisualizationLayerBuilder[TThis] =>
        other._frameWidth  = _frameWidth
        other._frameHeight = _frameHeight
        other._sink        = _sink.copy
      case _ =>
    }
  }

  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    sink.permuteSeeds(fn)
  }

}

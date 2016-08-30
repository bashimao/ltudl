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

package edu.latrobe.blaze.objectives.visual

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.objectives._
import edu.latrobe.io.vega._
import edu.latrobe.time._
import scala.collection._
import scala.util.hashing._

final class VisualizeCurves(override val builder: VisualizeCurvesBuilder,
                            override val seed:    InstanceSeed)
  extends ReportingObjective[VisualizeCurvesBuilder] {

  val frameWidth
  : Int = builder.width

  val frameHeight
  : Int = builder.height

  val xAxisType
  : XAxisType = builder.xAxisType

  val chart
  : ScatterPlot2D = ScatterPlot2D()

  if (frameWidth >= 0) {
    chart.width = frameWidth
  }

  if (frameHeight >= 0) {
    chart.height = frameHeight
  }

  // X-Axis
  chart.xAxisLabel = xAxisType.label

  // Y-Axis 0
  private var _yAxis0Curves
  : Array[Curve] = _

  private lazy val yAxis0Curves
  : Array[Curve] = {
    if (_yAxis0Curves == null) {
      _yAxis0Curves = builder.yAxis0Curves.map(builder => {
        // A little hacky...
        val tmp = builder.build(0, chart.nextColor, seed)
        chart.series += tmp.dataSeries
        tmp
      }).toArray
    }
    _yAxis0Curves
  }

  // Y-Axis 1
  private var _yAxis1Curves
  : Array[Curve] = _

  private lazy val yAxis1Curves
  : Array[Curve] = {
    if (_yAxis1Curves == null) {
      _yAxis1Curves = builder.yAxis1Curves.map(builder => {
        // A little hacky...
        val tmp = builder.build(1, chart.nextColor, seed)
        chart.series += tmp.dataSeries
        tmp
      }).toArray
    }
    _yAxis1Curves
  }

  override protected def doEvaluateEx(sink:                Sink,
                                      optimizer:           OptimizerLike,
                                      runBeginIterationNo: Long,
                                      runBeginTime:        Timestamp,
                                      runNoSamples:        Long,
                                      model:               Module,
                                      batch:               Batch,
                                      output:              Tensor,
                                      value:               Real)
  : Unit = {
    // Convert iteration number/or whatever to real.
    val xValue = xAxisType.valueFor(optimizer)

    // Update curves.
    ArrayEx.foreach(yAxis0Curves)(
      _.update(
        xValue,
        optimizer,
        runBeginIterationNo,
        runBeginTime,
        runNoSamples,
        model,
        batch,
        output,
        value
      )
    )
    ArrayEx.foreach(yAxis1Curves)(
      _.update(
        xValue,
        optimizer,
        runBeginIterationNo,
        runBeginTime,
        runNoSamples,
        model,
        batch,
        output,
        value
      )
    )

    // Publish to showoff server (or wherever sink points!).
    sink.write(chart)
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : ObjectiveState = VisualizeCurvesState(
    super.state,
    ArrayEx.map(
      yAxis0Curves
    )(c => ArrayEx.serialize(c.dataSeries)),
    ArrayEx.map(
      yAxis1Curves
    )(c => ArrayEx.serialize(c.dataSeries))
  )

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    require(_yAxis0Curves == null && _yAxis1Curves == null)
    state match {
      case state: VisualizeCurvesState =>
        // Restore the old curves.
        _yAxis0Curves = SeqEx.zip(
          builder.yAxis0Curves,
          state.yAxis0Curves
        )((builder, state) => {
          val dataSeries = ArrayEx.deserialize[DataSeries2D](
            state
          )
          builder.doBuild(dataSeries, seed)
        })

        // Restore the old curves.
        _yAxis1Curves = SeqEx.zip(
          builder.yAxis1Curves,
          state.yAxis1Curves
        )((builder, state) => {
          val dataSeries = ArrayEx.deserialize[DataSeries2D](
            state
          )
          builder.doBuild(dataSeries, seed)
        })

      case _ =>
        throw new MatchError(state)
    }
  }

}

final class VisualizeCurvesBuilder
  extends ReportingObjectiveBuilder[VisualizeCurvesBuilder] {

  override def repr
  : VisualizeCurvesBuilder = this

  private var _width
  : Int = -1

  def width
  : Int = _width

  def width_=(value: Int)
  : Unit = {
    require(value >= -1)
    _width = value
  }

  def setWidth(value: Int)
  : VisualizeCurvesBuilder = {
    width_=(value)
    repr
  }

  private var _height
  : Int = -1

  def height
  : Int = _height

  def height_=(value: Int)
  : Unit = {
    require(value >= -1)
    _height = value
  }

  def setHeight(value: Int)
  : VisualizeCurvesBuilder = {
    height_=(value)
    repr
  }

  private var _xAxisType
  : XAxisType = XAxisType.IterationNo

  def xAxisType
  : XAxisType = _xAxisType

  def xAxisType_=(value: XAxisType)
  : Unit = {
    require(value != null)
    _xAxisType = value
  }

  def setXAxisFormat(value: XAxisType)
  : VisualizeCurvesBuilder = {
    xAxisType_=(value)
    repr
  }

  val yAxis0Curves
  : mutable.Buffer[CurveBuilder] = mutable.Buffer.empty

  def +=(curve: CurveBuilder)
  : VisualizeCurvesBuilder = {
    yAxis0Curves += curve
    this
  }

  def ++=(curves: TraversableOnce[CurveBuilder])
  : VisualizeCurvesBuilder = {
    yAxis0Curves ++= curves
    this
  }

  val yAxis1Curves
  : mutable.Buffer[CurveBuilder] = mutable.Buffer.empty

  override protected def doToString()
  : List[Any] = {
    s"${_width} x ${_height}" :: _xAxisType :: yAxis0Curves.length :: yAxis1Curves.length :: super.doToString()
  }

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _width.hashCode())
    tmp = MurmurHash3.mix(tmp, _height.hashCode())
    tmp = MurmurHash3.mix(tmp, _xAxisType.hashCode())
    tmp = MurmurHash3.mix(tmp, yAxis0Curves.hashCode())
    tmp = MurmurHash3.mix(tmp, yAxis1Curves.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[VisualizeCurvesBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: VisualizeCurvesBuilder =>
      _width       == other._width       &&
      _height      == other._height      &&
      _xAxisType   == other._xAxisType   &&
      yAxis0Curves == other.yAxis0Curves &&
      yAxis1Curves == other.yAxis1Curves
    case _ =>
      false
  })

  override protected def doCopy()
  : VisualizeCurvesBuilder = VisualizeCurvesBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: VisualizeCurvesBuilder =>
        other._width  = _width
        other._height = _height
        other._xAxisType   = _xAxisType
        other.yAxis0Curves.clear()
        other.yAxis0Curves ++= yAxis0Curves.map(_.copy)
        other.yAxis1Curves.clear()
        other.yAxis1Curves ++= yAxis1Curves.map(_.copy)
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : VisualizeCurves = new VisualizeCurves(this, seed)

  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    yAxis0Curves.foreach(
      _.permuteSeeds(fn)
    )
    yAxis1Curves.foreach(
      _.permuteSeeds(fn)
    )
  }

}

object VisualizeCurvesBuilder {

  final def apply()
  : VisualizeCurvesBuilder = new VisualizeCurvesBuilder

}

final case class VisualizeCurvesState(override val parent: InstanceState,
                                      yAxis0Curves:        Array[Array[Byte]],
                                      yAxis1Curves:        Array[Array[Byte]])
  extends ObjectiveState {
}

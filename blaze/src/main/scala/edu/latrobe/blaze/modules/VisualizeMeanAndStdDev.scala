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
import edu.latrobe.io.vega._
import scala.collection._
import scala.util.hashing._

final class VisualizeMeanAndStdDev(override val builder:        VisualizeMeanAndStdDevBuilder,
                                   override val inputHints:     BuildHints,
                                   override val seed:           InstanceSeed,
                                   override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends VisualizationLayer[VisualizeMeanAndStdDevBuilder, ScatterPlot2D] {

  override protected def doInitializeChart()
  : ScatterPlot2D = {
    val chart = VisualizeMeanAndStdDev.chartCache.get(handle)
    if (chart.isDefined) {
      chart.get
    }
    else {
      val chart = ScatterPlot2D()
      val meanSeries = InfiniteWindow2D("Mean", chart.nextColor)
      meanSeries.addPoint(Real.zero, Real.nan)
      chart.series += meanSeries
      val stdDevSeries = InfiniteWindow2D("StdDev", chart.nextColor).setYAxisNo(1)
      stdDevSeries.addPoint(Real.zero, Real.nan)
      chart.series += stdDevSeries
      chart
    }
  }

  val meanSeries
  : DataSeries2D = chart.getSeries("Mean")

  val stdDevSeries
  : DataSeries2D = chart.getSeries("StdDev")

  val pointIndex
  : Int = meanSeries.noPoints

  // TODO: This will lead to problems if we restore the model. But probably not an issue right now because nobody uses this on Spark.
  meanSeries.addPoint(Real(pointIndex), Real.nan)

  stdDevSeries.addPoint(Real(pointIndex), Real.nan)

  val epsilon
  : Double = DoubleEx(builder.epsilon)

  protected def doUpdate(tensor: Tensor): Unit = {
    val mv = MeanAndVariance.derive(tensor.values)

    // Insert into showoff.
    meanSeries.replacePoint(pointIndex, Real(pointIndex), mv.mean)
    stdDevSeries.replacePoint(pointIndex, Real(pointIndex), mv.sampleStdDev(epsilon))
  }


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input: Tensor)
  : Unit = {
    doUpdate(input)

    // Redraw if this is the last contributor.
    if (pointIndex == meanSeries.noPoints - 1) {
      sink.write(chart)
    }
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(error: Tensor)
  : Unit = {
    doUpdate(error)

    // Redraw if this is the last contributor.
    if (pointIndex == 0) {
      sink.write(chart)
    }
  }

}

private object VisualizeMeanAndStdDev {

  final val chartCache
  : mutable.WeakHashMap[String, ScatterPlot2D] = mutable.WeakHashMap.empty

  def clearCache()
  : Unit = chartCache.clear()

}

final class VisualizeMeanAndStdDevBuilder
  extends VisualizationLayerBuilder[VisualizeMeanAndStdDevBuilder] {

  override def repr
  : VisualizeMeanAndStdDevBuilder = this

  private var _epsilon
  : Real = Real.zero

  def epsilon
  : Real = _epsilon

  def epsilon_=(value: Real)
  : Unit = {
    require(value >= Real.zero)
    _epsilon = value
  }

  def setEpsilon(value: Real)
  : VisualizeMeanAndStdDevBuilder = {
    epsilon_=(value)
    this
  }

  override def defaultFrameTitle()
  : String = s"$reportingPhase Mean & Standard Deviation - $handle"

  override protected def doToString()
  : List[Any] = f"${_epsilon}%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _epsilon.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[VisualizeMeanAndStdDevBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: VisualizeMeanAndStdDevBuilder =>
      _epsilon == other._epsilon
    case _ =>
      false
  })

  override protected def doCopy()
  : VisualizeMeanAndStdDevBuilder = VisualizeMeanAndStdDevBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: VisualizeMeanAndStdDevBuilder =>
        other._epsilon = _epsilon
      case _ =>
    }
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : VisualizeMeanAndStdDev = new VisualizeMeanAndStdDev(
    this, hints, seed, weightsBuilder
  )

}

object VisualizeMeanAndStdDevBuilder {

  final def apply()
  : VisualizeMeanAndStdDevBuilder = new VisualizeMeanAndStdDevBuilder

}

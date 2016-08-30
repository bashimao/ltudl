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

final class VisualizeHistogram(override val builder:        VisualizeHistogramBuilder,
                               override val inputHints:     BuildHints,
                               override val seed:           InstanceSeed,
                               override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends VisualizationLayer[VisualizeHistogramBuilder, BarChart2D] {

  val domain
  : TensorDomain = builder.domain

  lazy val noBins
  : Int = builder.noBins

  lazy val fixedRange
  : RealRange = builder.range

  def computeLimits(range: RealRange)
  : Array[Real] = {
    val step = range.length / (noBins - 1)
    val bins = new Array[Real](noBins + 1)
    bins(0) = range.min
    var i0 = 0
    while (i0 < noBins) {
      val i1 = i0 + 1
      bins(i1) = bins(i0) + step
      i0 = i1
    }
    bins
  }

  private lazy val fixedLimits
  : Array[Real] = {
    if (fixedRange.isInfinite) {
      null
    }
    else {
      computeLimits(fixedRange)
    }
  }

  val normalizeMeanAndVariance
  : Boolean = builder.normalizeMeanAndVariance

  override def doInitializeChart()
  : BarChart2D = {
    val chart = VisualizeHistogram.chartCache.get(handle)
    if (chart.isDefined) {
      chart.get
    }
    else {
      val chart = BarChart2D().setSpaceBetweenBars(1)
      if (fixedLimits == null) {
        chart.labelFormat = (".2g", "%")
      }
      else {
        chart.labelFormat = (".4g", "%")

        val ticks = List.newBuilder[Real]
        ticks += fixedLimits(0)
        // last
        if (noBins >= 2) {
          ticks += fixedLimits(noBins - 1)
        }
        // center
        if (noBins >= 3) {
          ticks += fixedLimits((noBins + 1) / 2 - 1)
        }
        // 25% and 75%
        if (noBins >= 5) {
          ticks += fixedLimits((noBins * 1 + 3) / 4 - 1)
          ticks += fixedLimits((noBins * 3 + 3) / 4 - 1)
        }
        chart.tickLabelsX = ticks.result()
      }
      chart
    }
  }

  val labelPrefix
  : String = builder.labelPrefix

  private def createSeries(labelPostfix: String, labelNo: Int)
  : FixedSizeWindow2D = {
    val s = FixedSizeWindow2D(
      noBins + 1,
      s"$labelPrefix $labelPostfix $labelNo",
      chart.nextColor
    )
    var i = 0
    while (i <= noBins) {
      s.addPoint(DataPoint2D.nan)
      i += 1
    }
    chart.series += s
    s
  }

  val series
  : Array[FixedSizeWindow2D] = domain match {
    case TensorDomain.Unit =>
      Array.tabulate(
        inputSizeHint.noValues
      )(createSeries("Unit", _))

    case TensorDomain.Channel =>
      Array.tabulate(
        inputSizeHint.noChannels
      )(createSeries("Channel", _))

    case TensorDomain.Sample =>
      Array.tabulate(
        inputLayoutHint.noSamples
      )(createSeries("Sample", _))

    case TensorDomain.Batch =>
      Array.tabulate(
        1
      )(createSeries("Batch", _))

    case _ =>
      throw new MatchError(domain)
  }

  def binNoFor(value: Real, limits: Array[Real]): Int = {
    var i = 0
    while (i < noBins) {
      if (value < limits(i)) {
        return i
      }
      i += 1
    }
    noBins
  }

  protected def doUpdate(tensor: Tensor)
  : Unit = {
    val ten = tensor.asOrToRealArrayTensor
    doUpdate(ten)

    // Cleanup
    if (ten ne tensor) {
      ten.close()
    }
  }

  protected def doUpdate(tensor: RealArrayTensor)
  : Unit = domain match {
    case TensorDomain.Unit =>
      require(tensor.layout.size == inputSizeHint)
      tensor.foreachUnit((off, stride, length) => {
        doUpdate(
          series(off),
          tensor.values, off, stride,
          length
        )
      })

    case TensorDomain.Channel =>
      require(tensor.layout.size.noChannels == inputSizeHint.noChannels)
      tensor.foreachChannel((off, stride, length) => {
        doUpdate(
          series(off),
          tensor.values, off, stride,
          length
        )
      })

    case TensorDomain.Sample =>
      require(tensor.layout.noSamples == inputLayoutHint.noSamples)
      tensor.foreachSamplePair((i, off, length) => {
        doUpdate(
          series(i),
          tensor.values, off, 1,
          length
        )
      })

    case TensorDomain.Batch =>
      tensor.foreachChunk(Int.MaxValue, (off, length) => {
        doUpdate(
          series(0),
          tensor.values, off, 1,
          length
        )
      })

    case _ =>
      throw new MatchError(domain)
  }

  protected def doUpdate(series: FixedSizeWindow2D,
                         input:  Array[Real], offset: Int, stride: Int,
                         length: Int)
  : Unit = {
    // Get values.
    val values = {
      if (normalizeMeanAndVariance) {
        // Compute mean and variance.
        val mv = MeanAndVariance()
        ArrayEx.foreach(
          input, offset, stride,
          length
        )(mv.update)

        // Normalize.
        val mu       = mv.mean
        val sigmaInv = Real.one / mv.sampleStdDev()
        ArrayEx.map(
          input, offset, stride,
          length
        )(_ * sigmaInv + mu)
      }
      else {
        ArrayEx.copy(
          input, offset, stride,
          length
        )
      }
    }

    val limits = {
      if (fixedLimits == null) {
        // Compute limits.
        val minMax = ArrayEx.minMax(
          input, offset, stride,
          length
        )
        computeLimits(minMax)
      }
      else {
        fixedLimits
      }
    }

    // Sort values into bins.
    val bins = new Array[Long](noBins + 1)
    ArrayEx.foreach(
      values
    )(value => {
      val binNo = binNoFor(value, limits)
      bins(binNo) += 1L
    })

    // Insert bins into vega.
    ArrayEx.foreachPair(
      limits,
      bins
    )((i, x, y) => series.replacePoint(i, x, y / Real(length)))
  }


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input: Tensor)
  : Unit = {
    doUpdate(input)

    // Redraw if this is the last contributor.
    if (chart.series.last eq series.last) {
      sink.write(chart)
    }
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(error: Tensor)
  : Unit = {
    doUpdate(error)

    // Update if this is the first contributor.
    if (chart.series.head eq series(0)) {
      sink.write(chart)
    }
  }

}

private object VisualizeHistogram {

  final val chartCache
  : mutable.WeakHashMap[String, BarChart2D] = mutable.WeakHashMap.empty

  def clearCache()
  : Unit = chartCache.clear()

}

final class VisualizeHistogramBuilder
  extends VisualizationLayerBuilder[VisualizeHistogramBuilder] {

  override def repr
  : VisualizeHistogramBuilder = this

  private var _domain
  : TensorDomain = TensorDomain.Batch

  def domain
  : TensorDomain = _domain

  def domain_=(value: TensorDomain)
  : Unit = {
    require(value != null)
    _domain = value
  }

  def setDomain(value: TensorDomain)
  : VisualizeHistogramBuilder = {
    domain_=(value)
    this
  }

  private var _noBins
  : Int = 25

  def noBins
  : Int = _noBins

  def noBins_=(value: Int)
  : Unit = {
    require(value > 0)
    _noBins = value
  }

  def setNoBins(value: Int)
  : VisualizeHistogramBuilder = {
    noBins_=(value)
    this
  }

  private var _range
  : RealRange = RealRange.infinite

  def range
  : RealRange = _range

  def range_=(value: RealRange)
  : Unit = {
    require(value != null)
    _range = value
  }

  def setRange(value: RealRange)
  : VisualizeHistogramBuilder = {
    range_=(value)
    this
  }

  def setRange(min: Real, max: Real)
  : VisualizeHistogramBuilder = setRange(RealRange(min, max))

  var normalizeMeanAndVariance
  : Boolean = false

  def setNormalizeMeanAndVariance(value: Boolean)
  : VisualizeHistogramBuilder = {
    normalizeMeanAndVariance_=(value)
    this
  }

  private var _labelPrefix
  : String = "???"

  def labelPrefix
  : String = _labelPrefix

  def labelPrefix_=(value: String)
  : Unit = {
    require(value != null)
    _labelPrefix = value
  }

  def setLabelPrefix(value: String)
  : VisualizeHistogramBuilder = {
    labelPrefix_=(value)
    this
  }

  override def defaultFrameTitle()
  : String = s"Distribution Histogram ($reportingPhase) - $handle"

  override protected def doToString()
  : List[Any] = {
    _domain :: _labelPrefix :: _noBins :: _range :: normalizeMeanAndVariance :: super.doToString()
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _domain.hashCode())
    tmp = MurmurHash3.mix(tmp, _noBins.hashCode())
    tmp = MurmurHash3.mix(tmp, _range.hashCode())
    tmp = MurmurHash3.mix(tmp, normalizeMeanAndVariance.hashCode())
    tmp = MurmurHash3.mix(tmp, _labelPrefix.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[VisualizeHistogramBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: VisualizeHistogramBuilder =>
      _domain                  == other._domain                  &&
      _noBins                  == other._noBins                  &&
      _range                   == other._range                   &&
      normalizeMeanAndVariance == other.normalizeMeanAndVariance &&
      _labelPrefix             == other._labelPrefix
    case _ =>
      false
  })

  override protected def doCopy()
  : VisualizeHistogramBuilder = VisualizeHistogramBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: VisualizeHistogramBuilder =>
        other._domain                  = _domain
        other._noBins                  = _noBins
        other._range                   = _range
        other.normalizeMeanAndVariance = normalizeMeanAndVariance
        other._labelPrefix             = _labelPrefix
      case _ =>
    }
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : VisualizeHistogram = new VisualizeHistogram(this, hints, seed, weightsBuilder)

}

object VisualizeHistogramBuilder {

  final def apply()
  : VisualizeHistogramBuilder = new VisualizeHistogramBuilder

  final def apply(labelPrefix: String)
  : VisualizeHistogramBuilder = apply().setLabelPrefix(labelPrefix)

  final def apply(labelPrefix: String, range: RealRange)
  : VisualizeHistogramBuilder = apply(labelPrefix).setRange(range)

  final def apply(labelPrefix: String, range: RealRange, noBins: Int)
  : VisualizeHistogramBuilder = apply(labelPrefix, range).setNoBins(noBins)

}

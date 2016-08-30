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

package edu.latrobe.blaze.parameters

import edu.latrobe._
import edu.latrobe.blaze._
import scala.collection._
import scala.util.hashing._

/**
  * Accumulates history, computes the trend and takes a step if trend is in the
  * trigger range.
  *
  * Linear trend (based on least squares):
  *
  * y = slope * x + intercept
  *
  *         ---                 ---       ---
  *         \                 1 \         \
  *         /   (x_i * y_i) - - /   x_i * /   y_i
  *         ---               m ---       ---
  *          i                   i         i
  * slope = -------------------------------------
  *                                      2
  *              ---          ( ---     )
  *              \      2   1 ( \       )
  *              /   x_i  - - ( /   x_i )
  *              ---        m ( ---     )
  *               i           (  i      )
  *
  *             ---               ---
  *             \                 \
  *             /   y_i - slope * /   x_i
  *             ---               ---
  *              i                 i
  * intercept = -------------------------
  *                         m
  *
  * However, our x's are integers (x_i = i). So we can simplify to:
  *
  *   m
  *  ---
  *  \       m * (m + 1)
  *  /   i = -----------
  *  ---          2
  *  i=1
  *
  *          m                 m       m
  *         ---               ---     ---
  *         \               1 \       \
  *         /   (i * y_i) - - /   i * /   y_i
  *         ---             m ---     ---
  *         i=1               i=1     i=1
  * slope = ---------------------------------
  *                                  2
  *               m         (  m    )
  *              ---        ( ---   )
  *              \    2   1 ( \     )
  *              /   i  - - ( /   i )
  *              ---      m ( ---   )
  *              i=1        ( i=1   )
  *
  *          m                       m
  *         ---                     ---
  *         \               m + 1   \
  *         /   (i * y_i) - ----- * /   y_i
  *         ---               2     ---
  *         i=1                     i=1
  *       = -------------------------------
  *              m
  *             ---                   2
  *             \    2       ( m + 1 )
  *             /   i  - m * ( ----- )
  *             ---          (   2   )
  *             i=1
  *
  *          m
  *         ---
  *         \   ( (     m + 1 )       )
  *         /   ( ( i - ----- ) * y_i )
  *         --- ( (       2   )       )
  *         i=1
  *       = ---------------------------
  *            m
  *           ---                   2
  *           \    2       ( m + 1 )
  *           /   i  - m * ( ----- )
  *           ---          (   2   )
  *           i=1
  *
  */
final class TrendBasedSteps(override val builder: TrendBasedStepsBuilder,
                            override val name:    String,
                            override val seed:    InstanceSeed)
  extends IndependentParameter[TrendBasedStepsBuilder] {

  private val history
  : RingBuffer[Real] = RingBuffer[Real](builder.historyLength)

  val triggerRange
  : RealRange = builder.triggerRange

  val delayAfterTrigger
  : Long = builder.delayAfterTrigger

  private val values
  : Array[Real] = builder.values.toArray

  private var valueNo
  : Int = 0

  private var lastTrigger
  : Long = -delayAfterTrigger

  override def get(phaseNo: Long)
  : Real = values(Math.min(valueNo, values.length - 1))

  override def update(phaseNo: Long, value: Real)
  : Unit = {
    // If trigger delay has passed.
    if (phaseNo >= lastTrigger + delayAfterTrigger) {
      // Fill up history.
      history += value

      // If history is full.
      if (history.isFull) {
        // Compute slope.
        val m          = history.length
        val mPlus1Div2 = (m + 1) * 0.5
        var a = 0.0
        var b = 0.0
        var i = 0
        while (i < m) {
          a += (i - mPlus1Div2) * history(i)
          b += i * i
          i += 1
        }
        val slope = Real(a / (b - m * mPlus1Div2 * mPlus1Div2))
        if (logger.isDebugEnabled) {
          logger.debug(f"Step#: $phaseNo%d -> Trend: $slope%.4g")
        }

        // Check trigger.
        if (triggerRange.contains(slope)) {
          valueNo += 1
          if (logger.isInfoEnabled) {
            logger.info(
              f"Iteration#: $phaseNo%d -> Trend: $slope%.4g is in $triggerRange; Value# is now $valueNo"
            )
          }

          // Reset history.
          history.clear()
          lastTrigger = phaseNo
        }
      }
    }
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state: TrendBasedStepsState = TrendBasedStepsState(
    super.state,
    history,
    valueNo,
    lastTrigger
  )

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: TrendBasedStepsState =>
        history.clear()
        history ++= state.history
        valueNo     = state.valueNo
        lastTrigger = state.lastTrigger
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class TrendBasedStepsBuilder
  extends IndependentParameterBuilder[TrendBasedStepsBuilder] {

  override def repr
  : TrendBasedStepsBuilder = this

  private var _historyLength
  : Int = 1000

  def historyLength
  : Int = _historyLength

  def historyLength_=(value: Int)
  : Unit = {
    require(value > 0)
    _historyLength = value
  }

  def setHistoryLength(value: Int)
  : TrendBasedStepsBuilder = {
    historyLength_=(value)
    this
  }

  private var _triggerRange
  : RealRange = RealRange(-0.005f, 0.005f)

  def triggerRange
  : RealRange = _triggerRange

  def triggerRange_=(value: RealRange)
  : Unit = {
    require(value != null)
    _triggerRange = value
  }

  def setTriggerRange(value: RealRange)
  : TrendBasedStepsBuilder = {
    triggerRange_=(value)
    this
  }

  private var _delayAfterTrigger
  : Long = 0L

  def delayAfterTrigger
  : Long = _delayAfterTrigger

  def delayAfterTrigger_=(value: Long)
  : Unit = {
    require(value >= 0L)
    _delayAfterTrigger = value
  }

  def setDelayAfterTrigger(value: Long)
  : TrendBasedStepsBuilder = {
    delayAfterTrigger_=(value)
    this
  }

  val values
  : mutable.Buffer[Real] = mutable.Buffer.empty

  override protected def doToString()
  : List[Any] = {
    _historyLength :: _triggerRange :: _delayAfterTrigger :: values.length :: super.doToString()
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _historyLength.hashCode())
    tmp = MurmurHash3.mix(tmp, _triggerRange.hashCode())
    tmp = MurmurHash3.mix(tmp, _delayAfterTrigger.hashCode())
    tmp = MurmurHash3.mix(tmp, values.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[TrendBasedStepsBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: TrendBasedStepsBuilder =>
      _historyLength     == other._historyLength &&
      _triggerRange      == other._triggerRange  &&
      _delayAfterTrigger == other._delayAfterTrigger  &&
      values             == other.values
    case _ =>
      false
  })

  override protected def doCopy()
  : TrendBasedStepsBuilder = TrendBasedStepsBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: TrendBasedStepsBuilder =>
        other._historyLength     = _historyLength
        other._triggerRange      = _triggerRange
        other._delayAfterTrigger = _delayAfterTrigger
        other.values.clear()
        other.values ++= values
      case _ =>
    }
  }

  override def build(name: String, seed: InstanceSeed)
  : TrendBasedSteps = new TrendBasedSteps(this, name, seed)

}

object TrendBasedStepsBuilder {

  final def apply()
  : TrendBasedStepsBuilder = new TrendBasedStepsBuilder

  final def apply(historyLength: Int)
  : TrendBasedStepsBuilder = apply().setHistoryLength(historyLength)

  final def apply(historyLength: Int,
                  triggerRange:  RealRange)
  : TrendBasedStepsBuilder = apply(
    historyLength
  ).setTriggerRange(triggerRange)

  final def apply(historyLength:     Int,
                  triggerRange:      RealRange,
                  delayAfterTrigger: Long)
  : TrendBasedStepsBuilder = apply(
    historyLength,
    triggerRange
  ).setDelayAfterTrigger(delayAfterTrigger)

}

final case class TrendBasedStepsState(override val parent: InstanceState,
                                      history:             RingBuffer[Real],
                                      valueNo:             Int,
                                      lastTrigger:         Long)
  extends InstanceState {
}

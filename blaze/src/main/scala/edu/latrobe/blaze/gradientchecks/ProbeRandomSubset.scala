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

package edu.latrobe.blaze.gradientchecks

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.time._
import scala.collection._
import scala.util.hashing._
import spire.implicits._

final class ProbeRandomSubset(override val builder: ProbeRandomSubsetBuilder,
                              override val seed:    InstanceSeed)
  extends NumericalGradientCheck[ProbeRandomSubsetBuilder] {

  val noTrialsMax
  : Long = builder._noTrialsMax

  override protected def doApply(weights:   ValueTensorBuffer,
                                 cost:      Real,
                                 gradients: ValueTensorBuffer,
                                 fnCost:    () => Real)
  : GradientDeviation = {
    val noTrials
    : Long = Math.min(noTrialsMax, weights.noValues)
    val invEpsilon2 = Real.one / (epsilon + epsilon)

    // Compute numerical gradients.
    val noParameters = weights.noValues
    var prevTime     = Timestamp.now()

    val numericalGradients = {
      val ng = SortedMap.newBuilder[(Int, Int, Int), ((Int, Int, Int), Real)]
      cfor(0L)(_ < noTrials, _ + 1L)(trailNo => {
        val linearParameterNo = rng.nextLong(noParameters)
        val parameterNo       = weights.indexOf(linearParameterNo)

        // Backup weight, pertube, compute cost and restore it.
        val weight = weights(parameterNo)
        weights.update(parameterNo, weight - epsilon)
        val cost0 = fnCost()
        weights.update(parameterNo, weight + epsilon)
        val cost1 = fnCost()
        weights.update(parameterNo, weight)

        // Compute gradient and save.
        if (dumpProgressInterval != null) {
          val now = Timestamp.now()
          if (TimeSpan(prevTime, now) >= dumpProgressInterval) {
            System.err.println(
              f"Gradients testing progress: $trailNo%d / $noTrials%d (${trailNo * 100.0 / noTrials}%.2f %%)"
            )
            prevTime = now
          }
        }

        // Add gradient to map.
        ng += Tuple2(parameterNo, (parameterNo, (cost1 - cost0) * invEpsilon2))
      })

      ng.result()
    }

    // Compare gradients.
    val differences = MapEx.mapValues(numericalGradients)(
      v => v._2 - gradients(v._1)
    )

    val norm0 = MapEx.foldLeftValues(0.0, differences)(
      (norm, diff) => norm + diff * diff
    )
    val norm1 = numericalGradients.foldLeft(0.0)((norm, kv) => {
      val g = gradients(kv._1)
      val diff = kv._2._2 + g
      norm + diff * diff
    })

    GradientDeviation(
      Real(Math.sqrt(norm0) / Math.sqrt(norm1)),
      MapEx.mapValues(
        differences
      )(Tuple2(_, 1L))
    )
  }

}

final class ProbeRandomSubsetBuilder
  extends NumericalGradientCheckBuilder[ProbeRandomSubsetBuilder] {

  override def repr: ProbeRandomSubsetBuilder = this

  var _noTrialsMax: Long = 1000L

  def noTrialsMax: Long = _noTrialsMax

  def noTrialsMax_=(value: Long): Unit = {
    require(value >= 0)
    _noTrialsMax = value
  }

  def setNoTrialsMax(value: Long): ProbeRandomSubsetBuilder = {
    noTrialsMax_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _noTrialsMax :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ProbeRandomSubsetBuilder]

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _noTrialsMax.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ProbeRandomSubsetBuilder =>
      _noTrialsMax == other._noTrialsMax
    case _ =>
      false
  })

  override protected def doCopy()
  : ProbeRandomSubsetBuilder = ProbeRandomSubsetBuilder()

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: ProbeRandomSubsetBuilder =>
        other._noTrialsMax = _noTrialsMax
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : ProbeRandomSubset = new ProbeRandomSubset(this, seed)

}

object ProbeRandomSubsetBuilder {

  final def apply(): ProbeRandomSubsetBuilder = new ProbeRandomSubsetBuilder

  final def apply(noTrialsMax: Long)
  : ProbeRandomSubsetBuilder = apply().setNoTrialsMax(noTrialsMax)

  final def apply(noTrialsMax:       Long,
                  reportingInterval: TimeSpan)
  : ProbeRandomSubsetBuilder = apply(
    noTrialsMax
  ).setReportingInterval(reportingInterval)

  /**
   * @param weightsGroupNo The buffer to test.
   * @param mode The mode to use to perform the check.
   * @param epsilon A small offset value to be used for computing the numerical
   *                gradient.
   * @return An object describing the gradient check operation.
   */
  final def apply(noTrialsMax:       Long,
                  reportingInterval: TimeSpan,
                  epsilon:           Real)
  : ProbeRandomSubsetBuilder = apply(
    noTrialsMax,
    reportingInterval
  ).setEpsilon(epsilon)

}

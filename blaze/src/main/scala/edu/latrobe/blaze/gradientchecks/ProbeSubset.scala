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

final class ProbeSubset(override val builder: ProbeSubsetBuilder,
                        override val seed:    InstanceSeed)
  extends NumericalGradientCheck[ProbeSubsetBuilder] {

  val parameters: Set[(Int, Int, Int)] = builder.parameters

  override protected def doApply(weights:   ValueTensorBuffer,
                                 cost:      Real,
                                 gradients: ValueTensorBuffer,
                                 fnCost:    () => Real)
  : GradientDeviation = {
    val invEpsilon2 = Real.one / (epsilon + epsilon)

    // Compute numerical gradients.
    val noParameters = weights.noValues
    var prevTime     = Timestamp.now()

    val numericalGradients = {
      val ng = SortedMap.newBuilder[(Int, Int, Int), ((Int, Int, Int), Real)]
      SeqEx.foreachPair(parameters)((i, parameterNo) => {

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
              f"Gradients testing progress: $i%d / ${parameters.size}%d (${i * 100.0 / parameters.size}%.2f %%)"
            )
            prevTime = now
          }
        }

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

final class ProbeSubsetBuilder
  extends NumericalGradientCheckBuilder[ProbeSubsetBuilder] {

  override def repr
  : ProbeSubsetBuilder = this

  val parameters
  : mutable.Set[(Int, Int, Int)] = mutable.Set.empty

  def +=(parameterNo: (Int, Int, Int))
  : ProbeSubsetBuilder = {
    parameters += parameterNo
    this
  }

  def ++=(parameters: TraversableOnce[(Int, Int, Int)])
  : ProbeSubsetBuilder = {
    this.parameters ++= parameters
    this
  }

  override protected def doToString()
  : List[Any] = parameters.size :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), parameters.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ProbeSubsetBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ProbeSubsetBuilder =>
      parameters == other.parameters
    case _ =>
      false
  })

  override protected def doCopy()
  : ProbeSubsetBuilder = ProbeSubsetBuilder()

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: ProbeSubsetBuilder =>
        other.parameters.clear()
        other.parameters ++= parameters
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : ProbeSubset = new ProbeSubset(this, seed)

}

object ProbeSubsetBuilder {

  final def apply(): ProbeSubsetBuilder = new ProbeSubsetBuilder

  final def apply(reportingInterval: TimeSpan)
  : ProbeSubsetBuilder = apply().setReportingInterval(reportingInterval)

  /**
   * @param weightsGroupNo The buffer to test.
   * @param probability The pobability of each index to be selected.
   * @param weightsIndices The indices to test.
   * @param mode The mode to use to perform the check.
   * @param epsilon A small offset value to be used for computing the numerical
   *                gradient.
   * @return An object describing the gradient check operation.
   */
  final def apply(reportingInterval: TimeSpan,
                  epsilon:           Real)
  : ProbeSubsetBuilder = apply(reportingInterval).setEpsilon(epsilon)

}
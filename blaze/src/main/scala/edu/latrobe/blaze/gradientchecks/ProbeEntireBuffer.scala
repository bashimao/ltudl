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

final class ProbeEntireBuffer(override val builder: ProbeEntireBufferBuilder,
                              override val seed:    InstanceSeed)
  extends NumericalGradientCheck[ProbeEntireBufferBuilder] {

  override protected def doApply(weights:   ValueTensorBuffer,
                                 cost:      Real,
                                 gradients: ValueTensorBuffer,
                                 fnCost:    () => Real)
  : GradientDeviation = {
    val invEpsilon2 = Real.one / (epsilon + epsilon)

    // Compute numerical gradients.
    val noParameters = weights.noValues
    var i            = 0
    var prevTime     = Timestamp.now()

    using(weights.allocateZeroedSibling())(numericalGradients => {
      numericalGradients.tabulate((g, s, p) => {
        // Backup weight, pertube, compute cost and restore it.
        val weight = weights(g, s, p)
        weights.update(g, s, p, weight - epsilon)
        val cost0 = fnCost()
        weights.update(g, s, p, weight + epsilon)
        val cost1 = fnCost()
        weights.update(g, s, p, weight)

        // Compute gradient and save it.
        if (dumpProgressInterval != null) {
          val now = Timestamp.now()
          if (TimeSpan(prevTime, now) >= dumpProgressInterval) {
            System.err.println(
              f"Gradients testing progress: $i%d / $noParameters%d (${i * 100.0 / noParameters}%.2f %%)"
            )
            prevTime = now
          }
        }

        i += 1
        (cost1 - cost0) * invEpsilon2
      })

      // Compare gradients.
      using(numericalGradients - gradients,
            numericalGradients + gradients
      )((diff0, diff1) => {
        //val norm0 = norm(diff0)
        //val norm1 = norm(diff1)
        val norm0 = Math.sqrt(diff0.dot(diff0))
        val norm1 = Math.sqrt(diff1.dot(diff1))

        // TODO: This wastes resources!
        GradientDeviation(
          Real(norm0 / Math.max(norm1, Real.minQuotient)),
          MapEx.mapValues(
            diff0.toPairMap
          )(Tuple2(_, 1L))
        )
      })
    })
  }

}

final class ProbeEntireBufferBuilder
  extends NumericalGradientCheckBuilder[ProbeEntireBufferBuilder] {

  override def repr
  : ProbeEntireBufferBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ProbeEntireBufferBuilder]

  override protected def doCopy()
  : ProbeEntireBufferBuilder = ProbeEntireBufferBuilder()

  override def build(seed: InstanceSeed)
  : ProbeEntireBuffer = new ProbeEntireBuffer(this, seed)

}

object ProbeEntireBufferBuilder {

  final def apply()
  : ProbeEntireBufferBuilder = new ProbeEntireBufferBuilder

  final def apply(reportingInterval: TimeSpan)
  : ProbeEntireBufferBuilder = apply().setReportingInterval(reportingInterval)

  // TODO: Fix this description!
  /**
   * This will test either a fraction or all gradients. Works best with
   * rand = null or rand = PseudoRNG.default.uniform.
   *
   * @param rand If =null, tests all gradients, otherwise uses this as the
   *             source for values to compare against threshold.
   * @param threshold If random value is below threshold, will do a gradient
   *                  test for the respective weight.
   * @return The first value is a relative measure of accuracy. If this value is
   *         very low, everything should be alright. The second value contains
   *         the finite differences between gradient derived from the model and
   *         an numerical approximation using the cost function at [-e, +e].
   */
  final def apply(reportingInterval: TimeSpan,
                  epsilon:           Real)
  : ProbeEntireBufferBuilder = apply(reportingInterval).setEpsilon(epsilon)

}


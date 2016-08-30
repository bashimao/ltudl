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

package edu.latrobe.blaze

import edu.latrobe.Equatable
import edu.latrobe.blaze._

import scala.util.hashing._

/**
  * Result of an optimizer run.
  *
  * @param exitCode An ID that represents the exit reason of the optimizer.
  * @param iterationNo Number of optimizer iterations.
  * @param runningNetCost Averaged computation result during optimizer run.
  *                       (Seen result!)
  * @param runningGrossCost Averaged computation result during optimizer run
  *                        (This subset of gross result that has actually been
  *                         considered for improving the weights!)
  *
  */
final class OptimizationResult(val exitCode:    OptimizerExitCode,
                               val iterationNo: Long,
                               val noSamples:   Long,
                               val noConverges: Int,
                               val noFailures:  Int)
  extends Equatable
    with Serializable {
  require(exitCode    != null)
  require(iterationNo >= 0L)
  require(noSamples   >= 0L)
  require(noConverges >= 0L)
  require(noFailures  >= 0L)

  override def toString
  : String = {
    s"ID: $exitCode, Iterations#: $iterationNo, NoSamples: $noSamples, Converge#: $noConverges, Failure#: $noFailures"
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, exitCode.hashCode())
    tmp = MurmurHash3.mix(tmp, iterationNo.hashCode())
    tmp = MurmurHash3.mix(tmp, noSamples.hashCode())
    tmp = MurmurHash3.mix(tmp, noConverges.hashCode())
    tmp = MurmurHash3.mix(tmp, noFailures.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[OptimizationResult]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: OptimizationResult =>
      exitCode    == other.exitCode    &&
      iterationNo == other.iterationNo &&
      noSamples   == other.noSamples   &&
      noConverges == other.noConverges &&
      noFailures  == other.noFailures

    case _ =>
      false
  })

  def +(other: OptimizationResult)
  : OptimizationResult = OptimizationResult(
    exitCode + other.exitCode,
    Math.max(iterationNo, other.iterationNo),
    noSamples + other.noSamples,
    noConverges + other.noConverges,
    noFailures + other.noFailures
  )

}

object OptimizationResult {

  final def apply(exitCode:    OptimizerExitCode,
                  iterationNo: Long,
                  noSamples:   Long,
                  noConverges: Int,
                  noFailures:  Int)
  : OptimizationResult = new OptimizationResult(
    exitCode,
    iterationNo,
    noSamples,
    noConverges,
    noFailures
  )

  final def derive(exitCode:    OptimizerExitCode,
                   iterationNo: Long,
                   noSamples:   Long)
  : OptimizationResult = apply(
    exitCode,
    iterationNo,
    noSamples,
    if (exitCode.indicatesConvergence) 1 else 0,
    if (exitCode.indicatesFailure) 1 else 0
  )

}
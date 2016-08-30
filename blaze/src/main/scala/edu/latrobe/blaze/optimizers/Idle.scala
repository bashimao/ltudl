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

package edu.latrobe.blaze.optimizers

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.optimizerexitcodes._
import edu.latrobe.time._
import java.util.UUID
import scala.collection._

/**
  * A dummy optimizer that satisfies all interface requirements but does... You
  * guessed it... nothing ... ;) except evaluating objectives!
  */
final class Idle(override val builder:   IdleBuilder,
                 override val model:     Module,
                 override val batchPool: BatchPool,
                 override val seed:      InstanceSeed)
  extends OptimizerEx[IdleBuilder] {

  override val buffers
  : List[ValueTensorBuffer] = super.buffers

  override val parameters
  : Map[UUID, Parameter] = super.parameters

  override protected def doRun(runBeginIterationNo: Long,
                               runBeginTime:        Timestamp)
  : OptimizationResult = {
    val clock = Stopwatch()

    // Improvement loop.
    while (_iterationNo < Long.MaxValue) {
      // Evaluate early objectives.
      doEvaluateEarlyObjectives(
        runBeginIterationNo,
        runBeginTime,
        0L
      ).foreach(exitCode => {
        return OptimizationResult.derive(exitCode, _iterationNo, 0L)
      })

      val t0 = clock.readAndResetAs("EvalEObj")

      // Fetch parameters.

      val t1 = clock.readAndResetAs("GetHyperP")

      // Evaluate objectives.
      doEvaluateObjectives(
        runBeginIterationNo,
        runBeginTime,
        0L,
        null, null, Real.zero
      ).foreach(exitCode => {
        return OptimizationResult.derive(exitCode, _iterationNo, 0L)
      })

      val t2 = clock.readAndResetAs("EvalObj")

      // Update value series and record series.
      doUpdateParameters(_iterationNo, Real.zero)

      val t3 = clock.readAndResetAs("UpdHyperP")

      if (logger.isDebugEnabled) {
        val tS = LabeledTimeSpan(
          "SUM", t0 + t1 + t2 + t3
        )
        logger.debug(
          s"$tS, $t0, $t1, $t2, $t3"
        )
      }

      // Update internal state.
      _iterationNo += 1L
    }

    OptimizationResult.derive(NoIterationsLimit(), _iterationNo, 0L)
  }

}

final class IdleBuilder
  extends OptimizerExBuilder[IdleBuilder] {

  override def repr
  : IdleBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[IdleBuilder]

  override protected def doCopy()
  : IdleBuilder = IdleBuilder()

  override def build(model:     Module,
                     batchPool: BatchPool,
                     seed:      InstanceSeed)
  : Idle = new Idle(this, model, batchPool, seed)

}

object IdleBuilder {

  final def apply()
  : IdleBuilder = new IdleBuilder

}

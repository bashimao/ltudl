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
  * Like idle, but computes gradients. Use this as template for new optimizers.
  */
final class Template(override val builder:   TemplateBuilder,
                     override val model:     Module,
                     override val batchPool: BatchPool,
                     override val seed:      InstanceSeed)
  extends OptimizerEx[TemplateBuilder] {

  override val buffers
  : List[ValueTensorBuffer] = super.buffers

  override val parameters
  : Map[UUID, Parameter] = super.parameters

  override protected def doRun(runBeginIterationNo: Long,
                               runBeginTime:        Timestamp)
  : OptimizationResult = {
    var noSamples = 0L
    using(
      weightBuffer.allocateSibling()
    )(gradients => {
      val clock = Stopwatch()

      // Improvement loop.
      while (iterationNo < Long.MaxValue) {
        // Evaluate early objectives.
        doEvaluateEarlyObjectives(
          runBeginIterationNo,
          runBeginTime,
          noSamples
        ).foreach(exitCode => {
          return OptimizationResult.derive(
            exitCode, iterationNo, noSamples
          )
        })

        val t0 = clock.readAndResetAs("EvalEObj")

        // Fetch parameters.
        val w  = determineCurrentScope()

        val t1 = clock.readAndResetAs("GetHyperP")

        using(batchPool.draw())(drawContext => {
          // Check for end of stream and compute gradients.
          if (drawContext.isEmpty) {
            return OptimizationResult.derive(
              NoMoreData(), iterationNo, noSamples
            )
          }
          val batch = drawContext.batch

          val t2 = clock.readAndResetAs("GetBatch")

          using(forwardProp(w, batch))(context => {

            val t3 = clock.readAndResetAs("FProp")

            // Evaluate objectives.
            doEvaluateObjectives(
              runBeginIterationNo,
              runBeginTime,
              noSamples,
              batch, context.output, context.value
            ).foreach(exitCode => {
              return OptimizationResult.derive(
                exitCode, iterationNo, noSamples
              )
            })

            val g = gradients.createIntersectionView(w)

            val t4 = clock.readAndResetAs("EvalObj")

            // Compute new gradients.
            doBackwardProp(w, context, g)

            val t5 = clock.readAndResetAs("BProp")

            // Update value series and record series.
            doUpdateParameters(_iterationNo, context.value)

            val t6 = clock.readAndResetAs("UpdHyperP")

            if (logger.isDebugEnabled) {
              val tS = LabeledTimeSpan(
                "SUM", t0 + t1 + t2 + t3 + t4 + t5 + t6
              )
              logger.debug(
                s"$tS, $t0, $t1, $t2, $t3, $t4, $t5, $t6"
              )
            }
          })

          // Update internal state.
          noSamples    += batch.noSamples
          _iterationNo += 1L
        })
      }
    })

    OptimizationResult.derive(NoIterationsLimit(), _iterationNo, noSamples)
  }

}

final class TemplateBuilder
  extends OptimizerExBuilder[TemplateBuilder] {

  override def repr
  : TemplateBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[TemplateBuilder]

  override protected def doCopy()
  : TemplateBuilder = TemplateBuilder()

  override def build(model:     Module,
                     batchPool: BatchPool,
                     seed:      InstanceSeed)
  : Template = new Template(this, model, batchPool, seed)

}

object TemplateBuilder {

  final def apply()
  : TemplateBuilder = new TemplateBuilder

}

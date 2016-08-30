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

package edu.latrobe.blaze.optimizers

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.optimizerexitcodes._
import edu.latrobe.time._
import java.util.UUID

import edu.latrobe.blaze.parameters.ConstantValueBuilder

import scala.collection._
import scala.util.hashing._

/**
 * Traditional SGD without extras.
 */
final class TraditionalSGD(override val builder:   TraditionalSGDBuilder,
                           override val model:     Module,
                           override val batchPool: BatchPool,
                           override val seed:      InstanceSeed)
  extends OptimizerEx[TraditionalSGDBuilder] {

  private val learningRate
  : Parameter = builder.learningRate.build("Learning Rate", seed)

  override val buffers
  : List[ValueTensorBuffer] = super.buffers

  override val parameters
  : Map[UUID, Parameter] = super.parameters + learningRate.toTuple

  override protected def doClose()
  : Unit = {
    learningRate.close()
    super.doClose()
  }

  override protected def doRun(runBeginIterationNo: Long,
                               runBeginTime:        Timestamp)
  : OptimizationResult = {
    val clock     = Stopwatch()
    var noSamples = 0L
    using(
      weightBuffer.allocateSibling()
    )(gradients => {
      // Improvement loop.
      while (_iterationNo < Long.MaxValue) {
        // Evaluate early objectives.
        doEvaluateEarlyObjectives(
          runBeginIterationNo,
          runBeginTime,
          noSamples
        ).foreach(exitCode => {
          return OptimizationResult.derive(
            exitCode, _iterationNo, noSamples
          )
        })

        val t0 = clock.readAndResetAs("EvalEObj")

        // Fetch parameters.
        val w  = determineCurrentScope()
        val lr = learningRate.get(_iterationNo, RealRange.zeroToInfinity)

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

            // Perform SGD step.
            w.add(g, -lr)

            val t6 = clock.readAndResetAs("UpdW")

            // Update value series and record series.
            doUpdateParameters(_iterationNo, context.value)

            val t7 = clock.readAndResetAs("UpdHyperP")

            if (logger.isDebugEnabled) {
              val tS = LabeledTimeSpan(
                "SUM", t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7
              )
              logger.debug(
                s"$tS, $t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7"
              )
            }
          })

          // Update internal state.
          noSamples    += batch.noSamples
          _iterationNo += 1L
        })
      }
    })

    OptimizationResult.derive(NoIterationsLimit(), iterationNo, noSamples)
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : TraditionalSGDState = TraditionalSGDState(
    super.state,
    learningRate.state
  )

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: TraditionalSGDState =>
        learningRate.restoreState(state.learningRate)
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class TraditionalSGDBuilder
  extends OptimizerExBuilder[TraditionalSGDBuilder] {

  override def repr
  : TraditionalSGDBuilder = this

  private var _learningRate
  : ParameterBuilder = ConstantValueBuilder(1e-3f)

  def learningRate
  : ParameterBuilder = _learningRate

  def learningRate_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _learningRate = value
  }

  def setLearningRate(value: ParameterBuilder)
  : TraditionalSGDBuilder = {
    learningRate_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _learningRate :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _learningRate.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[TraditionalSGDBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: TraditionalSGDBuilder =>
      _learningRate == other._learningRate
    case _ =>
      false
  })

  override protected def doCopy()
  : TraditionalSGDBuilder = TraditionalSGDBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: TraditionalSGDBuilder =>
        other._learningRate = _learningRate.copy
      case _ =>
    }
  }

  override def build(model:     Module,
                     batchPool: BatchPool,
                     seed:      InstanceSeed)
  : TraditionalSGD = new TraditionalSGD(this, model, batchPool, seed)


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _learningRate.permuteSeeds(fn)
  }

}

object TraditionalSGDBuilder {

  final def apply()
  : TraditionalSGDBuilder = new TraditionalSGDBuilder

  final def apply(learningRate: ParameterBuilder)
  : TraditionalSGDBuilder = apply().setLearningRate(learningRate)

}

final case class TraditionalSGDState(override val parent: OptimizerState,
                                     learningRate:        InstanceState)
  extends OptimizerState {
}

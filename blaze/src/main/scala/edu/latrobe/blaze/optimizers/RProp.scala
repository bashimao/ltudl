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
 * Train with individual weights with R-Prop.
 */
final class RProp(override val builder:   RPropBuilder,
                  override val model:     Module,
                  override val batchPool: BatchPool,
                  override val seed:      InstanceSeed)
  extends OptimizerEx[RPropBuilder] {

  protected val learningRate
  : Parameter = builder.learningRate.build("Learning Rate", seed)

  protected val rewardFactor
  : Parameter = builder.rewardFactor.build("Reward Factor", seed)

  protected val penaltyFactor
  : Parameter = builder.penaltyFactor.build("Penalty Factor", seed)

  protected val localFactorLimit
  : RealRange = builder.localFactorLimit

  protected val localFactors
  : ValueTensorBuffer = weightBuffer.allocateSibling(); localFactors := Real.one

  protected val prevGradients
  : ValueTensorBuffer = weightBuffer.allocateZeroedSibling()

  override val buffers
  : List[ValueTensorBuffer] = localFactors :: prevGradients :: super.buffers

  override val parameters
  : Map[UUID, Parameter] = {
    var tmp = super.parameters
    tmp += learningRate.toTuple
    tmp += rewardFactor.toTuple
    tmp += penaltyFactor.toTuple
    tmp
  }

  override protected def doClose()
  : Unit = {
    prevGradients.close()
    localFactors.close()
    penaltyFactor.close()
    rewardFactor.close()
    learningRate.close()
    super.doClose()
  }

  // TODO: Double check this code!
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
        val rf = rewardFactor.get(_iterationNo, RealRange.oneToInfinity)
        val pf = penaltyFactor.get(_iterationNo, RealRange.zeroToOne)

        val t1 = clock.readAndResetAs("GetHyperP")

        using(batchPool.draw())(drawContext => {
          // Check for end of stream and recompute.
          if (drawContext.isEmpty) {
            return OptimizationResult.derive(
              NoMoreData(), _iterationNo, noSamples
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

            val lf = localFactors.createIntersectionView(w)
            val pg = prevGradients.createIntersectionView(w)
            val g  = gradients.createIntersectionView(w)

            val t4 = clock.readAndResetAs("EvalObj")

            // Compute new gradients.
            doBackwardProp(w, context, g)

            val t5 = clock.readAndResetAs("BProp")

            // Apply local rates and perform RProp-style GD step.
            w.foreachSegment(g, lf)(
              (_w, g, lf) => {
                val w  = _w.asOrToRealArrayTensor

                w.transform(g, lf,
                  (w, g, lf) => w - Math.signum(g) * lr * lf
                )

                if (w ne _w) {
                  _w := w
                  w.close()
                }
              }
            )

            val t6 = clock.readAndResetAs("UpdW")

            // Use changes of sign to update local factors.
            if (_iterationNo > 0L) {
              lf.foreachSegment(g, pg)(
                (_lf, g, pg) => {
                  val lf = _lf.asOrToRealArrayTensor

                  lf.transform(g, pg,
                    // Compare signs of gradient to decide whether to reward or penalize.
                    (lrf, g, pg) => {
                      if (g * pg >= Real.zero) {
                        localFactorLimit.clip(lrf * rf)
                      }
                      else {
                        localFactorLimit.clip(lrf * pf)
                      }
                    }
                  )

                  if (lf ne _lf) {
                    _lf := lf
                    lf.close()
                  }
                }
              )
            }

            val t7 = clock.readAndResetAs("UpdLF")

            // Backup gradients.
            pg := g

            val t8 = clock.readAndResetAs("UpdPrevG")

            // Update value series and record series.
            doUpdateParameters(_iterationNo, context.value)

            val t9 = clock.readAndResetAs("UpdHyperP")

            if (logger.isDebugEnabled) {
              val tS = LabeledTimeSpan(
                "SUM", t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9
              )
              logger.debug(
                s"$tS, $t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7, $t8, $t9"
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


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : RPropState = RPropState(
    super.state,
    learningRate.state,
    rewardFactor.state,
    penaltyFactor.state
  )

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: RPropState =>
        learningRate.restoreState(state.learningRate)
        rewardFactor.restoreState(state.rewardFactor)
        penaltyFactor.restoreState(state.penaltyFactor)
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class RPropBuilder
  extends OptimizerExBuilder[RPropBuilder] {

  override def repr
  : RPropBuilder = this

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
  : RPropBuilder = {
    learningRate_=(value)
    this
  }

  private var _rewardFactor
  : ParameterBuilder = ConstantValueBuilder(1.2f)

  def rewardFactor
  : ParameterBuilder = _rewardFactor

  def rewardFactor_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _rewardFactor = value
  }

  def setRewardFactor(value: ParameterBuilder)
  : RPropBuilder = {
    rewardFactor_=(value)
    this
  }

  private var _penaltyFactor
  : ParameterBuilder = ConstantValueBuilder(Real.pointFive)

  def penaltyFactor
  : ParameterBuilder = _penaltyFactor

  def penaltyFactor_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _penaltyFactor = value
  }

  def setPenaltyFactor(value: ParameterBuilder)
  : RPropBuilder = {
    penaltyFactor_=(value)
    this
  }

  private var _localFactorLimit
  : RealRange = RealRange(0.0001f, 1000.0f)

  def localFactorLimit
  : RealRange = _localFactorLimit

  def localFactorLimit_=(value: RealRange)
  : Unit = {
    require(value.min > Real.zero)
    _localFactorLimit = value
  }

  def setLocalFactorLimit(value: RealRange)
  : RPropBuilder = {
    localFactorLimit_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    _learningRate :: _rewardFactor :: _penaltyFactor :: _localFactorLimit :: super.doToString()
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _learningRate.hashCode())
    tmp = MurmurHash3.mix(tmp, _rewardFactor.hashCode())
    tmp = MurmurHash3.mix(tmp, _penaltyFactor.hashCode())
    tmp = MurmurHash3.mix(tmp, _localFactorLimit.hashCode())
    tmp
  }

  override def canEqual(that: Any): Boolean = that.isInstanceOf[RPropBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: RPropBuilder =>
      _learningRate     == other._learningRate  &&
      _rewardFactor     == other._rewardFactor  &&
      _penaltyFactor    == other._penaltyFactor &&
      _localFactorLimit == other._localFactorLimit
    case _ =>
      false
  })

  override protected def doCopy()
  : RPropBuilder = RPropBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: RPropBuilder =>
        other._learningRate     = _learningRate.copy
        other._rewardFactor     = _rewardFactor.copy
        other._penaltyFactor    = _penaltyFactor.copy
        other._localFactorLimit = _localFactorLimit
      case _ =>
    }
  }

  override def build(model:     Module,
                     batchPool: BatchPool,
                     seed:      InstanceSeed)
  : RProp = new RProp(this, model, batchPool, seed)


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _learningRate.permuteSeeds(fn)
  }

}

object RPropBuilder {

  final def apply()
  : RPropBuilder = new RPropBuilder

}

final case class RPropState(override val parent: OptimizerState,
                            learningRate:        InstanceState,
                            rewardFactor:        InstanceState,
                            penaltyFactor:       InstanceState)
  extends OptimizerState {
}

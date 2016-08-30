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

import edu.latrobe.blaze.parameters.ConstantValueBuilder

import scala.collection._
import scala.util.hashing._

/**
  * Gradient descent variant that adapts individual rates automatically for each
  * weight.
  *
  * w  = w    - lr * lrf * f'( w    )
  *  t    t-1                (  t-1 )
  *
  * lrf  = 1
  *    0
  *
  * lrf  = if f'( w  )  * f'( w    ) > 0
  *    t        (  t )      (  t-1 )
  *
  *        then  lrf    + reward
  *                 t-1
  *
  *        else  lrf    * penalty
  *                 t-1
  *
  */
// TODO: Integrate into momentum optimizer and/or enable for GPU.
final class SGDWithLocalLearningRates(override val builder:   SGDWithLocalLearningRatesBuilder,
                                      override val model:     Module,
                                      override val batchPool: BatchPool,
                                      override val seed:      InstanceSeed)
  extends OptimizerEx[SGDWithLocalLearningRatesBuilder] {

  private val learningRate
  : Parameter = builder.learningRate.build("Learning Rate", seed)

  private val rewardIncrement
  : Parameter = builder.rewardIncrement.build("Reward Increment", seed)

  private val penaltyFactor
  : Parameter = builder.penaltyFactor.build("Penalty Factor", seed)

  private val localFactorLimit
  : RealRange = builder.localFactorLimit

  private val localFactors
  : ValueTensorBuffer = weightBuffer.allocateSibling(); localFactors := Real.one

  private val prevGradients
  : ValueTensorBuffer = weightBuffer.allocateZeroedSibling()

  override val buffers
  : List[ValueTensorBuffer] = localFactors :: prevGradients :: super.buffers

  override val parameters
  : Map[UUID, Parameter] = {
    var tmp = super.parameters
    tmp += learningRate.toTuple
    tmp += rewardIncrement.toTuple
    tmp += penaltyFactor.toTuple
    tmp
  }

  override protected def doClose()
  : Unit = {
    prevGradients.close()
    localFactors.close()
    penaltyFactor.close()
    rewardIncrement.close()
    learningRate.close()
    super.doClose()
  }

  // TODO: Implement switching between multiple buffers for scenarios with multiple alternating targets.
  override protected def doRun(runBeginIterationNo: Long,
                               runBeginTime:        Timestamp)
  : OptimizationResult = {
    var noSamples = 0L
    using(
      weightBuffer.allocateSibling(),
      weightBuffer.allocateSibling()
    )((gradients, temporary) => {
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
        val lr = learningRate.get(_iterationNo, RealRange.zeroToInfinity)
        val ri = rewardIncrement.get(_iterationNo, RealRange.zeroToInfinity)
        val pf = penaltyFactor.get(_iterationNo, RealRange.zeroToOne)

        val t1 = clock.readAndResetAs("GetHyperP")

        using(batchPool.draw())(drawContext => {
          // Check for end of stream and recompute.
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

            val lf  = localFactors.createIntersectionView(w)
            val pg  = prevGradients.createIntersectionView(w)
            val g   = gradients.createIntersectionView(w)
            val tmp = temporary.createIntersectionView(w)

            val t4 = clock.readAndResetAs("EvalObj")

            // Compute new gradients.
            doBackwardProp(w, context, g)

            val t5 = clock.readAndResetAs("BProp")

            // Apply global and local learning rates and perform SGD step.
            tmp := g

            val t6 = clock.readAndResetAs("g->tmp")

            tmp :*= lf

            val t7 = clock.readAndResetAs("tmp*lf")

            w.add(tmp, -lr)

            /*
            weightsBank.transformValues(gradientsBank, localFactorsBank,
              (w, g, lf) => w - lr * g * lf
            )
            */

            val t8 = clock.readAndResetAs("UpdW")

            // Compute new gradient and use changes of sign to update local factors.
            if (_iterationNo > 0L) {
              lf.foreachSegment(g, pg)(
                (_lf, g, pg) => {
                  // TODO: Remove this after we have integrated Aiden's code.
                  val lf = _lf.asOrToRealArrayTensor
                  lf.transform(g, pg,
                    (lf, g, pg) => {
                      localFactorLimit.clip(
                        if (g * pg >= Real.zero) {
                          lf + ri
                        }
                        else {
                          lf * pf
                        }
                      )
                    }
                  )

                  if (lf ne _lf) {
                    _lf := lf
                    lf.close()
                  }
                }
              )
            }

            val t9 = clock.readAndResetAs("UpdLF")

            // Backup gradients.
            pg := g

            val tA = clock.readAndResetAs("g->prevG")

            // Update value series and record series.
            doUpdateParameters(_iterationNo, context.value)

            val tB = clock.readAndResetAs("UpdHyperP")

            if (logger.isDebugEnabled) {
              val tS = LabeledTimeSpan(
                "SUM", t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + tA + tB
              )
              logger.debug(
                s"$tS, $t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7, $t8, $t9, $tA, $tB"
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
  : SGDWithLocalLearningRatesState = SGDWithLocalLearningRatesState(
    super.state,
    learningRate.state,
    rewardIncrement.state,
    penaltyFactor.state
  )

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: SGDWithLocalLearningRatesState =>
        learningRate.restoreState(state.learningRate)
        rewardIncrement.restoreState(state.rewardIncrement)
        penaltyFactor.restoreState(state.penaltyFactor)
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class SGDWithLocalLearningRatesBuilder
  extends OptimizerExBuilder[SGDWithLocalLearningRatesBuilder] {

  override def repr
  : SGDWithLocalLearningRatesBuilder = this

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
  : SGDWithLocalLearningRatesBuilder = {
    learningRate_=(value)
    this
  }

  private var _rewardIncrement
  : ParameterBuilder = ConstantValueBuilder(0.05f)

  def rewardIncrement
  : ParameterBuilder = _rewardIncrement

  def rewardIncrement_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _rewardIncrement = value
  }

  def setRewardIncrement(value: ParameterBuilder)
  : SGDWithLocalLearningRatesBuilder = {
    rewardIncrement_=(value)
    this
  }

  private var _penaltyFactor
  : ParameterBuilder = ConstantValueBuilder(0.95f)

  def penaltyFactor
  : ParameterBuilder = _penaltyFactor

  def penaltyFactor_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _penaltyFactor = value
  }

  def setPenaltyFactor(value: ParameterBuilder)
  : SGDWithLocalLearningRatesBuilder = {
    penaltyFactor_=(value)
    this
  }

  private var _localFactorLimit
  : RealRange = RealRange(0.01f, 100.0f)

  def localFactorLimit
  : RealRange = _localFactorLimit

  def localFactorLimit_=(value: RealRange)
  : Unit = {
    require(value.min > Real.zero)
    _localFactorLimit = value
  }

  def setLocalFactorLimit(value: RealRange)
  : SGDWithLocalLearningRatesBuilder = {
    localFactorLimit_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    _learningRate :: _rewardIncrement :: _penaltyFactor :: _localFactorLimit :: super.doToString()
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SGDWithLocalLearningRatesBuilder]

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _learningRate.hashCode())
    tmp = MurmurHash3.mix(tmp, _rewardIncrement.hashCode())
    tmp = MurmurHash3.mix(tmp, _penaltyFactor.hashCode())
    tmp = MurmurHash3.mix(tmp, _localFactorLimit.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: SGDWithLocalLearningRatesBuilder =>
      _learningRate     == other._learningRate    &&
      _rewardIncrement  == other._rewardIncrement &&
      _penaltyFactor    == other._penaltyFactor   &&
      _localFactorLimit == other._localFactorLimit
    case _ =>
      false
  })

  override protected def doCopy()
  : SGDWithLocalLearningRatesBuilder = SGDWithLocalLearningRatesBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: SGDWithLocalLearningRatesBuilder =>
        other._learningRate     = _learningRate.copy
        other._rewardIncrement  = _rewardIncrement.copy
        other._penaltyFactor    = _penaltyFactor.copy
        other._localFactorLimit = _localFactorLimit
      case _ =>
    }
  }

  override def build(model:     Module,
                     batchPool: BatchPool,
                     seed:      InstanceSeed)
  : SGDWithLocalLearningRates = new SGDWithLocalLearningRates(
    this, model, batchPool, seed
  )



  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _learningRate.permuteSeeds(fn)
    _rewardIncrement.permuteSeeds(fn)
    _penaltyFactor.permuteSeeds(fn)
  }

}

object SGDWithLocalLearningRatesBuilder {

  final def apply()
  : SGDWithLocalLearningRatesBuilder = new SGDWithLocalLearningRatesBuilder

}

final case class SGDWithLocalLearningRatesState(override val parent: OptimizerState,
                                                learningRate:        InstanceState,
                                                rewardIncrement:     InstanceState,
                                                penaltyFactor:       InstanceState)
  extends OptimizerState {
}

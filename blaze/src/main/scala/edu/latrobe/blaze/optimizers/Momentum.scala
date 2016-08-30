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

import edu.latrobe.blaze.parameters.{CMAFactorBuilder, ConstantValueBuilder}

import scala.collection._
import scala.util.hashing._

// TODO: RMS + Nesterov Momentum Sutskever 2012 (divide correction by RMS)

// TODO: Yann LeCun "No more pesky learning rates"

/**
  * Uses momentum method to speed up GD.
  *
  * dr = decay rate
  * lr = learning rate
  *
  *                         (    )
  * v    = dr * v  - lr * f'( w  )
  *  t+1         t          (  t )
  *
  * w    = w  + v
  *  t+1    t    t+1
  *
  */
final class Momentum(override val builder:   MomentumBuilder,
                     override val model:     Module,
                     override val batchPool: BatchPool,
                     override val seed:      InstanceSeed)
  extends OptimizerEx[MomentumBuilder] {

  private val learningRate
  : Parameter = builder.learningRate.build("Learning Rate", seed)

  private val decayRate
  : Parameter = builder.decayRate.build("Velocity Decay Rate", seed)

  private val dampeningFactor
  : Parameter = builder.dampeningFactor.build("Dampening Factor", seed)

  private val nesterovFactor
  : Parameter = builder.nesterovFactor.build("Nesterov Factor", seed)

  private val velocity
  : ValueTensorBuffer = weightBuffer.allocateZeroedSibling()

  override val buffers
  : List[ValueTensorBuffer] = velocity :: super.buffers

  override val parameters
  : Map[UUID, Parameter] = {
    var tmp = super.parameters
    tmp += learningRate.toTuple
    tmp += decayRate.toTuple
    tmp += dampeningFactor.toTuple
    tmp += nesterovFactor.toTuple
    tmp
  }

  override protected def doClose()
  : Unit = {
    velocity.close()
    nesterovFactor.close()
    dampeningFactor.close()
    decayRate.close()
    learningRate.close()
    super.doClose()
  }

  // TODO: Implement switching between multiple buffers for scenarios with multiple alternating targets.
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
        val dr = decayRate.get(_iterationNo, RealRange.zeroToOne)
        val df = dampeningFactor.get(_iterationNo, RealRange.zeroToOne)
        val nf = nesterovFactor.get(_iterationNo, RealRange.zeroToInfinity)

        val t1 = clock.readAndResetAs("GetHyperP")

        using(batchPool.draw())(drawContext => {
          // Check for end of stream and compute cost.
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

            val v = velocity.createIntersectionView(w)
            val g = gradients.createIntersectionView(w)

            val t4 = clock.readAndResetAs("EvalObj")

            // Compute new gradients.
            doBackwardProp(w, context, g)

            val t5 = clock.readAndResetAs("BProp")

            // Update momentum.
            v.add(dr, g, Real.one - df)

            val t6 = clock.readAndResetAs("UpdVel")

            // Update weights. (with nesterov this is the correction step).
            w.add(v, -lr)

            val t8 = clock.readAndResetAs("UpdW")

            // Do speculative nesterov step.
            if (nf > Real.zero) {
              w.add(v, -nf)
            }

            val t7 = clock.readAndResetAs("Nesterov")

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
  : MomentumState = MomentumState(
    super.state,
    learningRate.state,
    decayRate.state,
    dampeningFactor.state,
    nesterovFactor.state
  )

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: MomentumState =>
        learningRate.restoreState(state.learningRate)
        decayRate.restoreState(state.decayRate)
        dampeningFactor.restoreState(state.dampeningFactor)
        nesterovFactor.restoreState(state.nesterovFactor)
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class MomentumBuilder
  extends OptimizerExBuilder[MomentumBuilder] {

  override def repr
  : MomentumBuilder = this

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
  : MomentumBuilder = {
    learningRate_=(value)
    this
  }

  private var _decayRate
  : ParameterBuilder = CMAFactorBuilder().mirror(Real.one).clip(0.0f, 0.9f)

  def decayRate
  : ParameterBuilder = _decayRate

  def decayRate_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _decayRate = value
  }

  def setDecayRate(value: ParameterBuilder)
  : MomentumBuilder = {
    decayRate_=(value)
    this
  }

  private var _dampeningFactor
  : ParameterBuilder = ConstantValueBuilder(Real.zero)

  def dampeningFactor
  : ParameterBuilder = _dampeningFactor

  def dampeningFactor_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _dampeningFactor = value
  }

  def setDampeningFactor(value: ParameterBuilder)
  : MomentumBuilder = {
    dampeningFactor_=(value)
    this
  }

  private var _nesterovFactor
  : ParameterBuilder = ConstantValueBuilder(Real.zero)

  def nesterovFactor
  : ParameterBuilder = _nesterovFactor

  def nesterovFactor_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _nesterovFactor = value
  }

  def setNesterovFactor(value: ParameterBuilder)
  : MomentumBuilder = {
    nesterovFactor_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    _learningRate :: _decayRate :: _dampeningFactor :: _nesterovFactor :: super.doToString()
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _learningRate.hashCode())
    tmp = MurmurHash3.mix(tmp, _decayRate.hashCode())
    tmp = MurmurHash3.mix(tmp, _dampeningFactor.hashCode())
    tmp = MurmurHash3.mix(tmp, _nesterovFactor.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MomentumBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: MomentumBuilder =>
      _learningRate    == other._learningRate    &&
      _decayRate       == other._decayRate       &&
      _dampeningFactor == other._dampeningFactor &&
      _nesterovFactor  == other._nesterovFactor
    case _ =>
      false
  })

  override protected def doCopy()
  : MomentumBuilder = MomentumBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: MomentumBuilder =>
        other._learningRate    = _learningRate.copy
        other._decayRate       = _decayRate.copy
        other._dampeningFactor = _dampeningFactor.copy
        other._nesterovFactor  = _nesterovFactor.copy
      case _ =>
    }
  }

  override def build(model:     Module,
                     batchPool: BatchPool,
                     seed:      InstanceSeed)
  : Momentum = new Momentum(this, model, batchPool, seed)


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _learningRate.permuteSeeds(fn)
    _decayRate.permuteSeeds(fn)
    _dampeningFactor.permuteSeeds(fn)
    _nesterovFactor.permuteSeeds(fn)
  }

}

object MomentumBuilder {

  final def apply()
  : MomentumBuilder = new MomentumBuilder

}

final case class MomentumState(override val parent: OptimizerState,
                               learningRate:        InstanceState,
                               decayRate:           InstanceState,
                               dampeningFactor:     InstanceState,
                               nesterovFactor:      InstanceState)
  extends OptimizerState {
}

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

import java.util.UUID

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.optimizerexitcodes._
import edu.latrobe.blaze.parameters._
import edu.latrobe.time._
import scala.collection._
import scala.concurrent.Future
import scala.util.hashing._
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * A learning algorithm that automatically adjusts step sizes using knowledge
  * about the past step sizes. Has only a few very robust hyper parameters. Good
  * default if you have no clue what hyper parameters you want to use.
  *
  * dr = Decay Rate (of MS buffers)
  * e  = epsilon (makes sure we do not divide by zero and also caps minimum step size)
  *
  * gMS   = 0
  *    t0
  *
  * dMS   = 0
  *    t0
  *
  *      d f
  * g  = ---
  *  t   d x
  *
  *                               2
  * gMS   = dr gMS    + (1 - dr) g
  *    t          t-1             t
  *
  *            /----------
  *           / dMS    + e
  *         \/     t-1
  * d  = g  --------------
  *  t    t    /---------
  *           / gMS  + e
  *         \/     t
  *
  *
  *                               2
  * dMS   = dr dMS    + (1 - dr) d
  *    t          t-1             t
  *
  *
  * w  = w    - d
  *  t    t-1    t
  *
  */
final class AdaDelta(override val builder:   AdaDeltaBuilder,
                     override val model:     Module,
                     override val batchPool: BatchPool,
                     override val seed:      InstanceSeed)
  extends OptimizerEx[AdaDeltaBuilder] {

  private val decayRate
  : Parameter = builder.decayRate.build("DecRate", seed)

  val epsilon
  : Real = builder.epsilon

  private val gradientMeanSquares
  : ValueTensorBuffer = weightBuffer.allocateZeroedSibling()

  private val deltaMeanSquares
  : ValueTensorBuffer = weightBuffer.allocateZeroedSibling()

  override val buffers
  : List[ValueTensorBuffer] = {
    gradientMeanSquares :: deltaMeanSquares :: super.buffers
  }

  override val parameters
  : Map[UUID, Parameter] = super.parameters + decayRate.toTuple

  override protected def doClose()
  : Unit = {
    deltaMeanSquares.close()
    gradientMeanSquares.close()
    decayRate.close()
    super.doClose()
  }

  override protected def doRun(runBeginIterationNo: Long,
                               runBeginTime:        Timestamp)
  : OptimizationResult = {
    val clock     = Stopwatch()
    var noSamples = 0L
    using(
      weightBuffer.allocateSibling(),
      weightBuffer.allocateSibling()
    )((gradients, temporary) => {
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
        val dr = decayRate.get(_iterationNo, RealRange.zeroToOne)

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
                exitCode, _iterationNo, noSamples
              )
            })

            val gms = gradientMeanSquares.createIntersectionView(w)
            val dms = deltaMeanSquares.createIntersectionView(w)
            val g   = gradients.createIntersectionView(w)
            val tmp = temporary.createIntersectionView(w)

            val t4 = clock.readAndResetAs("EvalObj")

            // Compute new gradients.
            doBackwardProp(w, context, g)

            val t5 = clock.readAndResetAs("BProp")

            // Update mean squares of gradients.
            gms.lerp(g, g, Real.one - dr)

            val t6 = clock.readAndResetAs("UpdGradMS")

            // Compute delta
            g.foreachSegment(gms, dms, tmp)(
              (g, gms, dms, tmp) => {
                tmp := dms

                tmp.divide(epsilon, gms, epsilon)
                tmp.sqrt()

                g :*= tmp
              }
            )

            // Now gradients is delta!

            val t7 = clock.readAndResetAs("CompDelta")

            // Update mean squares of the deltas.
            dms.lerp(g, g, Real.one - dr)

            val t8 = clock.readAndResetAs("UpdDeltaMS")

            // Apply delta to weights.
            w -= g

            val tA = clock.readAndResetAs("UpdWeights")

            // Update value series and record series.
            doUpdateParameters(_iterationNo, context.value)

            val t9 = clock.readAndResetAs("UpdHyperP")

            if (logger.isDebugEnabled) {
              val tS = LabeledTimeSpan(
                "SUM", t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + tA
              )
              logger.debug(
                s"$tS, $t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7, $t8, $t9, $tA"
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
  : AdaDeltaState = AdaDeltaState(
    super.state,
    decayRate.state
  )

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: AdaDeltaState =>
        decayRate.restoreState(state.decayRate)
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class AdaDeltaBuilder
  extends OptimizerExBuilder[AdaDeltaBuilder] {

  override def repr
  : AdaDeltaBuilder = this

  override protected def doToString()
  : List[Any] =  _decayRate :: f"${_epsilon}%.4g" :: super.doToString()

  /**
    * Should be in [0, 1]
    */
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
  : AdaDeltaBuilder = {
    decayRate_=(value)
    this
  }

  /**
    * I took this default from some code I found on the web. Not sure whether
    * this is a good default. However, do not set epsilon to small or you will
    * make no progress.
    */
  private var _epsilon
  : Real = 1e-6f

  def epsilon
  : Real = _epsilon

  def epsilon_=(value: Real)
  : Unit = {
    require(value >= Real.zero)
    _epsilon = value
  }

  def setEpsilon(value: Real)
  : AdaDeltaBuilder = {
    epsilon_=(value)
    this
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _decayRate.hashCode())
    tmp = MurmurHash3.mix(tmp, _epsilon.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AdaDeltaBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AdaDeltaBuilder =>
      _decayRate == other._decayRate &&
      _epsilon   == other._epsilon
    case _ =>
      false
  })

  override protected def doCopy()
  : AdaDeltaBuilder = AdaDeltaBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: AdaDeltaBuilder =>
        other._decayRate = _decayRate.copy
        other._epsilon   = _epsilon
      case _ =>
    }
  }

  override def build(model:     Module,
                     batchPool: BatchPool,
                     seed:      InstanceSeed)
  : AdaDelta = new AdaDelta(this, model, batchPool, seed)


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _decayRate.permuteSeeds(fn)
  }

}

object AdaDeltaBuilder {

  final def apply()
  : AdaDeltaBuilder = new AdaDeltaBuilder

}

final case class AdaDeltaState(override val parent: OptimizerState,
                               decayRate:           InstanceState)
  extends OptimizerState {
}

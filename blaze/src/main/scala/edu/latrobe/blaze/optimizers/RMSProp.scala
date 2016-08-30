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
import edu.latrobe.blaze.parameters._
import edu.latrobe.time._
import java.util.UUID
import scala.collection._
import scala.util.hashing._

/**
  * Divides the delta by the running average mean squares.
  *
  * Hinton proposes this method for training networks.
  * Quoc Le also says that he used it for training in his 2012 paper.
  *
  * Pro: Only minimal history needed.
  * Con: Historical information extremely compressed!
  *
  * RMS(0) = ??? Not covered in hinton lecture. However the following makes sense:
  *               2
  * RMS(0) = delta
  *                                                2
  * RMS(t) = gamma * RMS(t-1) + (1 - gamma) * delta
  *
  * w(t) = alpha * delta / sqrt(RMS(t) + epsilon) * w(t-1)
  *
  *
  * Note: To create AdaGrad, you should sen
  *
  *
  */
final class RMSProp(override val builder:   RMSPropBuilder,
                    override val model:     Module,
                    override val batchPool: BatchPool,
                    override val seed:      InstanceSeed)
  extends OptimizerEx[RMSPropBuilder] {

  private val learningRate
  : Parameter = builder.learningRate.build("LR", seed)

  private val decayRate
  : Parameter = builder.decayRate.build("RMSDecay", seed)

  val epsilon
  : Real = builder.epsilon

  private val residualMeanSquares
  : ValueTensorBuffer = weightBuffer.allocateZeroedSibling()

  override val buffers
  : List[ValueTensorBuffer] = residualMeanSquares :: super.buffers

  override val parameters
  : Map[UUID, Parameter] = {
    super.parameters + learningRate.toTuple + decayRate.toTuple
  }

  override protected def doClose()
  : Unit = {
    residualMeanSquares.close()
    decayRate.close()
    learningRate.close()
    super.doClose()
  }

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

            // Evaluate targets.
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

            val rms = residualMeanSquares.createIntersectionView(w)
            val g   = gradients.createIntersectionView(w)
            val tmp = temporary.createIntersectionView(w)

            val t4 = clock.readAndResetAs("EvalObj")

            // Compute new gradients.
            doBackwardProp(w, context, g)

            val t5 = clock.readAndResetAs("BProp")

            // Update the current mean squares.
            rms.lerp(g, g, Real.one - dr)

            val t6 = clock.readAndResetAs("UpdRMS")

            // Scale gradients and update the weights.
            w.foreachSegment(g, rms, tmp)(
              (w, g, rms, tmp) => {
                tmp := rms

                tmp.sqrt()
                g.divide(tmp, epsilon)

                w.add(g, -lr)
              }
            )

            val t7 = clock.readAndResetAs("UpdW")

            // Update value series and record series.
            doUpdateParameters(_iterationNo, context.value)

            val t8 = clock.readAndResetAs("UpdHyperP")

            if (logger.isDebugEnabled) {
              val tS = LabeledTimeSpan(
                "SUM", t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
              )
              logger.debug(
                s"$tS, $t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7, $t8"
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
  : RMSPropState = RMSPropState(
    super.state,
    learningRate.state,
    decayRate.state
  )

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: RMSPropState =>
        learningRate.restoreState(state.learningRate)
        decayRate.restoreState(state.decayRate)
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class RMSPropBuilder
  extends OptimizerExBuilder[RMSPropBuilder] {

  override def repr
  : RMSPropBuilder = this

  override protected def doToString()
  : List[Any] = {
    _learningRate :: _decayRate :: f"${_epsilon}%.4g" :: super.doToString()
  }

  /**
    * Often smaller than this is better.
    */
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
  : RMSPropBuilder = {
    learningRate_=(value)
    this
  }

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
  : RMSPropBuilder = {
    decayRate_=(value)
    this
  }

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
  : RMSPropBuilder = {
    epsilon_=(value)
    this
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _learningRate.hashCode())
    tmp = MurmurHash3.mix(tmp, _decayRate.hashCode())
    tmp = MurmurHash3.mix(tmp, _epsilon.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RMSPropBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: RMSPropBuilder =>
      _learningRate == other._learningRate &&
      _decayRate    == other._decayRate    &&
      _epsilon      == other._epsilon
    case _ =>
      false
  })

  override protected def doCopy()
  : RMSPropBuilder = RMSPropBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: RMSPropBuilder =>
        other._learningRate = _learningRate.copy
        other._decayRate    = _decayRate.copy
        other._epsilon      = _epsilon
      case _ =>
    }
  }

  override def build(model:     Module,
                     batchPool: BatchPool,
                     seed:      InstanceSeed)
  : RMSProp = new RMSProp(this, model, batchPool, seed)


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _learningRate.permuteSeeds(fn)
    _decayRate.permuteSeeds(fn)
  }

}

object RMSPropBuilder {

  final def apply()
  : RMSPropBuilder = new RMSPropBuilder()

  final def apply(learningRate: ParameterBuilder)
  : RMSPropBuilder = apply().setLearningRate(learningRate)

  final def apply(learningRate: ParameterBuilder,
                  decayRate:    ParameterBuilder)
  : RMSPropBuilder = apply(learningRate).setDecayRate(decayRate)

  final def apply(learningRate: ParameterBuilder,
                  decayRate:    ParameterBuilder,
                  epsilon:      Real)
  : RMSPropBuilder = apply(learningRate, decayRate).setEpsilon(epsilon)

}

final case class RMSPropState(override val parent: OptimizerState,
                              learningRate:        InstanceState,
                              decayRate:           InstanceState)
  extends OptimizerState {
}

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
  * http://arxiv.org/pdf/1412.6980v8.pdf
  */
// TODO: Not tested with variable beta1 and beta2.
final class Adam(override val builder:   AdamBuilder,
                 override val model:     Module,
                 override val batchPool: BatchPool,
                 override val seed:      InstanceSeed)
  extends OptimizerEx[AdamBuilder] {

  private val learningRate
  : Parameter = builder.learningRate.build("Learning Rate", seed)

  private val beta1
  : Parameter = builder.beta1.build("Beta1", seed)

  private val beta2
  : Parameter = builder.beta2.build("Beta2", seed)

  private val epsilon
  : Real = builder.epsilon

  private val momentum
  : ValueTensorBuffer = weightBuffer.allocateZeroedSibling()

  private val meanSquares
  : ValueTensorBuffer = weightBuffer.allocateZeroedSibling()

  override val buffers
  : List[ValueTensorBuffer] = momentum :: meanSquares :: super.buffers

  override val parameters
  : Map[UUID, Parameter] = {
    super.parameters + learningRate.toTuple + beta1.toTuple + beta2.toTuple
  }

  override protected def doClose()
  : Unit = {
    meanSquares.close()
    momentum.close()
    beta2.close()
    beta1.close()
    learningRate.close()
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
        val lr = learningRate.get(_iterationNo, RealRange.zeroToInfinity)
        val b1 = beta1.get(_iterationNo, RealRange.zeroToOne)
        val b2 = beta2.get(_iterationNo, RealRange.zeroToOne)

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
                exitCode, _iterationNo, noSamples
              )
            })

            val mom = momentum.createIntersectionView(w)
            val rms = meanSquares.createIntersectionView(w)
            val g   = gradients.createIntersectionView(w)
            val tmp = temporary.createIntersectionView(w)

            val t4 = clock.readAndResetAs("EvalObj")

            // Compute new gradients.
            doBackwardProp(w, context, g)

            val t5 = clock.readAndResetAs("BProp")

            // Update momentum.
            mom.lerp(g, Real.one - b1)

            val t6 = clock.readAndResetAs("UpdMom")

            // Update RMS.
            rms.lerp(g, g, Real.one - b2)

            val t7 = clock.readAndResetAs("UpdRMS")

            val biasCorrection1 = 1.0 - Math.pow(DoubleEx(b1), _iterationNo + 1L)
            val biasCorrection2 = 1.0 - Math.pow(DoubleEx(b2), _iterationNo + 1L)
            val stepSize        = Real(lr * Math.sqrt(biasCorrection2) / biasCorrection1)

            w.foreachSegment(g, mom, rms, tmp)(
              (w, g, mom, rms, tmp) => {
                tmp := rms
                tmp.sqrt()

                g := mom
                g.divide(tmp, epsilon)

                w.add(g, -stepSize)
              }
            )

            val t8 = clock.readAndResetAs("UpdWeights")

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
  : AdamState = AdamState(
    super.state,
    learningRate.state,
    beta1.state,
    beta2.state
  )

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: AdamState =>
        learningRate.restoreState(state.learningRate)
        beta1.restoreState(state.beta1)
        beta2.restoreState(state.beta2)
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class AdamBuilder
  extends OptimizerExBuilder[AdamBuilder] {

  override def repr
  : AdamBuilder = this

  /**
    * Although this is the suggested rate, I found that a smaller value tends
    * to work better.
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
  : AdamBuilder = {
    learningRate_=(value)
    this
  }

  /**
    * Decay rate for the momentum term.
    */
  private var _beta1
  : ParameterBuilder = ConstantValueBuilder(0.9f)

  def beta1
  : ParameterBuilder = _beta1

  def beta1_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _beta1 = value
  }

  def setBeta1(value: ParameterBuilder)
  : AdamBuilder = {
    beta1_=(value)
    this
  }

  /**
    * Decay rate for the residual mean squares term.
    */
  private var _beta2
  : ParameterBuilder = ConstantValueBuilder(0.999f)

  def beta2
  : ParameterBuilder = _beta2

  def beta2_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _beta2 = value
  }

  def setBeta2(value: ParameterBuilder)
  : AdamBuilder = {
    beta2_=(value)
    this
  }

  /**
    * To avoid divide by zero. Value suggested in paper.
    */
  private var _epsilon
  : Real = 1e-8f

  def epsilon
  : Real = _epsilon

  def epsilon_=(value: Real)
  : Unit = {
    require(value >= Real.zero)
    _epsilon = value
  }

  def setEpsilon(value: Real)
  : AdamBuilder = {
    epsilon_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = {
    _learningRate :: _beta1 :: _beta2 :: f"${_epsilon}%.4g" :: super.doToString()
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _learningRate.hashCode())
    tmp = MurmurHash3.mix(tmp, _beta1.hashCode())
    tmp = MurmurHash3.mix(tmp, _beta2.hashCode())
    tmp = MurmurHash3.mix(tmp, _epsilon.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AdamBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AdamBuilder =>
      _learningRate == other._learningRate &&
      _beta1        == other._beta1        &&
      _beta2        == other._beta2        &&
      _epsilon      == other._epsilon
    case _ =>
      false
  })

  override protected def doCopy()
  : AdamBuilder = AdamBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: AdamBuilder =>
        other._learningRate = _learningRate.copy
        other._beta1        = _beta1.copy
        other._beta2        = _beta2.copy
        other._epsilon      = _epsilon
      case _ =>
    }
  }

  override def build(model:     Module,
                     batchPool: BatchPool,
                     seed:      InstanceSeed)
  : Adam = new Adam(this, model, batchPool, seed)

  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _learningRate.permuteSeeds(fn)
    _beta1.permuteSeeds(fn)
    _beta2.permuteSeeds(fn)
  }

}

object AdamBuilder {

  final def apply()
  : AdamBuilder = new AdamBuilder

  final def apply(learningRate: ParameterBuilder)
  : AdamBuilder = apply().setLearningRate(learningRate)

  final def apply(learningRate: ParameterBuilder,
                  beta1:        ParameterBuilder)
  : AdamBuilder = apply(learningRate).setBeta1(beta1)

  final def apply(learningRate: ParameterBuilder,
                  beta1:        ParameterBuilder,
                  beta2:        ParameterBuilder)
  : AdamBuilder = apply(learningRate, beta1).setBeta2(beta2)

}


final case class AdamState(override val parent: OptimizerState,
                           learningRate:        InstanceState,
                           beta1:               InstanceState,
                           beta2:               InstanceState)
  extends OptimizerState {
}

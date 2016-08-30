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
 *                           alpha
 * w(t)  = w(t-1)  - -------------------- .* g(t)
 *     i        i            /-----------       i
 *                          /  k
 *                         /  ----
 *                        /   \         2
 *                       /    /    g(t')
 *                   \  /     ----      i
 *                    \/      t'=1
 */
final class AdaGrad(override val builder:   AdaGradBuilder,
                    override val model:     Module,
                    override val batchPool: BatchPool,
                    override val seed:      InstanceSeed)
  extends OptimizerEx[AdaGradBuilder] {

  private val learningRate
  : Parameter = builder.learningRate.build("Learning Rate", seed)

  val epsilon
  : Real = builder.epsilon

  private val residualMeanSquares
  : ValueTensorBuffer = weightBuffer.allocateZeroedSibling()

  override val buffers
  : List[ValueTensorBuffer] = residualMeanSquares :: super.buffers

  override val parameters
  : Map[UUID, Parameter] = {
    super.parameters + learningRate.toTuple
  }

  override protected def doClose()
  : Unit = {
    residualMeanSquares.close()
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

            // Update the mean squares history.
            rms.add(g, g)

            val t6 = clock.readAndResetAs("UpdGradMS")

            // Scale gradients and add to weights.
            w.foreachSegment(g, rms, tmp)(
              (w, g, rms, tmp) => {
                tmp := rms

                tmp.sqrt()
                g.divide(tmp, epsilon)

                w.add(g, -lr)
              }
            )

            val t7 = clock.readAndResetAs("UpdWeights")

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
  : AdaGradState = AdaGradState(
    super.state,
    learningRate.state
  )

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: AdaGradState =>
        learningRate.restoreState(state.learningRate)
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class AdaGradBuilder
  extends OptimizerExBuilder[AdaGradBuilder] {

  override def repr
  : AdaGradBuilder = this

  override protected def doToString()
  : List[Any] = _learningRate :: f"${_epsilon}%.4g" :: super.doToString()

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
  : AdaGradBuilder = {
    learningRate_=(value)
    this
  }

  /**
    * Used 1-e8f before. 1-e10f I've seen in torch. Not sure which is better.
    */
  private var _epsilon
  : Real = 1e-10f

  def epsilon
  : Real = _epsilon

  def epsilon_=(value: Real)
  : Unit = {
    require(value >= Real.zero)
    _epsilon = value
  }

  def setEpsilon(value: Real)
  : AdaGradBuilder = {
    epsilon_=(value)
    this
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _learningRate.hashCode())
    tmp = MurmurHash3.mix(tmp, _epsilon.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AdaGradBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AdaGradBuilder =>
      _learningRate == other._learningRate &&
      _epsilon      == other._epsilon
    case _ =>
      false
  })

  override protected def doCopy()
  : AdaGradBuilder = AdaGradBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: AdaGradBuilder =>
        other._learningRate = _learningRate.copy
        other._epsilon      = _epsilon
      case _ =>
    }
  }

  override def build(model:     Module,
                     batchPool: BatchPool,
                     seed:      InstanceSeed)
  : AdaGrad = new AdaGrad(this, model, batchPool, seed)

  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _learningRate.permuteSeeds(fn)
  }

}

object AdaGradBuilder {

  final def apply()
  : AdaGradBuilder = new AdaGradBuilder

  final def apply(learningRate: ParameterBuilder)
  : AdaGradBuilder = apply().setLearningRate(learningRate)

  final def apply(learningRate: ParameterBuilder,
                  epsilon:      Real)
  : AdaGradBuilder = apply(learningRate).setEpsilon(epsilon)

}

final case class AdaGradState(override val parent: OptimizerState,
                              learningRate:        InstanceState)
  extends OptimizerState {
}

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

package edu.latrobe.blaze

import edu.latrobe._
import edu.latrobe.blaze.objectives._
import edu.latrobe.blaze.optimizerexitcodes._
import edu.latrobe.blaze.sinks.StdOutSinkBuilder
import edu.latrobe.time._
import java.util.UUID
import scala.collection._
import scala.util.hashing._

trait OptimizerLike
  extends Instance
    with ParameterizedInstance {

  override def builder
  : OptimizerLikeBuilder

  final protected var _iterationNo
  : Long = 0L

  final def iterationNo
  : Long = _iterationNo

  final private var _beginTime
  : Timestamp = Timestamp.now()

  final def beginTime
  : Timestamp = _beginTime

  final private var _runNo
  : Long = 0L

  final def runNo
  : Long = _runNo

  final private val earlyObjectives
  : Array[Objective] = builder.earlyObjectives.map(_.build(seed)).toArray

  final private val objectives
  : Array[Objective] = builder.objectives.map(_.build(seed)).toArray

  override protected def doClose()
  : Unit = {
    ArrayEx.foreach(
      objectives
    )(_.close())
    ArrayEx.foreach(
      earlyObjectives
    )(_.close())
    super.doClose()
  }

  final def run()
  : OptimizationResult = {
    val runBeginTime = Timestamp.now()
    if (_runNo == 0L) {
      _beginTime = runBeginTime
    }
    val result = doRun(_iterationNo, runBeginTime)
    _runNo += 1L
    result
  }

  protected def doRun(runBeginIterationNo: Long,
                      runBeginTime:        Timestamp)
  : OptimizationResult

  @transient
  final private lazy val defaultSink
  : Sink = StdOutSinkBuilder().build(seed)

  final protected def doEvaluateEarlyObjectives(runBeginIterationNo: Long,
                                                runBeginTime:        Timestamp,
                                                runNoSamples:        Long,
                                                model:               Module)
  : Option[OptimizerExitCode] = {
    val iter = earlyObjectives.iterator
    while (iter.hasNext) {
      val objective = iter.next()
      val result    = objective.evaluate(
        defaultSink,
        this, runBeginIterationNo, runBeginTime, runNoSamples,
        model,
        null, null, Real.nan
      )
      result.foreach(result => {
        return Some(ObjectiveReached(result))
      })
    }
    None
  }

  final protected def doEvaluateObjectives(runBeginIterationNo: Long,
                                           runBeginTime:        Timestamp,
                                           runNoSamples:        Long,
                                           model:               Module,
                                           batch:               Batch,
                                           output:              Tensor,
                                           value:               Real)
  : Option[OptimizerExitCode] = {
    val iter = objectives.iterator
    while (iter.hasNext) {
      val objective = iter.next()
      val result    = objective.evaluate(
        defaultSink,
        this, runBeginIterationNo, runBeginTime, runNoSamples,
        model,
        batch, output, value
      )
      result.foreach(result => {
        return Some(ObjectiveReached(result))
      })
    }
    None
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : InstanceState = OptimizerLikeState(
    super.state,
    iterationNo,
    beginTime,
    runNo,
    ArrayEx.map(
      earlyObjectives
    )(_.state),
    ArrayEx.map(
      objectives
    )(_.state)
  )

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: OptimizerLikeState =>
        _iterationNo = state.iterationNo
        _beginTime   = state.beginTime
        _runNo       = state.runNo
        ArrayEx.foreach(
          earlyObjectives,
          state.earlyObjectives
        )(_.restoreState(_))
        ArrayEx.foreach(
          objectives,
          state.objectives
        )(_.restoreState(_))
      case _ =>
        throw new MatchError(state)
    }
  }

}

trait OptimizerLikeBuilder
  extends InstanceBuilder {

  /**
    * Usually evaluated before FPROP. Not all fields populated.
    *
    * By default wait for key pressed. (For debugging! ;-)
    */
  final val earlyObjectives
  : mutable.Buffer[ObjectiveBuilder] = mutable.Buffer(KeyPressedBuilder())

  /**
    * Usually evaluated after FPROP. All fields populated!
    */
  final val objectives
  : mutable.Buffer[ObjectiveBuilder] = mutable.Buffer.empty

  override protected def doToString()
  : List[Any] = {
    earlyObjectives.length :: objectives.length :: super.doToString()
  }

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, earlyObjectives.hashCode())
    tmp = MurmurHash3.mix(tmp, objectives.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: OptimizerLikeBuilder =>
      earlyObjectives == other.earlyObjectives &&
      objectives      == other.objectives
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: OptimizerLikeBuilder =>
        other.earlyObjectives.clear()
        other.earlyObjectives ++= earlyObjectives.map(_.copy)
        other.objectives.clear()
        other.objectives ++= objectives.map(_.copy)
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    earlyObjectives.foreach(
      _.permuteSeeds(fn)
    )
    objectives.foreach(
      _.permuteSeeds(fn)
    )
  }

}

final case class OptimizerLikeState(override val parent: InstanceState,
                                    iterationNo:         Long,
                                    beginTime:           Timestamp,
                                    runNo:               Long,
                                    earlyObjectives:     Array[InstanceState],
                                    objectives:          Array[InstanceState])
  extends InstanceState

/**
 * Base class for all local optimizers.
 *
 * FailSafe minimize ensures that it's output is as least as good as the input
 * for processing the given dataset. (Note that fail-safe needs more memory!)
 * This feature might not work with optimizers that rely on third party libraries.
 */
abstract class Optimizer
  extends InstanceEx[OptimizerBuilder]
    with OptimizerLike {

  override def builder
  : OptimizerBuilder

  /**
   * Must be overwritten using a constructor parameter.
   */
  def model
  : Module

  /**
    * Must be overwritten using a constructor parameter.
    */
  def batchPool
  : BatchPool

  /**
    * Extract the optimizable portion of the weight buffer.
    */
  final val weightBuffer
  : ValueTensorBuffer = model.weightBuffer.createView((b, s, v) => b < 1000)

  final val scopeDelimiter
  : Option[ScopeDelimiter] = builder.scopeDelimiter.map(
    _.build(weightBuffer, seed)
  )

  final val regularizers
  : Seq[Regularizer] = builder.regularizers.map(
    _.build(None, seed)
  )

    /*
  final val weightDecayRateL2
  : Parameter = builder.weightDecayRateL2.build(seed)
  */

  def buffers
  : List[ValueTensorBuffer] = Nil

  override def parameters
  : Map[UUID, Parameter] = {
    var result = super.parameters
    result ++= model.parameters
    scopeDelimiter.foreach(
      result ++= _.parameters
    )
    regularizers.foreach(
      result ++= _.parameters
    )
    /*
    SortedMap(
      //("Bank#",    Real(scope.get(iterationNo, runNo))),
      ("WDecayL2", weightDecayRateL2.get(iterationNo))
    )
    */
    result
  }

  final private var _scopeLimitOverride
  : Option[NullBuffer] = None

  final def scopeLimitOverride
  : Option[NullBuffer] = _scopeLimitOverride

  /**
    * Takes the current weights and chisels stuff until the current scope
    * emerges.
    */
  final def determineCurrentScope()
  : ValueTensorBuffer = {
    var w = weightBuffer

    // Apply current delimiter.
    scopeDelimiter.foreach(
      sd => w = w.createIntersectionView(sd.get(_iterationNo))
    )

    // Consider the hard limits.
    _scopeLimitOverride.foreach(
      sl => w = w.createIntersectionView(sl)
    )

    w
  }

  override protected def doClose()
  : Unit = {
    scopeDelimiter.foreach(
      _.close()
    )
    regularizers.foreach(
      _.close()
    )
    //weightDecayRateL2.close()
    super.doClose()
  }

  final protected def forwardProp(weights: ValueTensorBuffer,
                                  batch:   Batch)
  : BackpropagationContext = {
    // TODO: Add dynamic global weight decay stuff.
    //val xx = Timestamp()
    model.refresh()

    //val yy = Timestamp()
    val res = model.predict(
      Training(_iterationNo),
      batch
    )
    //val zz = Timestamp()
    //println(f"--- ${TimeSpan(xx, yy).seconds} --- ${TimeSpan(yy, zz)}")
    //blazeLogger.trace(s"COST: ${res.cost}")

    // L2 weight decay.
    /*
    if (wdRateL2 != Real.zero) {
      val value = Real.pointFive * weights.foldLeftSegmentsParallel(Real.zero)(
        _ + _.squaredNorm,
        _ + _
      )
      res += value
    }*/

    // Apply regularizers.
    regularizers.foreach(
      res.value += _.evaluate(_iterationNo, weights, res)
    )
    res
  }

  final protected def doBackwardProp(weights: ValueTensorBuffer,
                                     context: BackpropagationContext,
                                     sink:    ValueTensorBuffer)
  : Unit = {
    // Activate if unsure whether previous computation overrides gradient properly.
    sink.clear()

    // Perform backprop.
    val error = model.deriveGradients(context, sink)
    error.close()

    // Apply regularizers.
    regularizers.foreach(
      _.deriveGradients(_iterationNo, weights, context, sink)
    )

    // L2 weight decay.
    /*
    if (wdRateL2 != Real.zero) {
      val w = model.weights.createFilteredView(sink)
      sink.add(w, wdRateL2)
    }
    */
  }

  final protected def doForwardAndBackwardProp(weights: ValueTensorBuffer,
                                               batch:   Batch,
                                               sink:    ValueTensorBuffer)
  : Real = {
    using(
      forwardProp(weights, batch)
    )(context => {
      doBackwardProp(weights, context, sink)
      context.value
    })
  }


  final protected def doEvaluateEarlyObjectives(runBeginIterationNo: Long,
                                                runBeginTime:        Timestamp,
                                                runNoSamples:        Long)
  : Option[OptimizerExitCode] = doEvaluateEarlyObjectives(
    runBeginIterationNo, runBeginTime, runNoSamples,
    model
  )


  final protected def doEvaluateObjectives(runBeginIterationNo: Long,
                                           runBeginTime:        Timestamp,
                                           runNoSamples:        Long,
                                           batch:               Batch,
                                           output:              Tensor,
                                           value:               Real)
  : Option[OptimizerExitCode] = doEvaluateObjectives(
    runBeginIterationNo, runBeginTime, runNoSamples,
    model,
    batch, output, value
  )

  final protected def doUpdateParameters(phaseNo: Long, value: Real)
  : Unit = {
    scopeDelimiter.foreach(
      _.update(_iterationNo, value)
    )
    MapEx.foreachValue(
      parameters
    )(_.update(phaseNo, value))
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : OptimizerState = OptimizerStateEx(
    super.state,
    scopeDelimiter.map(_.state),
    regularizers.map(_.state),
    _scopeLimitOverride
  )

  /**
   * Initializes this optimizer with a state returned by a previous call to
   * minimize. This is not implemented for all optimizers.
   */
  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: OptimizerStateEx =>
        SeqEx.foreach(
          scopeDelimiter,
          state.scopeDelimiter
        )(_.restoreState(_))
        SeqEx.foreach(
          regularizers,
          state.regularizers
        )(_.restoreState(_))
        _scopeLimitOverride = state.scopeLimitOverride
      case _ =>
        throw new MatchError(state)
    }
  }

  final def overrideState(iterationNo: Long,
                          scopeLimit:  NullBuffer)
  : Unit = overrideState(iterationNo, Option(scopeLimit))

  final def overrideState(iterationNo: Long,
                          scopeLimit:  Option[NullBuffer])
  : Unit = {
    require(iterationNo >= 0L && scopeLimit != null)
    _iterationNo = iterationNo
    _scopeLimitOverride  = scopeLimit
  }

  /*
  final def restoreState(state:       InstanceState,
                         iterationNo: Long,
                         buffers:     Array[RealTensorBuffer])
  : Unit = {
    restoreState(state)
    restoreState(iterationNo)
    ArrayEx.foreach(
      this.buffers(),
      buffers
    )(_ := _)
  }*/

}

abstract class OptimizerBuilder
  extends InstanceExBuilder2[OptimizerBuilder, Optimizer, Module, BatchPool]
    with OptimizerLikeBuilder {

  override def repr
  : OptimizerBuilder

  final private var _scopeDelimiter
  : Option[ScopeDelimiterBuilder] = None

  final def scopeDelimiter
  : Option[ScopeDelimiterBuilder] = _scopeDelimiter

  final def scopeDelimiter_=(value: Option[ScopeDelimiterBuilder])
  : Unit = {
    require(value != null)
    _scopeDelimiter = value
  }

  def setScopeDelimiter(value: Option[ScopeDelimiterBuilder])
  : OptimizerBuilder

  def setScopeDelimiter(value: ScopeDelimiterBuilder)
  : OptimizerBuilder

  final val regularizers
  : mutable.Buffer[RegularizerBuilder] = mutable.Buffer.empty

  /*
  final private var _weightsDecayRateL2
  : ParameterBuilder = ParameterBuilder.zeros

  final def weightDecayRateL2
  : ParameterBuilder = _weightsDecayRateL2

  final def weightDecayRateL2_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _weightsDecayRateL2 = value
  }

  def setWeightDecayRateL2(value: ParameterBuilder)
  : OptimizerBuilder
  */

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _scopeDelimiter.hashCode())
    tmp = MurmurHash3.mix(tmp, regularizers.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: OptimizerBuilder =>
      _scopeDelimiter == other._scopeDelimiter &&
      regularizers    == other.regularizers
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: OptimizerBuilder =>
        other._scopeDelimiter = _scopeDelimiter.map(_.copy)
        other.regularizers.clear()
        other.regularizers ++= regularizers.map(_.copy)
      case _ =>
    }
  }

  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _scopeDelimiter.foreach(
      _.permuteSeeds(fn)
    )
    regularizers.foreach(
      _.permuteSeeds(fn)
    )
  }

}

abstract class OptimizerEx[TBuilder <: OptimizerExBuilder[_]]
  extends Optimizer {

  override def builder
  : TBuilder

}

abstract class OptimizerExBuilder[TThis <: OptimizerExBuilder[_]]
  extends OptimizerBuilder {

  def repr
  : TThis

  override protected def doCopy()
  : TThis

  final override def setScopeDelimiter(value: Option[ScopeDelimiterBuilder])
  : TThis = {
    scopeDelimiter_=(value)
    repr
  }

  final override def setScopeDelimiter(value: ScopeDelimiterBuilder)
  : TThis = setScopeDelimiter(Option(value))

  override def build(model:     Module,
                     batchPool: BatchPool,
                     seed:      InstanceSeed)
  : OptimizerEx[TThis]

}

abstract class OptimizerState
  extends InstanceState {
}

final case class OptimizerStateEx(override val parent: InstanceState,
                                  scopeDelimiter:      Option[InstanceState],
                                  regularizers:        Seq[RegularizerState],
                                  scopeLimitOverride:  Option[NullBuffer])
  extends OptimizerState {
}

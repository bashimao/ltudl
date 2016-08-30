/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2015 Matthias Langer (t3l@threelights.de)
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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import scala.collection._
import scala.util.hashing._

/**
  * A variant of Sequence that takes the time required for executing
  * the underlying modules.
  */
final class Benchmark(override val builder:             BenchmarkBuilder,
                      override val inputHints:          BuildHints,
                      override val seed:                InstanceSeed,
                      override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends SequentialContainer[BenchmarkBuilder]
    with BenchmarkEnabled {

  override val (children, outputHints)
  : (Seq[Module], BuildHints) = {
    var tmpHints = inputHints
    val modules = builder.children.map(child => {
      val tmp = child.build(tmpHints, seed, weightBufferBuilder)
      tmpHints = tmp.outputHints
      tmp
    })
    (modules, tmpHints)
  }

  override protected def doClose()
  : Unit = {
    children.foreach(
      _.close()
    )
    super.doClose()
  }

  val caption
  : String = builder.caption


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override val requiresMaintainingInputDuringForwardPropagation
  : Boolean = false

  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor,
                                   onEnter:        OnEnterPredict,
                                   onLeave:        OnLeavePredict)
  : (Tensor, PredictContext) = {
    doBenchmark(
      s"$caption.predict(${input.layout})",
      super.doPredict(
        mode,
        inPlaceAllowed,
        input,
        reference,
        onEnter,
        onLeave
      )
    )
  }

  override protected def doPredictInv(output:   Tensor,
                                      context:  PredictContext,
                                      onLeave:  OnLeavePredict,
                                      contexts: mutable.Stack[PredictContext])
  : Tensor = {
    doBenchmark(
      s"$caption.predictInv(${output.layout})",
      super.doPredictInv(
        output,
        context,
        onLeave,
        contexts
      )
    )
  }


  // ---------------------------------------------------------------------------
  //    Backward propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  override protected def doDeriveGradients(context:  PredictContext,
                                           error:    NextError,
                                           sink:     ValueTensorBuffer,
                                           onEnter:  OnEnterDeriveGradients,
                                           onLeave:  OnLeaveDeriveGradients,
                                           tensors:  mutable.Stack[Tensor],
                                           contexts: mutable.Stack[PredictContext])
  : NextError = {
    doBenchmark(
      s"$caption.deriveGradients()",
      super.doDeriveGradients(
        context,
        error,
        sink,
        onEnter,
        onLeave,
        tensors,
        contexts
      )
    )
  }


  // ---------------------------------------------------------------------------
  //    State backup and retrieval.
  // ---------------------------------------------------------------------------
  override def state
  : BenchmarkState = BenchmarkState(
    super.state,
    children.map(_.state)
  )

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: BenchmarkState =>
        SeqEx.foreach(
          children,
          state.children
        )(_.restoreState(_))
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class BenchmarkBuilder
  extends SequentialContainerBuilder[BenchmarkBuilder] {

  override def repr
  : BenchmarkBuilder = this

  private var _caption
  : String = ""

  def caption
  : String = _caption

  def caption_=(value: String)
  : Unit = {
    require(value != null)
    _caption = value
  }

  def setCaption(value: String)
  : BenchmarkBuilder = {
    caption_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _caption :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _caption.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[BenchmarkBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BenchmarkBuilder =>
      _caption == other._caption
    case _ =>
      false
  })

  override protected def doCopy()
  : BenchmarkBuilder = BenchmarkBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: BenchmarkBuilder =>
        other._caption = _caption
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Benchmark = new Benchmark(this, hints, seed, weightsBuilder)

}

object BenchmarkBuilder {

  final def apply()
  : BenchmarkBuilder = new BenchmarkBuilder

  final def apply(caption: String)
  : BenchmarkBuilder = apply().setCaption(caption)

  final def apply(caption: String,
                  module0: ModuleBuilder)
  : BenchmarkBuilder = apply(caption) += module0

  final def apply(caption: String,
                  module0: ModuleBuilder,
                  modules: ModuleBuilder*)
  : BenchmarkBuilder = apply(caption, module0) ++= modules

  final def apply(caption: String,
                  modules: TraversableOnce[ModuleBuilder])
  : BenchmarkBuilder = apply(caption) ++= modules

  final def apply(caption: String,
                  modules: Array[ModuleBuilder])
  : BenchmarkBuilder = apply(caption) ++= modules

}

final case class BenchmarkState(override val parent: InstanceState,
                                children:            Seq[InstanceState])
  extends ModuleState {
}

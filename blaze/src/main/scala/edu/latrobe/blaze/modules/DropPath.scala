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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.io.graph._

import scala.Seq
import scala.collection._
import scala.util.hashing.MurmurHash3

final class DropPath(override val builder:             DropPathBuilder,
                     override val inputHints:          BuildHints,
                     override val seed:                InstanceSeed,
                     override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends SequentialContainer[DropPathBuilder] {

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

  val probability
  : Real = builder.probability

  @transient
  lazy val bernoulli
  : Distribution[Boolean] = rng.bernoulliDistribution(probability)

  override protected def doClose()
  : Unit = {
    children.foreach(
      _.close()
    )
    super.doClose()
  }


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
    val drop = {
      if (mode.reproducible) {
        // PseudoRNG(System.identityHashCode(this))
        // Since this is primarily used for gradient checking we avoid drawing
        // a random number and instead avoid dropping.
        false
      }
      else {
        !bernoulli.sample()
      }
    }

    // If dropping, do nothing.
    if (drop) {
      (input, DropPathContext)
    }
    else {
      super.doPredict(
        mode,
        inPlaceAllowed,
        input,
        reference,
        onEnter,
        onLeave
      )
    }
  }

  override protected def doPredictInv(output:   Tensor,
                                      context:  PredictContext,
                                      onLeave:  OnLeavePredict,
                                      contexts: mutable.Stack[PredictContext])
  : Tensor = context match {
    case DropPathContext =>
      output
    case _ =>
      super.doPredictInv(
        output,
        context,
        onLeave,
        contexts
      )
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
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
  : NextError = context match {
    case DropPathContext =>
      error
    case _ =>
      super.doDeriveGradients(
        context,
        error,
        sink,
        onEnter,
        onLeave,
        tensors,
        contexts
      )
  }


  // ---------------------------------------------------------------------------
  //    State backup and retrieval.
  // ---------------------------------------------------------------------------
  override def state
  : DropPathState = DropPathState(super.state, children.map(_.state))

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: DropPathState =>
        SeqEx.foreach(
          children,
          state.children
        )(_.restoreState(_))
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class DropPathBuilder
  extends SequentialContainerBuilder[DropPathBuilder] {

  override def repr
  : DropPathBuilder = this

  private var _probability
  : Real = Real.pointFive

  def probability
  : Real = _probability

  def probability_=(value: Real)
  : Unit = {
    require(value >= Real.zero && value <= Real.one)
    _probability = value
  }

  def setProbability(value: Real)
  : DropPathBuilder = {
    probability_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"${_probability}%.4g" :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[DropPathBuilder]

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _probability.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: DropPathBuilder =>
      _probability == other._probability
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: DropPathBuilder =>
        other._probability = _probability
      case _ =>
    }
  }

  override protected def doCopy()
  : DropPathBuilder = DropPathBuilder()


  // ---------------------------------------------------------------------------
  //    Weights and binding related
  // ---------------------------------------------------------------------------
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : DropPath = new DropPath(this, hints, seed, weightsBuilder)


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  override protected def doToGraphEx(node:      VertexGroup,
                                     hints:     Option[BuildHints],
                                     inputs:    Seq[Vertex],
                                     edgeStyle: LineStyle,
                                     nodeSink:  mutable.Buffer[Node],
                                     edgeSink:  mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex]) = {
    // Create dummy vertex to simulate switch.
    val inVertex = Vertex.derive("Switch").setShape(NodeShape.Point)
    nodeSink += inVertex
    for (input <- inputs) {
      val edge = Edge(input, inVertex, edgeStyle)
      for (hints <- hints) {
        edge.label = hints.toEdgeLabel
      }
      edgeSink += edge
    }

    // Process children.
    val (childHints, childOutputs) = super.doToGraphEx(
      node,
      hints,
      Seq(inVertex),
      LineStyle.Dotted,
      nodeSink,
      edgeSink
    )

    // Create dummy vertex to simulate switch.
    val outVertex = Vertex.derive("Gate").setShape(NodeShape.Point)
    nodeSink += outVertex
    for (childOutput <- childOutputs) {
      val edge = Edge(childOutput, outVertex, LineStyle.Dotted)
      for(childHints <- childHints) {
        edge.label = childHints.toEdgeLabel
      }
      edgeSink += edge
    }

    // Add a bypass connection.
    val shortcutEdge = Edge(inVertex, outVertex, LineStyle.Dotted)
    for (hints <- hints) {
      shortcutEdge.label = hints.toEdgeLabel
    }
    edgeSink += shortcutEdge

    (childHints, Seq(outVertex))
  }

}

object DropPathBuilder {

  final def apply()
  : DropPathBuilder = new DropPathBuilder

  final def apply(probability: Real)
  : DropPathBuilder = apply().setProbability(probability)

  final def apply(probability: Real,
                  module0:     ModuleBuilder)
  : DropPathBuilder = apply(probability) += module0

  final def apply(probability: Real,
                  module0:     ModuleBuilder,
                  modules:     ModuleBuilder*)
  : DropPathBuilder = apply(probability, module0) ++= modules

  final def apply(probability: Real,
                  modules:     TraversableOnce[ModuleBuilder])
  : DropPathBuilder = apply(probability) ++= modules

  final def apply(probability: Real,
                  modules:     Array[ModuleBuilder])
  : DropPathBuilder = apply(probability) ++= modules

}

case object DropPathContext
  extends PredictContext {
}

final case class DropPathState(override val parent: InstanceState,
                               children:            Seq[InstanceState])
  extends ModuleState

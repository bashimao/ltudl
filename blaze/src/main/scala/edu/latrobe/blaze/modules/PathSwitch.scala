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
import scala.collection._
import scala.util.hashing._

abstract class PathSwitch[TBuilder <: PathSwitchBuilder[_]]
  extends Container[TBuilder] {

  final override val children
  : Seq[Module] = builder.children.map(
    _.build(inputHints, seed, weightBufferBuilder)
  )

  final override val outputHints
  : BuildHints = children.headOption.map(
    _.outputHints
  ).getOrElse(inputHints)

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
  final override val requiresMaintainingInputDuringForwardPropagation
  : Boolean = false

  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor,
                                         onEnter:        OnEnterPredict,
                                         onLeave:        OnLeavePredict)
  : (Tensor, PredictContext) = {
    // Determine child number.
    val childNo = doPredictEx(
      mode,
      inPlaceAllowed,
      input,
      reference,
      onEnter,
      onLeave
    )

    // Pipe data through sub-path.
    val output = children(childNo).predictEx(
      mode,
      input,
      reference,
      onEnter,
      onLeave
    )
    (output, PathSwitchContext(childNo))
  }

  protected def doPredictEx(mode:           Mode,
                            inPlaceAllowed: Boolean,
                            input:          Tensor,
                            reference:      Tensor,
                            onEnter:        OnEnterPredict,
                            onLeave:        OnLeavePredict)
  : Int

  final override protected def doPredictInv(output:  Tensor,
                                            context:  PredictContext,
                                            onLeave:  OnLeavePredict,
                                            contexts: mutable.Stack[PredictContext])
  : Tensor = context match {
    case PathSwitchContext(childNo) =>
      children(childNo).predictInvEx(
        output,
        onLeave,
        contexts
      )
    case _ =>
      throw new MatchError(context)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  final override protected def doDeriveGradients(context:       PredictContext,
                                                 error:         NextError,
                                                 sink:          ValueTensorBuffer,
                                                 onEnter:       OnEnterDeriveGradients,
                                                 onLeave:       OnLeaveDeriveGradients,
                                                 intermediates: mutable.Stack[Tensor],
                                                 contexts:      mutable.Stack[PredictContext])
  : NextError = context match {
    case PathSwitchContext(childNo) =>
      children(childNo).deriveGradientsEx(
        error,
        sink,
        onEnter,
        onLeave,
        intermediates,
        contexts
      )
    case _ =>
      throw new MatchError(context)
  }


  // ---------------------------------------------------------------------------
  //    State backup and retrieval.
  // ---------------------------------------------------------------------------
  override def state
  : ModuleState = PathSwitchState(super.state, children.map(_.state))

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: PathSwitchState =>
        SeqEx.foreach(
          children,
          state.children
        )(_.restoreState(_))
      case _ =>
        throw new MatchError(state)
    }
  }

}

abstract class PathSwitchBuilder[TThis <: PathSwitchBuilder[_]]
  extends ContainerBuilder[TThis] {

  final val children
  : mutable.Buffer[ModuleBuilder] = mutable.Buffer.empty

  final def +=(module: ModuleBuilder)
  : TThis = {
    require(module != null)
    children += module
    repr
  }

  final def ++=(modules: TraversableOnce[ModuleBuilder])
  : TThis = {
    require(modules.forall(_ != null))
    children ++= modules
    repr
  }

  override protected def doToString()
  : List[Any] = children.length :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), children.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: PathSwitchBuilder[_] =>
      children == other.children
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: PathSwitchBuilder[_] =>
        other.children.clear()
        other.children ++= children.map(_.copy)
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //   Weights / Building related.
  // ---------------------------------------------------------------------------
  final override def weightLayoutFor(hints:   BuildHints,
                                     builder: TensorLayoutBufferBuilder)
  : BuildHints = {
    var tmpHints = hints

    // Keep output of first set of output hints.
    val iter = children.iterator
    if (iter.hasNext) {
      val child = iter.next()
      tmpHints = child.weightLayoutFor(hints, builder)
    }

    // For remaining connections just process them.
    while (iter.hasNext) {
      val child = iter.next()
      child.weightLayoutFor(hints, builder)
    }

    tmpHints
  }

  final override def outputHintsFor(hints: BuildHints)
  : BuildHints = {
    // TODO: Need error checking here?
    children.headOption.map(
      _.outputHintsFor(hints)
    ).getOrElse(hints)
  }


  // ---------------------------------------------------------------------------
  //    Checking related
  // ---------------------------------------------------------------------------
  final override protected def doCheckEx(hints:        BuildHints,
                                         indentLevel:  Int,
                                         indentString: String)
  : (BuildHints, Long) = {
    var tmpHints = hints
    var noErrors = 0L

    // Keep output of first set of output hints.
    val iter = children.iterator
    if (iter.hasNext) {
      val child = iter.next()
      val tmp = child.checkEx(hints, indentLevel + 1, indentString)
      tmpHints = tmp._1
      noErrors += tmp._2
    }

    // For remaining connections just process them.
    while (iter.hasNext) {
      val child = iter.next()
      noErrors += child.check(hints, indentLevel + 1, indentString)
    }

    (tmpHints, noErrors)
  }


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  final override protected def doToGraphEx(node:      VertexGroup,
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
      val edge = Edge(input,inVertex, edgeStyle)
      for (hints <- hints) {
        edge.label = hints.toEdgeLabel
      }
      edgeSink += edge
    }

    // Create dummy vertex to simulate switch.
    val outVertex = Vertex.derive("Gate").setShape(NodeShape.Point)
    nodeSink += outVertex

    // For the first child we keep the output hints.
    var outHints = hints
    val iter = children.iterator
    if (iter.hasNext) {
      val child = iter.next()

      val (childHints, childOutputs) = child.toGraphEx(
        hints,
        Seq(inVertex),
        LineStyle.Dotted,
        nodeSink,
        edgeSink
      )

      // Connect child outputs to output vertex.
      for (childOutput <- childOutputs) {
        val edge = Edge(childOutput, outVertex, LineStyle.Dotted)
        for (childHints <- childHints) {
          edge.label = childHints.toEdgeLabel
        }
        edgeSink += edge
      }

      outHints = childHints
    }

    // Process children.
    while (iter.hasNext) {
      val child = iter.next()

      val (childHints, childOutputs) = child.toGraphEx(
        hints,
        Seq(inVertex),
        LineStyle.Dotted,
        nodeSink,
        edgeSink
      )

      // Connect child outputs to output vertex.
      for (childOutput <- childOutputs) {
        val edge = Edge(childOutput, outVertex, LineStyle.Dotted)
        for (childHints <- childHints) {
          edge.label = childHints.toEdgeLabel
        }
        edgeSink += edge
      }
    }


    // Only publish the output node to the next node.
    (outHints, Seq(outVertex))
  }

}

final case class PathSwitchContext(childNo: Int)
  extends PredictContext {
}

final case class PathSwitchState(override val parent: InstanceState,
                                 children:            Seq[InstanceState])
  extends ModuleState {
}

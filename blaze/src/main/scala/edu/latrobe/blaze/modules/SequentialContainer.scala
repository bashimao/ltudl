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

abstract class SequentialContainer[TBuilder <: SequentialContainerBuilder[_]]
  extends Container[TBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor,
                                   onEnter:        OnEnterPredict,
                                   onLeave:        OnLeavePredict)
  : (Tensor, PredictContext) = {
    var output = input
    for (child <- children) {
      output = child.predictEx(
        mode,
        output,
        reference,
        onEnter,
        onLeave
      )
    }
    (output, EmptyContext)
  }

  override protected def doPredictInv(output:  Tensor,
                                      context:  PredictContext,
                                      onLeave:  OnLeavePredict,
                                      contexts: mutable.Stack[PredictContext])
  : Tensor = {
    var input = output
    for (child <- reversedChildren) {
      val inp = child.predictInvEx(
        input,
        onLeave,
        contexts
      )
      // TODO: Why do we do this here?
      if ((inp ne input) && (inp ne output)) {
        input.close()
      }
      input = inp
    }
    input
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveGradients(context:  PredictContext,
                                           error:    NextError,
                                           sink:     ValueTensorBuffer,
                                           onEnter:  OnEnterDeriveGradients,
                                           onLeave:  OnLeaveDeriveGradients,
                                           tensors:  mutable.Stack[Tensor],
                                           contexts: mutable.Stack[PredictContext])
  : NextError = {
    var err = error
    for (child <- reversedChildren) {
      err = child.deriveGradientsEx(
        err,
        sink,
        onEnter,
        onLeave,
        tensors,
        contexts
      )
    }
    err
  }

}

abstract class SequentialContainerBuilder[TThis <: SequentialContainerBuilder[_]]
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
    case other: SequentialContainerBuilder[_] =>
      children == other.children
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: SequentialContainerBuilder[_] =>
        other.children.clear()
        other.children ++= children.map(_.copy)
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Weights and binding related
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = {
    var tmpHints = hints
    for (child <- children) {
      child.weightLayoutFor(tmpHints, builder)
      tmpHints = child.outputHintsFor(tmpHints)
    }
    tmpHints
  }

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = {
    var tmpHints = hints
    for (child <- children) {
      tmpHints = child.outputHintsFor(tmpHints)
    }
    tmpHints
  }


  // ---------------------------------------------------------------------------
  //    Checking related
  // ---------------------------------------------------------------------------
  override protected def doCheckEx(hints:         BuildHints,
                                   indentLevel:   Int,
                                   indentString:  String)
  : (BuildHints, Long) = {
    var tmpHints = hints
    var noErrors = 0L
    for (child <- children) {
      val tmp = child.checkEx(tmpHints, indentLevel + 1, indentString)
      tmpHints = tmp._1
      noErrors += tmp._2
    }
    (tmpHints, noErrors)
  }


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
    var inpHints = hints
    var outputs  = inputs

    // Carry over edge style for first child.
    val iter = children.iterator
    if (iter.hasNext) {
      val child = iter.next()
      val tmp = child.toGraphEx(
        inpHints,
        outputs,
        edgeStyle,
        nodeSink,
        edgeSink
      )
      inpHints = tmp._1
      outputs  = tmp._2
    }

    // Remaining children have are directly connected.
    while (iter.hasNext) {
      val child = iter.next()
      val tmp = child.toGraphEx(
        inpHints,
        outputs,
        LineStyle.Solid,
        nodeSink,
        edgeSink
      )
      inpHints = tmp._1
      outputs  = tmp._2
    }

    (inpHints, outputs)
  }

}

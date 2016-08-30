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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.io.graph._
import scala.collection._

/**
 * Leaf layers implement a more strict interface. This is suitable for modules
 * without any children.
 */
abstract class Layer[TBuilder <: ModuleExBuilder[_]]
  extends ModuleEx[TBuilder] {

  // ---------------------------------------------------------------------------
  //    Traversal related.
  // --------------------------------------------------------------------------
  final override protected def doTouch(callbackFn: Module => Unit)
  : Unit = {}


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
  : (Tensor, PredictContext) = doPredict(mode, inPlaceAllowed, input, reference)

  protected def doPredict(mode:           Mode,
                          inPlaceAllowed: Boolean,
                          input:          Tensor,
                          reference:      Tensor)
  : (Tensor, PredictContext)

  final override protected def doPredictInv(output:   Tensor,
                                            context:  PredictContext,
                                            onLeave:  OnLeavePredict,
                                            contexts: mutable.Stack[PredictContext])
  : Tensor = doPredictInv(output, context)

  protected def doPredictInv(output: Tensor, context: PredictContext)
  : Tensor


  // ---------------------------------------------------------------------------
  //    Cost/gradient related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveGradients(input:     Tensor,
                                                 reference: Tensor,
                                                 output:    Tensor,
                                                 context:   PredictContext,
                                                 error:     NextError,
                                                 sink:      ValueTensorBuffer,
                                                 onEnter:   OnEnterDeriveGradients,
                                                 onLeave:   OnLeaveDeriveGradients,
                                                 tensors:   mutable.Stack[Tensor],
                                                 contexts:  mutable.Stack[PredictContext])
  : NextError = doDeriveGradients(
    input,
    reference,
    output,
    context,
    error,
    sink
  )

  protected def doDeriveGradients(input:     Tensor,
                                  reference: Tensor,
                                  output:    Tensor,
                                  context:   PredictContext,
                                  error:     NextError,
                                  sink:      ValueTensorBuffer)
  : NextError

}

abstract class LayerBuilder[TThis <: LayerBuilder[_]]
  extends ModuleExBuilder[TThis] {

  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  final override def toGraphEx(hints:     Option[BuildHints],
                               inputs:    Seq[Vertex],
                               edgeStyle: LineStyle,
                               nodeSink:  mutable.Buffer[Node],
                               edgeSink:  mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex]) = {
    // Create the self-vertex.
    val node = Vertex.derive(toString("\n", ""))
    nodeSink += node
    doToGraphEx(node)

    // Add the vertex and edges with all inputs.
    for (input <- inputs) {
      val edge = Edge(input, node, edgeStyle)
      for (hints <- hints) {
        edge.label = hints.toEdgeLabel
      }
      edgeSink += edge
    }

    // Since this is a layer, it is the only output vertex.
    val outHints = hints.map(outputHintsFor)
    (outHints, Seq(node))
  }

  // TODO: Colors!
  protected def doToGraphEx(node: Vertex)
  : Unit = {}


  // ---------------------------------------------------------------------------
  //    Checking related
  // ---------------------------------------------------------------------------
  override protected def doCheckEx(hints:        BuildHints,
                                   indentLevel:  Int,
                                   indentString: String)
  : (BuildHints, Long) = {
    var noErrors    = 0L
    var outputHints = hints
    try {
      outputHints = outputHintsFor(outputHints)
    }
    catch {
      case ex: Exception =>
        val builder = StringBuilder.newBuilder
        var i = 0
        while (i < indentLevel) {
          builder ++= indentString
          i += 1
        }
        builder ++= s"outputHintsFor($outputHints) Exception: $ex%s"
        logger.info(builder.result())
        noErrors += 1L
    }
    (outputHints, noErrors)
  }

}

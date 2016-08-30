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

package edu.latrobe.blaze.batchpools

import edu.latrobe.blaze._
import edu.latrobe.io.graph._
import scala.collection._

abstract class IndependentBatchPool[TBuilder <: IndependentBatchPoolBuilder[_]]
  extends BatchPoolEx[TBuilder] {

  final lazy val outputHints
  : BuildHints = inputHints

}

abstract class IndependentBatchPoolBuilder[TThis <: IndependentBatchPoolBuilder[_]]
  extends BatchPoolExBuilder[TThis] {

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
    val vertex = Vertex.derive(toString("\n", ""))
    nodeSink += vertex
    doToGraphEx(vertex)

    // Add the vertex and edges with all inputs.
    for (input <- inputs) {
      val edge = Edge(input, vertex, edgeStyle)
      for (hints <- hints) {
        edge.label = hints.toEdgeLabel
      }
      edgeSink += edge
    }

    // Since this is an independent pool. This is the root of the pipeline.
    (hints, Seq(vertex))
  }

  // TODO: Colors!
  protected def doToGraphEx(vertex: Vertex)
  : Unit = {}

}

final class IndependentBatchPoolDrawContext(override val batch: Batch)
  extends BatchPoolDrawContext {

  override def close()
  : Unit = {}

}

object IndependentBatchPoolDrawContext {

  final def apply(batch: Batch)
  : IndependentBatchPoolDrawContext = new IndependentBatchPoolDrawContext(batch)

}

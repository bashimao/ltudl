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
import scala.util.hashing._

/**
  * Wraps around another batch series and takes the time it needs.
  */
final class BenchmarkedBatchPool(override val builder: BenchmarkedBatchPoolBuilder,
                                 override val seed:    InstanceSeed,
                                 override val source:  BatchPool)
  extends DependentBatchPool[BenchmarkedBatchPoolBuilder]
    with BenchmarkEnabled {

  override val outputHints
  : BuildHints = inputHints

  override def draw()
  : BatchPoolDrawContext = doBenchmark("draw", source.draw())

}

final class BenchmarkedBatchPoolBuilder
  extends DependentBatchPoolBuilder[BenchmarkedBatchPoolBuilder] {

  override def repr
  : BenchmarkedBatchPoolBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[BenchmarkedBatchPoolBuilder]


  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), source.hashCode())

  override protected def doCopy()
  : BenchmarkedBatchPoolBuilder = BenchmarkedBatchPoolBuilder()


  // ---------------------------------------------------------------------------
  //     Build functions.
  // ---------------------------------------------------------------------------
  override protected def doBuild(source: BatchPool,
                                 seed:   InstanceSeed)
  : BenchmarkedBatchPool = new BenchmarkedBatchPool(this, seed, source)


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  override protected def doToGraphEx(hints:    Option[BuildHints],
                                     inputs:   Seq[Vertex],
                                     nodeSink: mutable.Buffer[Node],
                                     edgeSink: mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex]) = {
    // Create self-vertex.
    val outVertex = Vertex.derive(toString("\n", ""))
    nodeSink += outVertex

    // Add an edge to the vertex.
    for (input <- inputs) {
      val edge = Edge(input, outVertex, LineStyle.Solid)
      for (hints <- hints) {
        edge.label = hints.toEdgeLabel
      }
      edgeSink += edge
    }

    (hints, Seq(outVertex))
  }

}

object BenchmarkedBatchPoolBuilder {

  final def apply()
  : BenchmarkedBatchPoolBuilder = new BenchmarkedBatchPoolBuilder

  final def apply(source: BatchPoolBuilder)
  : BenchmarkedBatchPoolBuilder = apply().setSource(source)

}

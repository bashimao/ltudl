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
import edu.latrobe.io.graph._
import scala.collection._

/**
 * Whatever you do, do not make this a traversable, because the IntelliJ
 * debugger would execute that when it hits a breakpoint.
 */
abstract class BatchPool
  extends InstanceEx[BatchPoolBuilder] {

  /**
    * Input hints passed through at creation time. Might be interesting to augmenters.
    * Should be implemented as a val.
    */
  def inputHints
  : BuildHints

  /**
    * Output hints from this pool. Can be implemented any way you want.
    */
  def outputHints
  : BuildHints


  /**
    * Draws the next batch from the pool.
    *
    * Returns None if the pool has been depleted.
    */
  def draw()
  : BatchPoolDrawContext

  /**
   * Grab works similar to take, but it will instigate updates on the record
   * series and does not ensure the number of records returned.
   */
  /*
  final def take(noBatchesMax: Int)
  : Array[Batch] = {
    val buffer = mutable.ArrayBuffer.empty[Batch]
    buffer.sizeHint(noBatchesMax)
    while (buffer.length < noBatchesMax) {
      draw() match {
        case Some(batch) =>
          val context =
          buffer +=
        case None =>
          return buffer.toArray
      }
    }
    buffer.toArray
  }
  */


  // ---------------------------------------------------------------------------
  //    Handy functions for batch jobs.
  // ---------------------------------------------------------------------------
  final def foldLeft[T](z: T)
                       (fn: (T, Batch) => T)
  : T = {
    var result = z
    foreach(
      batch => result = fn(result, batch)
    )
    result
  }

  final def foreach(fn: Batch => Unit)
  : Unit = {
    while (true) {
      using(draw())(ctx => {
        if (ctx.isEmpty) {
          return
        }
        fn(ctx.batch)
      })
    }
    throw new UnknownError
  }

  /**
   * Use with caution. Some pools are infinite!
   */
  /*
  final def toArray: Array[Batch] = {
    val builder = Array.newBuilder[Batch]
    foreach(builder += _)
    builder.result()
  }
  */

  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : BatchPoolState = BatchPoolStateEx(super.state)

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: BatchPoolStateEx =>
      case _ =>
        throw new MatchError(state)
    }
  }

}

abstract class BatchPoolBuilder
  extends InstanceExBuilder2[BatchPoolBuilder, BatchPool, TensorLayout, Iterable[Batch]] {

  final def build(layoutHint: TensorLayout,
                  sample:     Batch)
  : BatchPool = build(layoutHint, sample, InstanceSeed.default)

  final def build(layoutHint: TensorLayout,
                  sample:     Batch,
                  seed:       InstanceSeed)
  : BatchPool = build(layoutHint, Array(sample), seed)


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  final def toGraph(hints: Option[BuildHints] = None)
  : Graph = {
    val result = Graph()
    val input = Vertex.derive("Data Source")
    toGraphEx(hints, Seq(input), LineStyle.Solid, result.nodes, result.edges)
    result
  }

  /**
    * @param nodeSink Vertices and vertex groups will end up here.
    * @param edgeSink Edge information ends up here.
    * @return The vertex for the current object.
    */
  def toGraphEx(hints:     Option[BuildHints],
                inputs:    Seq[Vertex],
                edgeStyle: LineStyle,
                nodeSink:  mutable.Buffer[Node],
                edgeSink:  mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex])

}

abstract class BatchPoolEx[TBuilder <: BatchPoolExBuilder[_]]
  extends BatchPool {

  override def builder
  : TBuilder

}

abstract class BatchPoolExBuilder[TThis <: BatchPoolExBuilder[_]]
  extends BatchPoolBuilder {

  override def repr
  : TThis

  override protected def doCopy()
  : TThis

}

abstract class BatchPoolState
  extends InstanceState

final case class BatchPoolStateEx(override val parent: InstanceState)
  extends BatchPoolState

abstract class BatchPoolDrawContext
  extends AutoCloseable {

  /**
    * Should implement this as constructor argument.
    */
  def batch
  : Batch

  final def isEmpty
  : Boolean = batch == null

  final def nonEmpty
  : Boolean = batch != null

}

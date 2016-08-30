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

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.io.graph._

import scala.Seq
import scala.collection._
import scala.util.hashing._

/**
  * Encapsulates the concept of repeating that was previously coded in the
  * baseclass.
  */
final class Repeat(override val builder: RepeatBuilder,
                   override val seed:    InstanceSeed,
                   override val source:  BatchPool)
  extends DependentBatchPool[RepeatBuilder] {

  val noRepetitions
  : Int = builder.noRepetitions

  override val outputHints
  : BuildHints = {
    val noSamples = noRepetitions * inputHints.layout.noSamples
    inputHints.derive(
      inputHints.platform,
      inputHints.layout.derive(noSamples),
      inputHints.referencePlatform,
      inputHints.referenceLayout.derive(noSamples)
    )
  }

  override def draw()
  : BatchPoolDrawContext = {
    // Pull one batch from source.
    val ctx = source.draw()
    if (ctx.isEmpty) {
      return ctx
    }
    val inp = ctx.batch

    // Fill up array.
    val srcN = ArrayEx.fill(
      noRepetitions - 1,
      inp
    )

    // Concatenate batches.
    val out = inp.concat(srcN)

    // Concat forces copy. Hence, decoupling from source is safe.
    ctx.close()

    DependentBatchPoolDrawContext(out)
  }

}

final class RepeatBuilder
  extends DependentBatchPoolBuilder[RepeatBuilder] {

  override def repr
  : RepeatBuilder = this

  private var _noRepetitions
  : Int = 10

  def noRepetitions
  : Int = _noRepetitions

  def noRepetitions_=(value: Int)
  : Unit = {
    require(value > 0)
    _noRepetitions = value
  }

  def setNoRepetitions(value: Int)
  : RepeatBuilder = {
    noRepetitions_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _noRepetitions :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _noRepetitions.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RepeatBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: RepeatBuilder =>
      _noRepetitions == other._noRepetitions
    case _ =>
      false
  })

  override protected def doCopy()
  : RepeatBuilder = RepeatBuilder()

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: RepeatBuilder =>
        // TODO: Fix this: source.copyTo(other.source)
        other._noRepetitions = _noRepetitions
      case _ =>
    }
  }

  override protected def doBuild(source: BatchPool, seed: InstanceSeed)
  : Repeat = new Repeat(this, seed, source)


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  override protected def doToGraphEx(hints:   Option[BuildHints],
                                     inputs:   Seq[Vertex],
                                     nodeSink: mutable.Buffer[Node],
                                     edgeSink: mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex]) = {
    // Create self-vertex.
    val vertex = Vertex.derive(toString("\n", ""))
    nodeSink += vertex

    // Add the vertex and edges with all inputs.
    for (input <- inputs) {
      val edge = Edge(input, vertex, LineStyle.Solid)
      for (hints <- hints) {
        edge.label = hints.toEdgeLabel
      }
      edgeSink += edge
    }

    // Compute output hints.
    val outHints = hints.map(hints => hints.derive(
      hints.platform,
      hints.layout.derive(
        hints.layout.noSamples * _noRepetitions
      ),
      hints.referencePlatform,
      hints.referenceLayout.derive(
        hints.referenceLayout.noSamples * _noRepetitions
      )
    ))

    (outHints, Seq(vertex))
  }

}

object RepeatBuilder {

  final def apply()
  : RepeatBuilder = new RepeatBuilder

  final def apply(source: BatchPoolBuilder)
  : RepeatBuilder = apply().setSource(source)

  final def apply(source: BatchPoolBuilder, noRepetitions: Int)
  : RepeatBuilder = apply(source).setNoRepetitions(noRepetitions)

}

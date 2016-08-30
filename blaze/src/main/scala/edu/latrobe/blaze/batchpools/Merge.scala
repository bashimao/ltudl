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
import edu.latrobe.time._
import scala.collection._
import scala.util.hashing._

final class Merge(override val builder: MergeBuilder,
                  override val seed:    InstanceSeed,
                  override val source:  BatchPool)
  extends DependentBatchPool[MergeBuilder] {

  private val noBatches
  : Int = builder.noBatches

  override val outputHints
  : BuildHints = {
    val noSamples = noBatches * inputHints.layout.noSamples
    inputHints.derive(
      inputHints.platform,
      inputHints.layout.derive(noSamples),
      inputHints.referencePlatform,
      inputHints.referenceLayout.derive(noSamples)
    )
  }

  override def draw()
  : BatchPoolDrawContext = {
    // Pull first batch from source.
    val ctx0 = source.draw()
    if (ctx0.isEmpty) {
      return ctx0
    }
    val inp0 = ctx0.batch

    // Fill up array with more batches.
    val ctxBuilderN = Array.newBuilder[BatchPoolDrawContext]
    var i           = 1
    while (i < noBatches) {
      val ctx = source.draw()
      if (ctx.nonEmpty) {
        ctxBuilderN += ctx
      }
      i += 1
    }
    val ctxN = ctxBuilderN.result()
    val inpN = ArrayEx.map(
      ctxN
    )(_.batch)

    // Concatenate batches.
    val out = inp0.concat(inpN)

    // Concat forces copy. Hence, decoupling from source is safe.
    ctx0.close()
    ArrayEx.foreach(
      ctxN
    )(_.close())

    DependentBatchPoolDrawContext(out)
  }

}

final class MergeBuilder
  extends DependentBatchPoolBuilder[MergeBuilder] {

  override def repr
  : MergeBuilder = this

  private var _noBatches
  : Int = 128

  def noBatches
  : Int = _noBatches

  def noBatches_=(value: Int)
  : Unit = {
    require(value > 0)
    _noBatches = value
  }

  def setNoBatches(value: Int)
  : MergeBuilder = {
    noBatches_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _noBatches :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _noBatches.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MergeBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: MergeBuilder =>
      _noBatches == other._noBatches
    case _ =>
      false
  })

  override protected def doCopy()
  : MergeBuilder = MergeBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: MergeBuilder =>
        other._noBatches = _noBatches
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //   Record set construction
  // ---------------------------------------------------------------------------
  override protected def doBuild(source: BatchPool, seed: InstanceSeed)
  : Merge = new Merge(this, seed, source)


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  override protected def doToGraphEx(hints:    Option[BuildHints],
                                     inputs:   Seq[Vertex],
                                     nodeSink: mutable.Buffer[Node],
                                     edgeSink: mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex]) = {
    // Create self-vertex.
    val vertex = Vertex.derive(toString("\n", ""))
    nodeSink += vertex

    // Add the vertex and edges with all inputs.
    inputs.foreach(input => {
      val edge = Edge(input, vertex, LineStyle.Solid)
      edgeSink += edge
    })

    // Compute output hints.
    val outHints = hints.map(hints => {
      val noSamples = noBatches * hints.layout.noSamples
      hints.derive(
        hints.platform,
        hints.layout.derive(noSamples),
        hints.referencePlatform,
        hints.referenceLayout.derive(noSamples)
      )
    })

    (outHints, Seq(vertex))
  }

}

object MergeBuilder {

  final def apply()
  : MergeBuilder = new MergeBuilder

  final def apply(source: BatchPoolBuilder)
  : MergeBuilder = apply().setSource(source)

  final def apply(source: BatchPoolBuilder, noBatches: Int)
  : MergeBuilder = apply(source).setNoBatches(noBatches)

}

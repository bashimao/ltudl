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
import scala.collection._
import scala.util.hashing._

abstract class DependentBatchPool[TBuilder <: DependentBatchPoolBuilder[_]]
  extends BatchPoolEx[TBuilder] {

  /**
    * Must be provided as constructor argument.
    */
  def source
  : BatchPool

  final override val inputHints
  : BuildHints = source.outputHints

  final val inputLayout
  : TensorLayout = inputHints.layout

  final val inputReferenceLayout
  : TensorLayout = inputHints.referenceLayout

  override protected def doClose()
  : Unit = {
    source.close()
    super.doClose()
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : BatchPoolState = DependentBatchPoolState(super.state, source.state)

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: DependentBatchPoolState =>
        source.restoreState(state.source)
      case _ =>
        throw new MatchError(state)
    }
  }

}

abstract class DependentBatchPoolBuilder[TThis <: DependentBatchPoolBuilder[_]]
  extends BatchPoolExBuilder[TThis] {

  final private var _source
  : BatchPoolBuilder = ChooseAtRandomBuilder()

  final def source
  : BatchPoolBuilder = _source

  final def source_=(value: BatchPoolBuilder)
  : Unit = {
    require(value != null)
    _source = value
  }

  final def setSource(value: BatchPoolBuilder)
  : TThis = {
    source_=(value)
    repr
  }

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _source.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: DependentBatchPoolBuilder[TThis] =>
      _source == other._source
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: DependentBatchPoolBuilder[TThis] =>
        other._source = _source.copy
      case _ =>
    }
  }

  // ---------------------------------------------------------------------------
  //   Record set construction
  // ---------------------------------------------------------------------------
  final override def build(layoutHint: TensorLayout,
                           samples:    Iterable[Batch],
                           seed:       InstanceSeed)
  : BatchPool = doBuild(
    _source.build(layoutHint, samples, seed),
    seed
  )

  protected def doBuild(source: BatchPool, seed: InstanceSeed)
  : BatchPool


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _source.permuteSeeds(fn)
  }


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  final override def toGraphEx(hints:     Option[BuildHints],
                               inputs:    Seq[Vertex],
                               edgeStyle: LineStyle,
                               nodeSink:  mutable.Buffer[Node],
                               edgeSink:  mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex]) = {
    // Create source nodes.
    val (srcHints, srcOutputs) = _source.toGraphEx(
      hints,
      inputs,
      edgeStyle,
      nodeSink,
      edgeSink
    )

    // Create the current node.
    doToGraphEx(
      srcHints,
      srcOutputs,
      nodeSink,
      edgeSink
    )
  }

  protected def doToGraphEx(hints:    Option[BuildHints],
                            inputs:   Seq[Vertex],
                            nodeSink: mutable.Buffer[Node],
                            edgeSink: mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex])

}

final case class DependentBatchPoolState(override val parent: InstanceState,
                                         source:              InstanceState)
  extends BatchPoolState

final class DependentBatchPoolDrawContext(override val batch: Batch)
  extends BatchPoolDrawContext {
  require(batch != null)

  override def close()
  : Unit = batch.close()

}

object DependentBatchPoolDrawContext {

  final def apply(batch: Batch)
  : DependentBatchPoolDrawContext = new DependentBatchPoolDrawContext(batch)

}

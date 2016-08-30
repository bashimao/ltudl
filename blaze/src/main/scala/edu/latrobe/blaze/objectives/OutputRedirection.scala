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

package edu.latrobe.blaze.objectives

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.sinks._
import edu.latrobe.io.FileHandle
import edu.latrobe.time._

import scala.util.hashing._

final class OutputRedirection(override val builder: OutputRedirectionBuilder,
                              override val seed:    InstanceSeed)
  extends DependentObjectiveEx[OutputRedirectionBuilder] {
  require(builder != null && seed != null)

  val sink
  : Sink = builder.sink.build(seed)

  override protected def doClose()
  : Unit = {
    sink.close()
    super.doClose()
  }

  override protected def doEvaluate(sink:                Sink,
                                    optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Option[ObjectiveEvaluationResult] = {
    super.doEvaluate(
      this.sink,
      optimizer, runBeginIterationNo, runBeginTime, runNoSamples,
      model,
      batch, output, value
    )
  }

}

final class OutputRedirectionBuilder
  extends DependentObjectiveExBuilder[OutputRedirectionBuilder] {

  override def repr
  : OutputRedirectionBuilder = this

  private var _sink
  : SinkBuilder = StdErrSinkBuilder()

  def sink
  : SinkBuilder = _sink

  def sink_=(value: SinkBuilder)
  : Unit = {
    require(value != null)
    _sink = value
  }

  def setSink(value: SinkBuilder)
  : OutputRedirectionBuilder = {
    sink_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _sink :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _sink.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[OutputRedirectionBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: OutputRedirectionBuilder =>
      _sink == other._sink
    case _ =>
      false
  })

  override protected def doCopy()
  : OutputRedirectionBuilder = OutputRedirectionBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: OutputRedirectionBuilder =>
        other._sink = _sink.copy
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : OutputRedirection = new OutputRedirection(this, seed)

  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _sink.permuteSeeds(fn)
  }

}

object OutputRedirectionBuilder {

  final def apply()
  : OutputRedirectionBuilder = new OutputRedirectionBuilder

  final def apply(sink: SinkBuilder)
  : OutputRedirectionBuilder = apply().setSink(sink)

  final def apply(sink:   SinkBuilder,
                  child0: ObjectiveBuilder)
  : OutputRedirectionBuilder = apply(sink) += child0

  final def apply(sink:   SinkBuilder,
                  child0: ObjectiveBuilder,
                  childN: ObjectiveBuilder*)
  : OutputRedirectionBuilder = apply(sink, child0) ++= childN

  final def apply(sink:   SinkBuilder,
                  childN: TraversableOnce[ObjectiveBuilder])
  : OutputRedirectionBuilder = apply(sink) ++= childN

}
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
import edu.latrobe.blaze.modules._
import edu.latrobe.io.graph._
import scala.collection._
import scala.util.hashing._

/**
  * Applies all child augmenters one after another.
  */
final class BatchAugmenter(override val builder: BatchAugmenterBuilder,
                           override val seed:    InstanceSeed,
                           override val source:  BatchPool)
  extends Augmenter[BatchAugmenterBuilder] {

  val module
  : Module = builder.module.build(inputHints, seed)

  override val outputHints
  : BuildHints = module.outputHints

  override protected def doClose()
  : Unit = {
    module.close()
    super.doClose()
  }

  override def draw()
  : BatchPoolDrawContext = {
    val ctx = source.draw()
    if (ctx.isEmpty) {
      return ctx
    }

    // Project
    val inp = ctx.batch
    val out = module.project(mode, inp)

    BatchAugmenterDrawContext(out, ctx)
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : AugmenterState = AugmenterState(
    super.state,
    module.state
  )

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: AugmenterState =>
        module.restoreState(state.module)
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class BatchAugmenterBuilder
  extends AugmenterBuilder[BatchAugmenterBuilder] {

  override def repr
  : BatchAugmenterBuilder = this

  private var _module
  : ModuleBuilder = IdentityBuilder()

  def module
  : ModuleBuilder = _module

  def module_=(value: ModuleBuilder)
  : Unit = {
    require(value != null)
    _module = value
  }

  def setModule(value: ModuleBuilder)
  : BatchAugmenterBuilder = {
    module_=(value)
    this
  }

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _module.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[BatchAugmenterBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BatchAugmenterBuilder =>
      _module == other._module
    case _ =>
      false
  })

  override protected def doCopy()
  : BatchAugmenterBuilder = BatchAugmenterBuilder()

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: BatchAugmenterBuilder =>
        other._module = _module.copy
      case _ =>
    }
  }

  override protected def doBuild(source: BatchPool,
                                 seed:   InstanceSeed)
  : BatchAugmenter = new BatchAugmenter(this, seed, source)

  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _module.permuteSeeds(fn)
  }


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  override protected def doToGraphExEx(hints:     Option[BuildHints],
                                       inputs:   Seq[Vertex],
                                       nodeSink: mutable.Buffer[Node],
                                       edgeSink: mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex]) = {
    // Connect incoming edges to model.
    _module.toGraphEx(hints, inputs, LineStyle.Solid, nodeSink, edgeSink)
  }

}

object BatchAugmenterBuilder {

  final def apply()
  : BatchAugmenterBuilder = new BatchAugmenterBuilder

  final def apply(source: BatchPoolBuilder)
  : BatchAugmenterBuilder = apply().setSource(source)

  final def apply(source:  BatchPoolBuilder,
                  module0: ModuleBuilder)
  : BatchAugmenterBuilder = apply(source).setModule(module0)

  final def apply(source:  BatchPoolBuilder,
                  module0: ModuleBuilder,
                  modules: ModuleBuilder*)
  : BatchAugmenterBuilder = apply(source, SequenceBuilder(module0) ++= modules)

  final def apply(source:   BatchPoolBuilder,
                  modules: TraversableOnce[ModuleBuilder])
  : BatchAugmenterBuilder = apply(source, SequenceBuilder(modules))

}

final class BatchAugmenterDrawContext(override val batch: Batch,
                                      private val source: BatchPoolDrawContext)
  extends BatchPoolDrawContext {

  override def close()
  : Unit = {
    // TODO: Think this though carefully. May fail in some situations!
    if (batch.input ne source.batch.input) {
      if (batch.input ne source.batch.output) {
        batch.close()
      }
    }
    source.close()
  }

}

object BatchAugmenterDrawContext {

  final def apply(batch: Batch, source: BatchPoolDrawContext)
  : BatchAugmenterDrawContext = new BatchAugmenterDrawContext(batch, source)

}
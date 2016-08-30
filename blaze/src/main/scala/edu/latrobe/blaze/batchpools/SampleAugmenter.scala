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
final class SampleAugmenter(override val builder: SampleAugmenterBuilder,
                            override val seed:    InstanceSeed,
                            override val source:  BatchPool)
  extends Augmenter[SampleAugmenterBuilder] {

  val moduleInputHints
  : BuildHints = inputHints.derive(
    inputHints.platform,          inputLayout.derive(1),
    inputHints.referencePlatform, inputReferenceLayout.derive(1)
  )

  val module
  : Module = builder.module.build(moduleInputHints, seed)

  val moduleOutputHints
  : BuildHints = module.outputHints

  val moduleOutputLayout
  : TensorLayout = moduleOutputHints.layout

  val moduleOutputReferenceLayout
  : TensorLayout = moduleOutputHints.referenceLayout

  override val outputHints
  : BuildHints = moduleOutputHints.derive(
    moduleOutputHints.platform,
    moduleOutputLayout.derive(
      inputLayout.noSamples * moduleOutputLayout.noSamples
    ),
    moduleOutputHints.referencePlatform,
    moduleOutputReferenceLayout.derive(
      inputReferenceLayout.noSamples * moduleOutputReferenceLayout.noSamples
    )
  )

  override def draw()
  : BatchPoolDrawContext = {
    val ctx = source.draw()
    if (ctx.isEmpty) {
      return ctx
    }
    val inp = ctx.batch

    val tmpSamples = inp.split()
    val tmpOutputs = ArrayEx.map(tmpSamples)(sample => {
      val prediction = module.predict(mode, sample).dropIntermediates()
      prediction.output
    })

    val out = inp.derive(
      tmpOutputs(0).concat(tmpOutputs.tail)
    )

    // Safe after concat! Must actually do this to avoid memory issues.
    ArrayEx.foreach(tmpSamples)(_.close())
    ArrayEx.foreach(tmpOutputs)(_.tryClose())

    DependentBatchPoolDrawContext(out)
  }

}

final class SampleAugmenterBuilder
  extends AugmenterBuilder[SampleAugmenterBuilder] {

  override def repr
  : SampleAugmenterBuilder = this

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
  : SampleAugmenterBuilder = {
    module_=(value)
    this
  }

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _module.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SampleAugmenterBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: SampleAugmenterBuilder =>
      _module == other._module
    case _ =>
      false
  })

  override protected def doCopy()
  : SampleAugmenterBuilder = SampleAugmenterBuilder()

  override protected def doBuild(source: BatchPool,
                                 seed:   InstanceSeed)
  : SampleAugmenter = new SampleAugmenter(this, seed, source)

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
    // Compute sample hints.
    val sampleInputHints = hints.map(hints => hints.derive(
      hints.platform,          hints.layout.derive(1),
      hints.referencePlatform, hints.referenceLayout.derive(1)
    ))

    // Connect incoming edges to model.
    val (sOutHints, outputs) = _module.toGraphEx(
      sampleInputHints,
      inputs,
      LineStyle.Solid,
      nodeSink,
      edgeSink
    )

    // Now model edges to dummy node representing merge.
    val outVertex = Vertex.derive("Fuse").setShape(NodeShape.Point)
    nodeSink += outVertex
    for (output <- outputs) {
      val edge = Edge(output, outVertex, LineStyle.Solid)
      for (sOutHints <- sOutHints) {
        edge.label = sOutHints.toEdgeLabel
      }
      edgeSink += edge
    }

    // Compute output hints.
    val outHints = sOutHints.map(sOutHints => {
      val inpHints = hints.get
      sOutHints.derive(
        sOutHints.platform,
        sOutHints.layout.derive(
          sOutHints.layout.noSamples * inpHints.layout.noSamples
        ),
        sOutHints.referencePlatform,
        sOutHints.referenceLayout.derive(
          sOutHints.referenceLayout.noSamples * inpHints.referenceLayout.noSamples
        )
      )
    })

    // Present the outVertex to the next node.
    (outHints, Seq(outVertex))
  }

}

object SampleAugmenterBuilder {

  final def apply()
  : SampleAugmenterBuilder = new SampleAugmenterBuilder

  final def apply(source: BatchPoolBuilder)
  : SampleAugmenterBuilder = apply().setSource(source)

  final def apply(source: BatchPoolBuilder, module: ModuleBuilder)
  : SampleAugmenterBuilder = apply(source).setModule(module)

}

/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2015 Matthias Langer (t3l@threelights.de)
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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.io.graph._
import java.util.UUID
import scala.collection._
import scala.util.hashing._

/**
  * A module that uses another module to determine a new reference during FPROP
  * which it then injects as reference while executing yet another module.
  *
  * You should provide descent cleanup code in order to make this not spilling
  * memory when using custom converter functions. Actually this is super
  * tricky to use in some situations. In-depth understanding of the tensor
  * dependency engine is highly recommended when using custom functions.
  */
final class ToggleReference(override val builder:             ToggleReferenceBuilder,
                            override val inputHints:          BuildHints,
                            override val seed:                InstanceSeed,
                            override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends SequentialContainer[ToggleReferenceBuilder] {

  val converter
  : Module = builder.converter.build(inputHints, seed, weightBufferBuilder)

  val immediateHints
  : BuildHints = {
    val tmpHints = converter.outputHints
    inputHints.derive(
      inputHints.platform, inputHints.layout,
      tmpHints.platform,   tmpHints.layout
    )
  }

  override val (children, outputHints)
  : (Seq[Module], BuildHints) = {
    var tmpHints = immediateHints
    val modules = builder.children.map(child => {
      val tmp = child.build(tmpHints, seed, weightBufferBuilder)
      tmpHints = tmp.outputHints
      tmp
    })

    tmpHints = tmpHints.derive(
      tmpHints.platform,            tmpHints.layout,
      inputHints.referencePlatform, inputHints.referenceLayout
    )

    (modules, tmpHints)
  }

  override def parameters
  : Map[UUID, Parameter] = super.parameters ++ converter.parameters

  override protected def doClose()
  : Unit = {
    children.foreach(
      _.close()
    )
    converter.close()
    super.doClose()
  }


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override val requiresMaintainingInputDuringForwardPropagation
  : Boolean = true

  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor,
                                   onEnter:        OnEnterPredict,
                                   onLeave:        OnLeavePredict)
  : (Tensor, PredictContext) = {
    // Run through converter.
    val pred = converter.predict(mode, input, reference).dropIntermediates()

    // Inject into parent.
    val (out, ctx) = super.doPredict(
      mode,
      inPlaceAllowed,
      input,
      pred.output,
      onEnter,
      onLeave
    )

    out -> ToggleReferenceContext(ctx, pred)
  }

  override protected def doPredictInv(output:   Tensor,
                                      context:  PredictContext,
                                      onLeave:  OnLeavePredict,
                                      contexts: mutable.Stack[PredictContext])
  : Tensor = context match {
    case ToggleReferenceContext(ctx, pred) =>
      super.doPredictInv(output, ctx, onLeave, contexts)
    case _ =>
      throw new UnsupportedOperationException
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  override protected def doDeriveGradients(context:  PredictContext,
                                           error:    NextError,
                                           sink:     ValueTensorBuffer,
                                           onEnter:  OnEnterDeriveGradients,
                                           onLeave:  OnLeaveDeriveGradients,
                                           tensors:  mutable.Stack[Tensor],
                                           contexts: mutable.Stack[PredictContext])
  : NextError = context match {
    case ToggleReferenceContext(ctx, pred) =>
      super.doDeriveGradients(
        ctx, error, sink, onEnter, onLeave, tensors, contexts
      )
    case _ =>
      throw new UnsupportedOperationException
  }


  // ---------------------------------------------------------------------------
  //    State backup and retrieval.
  // ---------------------------------------------------------------------------
  override def state
  : ToggleReferenceState = ToggleReferenceState(
    super.state,
    converter.state,
    children.map(_.state)
  )

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: ToggleReferenceState =>
        converter.restoreState(state.converter)
        SeqEx.foreach(
          children,
          state.children
        )(_.restoreState(_))
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class ToggleReferenceBuilder
  extends SequentialContainerBuilder[ToggleReferenceBuilder] {

  override def repr
  : ToggleReferenceBuilder = this

  private var _converter
  : ModuleBuilder = CopyBuilder()

  def converter
  : ModuleBuilder = _converter

  def converter_=(value: ModuleBuilder)
  : Unit = {
    require(value != null)
    _converter = value
  }

  def setConverter(value: ModuleBuilder)
  : ToggleReferenceBuilder = {
    converter_=(value)
    repr
  }

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _converter.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ToggleReferenceBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ToggleReferenceBuilder =>
      _converter == other._converter
    case _ =>
      false
  })

  override protected def doCopy()
  : ToggleReferenceBuilder = ToggleReferenceBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ToggleReferenceBuilder =>
        other._converter = _converter.copy
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = {
    var tmpHints = hints

    // Run though converter and adjust hints. (note that weights are not supported in the converter!)
    tmpHints = _converter.outputHintsFor(hints)
    tmpHints = hints.derive(
      hints.platform,    hints.layout,
      tmpHints.platform, tmpHints.layout
    )

    tmpHints = super.weightLayoutFor(tmpHints, builder)

    // Re-inject old reference and return.
    tmpHints = tmpHints.derive(
      tmpHints.platform,       tmpHints.layout,
      hints.referencePlatform, hints.referenceLayout
    )
    tmpHints
  }

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = {
    var tmpHints = hints

    // Run though converter and adjust hints.
    tmpHints = _converter.outputHintsFor(hints)
    tmpHints = hints.derive(
      hints.platform,    hints.layout,
      tmpHints.platform, tmpHints.layout
    )

    tmpHints = super.outputHintsFor(tmpHints)

    // Re-inject old reference and return.
    tmpHints = tmpHints.derive(
      tmpHints.platform,       tmpHints.layout,
      hints.referencePlatform, hints.referenceLayout
    )
    tmpHints
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ToggleReference = new ToggleReference(this, hints, seed, weightsBuilder)


  // ---------------------------------------------------------------------------
  //    Checking related
  // ---------------------------------------------------------------------------
  override protected def doCheckEx(hints:        BuildHints,
                                   indentLevel:  Int,
                                   indentString: String)
  : (BuildHints, Long) = {
    var tmpHints = hints
    var noErrors = 0L

    // Run though converter and adjust hints.
    val tmp0  = _converter.checkEx(hints)
    tmpHints  = tmp0._1
    noErrors += tmp0._2
    tmpHints  = hints.derive(
      hints.platform,    hints.layout,
      tmpHints.platform, tmpHints.layout
    )

    val tmp1  = super.doCheckEx(hints, indentLevel, indentString)
    tmpHints  = tmp1._1
    noErrors += tmp1._2

    // Re-inject old reference and return.
    tmpHints = tmpHints.derive(
      tmpHints.platform,       tmpHints.layout,
      hints.referencePlatform, hints.referenceLayout
    )
    (tmpHints, noErrors)
  }


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  override protected def doToGraphEx(node:      VertexGroup,
                                     hints:     Option[BuildHints],
                                     inputs:    Seq[Vertex],
                                     edgeStyle: LineStyle,
                                     nodeSink:  mutable.Buffer[Node],
                                     edgeSink:  mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex]) = {
    var tmpHints  = hints
    var tmpInputs = inputs

    // Add special vertex to model the converter.
    val tmp0 = _converter.toGraphEx(
      hints, inputs, LineStyle.Solid, nodeSink, edgeSink
    )
    tmpHints  = tmp0._1
    tmpInputs = tmp0._2

    // Add special vertex to model reference toggle and connect input and converter output.
    val inVertex = Vertex.derive("Toggle").setShape(NodeShape.Point)
    nodeSink += inVertex
    for (input <- inputs) {
      val edge = Edge(input, inVertex, edgeStyle)
      for (hints <- hints) {
        edge.label = hints.toEdgeLabel
      }
      edgeSink += edge
    }
    for (tmpInput <- tmpInputs) {
      val edge = Edge(tmpInput, inVertex, edgeStyle)
      for (hints <- hints) {
        edge.label = hints.toEdgeLabel
      }
      edgeSink += edge
    }

    // Toggle hints.
    tmpHints = hints.map(hints => {
      hints.derive(
        hints.platform,        hints.layout,
        tmpHints.get.platform, tmpHints.get.layout
      )
    })
    tmpInputs = Seq(inVertex)

    // Process children.
    val tmp1 = super.doToGraphEx(
      node,
      tmpHints,
      tmpInputs,
      LineStyle.Solid,
      nodeSink,
      edgeSink
    )
    tmpHints  = tmp1._1
    tmpInputs = tmp1._2

    // Add special vertex to model reference toggle and connect outputs.
    val outVertex = Vertex.derive("Toggle").setShape(NodeShape.Point)
    nodeSink += outVertex
    for (tmpInput <- tmpInputs) {
      val edge = Edge(tmpInput, outVertex, LineStyle.Solid)
      for (hints <- hints) {
        edge.label = hints.toEdgeLabel
      }
      edgeSink += edge
    }

    // Re-inject old reference and return.
    tmpHints = tmpHints.map(tmpHints => {
      tmpHints.derive(
        tmpHints.platform,           tmpHints.layout,
        hints.get.referencePlatform, hints.get.referenceLayout
      )
    })
    (tmpHints, Seq(outVertex))
  }

}

object ToggleReferenceBuilder {

  final def apply()
  : ToggleReferenceBuilder = new ToggleReferenceBuilder

  final def apply(module0: ModuleBuilder)
  : ToggleReferenceBuilder = apply() += module0

  final def apply(module0: ModuleBuilder, modules: ModuleBuilder*)
  : ToggleReferenceBuilder = apply(module0) ++= modules

  final def apply(modules: TraversableOnce[ModuleBuilder])
  : ToggleReferenceBuilder = apply() ++= modules

  final def apply(modules: Array[ModuleBuilder])
  : ToggleReferenceBuilder = apply() ++= modules

}

final case class ToggleReferenceState(override val parent:  InstanceState,
                                      converter:            InstanceState,
                                      children:             Seq[InstanceState])
  extends ModuleState {
}

final case class ToggleReferenceContext(parent:    PredictContext,
                                        converter: Prediction)
  extends PredictContext {

  override protected def doClose()
  : Unit = {
    converter.close()
    parent.close()
    super.doClose()
  }

}

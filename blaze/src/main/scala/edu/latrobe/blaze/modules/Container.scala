/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.io.graph._
import java.util.UUID
import scala.collection._

abstract class Container[TBuilder <: ContainerBuilder[_]]
  extends ModuleEx[TBuilder]
    with NonPenalizing {

  /**
   * Should be implemented as val.
    *
    * @return All direct children of this container. (not necessarily in any particular order!)
   */
  def children
  : Seq[Module]

  @transient
  final lazy val reversedChildren
  : Seq[Module] = children.reverse

  override def parameters
  : Map[UUID, Parameter] = {
    var result = super.parameters
    children.foreach(
      result ++= _.parameters
    )
    result
  }


  // ---------------------------------------------------------------------------
  //    Traversal related.
  // --------------------------------------------------------------------------
  final override protected def doTouch(callbackFn: Module => Unit)
  : Unit = {
    children.foreach(
      _.touch(callbackFn)
    )
  }


  // ---------------------------------------------------------------------------
  //    Weights related
  // ---------------------------------------------------------------------------
  @transient
  final override lazy val weightReferences
  : Set[LabeledBufferReference] = {
    val builder = Set.newBuilder[LabeledBufferReference]
    children.foreach(
      builder ++= _.weightReferences
    )
    builder.result()
  }

  final override def reset(initializer: Initializer)
  : Unit = {
    children.foreach(
      _.reset(initializer)
    )
  }

  final override def refresh()
  : Unit = {
    children.foreach(
      _.refresh()
    )
  }


  // ---------------------------------------------------------------------------
  //   Statistics
  // ---------------------------------------------------------------------------
  @transient
  final override lazy val noNeurons
  : Long = {
    children.foldLeft(
      0L
    )(_ + _.noNeurons)
  }


  // ---------------------------------------------------------------------------
  //    Weights buffer handling related.
  // ---------------------------------------------------------------------------
  final def childFor(neuronNo: Long)
  : (Module, Long) = {
    var localNeuronNo = neuronNo
    val iter = children.iterator
    while (iter.hasNext) {
      val child = iter.next()
      if (localNeuronNo < child.noNeurons) {
        return (child, localNeuronNo)
      }
      localNeuronNo -= child.noNeurons
    }
    throw new IndexOutOfBoundsException
  }

  final override def extractWeightsFor(neuronNo: Long)
  : Array[Real] = {
    val (child, i) = childFor(neuronNo)
    child.extractWeightsFor(i)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveGradients(input:     Tensor,
                                                 reference: Tensor,
                                                 output:    Tensor,
                                                 context:   PredictContext,
                                                 error:     NextError,
                                                 sink:      ValueTensorBuffer,
                                                 onEnter:   OnEnterDeriveGradients,
                                                 onLeave:   OnLeaveDeriveGradients,
                                                 tensors:   mutable.Stack[Tensor],
                                                 contexts:  mutable.Stack[PredictContext])
  : NextError = {
    doDeriveGradients(
      context,
      error,
      sink,
      onEnter,
      onLeave,
      tensors,
      contexts
    )
  }


  protected def doDeriveGradients(context:       PredictContext,
                                  error:         NextError,
                                  sink:          ValueTensorBuffer,
                                  onEnter:       OnEnterDeriveGradients,
                                  onLeave:       OnLeaveDeriveGradients,
                                  intermediates: mutable.Stack[Tensor],
                                  contexts:      mutable.Stack[PredictContext])
  : NextError

}

abstract class ContainerBuilder[TThis <: ContainerBuilder[_]]
  extends ModuleExBuilder[TThis] {

  def children
  : Seq[ModuleBuilder]


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  override protected def doPermuteHandles(fn: String => String)
  : Unit = {
    super.doPermuteHandles(fn)
    children.foreach(
      _.permuteHandles(fn)
    )
  }

  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    children.foreach(
      _.permuteSeeds(fn)
    )
  }

  override protected def doPermutePreferredPlatforms(fn: Option[Platform] => Option[Platform])
  : Unit = {
    super.doPermutePreferredPlatforms(fn)
    children.foreach(
      _.permutePreferredPlatforms(fn)
    )
  }

  override protected def doPermutePreferredLibraries(fn: Option[String] => Option[String])
  : Unit = {
    super.doPermutePreferredLibraries(fn)
    children.foreach(
      _.permutePreferredLibraries(fn)
    )
  }

  override protected def doPermutePreferredMethods(fn: Option[String] => Option[String])
  : Unit = {
    super.doPermutePreferredMethods(fn)
    children.foreach(
      _.permutePreferredMethods(fn)
    )
  }

  override protected def doPermuteWeightReferences(fn: LabeledBufferReference => LabeledBufferReference)
  : Unit = {
    super.doPermuteWeightReferences(fn)
    children.foreach(
      _.permuteWeightReferences(fn)
    )
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
    // Create a vertex group for the container.
    val node = VertexGroup.derive(toString("\n", ""))
    nodeSink += node

    // Add the current vertex and all children to this group.
    doToGraphEx(node, hints, inputs, edgeStyle, node.children, edgeSink)
  }

  protected def doToGraphEx(node:      VertexGroup,
                            hints:     Option[BuildHints],
                            inputs:    Seq[Vertex],
                            edgeStyle: LineStyle,
                            nodeSink:  mutable.Buffer[Node],
                            edgeSink:  mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex])

}

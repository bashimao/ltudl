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
import scala.collection._
import scala.util.hashing._

/**
  *      branch          call           combine
  *                   sub-modules
  *
  *            -----               -----
  *      ---> | x_0 | --> f_0 --> | y_0 | -
  *    /       -----               -----   \
  *  ---       -----               -----    ->  ---
  * | x | --> | x_1 | --> f_1 --> | y_1 | ---> | y |
  *  ---       -----               -----    ->  ---
  *    \       -----               -----   /
  *      ---> | x_2 | --> f_2 --> | y_2 | -
  *            -----               -----
  *
  *
  * Branch phase:
  *
  * x_i = n * x / n = x
  *
  * d x_i
  * ----- = 1
  *  d x
  *
  *       ---
  * d y   \   d x_i    d y
  * --- = /   ----- * -----
  * d x   ---  d x    d x_i
  *        i
  *
  *
  * Sub-Modules phase:
  *
  * y_i = f_i(x_i)
  *
  *  d y    d f_i    d y
  * ----- = ----- * -----
  * d x_i   d x_i   d y_i
  *
  *
  * Combine phase:
  *
  * if combineOp = Add
  *
  *     ---
  *     \
  * y = /   y_i
  *     ---
  *      i
  *
  *  d y
  * ----- = 1
  * d y_i
  *
  *
  * if combineOp = Concatenate
  *
  * y = y_0 ++ y_1 ++ ... ++ y_2
  *
  *    if noChannels = 2
  *
  *        [ y_0_a ]
  *        [ y_0_b ]
  *        [ y_1_a ]
  *    y = [ y_1_b ]
  *        [  ...  ]
  *        [ y_2_a ]
  *        [ y_2_b ]
  *
  *  d y
  * ----- = slice(ones(size(y)), i * size(y_i), (i + 1) * size(y_i))
  * d y_i
  *
  *    if noChannels = 2
  *
  *          [ d y_0_a ]
  *          [ d y_0_b ]
  *          --- chop ---
  *          [ d y_1_a ]
  *    d y = [ d y_1_b ]
  *          --- chop ---
  *          [   ...   ]
  *          --- chop ---
  *          [ d y_2_a ]
  *          [ d y_2_b ]
  *
  *
  * if combineOp = ConcatenateChannels
  *
  * y = y_0 :++ y_1 :++ ... :++ y_2
  *
  *    if noTuples = 2
  *
  *        [ y_0_a ]
  *        [ y_1_a ]
  *        [ y_2_a ]
  *        [ y_0_b ]
  *    y = [ y_1_b ]
  *        [ y_2_b ]
  *        [  ...  ]
  *        [ y_0_z ]
  *        [ y_1_z ]
  *        [ y_2_z ]
  *
  *
  *  d y
  * ----- = sliceChannels(ones(size(y)), i * noChannels(y_i), (i + 1) * noChannels(y_i))
  * d y_i
  *
  *    if noTuples = 2
  *
  *                         [ d y_0_a ]
  *                         [ d y_0_b ]
  *          [ d y_0_a ] => [   ...   ]
  *          [ d y_1_a ]    [ d y_0_z ]
  *          [ d y_2_a ]
  *          [ d y_0_b ]    [ d y_1_a ]
  *          [ d y_1_b ]    [ d y_1_b ]
  *    d y = [ d y_2_b ] => [   ...   ]
  *          [   ...   ]    [ d y_1_z ]
  *          [ d y_0_z ]
  *          [ d y_1_z ]    [ d y_2_a ]
  *          [ d y_2_z ] => [ d y_2_b ]
  *                         [   ...   ]
  *                         [ d y_2_z ]
  *
  *
  * if combineOp = Lerp(t)
  *
  * y = y_0 * (1 - t) + y_1 * t
  *
  *  d y
  * ----- = 1 - t
  * d y_0
  *
  *  d y
  * ----- = t
  * d y_1
  *
  */
final class Branch(override val builder:        BranchBuilder,
                   override val inputHints:     BuildHints,
                   override val seed:           InstanceSeed,
                   override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Container[BranchBuilder] {

  override val children
  : Seq[Module] = builder.children.map(
    _.build(inputHints, seed, weightBufferBuilder)
  )

  override val outputHints
  : BuildHints = {
    val platforms = Array.newBuilder[Platform]
    val layouts   = Array.newBuilder[TensorLayout]
    for (child <- children) {
      val outHints = child.outputHints
      platforms += outHints.platform
      layouts   += outHints.layout
    }
    val platform = PlatformTable(platforms.result())
    val layout   = TensorLayoutTable(layouts.result())

    inputHints.derive(platform, layout)
  }

  override protected def doClose()
  : Unit = {
    for (child <- children) {
      child.close()
    }
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
    val outputs = Array.newBuilder[Tensor]
    outputs.sizeHint(children.length)

    for (child <- children) {
      // Compute child output and add it to the outputs list.
      val tmp = child.predictEx(
        mode,
        input,
        reference,
        onEnter,
        onLeave
      )
      // We can avoid copying if we set dependenceOnOutputForBackpropagation to true.
      // TODO: Find a more flexible variant. This is kind of bad design here.
      outputs += tmp
    }

    val output = TensorTable(outputs.result())
    (output, EmptyContext)
  }

  override protected def doPredictInv(output:   Tensor,
                                      context:  PredictContext,
                                      onLeave:  OnLeavePredict,
                                      contexts: mutable.Stack[PredictContext])
  : Tensor = throw new NotImplementedError


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  // This will make sure that the next layer will not discard the hyper tensor, in case it was needed by any of the children.
  // TODO: We could solve this by adding a code path in the checking algorithm that is aware of hyper tensors. But well... I need to find time to do this.
  // TODO: Still valid? Wastes a lot of memory.
  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.Required

  override protected def doDeriveGradients(context:       PredictContext,
                                           error:         NextError,
                                           sink:          ValueTensorBuffer,
                                           onEnter:       OnEnterDeriveGradients,
                                           onLeave:       OnLeaveDeriveGradients,
                                           intermediates: mutable.Stack[Tensor],
                                           contexts:      mutable.Stack[PredictContext])
  : NextError = {
    val oldError = error.compute()
    oldError match {
      case oldError: TensorTable =>
        // Inject error into each child and process.
        var i = oldError.length - 1
        val newErrors = reversedChildren.map(child => {
          val oldErr = IndependentError(oldError.getEntry(i), x => x)
          i -= 1

          child.deriveGradientsEx(
            oldErr, sink, onEnter, onLeave, intermediates, contexts
          )
          /*
          var newErr = terminator.deriveGradientsEx(
            oldErr, bankNo, sink, onEnter, onLeave, tensors, contexts
          )
          newErr = */
            /*newErr,*/
          //newErr
        })

        // Forge new error melting together the next errors.
        DependentErrorEx(
          newErrors.toArray,
          newErrors => {
            // TODO: Use reduceLeft
            val iter   = newErrors.iterator
            val result = iter.next()
            while (iter.hasNext) {
              result += iter.next()
            }
            result
          }
        )

      case _ =>
        throw new MatchError(context)
    }
  }


  // ---------------------------------------------------------------------------
  //    State backup and retrieval.
  // ---------------------------------------------------------------------------
  override def state
  : BranchState = BranchState(super.state, children.map(_.state))

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: BranchState =>
        SeqEx.foreach(
          children,
          state.children
        )(_.restoreState(_))
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class BranchBuilder
  extends ContainerBuilder[BranchBuilder] {

  override def repr
  : BranchBuilder = this

  val children
  : mutable.Buffer[ModuleBuilder] = mutable.Buffer.empty

  def +=(module: ModuleBuilder)
  : BranchBuilder = {
    require(module != null)
    children += module
    this
  }

  def ++=(modules: TraversableOnce[ModuleBuilder])
  : BranchBuilder = {
    require(modules.forall(_ != null))
    children ++= modules
    this
  }

  override protected def doToString()
  : List[Any] = children.length :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), children.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[BranchBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BranchBuilder =>
      children == other.children
    case _ =>
      false
  })

  override protected def doCopy()
  : BranchBuilder = BranchBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: BranchBuilder =>
        other.children.clear()
        other.children ++= children.map(_.copy)
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //   Weights / Building related.
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = {
    val platforms = Array.newBuilder[Platform]
    val layouts   = Array.newBuilder[TensorLayout]

    for (child <- children) {
      val outHints = child.weightLayoutFor(hints, builder)
      platforms += outHints.platform
      layouts   += outHints.layout
    }

    val platform = PlatformTable(platforms.result())
    val layout   = TensorLayoutTable(layouts.result())
    hints.derive(platform, layout)
  }

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = {
    val platforms = Array.newBuilder[Platform]
    val layouts   = Array.newBuilder[TensorLayout]

    for (child <- children) {
      val outHints = child.outputHintsFor(hints)
      platforms += outHints.platform
      layouts   += outHints.layout
    }

    val platform = PlatformTable(platforms.result())
    val layout   = TensorLayoutTable(layouts.result())
    hints.derive(platform, layout)
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = new Branch(this, hints, seed, weightsBuilder)


  // ---------------------------------------------------------------------------
  //    Checking related
  // ---------------------------------------------------------------------------
  override protected def doCheckEx(hints:        BuildHints,
                                   indentLevel:  Int,
                                   indentString: String)
  : (BuildHints, Long) = {
    val platforms = Array.newBuilder[Platform]
    val layouts   = Array.newBuilder[TensorLayout]
    var noErrors  = 0L

    for (child <- children) {
      val tmp = child.checkEx(hints, indentLevel + 1, indentString)
      platforms += tmp._1.platform
      layouts   += tmp._1.layout
      noErrors  += tmp._2
    }

    val platform = PlatformTable(platforms.result())
    val layout   = TensorLayoutTable(layouts.result())
    val outHints = hints.derive(platform, layout)
    (outHints, noErrors)
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
    val platforms = Array.newBuilder[Platform]
    val layouts   = Array.newBuilder[TensorLayout]

    // Create a dummy vertex to visualize the table creation.
    val outVertex = Vertex.derive("Table").setShape(NodeShape.Point)
    nodeSink += outVertex

    // Collect outputs.
    for (child <- children) {
      val (outHints, outNodes) = child.toGraphEx(
        hints,
        inputs,
        edgeStyle,
        nodeSink,
        edgeSink
      )

      // Connect edges.
      for (outNode <- outNodes) {
        val edge = Edge(outNode, outVertex, LineStyle.Solid)
        for (outHints <- outHints) {
          edge.label = outHints.toEdgeLabel
        }
        edgeSink += edge
      }

      // Create output build hints.
      outHints.foreach(outHints => {
        platforms += outHints.platform
        layouts   += outHints.layout
      })
    }

    // Build output hints.
    val platform = PlatformTable(platforms.result())
    val layout   = TensorLayoutTable(layouts.result())
    val outHints = hints.map(_.derive(platform, layout))
    (outHints, Seq(outVertex))
  }

}

object BranchBuilder {

  final def apply()
  : BranchBuilder = new BranchBuilder

  final def apply(module0: ModuleBuilder)
  : BranchBuilder = apply() += module0

  final def apply(module0: ModuleBuilder,
                  modules: ModuleBuilder*)
  : BranchBuilder = apply(module0) ++= modules

  final def apply(modules: TraversableOnce[ModuleBuilder])
  : BranchBuilder = apply() ++= modules

  final def apply(modules: Array[ModuleBuilder])
  : BranchBuilder = apply() ++= modules

}

final case class BranchState(override val parent: InstanceState,
                             children:            Seq[InstanceState])
  extends ModuleState {
}

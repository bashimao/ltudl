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
import scala.collection._

final class Sequence(override val builder:             SequenceBuilder,
                     override val inputHints:          BuildHints,
                     override val seed:                InstanceSeed,
                     override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends SequentialContainer[SequenceBuilder] {

  override val (children, outputHints)
  : (Seq[Module], BuildHints) = {
    var tmpHints = inputHints
    val modules = builder.children.map(child => {
      val tmp = child.build(tmpHints, seed, weightBufferBuilder)
      tmpHints = tmp.outputHints
      tmp
    })
    (modules, tmpHints)
  }

  override protected def doClose()
  : Unit = {
    children.foreach(
      _.close()
    )
    super.doClose()
  }


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override val requiresMaintainingInputDuringForwardPropagation
  : Boolean = false

  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired


  // ---------------------------------------------------------------------------
  //    State backup and retrieval.
  // ---------------------------------------------------------------------------
  override def state
  : SequenceState = SequenceState(super.state, children.map(_.state))

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: SequenceState =>
        SeqEx.foreach(
          children,
          state.children
        )(_.restoreState(_))
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class SequenceBuilder
  extends SequentialContainerBuilder[SequenceBuilder] {

  override def repr
  : SequenceBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SequenceBuilder]

  override protected def doCopy()
  : SequenceBuilder = SequenceBuilder()


  // ---------------------------------------------------------------------------
  //    Weights and binding related
  // ---------------------------------------------------------------------------
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Sequence = new Sequence(this, hints, seed, weightsBuilder)

}

object SequenceBuilder {

  final def apply()
  : SequenceBuilder = new SequenceBuilder

  final def apply(module0: ModuleBuilder)
  : SequenceBuilder = apply() += module0

  final def apply(module0: ModuleBuilder, modules: ModuleBuilder*)
  : SequenceBuilder = apply(module0) ++= modules

  final def apply(modules: TraversableOnce[ModuleBuilder])
  : SequenceBuilder = apply() ++= modules

  final def apply(modules: Array[ModuleBuilder])
  : SequenceBuilder = apply() ++= modules

}

final case class SequenceState(override val parent:  InstanceState,
                               children:             Seq[InstanceState])
  extends ModuleState {
}

/*
Delete if new exception based check works!

protected override def checkCallback(errors: StringBuilder, indentLevel: Int)
: Int = {
  var noErrors = super.checkCallback(errors, indentLevel)

  var prev: LayerDesc = null
  while (iter.hasNext) {
    val act = iter.next()
    noErrors += act.check(errors, indentLevel)
    //final def supports(inputSize: Int)
    //: Boolean = size == kernel.inputSize * noChannels
    // TODO: Fix this!
    if (prev != null && act.inputSize != prev.outputSize) {
      var i = 0
      while (i < indentLevel) {
        errors ++= "  "
        i += 1
      }
      errors ++= "Layer %s: InputSize (%d) does not match output size of previous layer (%d)!%n".format(
        act, act.inputSize, prev.outputSize
      )
      noErrors += 1
    }
    prev = act
  }
  noErrors
}
*/

/*
final def getLayerForWeight(weightIndex: Long): (LayerDescLike, Long) = {
  var offset = noWeights
  val iter   = layersReverseIterator
  while (iter.hasNext) {
    val layer = iter.next()
    offset -= layer.noWeights
    if (offset <= weightIndex) {
      return (layer, weightIndex - offset)
    }
  }
  throw new IndexOutOfBoundsException()
}
*/

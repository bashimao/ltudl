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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._

import scala.util.hashing.MurmurHash3

/**
  * See description of Branch-layer for equations and more information.
  *
  * Tensor dimensions must match.
  */
final class Merge(override val builder:        MergeBuilder,
                  override val inputHints:     BuildHints,
                  override val seed:           InstanceSeed,
                  override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Combiner[MergeBuilder] {

  val op
  : MergeOp = builder.op


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input: TensorTable)
  : Tensor = {
    // TODO: Why don't we work in place here?!
    val iter = input.iterator
    val out  = iter.next().copy

    op match {
      case MergeOp.Add =>
        while (iter.hasNext) {
          val inp = iter.next ()
          out += inp
        }

      case MergeOp.Mean =>
        val t = Real.one / input.length
        out *= t
        while (iter.hasNext) {
          val inp = iter.next()
          out.add(inp, t)
        }

      case MergeOp.Lerp(t) =>
        require(input.length == 2)
        val inp = iter.next()
        out.lerp(inp, t)

      case _ =>
        throw new MatchError(op)
    }

    out
  }

  override protected def doPredictInv(output: Tensor,
                                      input:  TensorTable)
  : Unit = op match {

    case MergeOp.Add =>
      val t = Real.one / input.length
      input.foreachTensor(
        _.set(output, t)
      )

    case MergeOp.Mean =>
      input.foreachTensor(
        _ := output
      )

    case MergeOp.Lerp(t) =>
      input.getEntry(0).set(output, Real.one - t)
      input.getEntry(1).set(output, t)

    case _ =>
      throw new MatchError(op)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  override protected def doDeriveInputError(input:     Tensor,
                                            reference: Tensor,
                                            output:    Tensor,
                                            oldError:  Tensor,
                                            newError:  TensorTable)
  : Unit = op match {

    case MergeOp.Add =>
      newError.foreachTensor(
        _ := oldError
      )

    case MergeOp.Mean =>
      val t = Real.one / newError.length
      newError.foreachTensor(
        _.set(oldError, t)
      )

    case MergeOp.Lerp(t) =>
      newError.getEntry(0).set(oldError, Real.one - t)
      newError.getEntry(1).set(oldError, t)

    case _ =>
      throw new MatchError(op)
  }

}

final class MergeBuilder
  extends CombinerBuilder[MergeBuilder] {

  override def repr
  : MergeBuilder = this

  private var _op
  : MergeOp = MergeOp.Add

  def op
  : MergeOp = _op

  def op_=(value: MergeOp)
  : Unit = {
    require(value != null)
    _op = value
  }

  def setOp(value: MergeOp)
  : MergeBuilder = {
    op_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _op :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _op.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MergeBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: MergeBuilder =>
      _op == other._op
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
        other._op = _op
      case _ =>
    }
  }

  override def outputPlatformFor(platformHint: PlatformTable)
  : Platform = platformHint.getEntry(0)

  override def outputLayoutFor(layoutHint: TensorLayoutTable)
  : TensorLayout = {
    val iter   = layoutHint.iterator
    val result = iter.next()
    while (iter.hasNext) {
      require(
        iter.next() == result,
        "Outputs of parallel sub-modules can only be merged if they have the same dimensions!"
      )
    }
    result
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Merge = new Merge(this, hints, seed, weightsBuilder)

}

object MergeBuilder {

  final def apply()
  : MergeBuilder = new MergeBuilder

}

abstract class MergeOp
  extends Serializable

object MergeOp {

  case object Add
    extends MergeOp

  case object Mean
    extends MergeOp

  case class Lerp(t: Real)
    extends MergeOp

}

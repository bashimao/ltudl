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
import scala.util.hashing._

/**
  * Select the n'th element. Use negative indices to select from end.
  */
final class Select(override val builder:        SelectBuilder,
                   override val inputHints:     BuildHints,
                   override val seed:           InstanceSeed,
                   override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Combiner[SelectBuilder] {

  val op
  : SelectOp = builder.op


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input: TensorTable)
  : Tensor = op match {

    case SelectOp.Entry(index) =>
      input.getEntry(index)

    case SelectOp.Subset(indices) =>
      // TODO: Could handle this more elegantly.
      // Avoids the tensor from being overwritten by the next layer.
      val out = ArrayEx.map(
        input.getEntries(indices)
      )(_.copy)
      TensorTable(out)

    case _ =>
      throw new MatchError(op)
  }

  override protected def doPredictInv(output: Tensor,
                                      input:  TensorTable)
  : Unit = throw new UnsupportedOperationException


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

    case SelectOp.Entry(index) =>
      val newErr = newError.getEntry(index)
      newErr := oldError

    case SelectOp.Subset(indices) =>
      oldError match {
        case oldErr: TensorTable =>
          val iter = oldErr.iterator
          RangeEx.map(indices)(i => {
            val newErr = newError.getEntry(i)
            val oldErr = iter.next()
            newErr := oldErr
          })
        case _ =>
          throw new MatchError(oldError)
      }

    case _ =>
      throw new MatchError(op)
  }

}

final class SelectBuilder
  extends CombinerBuilder[SelectBuilder] {

  override def repr
  : SelectBuilder = this

  private var _op
  : SelectOp = SelectOp.First

  def op
  : SelectOp = _op

  def op_=(value: SelectOp)
  : Unit = {
    require(value != null)
    _op = value
  }

  def setOp(value: SelectOp)
  : SelectBuilder = {
    op_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _op :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _op.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SelectBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: SelectBuilder =>
      _op == other._op
    case _ =>
      false
  })

  override protected def doCopy()
  : SelectBuilder = SelectBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: SelectBuilder =>
        other._op = _op
      case _ =>
    }
  }

  override def outputPlatformFor(platformHint: PlatformTable)
  : Platform = _op match {
    case SelectOp.Entry(index) =>
      platformHint.getEntry(index)

    case SelectOp.Subset(indices) =>
      val platforms = platformHint.getEntries(indices)
      PlatformTable(platforms)

    case _ =>
      throw new MatchError(_op)
  }

  override def outputLayoutFor(layoutHint: TensorLayoutTable)
  : TensorLayout = _op match {
    case SelectOp.Entry(index) =>
      layoutHint.getEntry(index)

    case SelectOp.Subset(indices) =>
      val layouts = layoutHint.getEntries(indices)
      TensorLayoutTable(layouts)

    case _ =>
      throw new MatchError(_op)
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Select = new Select(this, hints, seed, weightsBuilder)

}

object SelectBuilder {

  final def apply()
  : SelectBuilder = new SelectBuilder

  final def apply(op: SelectOp)
  : SelectBuilder = apply().setOp(op)

  final def derive(index: Int)
  : SelectBuilder = apply(SelectOp.Entry(index))

}

abstract class SelectOp
  extends Serializable

object SelectOp {

  /**
    * [0 Int.MaxValue] => index from first
    * [Int.MinValue -1] => index from last
    */
  case class Entry(index: Int)
    extends SelectOp

  case class Subset(indices: Range)
    extends SelectOp

  object First
    extends Entry(0)

  object Last
    extends Entry(-1)

}

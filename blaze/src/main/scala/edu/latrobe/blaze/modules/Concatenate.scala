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
  * See description of Branch-layer for equations and more information.
  *
  * Tensors must have the same number of samples and sizes must compatible.
  */
final class Concatenate(override val builder:        ConcatenateBuilder,
                        override val inputHints:     BuildHints,
                        override val seed:           InstanceSeed,
                        override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Combiner[ConcatenateBuilder] {

  val domain
  : TensorDomain = builder.domain

  @inline
  protected def fuse(input0: Tensor, input1: Tensor)
  : Tensor = domain match {
    case TensorDomain.Channel =>
      input0 :++ input1
    case TensorDomain.Sample =>
      input0 ++ input1
    case TensorDomain.Batch =>
      input0.concat(input1)
    case _ =>
      throw new MatchError(domain)
  }

  @inline
  protected def extract(output:  Tensor,
                        offset0: Int,
                        input:   Tensor)
  : Int = domain match {
    case TensorDomain.Channel =>
      output.sliceChannels(offset0, input)
      offset0 + input.layout.size.noChannels
    case TensorDomain.Sample =>
      output.slice(offset0, input)
      offset0 + input.layout.size.noTuples
    case TensorDomain.Batch =>
      // TODO: Make this faster!
      logger.trace("Performance warning: Extracting samples is super-slow right now.")
      val offset1 = offset0 + input.layout.noSamples
      /*using(output(offset0 until offset1))(
        input := _
      )*/
      throw new NotImplementedError
    case _ =>
      throw new MatchError(domain)
  }


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input: TensorTable)
  : Tensor = {
    val iter = input.iterator
    var out  = iter.next().copy
    while (iter.hasNext) {
      val inp = iter.next()
      val tmp = fuse(out, inp)
      // Destroy previous output and switch.
      out.close()
      out = tmp
    }
    out
  }

  override protected def doPredictInv(output: Tensor,
                                      input:  TensorTable)
  : Unit = {
    input.foldLeftTensors(0)(
      extract(output, _, _)
    )
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
  : Unit = {
    newError.foldLeftTensors(0)(
      extract(oldError, _, _)
    )
  }

}

final class ConcatenateBuilder
  extends CombinerBuilder[ConcatenateBuilder] {

  override def repr
  : ConcatenateBuilder = this

  private var _domain
  : TensorDomain = TensorDomain.Batch

  def domain
  : TensorDomain = _domain

  def domain_=(value: TensorDomain): Unit = {
    require(value != null)
    _domain = value
  }

  def setDomain(value: TensorDomain)
  : ConcatenateBuilder = {
    domain_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _domain :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _domain.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ConcatenateBuilder =>
      _domain == other._domain
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ConcatenateBuilder =>
        other._domain = _domain
      case _ =>
    }
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ConcatenateBuilder]

  override protected def doCopy()
  : ConcatenateBuilder = ConcatenateBuilder()


  // ---------------------------------------------------------------------------
  //   Weights / Building related.
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(platformHint: PlatformTable)
  : Platform = platformHint.getEntry(0)

  override def outputLayoutFor(layoutHint: TensorLayoutTable)
  : TensorLayout = _domain match {

    case TensorDomain.Channel =>
      val iter   = layoutHint.iterator
      var result = iter.next()
      while (iter.hasNext) {
        result :++= iter.next()
      }
      result

    case TensorDomain.Sample =>
      val iter   = layoutHint.iterator
      var result = iter.next()
      while (iter.hasNext) {
        result ++= iter.next()
      }
      result

    case TensorDomain.Batch =>
      val iter   = layoutHint.iterator
      var result = iter.next()
      while (iter.hasNext) {
        result = result.concat(iter.next())
      }
      result

    case _ =>
      throw new MatchError(_domain)
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Concatenate = new Concatenate(this, hints, seed, weightsBuilder)

}

object ConcatenateBuilder {

  final def apply()
  : ConcatenateBuilder = new ConcatenateBuilder

  final def apply(domain: TensorDomain)
  : ConcatenateBuilder = apply().setDomain(domain)

}

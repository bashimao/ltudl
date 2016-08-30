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

package edu.latrobe.blaze

import edu.latrobe._
import edu.latrobe.sizes._
import edu.latrobe.time._
import scala.util.hashing._

/**
 * Represents a mini-batch for either supervised or unsupervised learning.
 * Shape of activations matrix:
 *
 * SS ... S
 * AA ... A
 * MM ... M
 * PP ... P
 * LL ... L
 * EE ... E
 * 01 ....n
 */
final class Batch(val input:  Tensor,
                  val output: Tensor,
                  val tags:   Array[JSerializable])
  extends Serializable
    with Equatable
    with CopyableEx[Batch]
    with AutoCloseable {
  require(
    input != null/* &&
    input.layout.noSamples == output.layout.noSamples &&
    tags.length == input.layout.noSamples*/
  )

  override def toString
  : String = s"Batch[$input -> $output, Tags: $tags]"

  override def canEqual(that: Any): Boolean = that.isInstanceOf[Batch]

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, input.hashCode())
    tmp = MurmurHash3.mix(tmp, output.hashCode())
    tmp = MurmurHash3.mix(tmp, tags.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Batch =>
      input  == other.input &&
      output == other.output &&
      tags.sameElements(other.tags)
    case _ =>
      false
  })

  override def close()
  : Unit = {
    if (!input.closed) {
      input.close()
    }
    if (!output.closed) {
      output.close()
    }
  }

  override def copy
  : Batch = Batch(
    input.copy,
    output.copy,
    tags
  )

  def derive(newInput: Tensor)
  : Batch = Batch(newInput, output, tags)

  def noSamples
  : Int = input.layout.noSamples


  // ---------------------------------------------------------------------------
  //   Basic operations.
  // ---------------------------------------------------------------------------
  def apply(index: Int): Batch = Batch(
    input(index),
    output(index),
    Array(tags(index))
  )

  def apply(indices: Range): Batch = Batch(
    input(indices),
    output(indices),
    ArrayEx.slice(tags, indices)
  )

  def apply(index0: Int, indices: Int*)
  : Array[Batch] = apply(SeqEx.concat(index0, indices))

  def apply(indices: Array[Int])
  : Array[Batch] = ArrayEx.map(indices)(apply)

  def split(): Array[Batch] = {
    if (noSamples == 1) {
      Array(this)
    }
    else {
      val inp = input.splitSamples
      val out = output.splitSamples
      val tag = ArrayEx.map(tags)(Array(_))
      ArrayEx.zip(inp, out, tag)(Batch.apply)
    }
  }

  def concat(other: Batch): Batch = Batch(
    input.concat(other.input),
    output.concat(other.output),
    ArrayEx.concat(tags, other.tags)
  )

  def concat(other0: Batch, others: Batch*)
  : Batch = concat(SeqEx.concat(other0, others))

  def concat(others: Array[Batch]): Batch = Batch(
    input.concat(ArrayEx.map(others)(_.input)),
    output.concat(ArrayEx.map(others)(_.output)),
    ArrayEx.concat(
      tags,
      ArrayEx.map(others)(_.tags)
    )
  )

}


object Batch {

  final def apply(input: Tensor)
  : Batch = apply(input, RealArrayTensor.zeros(input.layout.derive(Size1.zero)))

  final def apply(input: Tensor, output: Tensor)
  : Batch = apply(
    input,
    output,
    new Array[JSerializable](input.layout.noSamples)
  )

  final def apply(input:  Tensor,
                  output: Tensor,
                  tags:   Array[JSerializable])
  : Batch = new Batch(input, output, tags)

  final def derive(input: Tensor, tags: Array[JSerializable])
  : Batch = apply(
    input,
    RealArrayTensor.zeros(input.layout.derive(Size1.zero)),
    tags
  )

  final def concat(batches: Array[Batch])
  : Batch = batches(0).concat(batches.tail)

}
/*
/**
 * A simpler less efficient batch type (with almost no overhead).
 */
final class BatchSet(val batches: IndexedSeq[Batch])
  extends BatchLikeEx[TensorSet] {

  override def toString: String = s"BatchSet[${batches.length}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), batches.hashCode())

  override def canEqual(that: Any): Boolean = that match {
    case that: BatchSet =>
      true
    case _ =>
      false
  }

  override def equals(obj: Any): Boolean = super.equals(obj) && (obj match {
    case obj: BatchSet =>
      batches == obj.batches
    case _ =>
      false
  })

  override def copy: BatchSet = BatchSet(batches.map(_.copy))

  override def noSamples: Int = batches.foldLeft(0)(_ + _.noSamples)

  override def input
  : TensorSet = TensorSet(batches.map(_.input))

  override def output
  : TensorSet = TensorSet(batches.map(_.output))

  def tags: IndexedSeq[JSerializable] = batches.flatMap(_.tags)

  override def derive(newInput: TensorSet)
  : BatchSet = BatchSet(batches.fastZip(newInput.tensors)(_.derive(_)))

  def mapSamples(fn: Batch => Batch)
  : BatchSet = BatchSet(batches.map(fn))

  override def apply(index: Int): Batch = batches(index)

  override def apply(indices: Range): BatchSet = {
    // TODO: Remove this constraint. See fastSlice for arrays!
    require(indices.step == 1)
    BatchSet(batches.slice(indices.start, indices.end))
  }

  //def toBatch: Batch = Batch(batches)

}

object BatchSet {

  final def apply(batch0: Batch): BatchSet = apply(Array(batch0))

  final def apply(batch0: Batch, batches: Batch*): BatchSet = {
    val tmp = Array.newBuilder[Batch]
    tmp += batch0
    tmp ++= batches
    apply(tmp.result())
  }

  final def apply(batches: IndexedSeq[Batch])
  : BatchSet = new BatchSet(batches)

  final def apply(input: TensorSet, output: TensorSet): BatchSet = {
    if (output != null) {
      apply(input.tensors.fastZip(output.tensors)(Batch.apply))
    }
    else {
      apply(input.tensors.map(Batch.apply))
    }
  }

}
*/
/*
trait BatchTensorLike extends TensorLike {

  def noSamples: Int

  override def copy: BatchTensorLike


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  //override def copyWithoutMetaData: BatchTensorLike

  override def mapValues(fn: Real => Real): BatchTensorLike

  //override def dropMetaData: BatchTensorLike


  // ---------------------------------------------------------------------------
  //    Conversion & extraction methods.
  // ---------------------------------------------------------------------------
  /*
  def toSeq: Seq[SampleTensor]

  def toArray: Array[SampleTensor]
  */

}
*/
/*
trait BatchTensor extends Tensor with BatchTensorLike {
/*
  override def copy: BatchTensor

  override def values: DMat

  override def valuesEx: Mat with Serializable


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  override def reshape(size: Size): BatchTensor

  //override def withMetaData(metaData: JSerializable): BatchTensor

  //override def dropMetaData: BatchTensor

  //override def copyWithoutMetaData: BatchTensor

  override def mapValues(fn: Real => Real): BatchTensor

    //override def mapValues(fn: Real => Real, metaData: JSerializable)
  //: BatchTensor
*/



}
*/


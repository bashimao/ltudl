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

package edu.latrobe

import org.json4s.JsonAST._
import scala.collection._
import scala.util.hashing._

/**
  * A sample that was encoded in memory using some method. It must be
  * decoded before it can be used (augmentors are meant to do that). Use this
  * if you want to keep things compressed and only uncompress on the fly when you
  * access the items.
  */
final class ByteArrayTensor(override val sizeHint: Size,
                            override val bytes:    Array[Array[Byte]])
  extends TensorEx[ByteArrayTensor]
    with RawTensor
    with Serializable {
  require(sizeHint != null && !ArrayEx.contains(bytes, null))

  override def repr
  : ByteArrayTensor = this

  override def toString
  : String = s"ByteArrayTensor[$sizeHint x ${bytes.length}]"

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, sizeHint.hashCode())
    tmp = MurmurHash3.mix(tmp, bytes.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ByteArrayTensor =>
      sizeHint   == other.sizeHint &&
      bytes.deep == other.bytes.deep
    case _ =>
      false
  })

  override def createSibling(newLayout: TensorLayout)
  : ByteArrayTensor = allocateZeroedSibling(
    newLayout.size,
    newLayout.noSamples
  )

  override def createSiblingAndClear(newLayout: TensorLayout)
  : ByteArrayTensor = allocateZeroedSibling(
    newLayout.size,
    newLayout.noSamples
  )

  def allocateZeroedSibling(size: Size, noSamples: Int)
  : ByteArrayTensor = ByteArrayTensor(
    size,
    ArrayEx.fill(
      noSamples
    )(new Array[Byte](0))
  )

  override def copy
  : ByteArrayTensor = ByteArrayTensor(sizeHint, ArrayEx.map(bytes)(_.clone()))

  override protected def doClose()
  : Unit = {
    ArrayEx.fill(
      bytes
    )(null)
    super.doClose()
  }

  override def platform
  : JVM.type = JVM

  @transient
  override lazy val layout
  : IndependentTensorLayout = IndependentTensorLayout(sizeHint, bytes.length)

  override def clear()
  : Unit = ArrayEx.foreach(
    bytes
  )(ArrayEx.fill(_, 0.toByte))


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  override def reshape(newSize: Size)
  : ByteArrayTensor = ByteArrayTensor(newSize, bytes)

  override def apply(index: Int)
  : ByteArrayTensor = ByteArrayTensor.derive(sizeHint, bytes(index))

  override def apply(indices: Range)
  : ByteArrayTensor = ByteArrayTensor(
    sizeHint,
    ArrayEx.slice(bytes, indices)
  )

  override def splitSamples
  : Array[Tensor] = ArrayEx.map(
    bytes
  )(ByteArrayTensor.derive(sizeHint, _))

  override def concat(other: Tensor)
  : ByteArrayTensor = other match {
    case other: RawTensor =>
      ByteArrayTensor(
        sizeHint,
        ArrayEx.concat(bytes, other.bytes)
      )
    case _ =>
      throw new MatchError(other)
  }

  override def concat[T <: Tensor](others: Array[T])
  : ByteArrayTensor = {
    def getBytes(tensor: Tensor)
    : Array[Array[Byte]] = tensor match {
      case tensor: RawTensor =>
        tensor.bytes
      case _ =>
        throw new MatchError(tensor)
    }

    val newBytes = ArrayEx.concat(
      bytes,
      ArrayEx.map(
        others
      )(getBytes)
    )
    ByteArrayTensor(sizeHint, newBytes)
  }

  override protected def doSet(other: Tensor)
  : Unit = other match {
    case other: RawTensor =>
      ArrayEx.fill(
        bytes,
        other.bytes
      )(_.clone())
    case _ =>
      throw new UnsupportedOperationException
  }

  override def ++(other: Tensor)
  : ByteArrayTensor = other match {
    case other: RawTensor =>
      val newSizeHint = sizeHint ++ other.sizeHint
      val newBytes    = ArrayEx.zip(
        bytes,
        other.bytes
      )((a, b) => ArrayEx.concat(a, b))
      ByteArrayTensor(newSizeHint, newBytes)
    case _ =>
      throw new MatchError(other)
  }

  override def :++(other: Tensor)
  : ByteArrayTensor = throw new UnsupportedOperationException

  override protected def doToJson()
  : List[JField] = List(
    Json.field("sizeHint", sizeHint),
    Json.field("bytes",    bytes)
  )

}

object ByteArrayTensor {

  final def apply(sizeHint: Size, bytes: Array[Array[Byte]])
  : ByteArrayTensor = new ByteArrayTensor(sizeHint, bytes)

  final def derive(sizeHint: Size, bytes0: Array[Byte])
  : ByteArrayTensor = apply(sizeHint, Array(bytes0))

  final def derive(sizeHint: Size, bytes0: Array[Byte], bytes: Array[Byte]*)
  : ByteArrayTensor = apply(sizeHint, SeqEx.concat(bytes0, bytes))

  final def derive(sizeHint: Size, bytes: Seq[Array[Byte]])
  : ByteArrayTensor = apply(sizeHint, bytes.toArray)

}

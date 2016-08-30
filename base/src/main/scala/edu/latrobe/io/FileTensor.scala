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

package edu.latrobe.io

import edu.latrobe._
import org.json4s.JsonAST._

import scala.collection._
import scala.reflect.ClassTag
import scala.util.hashing._

trait FileTensorLike
  extends RawTensor
    with Serializable {
}

/**
  * We keep the configuration as bytes and only build the configuration object
  * on demand. Thus allowing quick repartitioning across the cluster.
  *
  * cacheBytes = true will keep the data read from the file in cached in memory.
  */
final class FileTensor(override val sizeHint: Size,
                       val          handles:  Array[FileHandle])
  extends TensorEx[FileTensor]
    with FileTensorLike {
  require(sizeHint != null && !ArrayEx.contains(handles, null))

  override def repr
  : FileTensor = this

  override def toString
  : String = s"FileTensor[$sizeHint x ${handles.length}]"

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, sizeHint.hashCode())
    tmp = MurmurHash3.mix(tmp, ArrayEx.hashCode(handles))
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: FileTensor =>
      sizeHint == other.sizeHint &&
      ArrayEx.compare(handles, other.handles)
    case _ =>
      false
  })

  override def createSibling(newLayout: TensorLayout)
  : FileTensor = throw new UnsupportedOperationException

  override def createSiblingAndClear(newLayout: TensorLayout)
  : FileTensor = throw new UnsupportedOperationException

  override def copy
  : FileTensor = FileTensor(sizeHint, handles.clone())

  override protected def doClose()
  : Unit = {
    ArrayEx.fill(
      handles
    )(null)
    super.doClose()
  }

  override def platform
  : JVM.type = JVM

  @transient
  override lazy val layout
  : IndependentTensorLayout = IndependentTensorLayout(sizeHint, handles.length)

  override def bytes
  : Array[Array[Byte]] = ArrayEx.map(
    handles
  )(_.readAsArray())

  override def clear()
  : Unit = {
    ArrayEx.fill(
      handles
    )(null)
  }


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  override def reshape(size: Size)
  : FileTensor = FileTensor(size, handles.clone())

  override def apply(index: Int)
  : FileTensor = FileTensor(sizeHint, Array(handles(index)))

  override def apply(indices: Range)
  : FileTensor = FileTensor(sizeHint, ArrayEx.slice(handles, indices))

  override def splitSamples
  : Array[Tensor] = ArrayEx.map(
    handles
  )(handle => FileTensor(sizeHint, Array(handle)))

  override def concat(other: Tensor)
  : FileTensor = other match {
    case other: FileTensor =>
      FileTensor(sizeHint, ArrayEx.concat(handles, other.handles))
    case _ =>
      throw new MatchError(other)
  }

  override def concat[T <: Tensor](others: Array[T])
  : FileTensor = {
    def getHandles(tensor: Tensor)
    : Array[FileHandle] = tensor match {
      case tensor: FileTensor =>
        tensor.handles
      case _ =>
        throw new MatchError(tensor)
    }
    val result = ArrayEx.concat(
      handles,
      ArrayEx.map(
        others
      )(getHandles)
    )
    FileTensor(sizeHint, result)
  }

  override protected def doSet(other: Tensor)
  : Unit = other match {
    case other: FileTensor =>
      ArrayEx.set(
        handles,
        other.handles
      )
    case _ =>
      throw new UnsupportedOperationException
  }

  def mapSampleHandles[T](fn: FileHandle => T)
                         (implicit tagT: ClassTag[T])
  : Array[T] = {
    if (handles.length > 1) {
      ArrayEx.mapParallel(
        handles
      )(fn)
    }
    else {
      ArrayEx.map(
        handles
      )(fn)
    }
  }

  override def ++(other: Tensor)
  : FileTensor = throw new UnsupportedOperationException

  override def :++(other: Tensor)
  : FileTensor = throw new UnsupportedOperationException

  override protected def doToJson()
  : List[JField] = List(
    Json.field("sizeHint", sizeHint),
    Json.field("handles",  handles)
  )

}

object FileTensor {

  final def apply(sizeHint: Size,
                  handles:  Array[FileHandle])
  : FileTensor = new FileTensor(sizeHint, handles)

  final def derive(sizeHint: Size,
                   handle0:  FileHandle)
  : FileTensor = apply(sizeHint, Array(handle0))

  final def derive(sizeHint: Size,
                   handle0:  FileHandle,
                   handles:  FileHandle*)
  : FileTensor = apply(sizeHint, SeqEx.concat(handle0, handles))

  final def derive(sizeHint: Size,
                   handles:  Seq[FileHandle])
  : FileTensor = apply(sizeHint, handles.toArray)

}



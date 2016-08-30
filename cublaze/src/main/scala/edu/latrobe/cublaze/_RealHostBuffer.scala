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

package edu.latrobe.cublaze

import edu.latrobe._
import edu.latrobe.native._
import org.bytedeco.javacpp._

final class _RealHostBuffer private(override protected val _ptr: RealPointer)
  extends _HostBuffer[RealPointer] {

  override def capacityInBytes
  : Long = _ptr.capacity() * Real.size

  // ---------------------------------------------------------------------------
  //    Content read/write.
  // ---------------------------------------------------------------------------
  /*
  def get(result: DenseVector[Real])
  : Unit = {
    require(result.length <= capacity)

    if (result.stride == 1) {
      get(result.data, result.offset, result.length)
    }
    else {
      VectorEx.tabulate(result)(get)
    }
  }

  def get(result: DenseMatrix[Real])
  : Unit = {
    require(!result.isTranspose)

    // Cache frequently used values.
    val size = result.size
    val min  = MatrixEx.minor(result)
    val gap  = result.majorStride - min

    require(size <= capacity())

    if (result.majorStride == result.rows) {
      get(result.data, result.offset, size)
    }
    else
    {
      val data   = result.data
      var offset = result.offset
      var i      = 0
      while (i < size) {
        val nextGap = i + min
        while (i < nextGap) {
          data(offset) = get(i)
          offset += 1
          i      += 1
        }
        offset += gap
      }
    }
  }
  */

  /*
  def addTo(result: DenseVector[Real]): Unit = {
    require(result.length <= capacity())

    val data   = result.data
    var offset = result.offset
    var i      = 0
    while (i < result.length) {
      data(offset) += get(i)
      offset += result.stride
      i      += 1
    }
  }

  def addTo(result: DenseMatrix[Real]): Unit = {
    require(!result.isTranspose)

    // Cache frequently used values.
    val size = result.size
    val min  = MatrixEx.minor(result)
    val gap  = result.majorStride - min

    require(size <= capacity())

    val data   = result.data
    var offset = result.offset
    var i      = 0
    while (i < size) {
      val nextGap = i + min
      while (i < nextGap) {
        data(offset) += get(i)
        offset += 1
        i      += 1
      }
      offset += gap
    }
  }
  */

  /*
  def put(vector: DenseVector[Real])
  : HostRealPtr = {
    require(vector.length <= capacity)

    if (vector.stride == 1) {
      put(vector.data, vector.offset, vector.length)
    }
    else {
      VectorEx.foreachPair(vector)(
        put(_, _)
      )
    }
    this
  }

  def put(matrix: DenseMatrix[Real]): HostRealPtr = {
    // Cache frequently used values.
    val size = matrix.size
    val min  = MatrixEx.minor(matrix)
    val gap  = matrix.majorStride - min

    require(size <= capacity())

    if (gap == 0) {
      put(matrix.data, matrix.offset, size)
    }
    else {
      // TODO: Change to MatrixEx!
      val data   = matrix.data
      var offset = matrix.offset
      var i      = 0
      while (i < size) {
        val nextGap = i + min
        while (i < nextGap) {
          put(i, data(offset))
          offset += 1
          i      += 1
        }
        offset += gap
      }
    }
    this
  }
  */

  def apply(index: Int)
  : Real = _ptr.get(index)

  def update(index: Int, value: Real)
  : Unit = _ptr.put(index, value)

  def get(array: Array[Real])
  : Unit = get(array, 0, array.length)

  def get(array: Array[Real], offset: Int, length: Int)
  : Unit = _ptr.get(array, offset, length)

  def put(array: Array[Real])
  : Unit = put(array, 0, array.length)

  def put(array: Array[Real], offset: Int, length: Int)
  : Unit = _ptr.put(array, offset, length)

  def toArray(length: Int)
  : Array[Real] = {
    require(length <= _ptr.capacity())
    val result = new Array[Real](length)
    _ptr.get(result, 0, result.length)
    result
  }


  // ---------------------------------------------------------------------------
  //    Copying
  // ---------------------------------------------------------------------------
  override def copy
  : _RealHostBuffer = {
    val result = _RealHostBuffer(_ptr.capacity())
    copyTo(result)
    result
  }

}

private[cublaze] object _RealHostBuffer {

  final def apply(capacity: Long)
  : _RealHostBuffer = {
    require(capacity <= ArrayEx.maxSize)
    val ptr = new RealPointer()
    _CUDA.mallocHost(ptr, capacity * Real.size)
    ptr.capacity(capacity)
    new _RealHostBuffer(ptr)
  }

}
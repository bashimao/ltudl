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
import edu.latrobe.sizes._
import org.bytedeco.javacpp.cudnn._

final class _TensorStruct
  extends AutoClosingPointerEx[cudnnTensorStruct] {

  override protected val _ptr
  : cudnnTensorStruct = _CUDNN.createTensorDescriptor()

  override protected def doClose()
  : Unit = {
    _CUDNN.destroyTensorDescriptor(this)
    super.doClose()
  }

  def noDims
  : Int = _CUDNN.getTensorNdDescriptor(this)._1

  def dataType
  : Int = _CUDNN.getTensorNdDescriptor(this)._2

  def dims
  : Array[Int] = {
    val noDims = _CUDNN.getTensorNdDescriptor(this)._1
    val dims   = new Array[Int](noDims)
    val stride = new Array[Int](noDims)
    _CUDNN.getTensorNdDescriptor(
      this, dims, stride
    )
    dims
  }

  def stride
  : Array[Int] = {
    val noDims = _CUDNN.getTensorNdDescriptor(this)._1
    val dims   = new Array[Int](noDims)
    val stride = new Array[Int](noDims)
    _CUDNN.getTensorNdDescriptor(
      this, dims, stride
    )
    stride
  }

}

private[cublaze] object _TensorStruct {

  final def nchw(layout:   IndependentTensorLayout,
                 dataType: Int)
  : _TensorStruct = nchw(
    layout.size,
    layout.noSamples,
    dataType
  )

  /**
   * Preferred format of cuDNN.
   */
  final def nchw(size:      Size,
                 noSamples: Int,
                 dataType:  Int)
  : _TensorStruct = {
    val result = new _TensorStruct()
    if (size.noChannels > 0) {
      size match {
        case size: Size1 =>
          _CUDNN.setTensor4dDescriptor(
            result,
            CUDNN_TENSOR_NCHW,
            dataType,
            noSamples,
            size.noChannels,
            1,
            size.noTuples
          )

        case size: Size2 =>
          _CUDNN.setTensor4dDescriptor(
            result,
            CUDNN_TENSOR_NCHW,
            dataType,
            noSamples,
            size.noChannels,
            size.dims._2,
            size.dims._1
          )

        case size: Size3 =>
          val dims = Array(
            noSamples,
            size.noChannels,
            size.dims._3,
            size.dims._2,
            size.dims._1
          )
          val stride = Array(
            size.noValues,
            size.noTuples,
            size.dims._1 * size.dims._2,
            size.dims._1,
            1
          )
          _CUDNN.setTensorNdDescriptor(
            result,
            dataType,
            dims,
            stride
          )

        case size: Size4 =>
          val dims = Array(
            noSamples,
            size.noChannels,
            size.dims._4,
            size.dims._3,
            size.dims._2,
            size.dims._1
          )
          val stride = Array(
            size.noValues,
            size.noTuples,
            size.dims._1 * size.dims._2 * size.dims._3,
            size.dims._1 * size.dims._2,
            size.dims._1,
            1
          )
          _CUDNN.setTensorNdDescriptor(
            result,
            dataType,
            dims,
            stride
          )

        case _ =>
          throw new MatchError(size)
      }
    }
    result
  }

  final def nchw(layout:     IndependentTensorLayout,
                 memorySize: Size,
                 dataType:   Int)
  : _TensorStruct = nchw(
    layout.size,
    layout.noSamples,
    memorySize,
    dataType
  )

  final def nchw(size:       Size,
                 noSamples:  Int,
                 memorySize: Size,
                 dataType:   Int)
  : _TensorStruct = {
    val result = new _TensorStruct()
    if (size.noChannels > 0) {
      (size, memorySize) match {
        case (size: Size1, memorySize: Size1) =>
          _CUDNN.setTensor4dDescriptor(
            result,
            dataType,
            noSamples,
            size.noChannels,
            1,
            size.noTuples,
            memorySize.noValues,
            memorySize.noTuples,
            1,
            1
          )

        case (size: Size2, memorySize: Size2) =>
          _CUDNN.setTensor4dDescriptor(
            result,
            dataType,
            noSamples,
            size.noChannels,
            size.dims._2,
            size.dims._1,
            memorySize.noValues,
            memorySize.noTuples,
            memorySize.dims._1,
            1
          )

        case (size: Size3, memorySize: Size3) =>
          val dims = Array(
            noSamples,
            size.noChannels,
            size.dims._3,
            size.dims._2,
            size.dims._1
          )
          val stride = Array(
            memorySize.noValues,
            memorySize.noTuples,
            memorySize.dims._1 * size.dims._2,
            memorySize.dims._1,
            1
          )
          _CUDNN.setTensorNdDescriptor(
            result,
            dataType,
            dims,
            stride
          )

        case (size: Size4, memorySize: Size4) =>
          val dims = Array(
            noSamples,
            size.noChannels,
            size.dims._4,
            size.dims._3,
            size.dims._2,
            size.dims._1
          )
          val stride = Array(
            size.noValues,
            size.noTuples,
            size.dims._1 * size.dims._2 * size.dims._3,
            size.dims._1 * size.dims._2,
            size.dims._1,
            1
          )
          _CUDNN.setTensorNdDescriptor(
            result,
            dataType,
            dims,
            stride
          )

        case _ =>
          throw new MatchError(size)
      }
    }
    result
  }


  final def nhwc(layout:   IndependentTensorLayout,
                 dataType: Int)
  : _TensorStruct = nhwc(
    layout.size,
    layout.noSamples,
    dataType
  )

  final def nhwc(size:      Size,
                 noSamples: Int,
                 dataType:  Int)
  : _TensorStruct = {
    val result = new _TensorStruct()
    size match {
      case size: Size1 =>
        _CUDNN.setTensor4dDescriptor(
          result,
          CUDNN_TENSOR_NHWC,
          dataType,
          noSamples,
          size.noChannels,
          1,
          size.noTuples
        )

      case size: Size2 =>
        _CUDNN.setTensor4dDescriptor(
          result,
          CUDNN_TENSOR_NHWC,
          dataType,
          noSamples,
          size.noChannels,
          size.dims._2,
          size.dims._1
        )

      case size: Size3 =>
        val dims = Array(
          noSamples,
          size.noChannels,
          size.dims._3,
          size.dims._2,
          size.dims._1
        )
        val stride = Array(
          size.noValues,
          1,
          size.strideZ,
          size.strideY,
          size.noChannels
        )
        _CUDNN.setTensorNdDescriptor(
          result,
          dataType,
          dims,
          stride
        )

      case size: Size4 =>
        val dims = Array(
          noSamples,
          size.noChannels,
          size.dims._4,
          size.dims._3,
          size.dims._2,
          size.dims._1
        )
        val stride = Array(
          size.noValues,
          1,
          size.strideW,
          size.strideZ,
          size.strideY,
          size.noChannels
        )
        _CUDNN.setTensorNdDescriptor(
          result,
          dataType,
          dims,
          stride
        )

      case _ =>
        throw new MatchError(size)
    }
    result
  }

}

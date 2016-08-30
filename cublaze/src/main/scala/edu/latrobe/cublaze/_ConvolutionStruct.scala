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
import edu.latrobe.kernels._
import edu.latrobe.native._
import org.bytedeco.javacpp.cudnn._

final class _ConvolutionStruct private
  extends AutoClosingPointerEx[cudnnConvolutionStruct] {

  override protected val _ptr
  : cudnnConvolutionStruct = _CUDNN.createConvolutionDescriptor()

  override protected def doClose()
  : Unit = {
    _CUDNN.destroyConvolutionDescriptor(this)
    super.doClose()
  }

  @transient
  lazy val noDims
  : Int = _CUDNN.getConvolutionNdDescriptor(this)._1

  @transient
  lazy val convMode
  : Int = _CUDNN.getConvolutionNdDescriptor(this)._2

  @transient
  lazy val padding
  : Array[Int] = {
    val noDims   = _CUDNN.getConvolutionNdDescriptor(this)._1
    val padding  = new Array[Int](noDims)
    val stride   = new Array[Int](noDims)
    val upScale  = new Array[Int](noDims)
    _CUDNN.getConvolutionNdDescriptor(
      this, padding, stride, upScale
    )
    padding
  }

  @transient
  lazy val stride
  : Array[Int] = {
    val noDims   = _CUDNN.getConvolutionNdDescriptor(this)._1
    val padding  = new Array[Int](noDims)
    val stride   = new Array[Int](noDims)
    val upScale  = new Array[Int](noDims)
    _CUDNN.getConvolutionNdDescriptor(
      this, padding, stride, upScale
    )
    stride
  }

  @transient
  lazy val upScale
  : Array[Int] = {
    val noDims   = _CUDNN.getConvolutionNdDescriptor(this)._1
    val padding  = new Array[Int](noDims)
    val stride   = new Array[Int](noDims)
    val upScale  = new Array[Int](noDims)
    _CUDNN.getConvolutionNdDescriptor(
      this, padding, stride, upScale
    )
    upScale
  }

}

private[cublaze] object _ConvolutionStruct {

  // TODO: "dataType" not yet used in all convolution settings!
  final def apply(kernel: Kernel)
  : _ConvolutionStruct = {
    val result = new _ConvolutionStruct

    // TODO: Can we replace all these with NdDescriptor?
    kernel match {
      case kernel: Kernel1 =>
        require(kernel.padding0 == kernel.padding1)
        _CUDNN.setConvolution2dDescriptor(
          result,
          0, kernel.padding0._1, // padding
          1, kernel.stride._1, // stride
          1, 1, // upScale
          CUDNN_CROSS_CORRELATION
        )

      case kernel: Kernel2 =>
        require(kernel.padding0 == kernel.padding1)
        // TODO: What is upScaleX, upScaleY?
        _CUDNN.setConvolution2dDescriptor(
          result,
          kernel.padding0._2, kernel.padding0._1, // padding
          kernel.stride._2,   kernel.stride._1, // stride
          1, 1, // upScale
          CUDNN_CROSS_CORRELATION
        )

      case kernel: Kernel3 =>
        require(kernel.padding0 == kernel.padding1)
        val padding = Array(
          kernel.padding0._3,
          kernel.padding0._2,
          kernel.padding0._1
        )
        val stride = Array(
          kernel.stride._3,
          kernel.stride._2,
          kernel.stride._1
        )
        val upScale = Array(1, 1, 1)
        _CUDNN.setConvolutionNdDescriptor(
          result,
          padding,
          stride,
          upScale,
          CUDNN_CROSS_CORRELATION,
          _RealTensorDeviceBuffer.dataType
        )

      case kernel: Kernel4 =>
        require(kernel.padding0 == kernel.padding1)
        val padding = Array(
          kernel.padding0._4,
          kernel.padding0._3,
          kernel.padding0._2,
          kernel.padding0._1
        )
        val stride = Array(
          kernel.stride._4,
          kernel.stride._3,
          kernel.stride._2,
          kernel.stride._1
        )
        val upScale = Array(1, 1, 1, 1)
        _CUDNN.setConvolutionNdDescriptor(
          result,
          padding,
          stride,
          upScale,
          CUDNN_CROSS_CORRELATION,
          _RealTensorDeviceBuffer.dataType
        )

      case _ =>
        throw new MatchError(kernel)
    }

    result
  }

}
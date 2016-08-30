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

package edu.latrobe.cublaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.cublaze._
import edu.latrobe.native._
import org.bytedeco.javacpp.cudnn._

import scala.collection.mutable

/**
  * An implementation that favors CUDA.
  */
final class ConvolutionFilter_CUDA_CUDNN(override val builder:        ConvolutionFilterBuilder,
                                         override val inputHints:     BuildHints,
                                         override val seed:           InstanceSeed,
                                         override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ConvolutionFilter_CUDA {

  // Move filter to device (converts to NCHW on the way).
  protected val filterDescPtr
  : _FilterStruct = _FilterStruct(noMaps, kernel, inputSizeHint.noChannels)

  private val defaultConvDescPtr
  : _ConvolutionStruct = _ConvolutionStruct(kernel)

  private val (defaultForwardAlgorithm, defaultBackwardAlgorithmForFilter, defaultBackwardAlgorithmForData)
  : (Int, Int, Int) = using(
    _TensorStruct.nchw(inputSizeHint, inputLayoutHint.noSamples, _RealTensorDeviceBuffer.dataType),
    _TensorStruct.nchw(outputSizeHint, outputLayoutHint.noSamples, _RealTensorDeviceBuffer.dataType)
  )((xDescPtr, yDescPtr) => {
    val forwardAlgorithm = getForwardAlgorithm(
      defaultConvDescPtr, xDescPtr, yDescPtr
    )
    val backwardAlgorithmForFilter = getBackwardAlgorithmForFilter(
      defaultConvDescPtr, xDescPtr, yDescPtr
    )
    val backwardAlgorithmForData = getBackwardAlgorithmForData(
      defaultConvDescPtr, xDescPtr, yDescPtr
    )
    (forwardAlgorithm, backwardAlgorithmForFilter, backwardAlgorithmForData)
  })

  override protected def doClose()
  : Unit = {
    filterDescPtr.close()
    defaultConvDescPtr.close()
    super.doClose()
  }

  @inline
  private def getForwardAlgorithm(convDescPtr: _ConvolutionStruct,
                                  xDescPtr:    _TensorStruct,
                                  yDescPtr:    _TensorStruct)
  : Int = _CUDNN.getConvolutionForwardAlgorithm(
    device,
    xDescPtr,
    filterDescPtr,
    convDescPtr,
    yDescPtr,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
    CUBLAZE_SCRATCH_BUFFER_SIZE
  )

  @inline
  private def getBackwardAlgorithmForFilter(convDescPtr: _ConvolutionStruct,
                                            xDescPtr:    _TensorStruct,
                                            dyDescPtr:   _TensorStruct)
  : Int = _CUDNN.getConvolutionBackwardFilterAlgorithm(
    device,
    xDescPtr,
    dyDescPtr,
    convDescPtr,
    filterDescPtr,
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
    CUBLAZE_SCRATCH_BUFFER_SIZE
  )

  @inline
  private def getBackwardAlgorithmForData(convDescPtr: _ConvolutionStruct,
                                          dxDescPtr:   _TensorStruct,
                                          dyDescPtr:   _TensorStruct)
  : Int = _CUDNN.getConvolutionBackwardDataAlgorithm(
    device,
    filterDescPtr,
    dyDescPtr,
    convDescPtr,
    dxDescPtr,
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
    CUBLAZE_SCRATCH_BUFFER_SIZE
  )


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  override def refresh(): Unit = {}


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input:  CUDARealTensor,
                                   output: CUDARealTensor)
  : PredictContext = {
    // Create descriptors for convolution.
    val convDescPtr = {
      if (input.layout == inputLayoutHint) {
        defaultConvDescPtr
      }
      else {
        _ConvolutionStruct(kernel)
      }
    }

    // Request parameters for convolution.
    val algorithm = {
      if (convDescPtr eq defaultConvDescPtr) {
        defaultForwardAlgorithm
      }
      else {
        getForwardAlgorithm(convDescPtr, input.desc, output.desc)
      }
    }
    if (logger.isTraceEnabled) {
      logger.trace(
        s"CUDNN forward algorithm = $algorithm${if (convDescPtr eq defaultConvDescPtr) " (default)" else ""}"
      )
    }

    // Invoke CUDA.
    _CUDNN.convolutionForward(
      device,
      _RealTensorNativeReal.one,
      input.desc, input.data.ptr,
      filterDescPtr, filter.data.ptr,
      convDescPtr,
      algorithm,
      _RealTensorNativeReal.zero,
      output.desc, output.data.ptr
    )

    // Store input tensor in context if newly created during fprop.
    if (convDescPtr eq defaultConvDescPtr) {
      EmptyContext
    }
    else {
      ConvolutionFilter_CUDA_CUDNN_Context(convDescPtr)
    }
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveFilterGradients(input:   CUDARealTensor,
                                                 context: PredictContext,
                                                 error:   CUDARealTensor,
                                                 sink:    CUDARealTensor)
  : Unit = {
    // Get convolution descriptor.
    val convDescPtr = context match {
      case EmptyContext =>
        defaultConvDescPtr
      case context: ConvolutionFilter_CUDA_CUDNN_Context =>
        context.convDescPtr
      case _ =>
        throw new MatchError(context)
    }

    // Request parameters for convolution.
    val algorithm = {
      if (convDescPtr eq defaultConvDescPtr) {
        defaultBackwardAlgorithmForFilter
      }
      else {
        getBackwardAlgorithmForFilter(convDescPtr, input.desc, error.desc)
      }
    }
    if (logger.isTraceEnabled) {
      logger.trace(
        s"CUDNN backward filter algorithm = $algorithm${if (convDescPtr eq defaultConvDescPtr) " (default)" else ""}"
      )
    }

    // Invoke CUDA.
    _CUDNN.convolutionBackwardFilter(
      device,
      _RealTensorNativeReal.one,
      input.desc, input.data.ptr,
      error.desc, error.data.ptr,
      convDescPtr,
      algorithm,
      _RealTensorNativeReal.one,
      filterDescPtr, sink.data.ptr
    )
  }

  override protected def doDeriveInputError(context:  PredictContext,
                                            oldError: CUDARealTensor,
                                            newError: CUDARealTensor)
  : Unit = {
    // Get convolution descriptor.
    val convDescPtr = context match {
      case EmptyContext =>
        defaultConvDescPtr
      case context: ConvolutionFilter_CUDA_CUDNN_Context =>
        context.convDescPtr
      case _ =>
        throw new MatchError(context)
    }

    // Request parameters for convolution.
    val algorithm = {
      if (convDescPtr eq defaultConvDescPtr) {
        defaultBackwardAlgorithmForData
      }
      else {
        getBackwardAlgorithmForData(convDescPtr, newError.desc, oldError.desc)
      }
    }
    if (logger.isTraceEnabled) {
      val defStr = if (convDescPtr eq defaultConvDescPtr) " (default)" else ""
      logger.trace(s"CUDNN backward data algorithm = $algorithm$defStr")
    }

    // Invoke CUDA.
    _CUDNN.convolutionBackwardData(
      device,
      _RealTensorNativeReal.one,
      filterDescPtr, filter.data.ptr,
      oldError.desc, oldError.data.ptr,
      convDescPtr,
      algorithm,
      _RealTensorNativeReal.zero,
      newError.desc, newError.data.ptr
    )
  }

}

object ConvolutionFilter_CUDA_CUDNN_Description
  extends ModuleVariant_CUDA_Description[ConvolutionFilterBuilder] {

  override protected def doScore(builder:   ConvolutionFilterBuilder,
                                 hints:     BuildHints,
                                 scorePrev: Int,
                                 reasons:   mutable.ArrayBuilder[String])
  : Int = {
    // TODO: Check for unsupported combinations of settings.
    super.doScore(builder, hints, scorePrev, reasons)
  }

  override def build(builder:        ConvolutionFilterBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ConvolutionFilter_CUDA_CUDNN = new ConvolutionFilter_CUDA_CUDNN(
    builder, hints, seed, weightsBuilder
  )

}

final case class ConvolutionFilter_CUDA_CUDNN_Context(convDescPtr: _ConvolutionStruct)
  extends PredictContext
    with AutoClosing {

  override protected def doClose()
  : Unit = {
    convDescPtr.close()
    super.doClose()
  }

}

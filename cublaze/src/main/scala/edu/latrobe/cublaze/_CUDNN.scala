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

import java.nio.charset._

import edu.latrobe._
import edu.latrobe.native._
import org.bytedeco.javacpp.{DoublePointer, FloatPointer, Pointer}
import org.bytedeco.javacpp.cudnn._

private[cublaze] object _CUDNN {

  final val version
  : Long = cudnnGetVersion()

  @inline
  final private def check(resultCode: Int)
  : Unit = {
    if (resultCode != CUDNN_STATUS_SUCCESS) {
      using(cudnnGetErrorString(resultCode))(ptr => {
        val str = ptr.getString(StandardCharsets.US_ASCII.name)
        throw new InternalError(s"CUDNN: $str")
      })
    }
  }

  @inline
  final def activationBackward(device:  LogicalDevice,
                               actDesc: _ActivationStruct,
                               alpha:   _RealTensorNativeReal,
                               yDesc:   _TensorStruct, yPtr:  _RealTensorPointer,
                               dyDesc:  _TensorStruct, dyPtr: _RealTensorPointer,
                               xDesc:   _TensorStruct, xPtr:  _RealTensorPointer,
                               beta:    _RealTensorNativeReal,
                               dxDesc:  _TensorStruct, dxPtr: _RealTensorPointer)
  : Unit = {
    val result = cudnnActivationBackward(
      device.dnnContextPtr,
      actDesc.ptr,
      alpha.ptr,
      yDesc.ptr,  yPtr,
      dyDesc.ptr, dyPtr,
      xDesc.ptr,  xPtr,
      beta.ptr,
      dxDesc.ptr, dxPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def activationForward(device:  LogicalDevice,
                              actDesc: _ActivationStruct,
                              alpha:   _RealTensorNativeReal,
                              xDesc:   _TensorStruct, xPtr: _RealTensorPointer,
                              beta:    _RealTensorNativeReal,
                              yDesc:   _TensorStruct, yPtr: _RealTensorPointer)
  : Unit = {
    val result = cudnnActivationForward(
      device.dnnContextPtr,
      actDesc.ptr,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def addTensor(device: LogicalDevice,
                      alpha: _RealTensorNativeReal,
                      xDesc: _TensorStruct, xPtr: _RealTensorPointer,
                      beta:  _RealTensorNativeReal,
                      yDesc: _TensorStruct, yPtr: _RealTensorPointer)
  : Unit = {
    val result = cudnnAddTensor(
      device.dnnContextPtr,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def batchNormalizationBackward(device:        LogicalDevice,
                                       mode:          Int,
                                       alpha:         _RealTensorNativeReal,
                                       xDesc:         _TensorStruct, xPtr:  _RealTensorPointer,
                                       dyDesc:        _TensorStruct, dyPtr: _RealTensorPointer,
                                       beta:          _RealTensorNativeReal,
                                       dxDesc:        _TensorStruct, dxPtr: _RealTensorPointer,
                                       bnDesc:        _TensorStruct,
                                       bnAlpha:       _RealTensorNativeReal,
                                       bnScale:       _RealTensorDeviceBuffer,
                                       bnBeta:        _RealTensorNativeReal,
                                       bnDScale:      _RealTensorDeviceBuffer,
                                       bnDBias:       _RealTensorDeviceBuffer,
                                       epsilon:       Real,
                                       meanCache:     _RealTensorDeviceBuffer,
                                       varianceCache: _RealTensorDeviceBuffer)
  : Unit = {
    val eps = Math.max(CUDNN_BN_MIN_EPSILON, epsilon)
    if (eps != epsilon) {
      logger.warn("Selected epsilon for BN was below CUDNN_BN_MIN_EPSILON.")
    }

    val result = cudnnBatchNormalizationBackward(
      device.dnnContextPtr,
      mode,
      alpha.ptr,
      beta.ptr,
      bnAlpha.ptr,
      bnBeta.ptr,
      xDesc.ptr,  xPtr,
      dyDesc.ptr, dyPtr,
      dxDesc.ptr, dxPtr,
      bnDesc.ptr,
      bnScale.ptr,
      bnDScale.ptr,
      bnDBias.ptr,
      eps,
      meanCache.ptr,
      varianceCache.ptr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def batchNormalizationForwardInference(device:   LogicalDevice,
                                               mode:     Int,
                                               alpha:    _RealTensorNativeReal,
                                               xDesc:    _TensorStruct, xPtr: _RealTensorPointer,
                                               beta:     _RealTensorNativeReal,
                                               yDesc:    _TensorStruct, yPtr: _RealTensorPointer,
                                               bnDesc:   _TensorStruct,
                                               mu:       _RealTensorDeviceBuffer,
                                               variance: _RealTensorDeviceBuffer,
                                               bnScale:  _RealTensorDeviceBuffer,
                                               bnBias:   _RealTensorDeviceBuffer,
                                               epsilon:  Real)
  : Unit = {
    val eps = Math.max(CUDNN_BN_MIN_EPSILON, epsilon)
    if (eps != epsilon) {
      logger.warn("Selected epsilon for BN was below CUDNN_BN_MIN_EPSILON.")
    }

    val result = cudnnBatchNormalizationForwardInference(
      device.dnnContextPtr,
      mode,
      alpha.ptr,
      beta.ptr,
      xDesc.ptr, xPtr,
      yDesc.ptr, yPtr,
      bnDesc.ptr,
      bnScale.ptr,
      bnBias.ptr,
      mu.ptr,
      variance.ptr,
      eps
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def batchNormalizationForwardTraining(device:          LogicalDevice,
                                              mode:            Int,
                                              alpha:           _RealTensorNativeReal,
                                              xDesc:           _TensorStruct, xPtr: _RealTensorPointer,
                                              beta:            _RealTensorNativeReal,
                                              yDesc:           _TensorStruct, yPtr: _RealTensorPointer,
                                              bnDesc:          _TensorStruct,
                                              runningMean:     _RealTensorDeviceBuffer,
                                              runningVariance: _RealTensorDeviceBuffer,
                                              t:               Real,
                                              bnScale:         _RealTensorDeviceBuffer,
                                              bnBias:          _RealTensorDeviceBuffer,
                                              epsilon:         Real)
  : (_RealTensorDeviceBuffer, _RealTensorDeviceBuffer) = {
    val meanCache     = runningMean.allocateSibling()
    val varianceCache = runningVariance.allocateSibling()
    val eps           = Math.max(CUDNN_BN_MIN_EPSILON, epsilon)
    if (eps != epsilon) {
      logger.warn("Selected epsilon for BN was below CUDNN_BN_MIN_EPSILON.")
    }

    val result = cudnnBatchNormalizationForwardTraining(
      device.dnnContextPtr,
      mode,
      alpha.ptr,
      beta.ptr,
      xDesc.ptr, xPtr,
      yDesc.ptr, yPtr,
      bnDesc.ptr,
      bnScale.ptr,
      bnBias.ptr,
      DoubleEx(t),
      runningMean.ptr,
      runningVariance.ptr,
      eps,
      meanCache.ptr,
      varianceCache.ptr
    )
    check(result)
    device.trySynchronize()

    (meanCache, varianceCache)
  }

  @inline
  final def convolutionBackwardBias(device: LogicalDevice,
                                    alpha:  _RealTensorNativeReal,
                                    dyDesc: _TensorStruct, dyPtr: _RealTensorPointer,
                                    beta:   _RealTensorNativeReal,
                                    dwDesc: _TensorStruct, dwPtr: _RealTensorPointer)
  : Unit = {
    val result = cudnnConvolutionBackwardBias(
      device.dnnContextPtr,
      alpha.ptr,
      dyDesc.ptr, dyPtr,
      beta.ptr,
      dwDesc.ptr, dwPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def convolutionBackwardData(device:    LogicalDevice,
                                    alpha:     _RealTensorNativeReal,
                                    wDesc:     _FilterStruct, wPtr:  _RealTensorPointer,
                                    dyDesc:    _TensorStruct, dyPtr: _RealTensorPointer,
                                    convDesc:  _ConvolutionStruct,
                                    algorithm: Int,
                                    beta:      _RealTensorNativeReal,
                                    dxDesc:    _TensorStruct, dxPtr: _RealTensorPointer)
  : Unit = {
    // Make sure scratch buffer is large enough.
    val wsSize = {
      using(NativeSizeT.allocate(1L))(tmp => {
        val tmpPtr = tmp.ptr
        val result = cudnnGetConvolutionBackwardDataWorkspaceSize(
          device.dnnContextPtr,
          wDesc.ptr,
          dyDesc.ptr,
          convDesc.ptr,
          dxDesc.ptr,
          algorithm,
          tmpPtr
        )
        check(result)
        tmpPtr.get()
      })
    }
    assume(wsSize <= CUBLAZE_SCRATCH_BUFFER_SIZE)

    // Do convolution.
    if (wsSize == 0L) {
      val result = cudnnConvolutionBackwardData(
        device.dnnContextPtr,
        alpha.ptr,
        wDesc.ptr,  wPtr,
        dyDesc.ptr, dyPtr,
        convDesc.ptr,
        algorithm,
        NativeByte.NULL.ptr, 0L,
        beta.ptr,
        dxDesc.ptr, dxPtr
      )
      check(result)
      device.trySynchronize()
    }
    else {
      val wsPtr = device.scratchBuffer.asBytePtr
      val result = cudnnConvolutionBackwardData(
        device.dnnContextPtr,
        alpha.ptr,
        wDesc.ptr,  wPtr,
        dyDesc.ptr, dyPtr,
        convDesc.ptr,
        algorithm,
        wsPtr, wsPtr.capacity(),
        beta.ptr,
        dxDesc.ptr, dxPtr
      )
      check(result)
      device.trySynchronize()
    }
  }

  @inline
  final def convolutionBackwardFilter(device:    LogicalDevice,
                                      alpha:     _RealTensorNativeReal,
                                      xDesc:     _TensorStruct, xPtr:  _RealTensorPointer,
                                      dyDesc:    _TensorStruct, dyPtr: _RealTensorPointer,
                                      convDesc:  _ConvolutionStruct,
                                      algorithm: Int,
                                      beta:      _RealTensorNativeReal,
                                      dwDesc:    _FilterStruct, dwPtr: _RealTensorPointer)
  : Unit = {
    // Make sure scratch buffer is large enough.
    val wsSize = {
      using(NativeSizeT.allocate(1L))(tmp => {
        val tmpPtr = tmp.ptr
        val result = cudnnGetConvolutionBackwardFilterWorkspaceSize(
          device.dnnContextPtr,
          xDesc.ptr,
          dyDesc.ptr,
          convDesc.ptr,
          dwDesc.ptr,
          algorithm,
          tmpPtr
        )
        check(result)
        tmpPtr.get()
      })
    }
    assume(wsSize <= CUBLAZE_SCRATCH_BUFFER_SIZE)

    // Do convolution.
    if (wsSize == 0L) {
      val result = cudnnConvolutionBackwardFilter(
        device.dnnContextPtr,
        alpha.ptr,
        xDesc.ptr,  xPtr,
        dyDesc.ptr, dyPtr,
        convDesc.ptr,
        algorithm,
        NativeByte.NULL.ptr, 0L,
        beta.ptr,
        dwDesc.ptr, dwPtr
      )
      check(result)
      device.trySynchronize()
    }
    else {
      val wsPtr  = device.scratchBuffer.asBytePtr
      val result = cudnnConvolutionBackwardFilter(
        device.dnnContextPtr,
        alpha.ptr,
        xDesc.ptr,  xPtr,
        dyDesc.ptr, dyPtr,
        convDesc.ptr,
        algorithm,
        wsPtr, wsPtr.capacity(),
        beta.ptr,
        dwDesc.ptr, dwPtr
      )
      check(result)
      device.trySynchronize()
    }
  }

  @inline
  final def convolutionForward(device:    LogicalDevice,
                               alpha:     _RealTensorNativeReal,
                               xDesc:     _TensorStruct, xPtr:  _RealTensorPointer,
                               wDesc:     _FilterStruct, wPtr: _RealTensorPointer,
                               convDesc:  _ConvolutionStruct,
                               algorithm: Int,
                               beta:      _RealTensorNativeReal,
                               yDesc:     _TensorStruct, yPtr: _RealTensorPointer)
  : Unit = {
    // Make sure scratch buffer is large enough.
    val wsSize = {
      using(NativeSizeT.allocate(1L))(tmp => {
        val tmpPtr = tmp.ptr
        val result = cudnnGetConvolutionForwardWorkspaceSize(
          device.dnnContextPtr,
          xDesc.ptr,
          wDesc.ptr,
          convDesc.ptr,
          yDesc.ptr,
          algorithm,
          tmpPtr
        )
        check(result)
        tmpPtr.get()
      })
    }
    assume(wsSize <= CUBLAZE_SCRATCH_BUFFER_SIZE)

    // Do convolution.
    if (wsSize == 0L) {
      val result = cudnnConvolutionForward(
        device.dnnContextPtr,
        alpha.ptr,
        xDesc.ptr, xPtr,
        wDesc.ptr, wPtr,
        convDesc.ptr,
        algorithm,
        NativeByte.NULL.ptr, 0L,
        beta.ptr,
        yDesc.ptr, yPtr
      )
      check(result)
      device.trySynchronize()
    }
    else {
      val wsPtr = device.scratchBuffer.asBytePtr
      val result = cudnnConvolutionForward(
        device.dnnContextPtr,
        alpha.ptr,
        xDesc.ptr, xPtr,
        wDesc.ptr, wPtr,
        convDesc.ptr,
        algorithm,
        wsPtr, wsPtr.capacity(),
        beta.ptr,
        yDesc.ptr, yPtr
      )
      check(result)
      device.trySynchronize()
    }
  }

  @inline
  final def create()
  : cudnnContext = {
    val ptr    = new cudnnContext
    val result = cudnnCreate(ptr)
    _CUDNN.check(result)
    ptr
  }

  @inline
  final def createActivationDescriptor()
  : cudnnActivationStruct = {
    val ptr    = new cudnnActivationStruct
    val result = cudnnCreateActivationDescriptor(ptr)
    check(result)
    ptr
  }

  @inline
  final def createConvolutionDescriptor()
  : cudnnConvolutionStruct = {
    val ptr    = new cudnnConvolutionStruct
    val result = cudnnCreateConvolutionDescriptor(ptr)
    check(result)
    ptr
  }

  @inline
  final def createDropoutDescriptor()
  : cudnnDropoutStruct = {
    val ptr    = new cudnnDropoutStruct
    val result = cudnnCreateDropoutDescriptor(ptr)
    check(result)
    ptr
  }

  @inline
  final def createFilterDescriptor()
  : cudnnFilterStruct = {
    val ptr    = new cudnnFilterStruct
    val result = cudnnCreateFilterDescriptor(ptr)
    check(result)
    ptr
  }

  @inline
  final def createLRNDescriptor()
  : cudnnLRNStruct = {
    val ptr    = new cudnnLRNStruct
    val result = cudnnCreateLRNDescriptor(ptr)
    check(result)
    ptr
  }

  @inline
  final def createOpTensorDescriptor()
  : cudnnOpTensorStruct = {
    val ptr    = new cudnnOpTensorStruct
    val result = cudnnCreateOpTensorDescriptor(ptr)
    check(result)
    ptr
  }

  @inline
  final def createPoolingDescriptor()
  : cudnnPoolingStruct = {
    val ptr    = new cudnnPoolingStruct
    val result = cudnnCreatePoolingDescriptor(ptr)
    check(result)
    ptr
  }

  @inline
  final def createTensorDescriptor()
  : cudnnTensorStruct = {
    val ptr    = new cudnnTensorStruct
    val result = cudnnCreateTensorDescriptor(ptr)
    check(result)
    ptr
  }

  @inline
  final def destroy(device: LogicalDevice)
  : Unit = {
    val result = cudnnDestroy(device.dnnContextPtr)
    check(result)
  }

  @inline
  final def destroyActivationDescriptor(actDesc: _ActivationStruct)
  : Unit = {
    val result = cudnnDestroyActivationDescriptor(actDesc.ptr)
    check(result)
  }

  @inline
  final def destroyConvolutionDescriptor(convDesc: _ConvolutionStruct)
  : Unit = {
    val result = cudnnDestroyConvolutionDescriptor(convDesc.ptr)
    check(result)
  }

  @inline
  final def destroyDropoutDescriptor(dropDesc: _DropoutStruct)
  : Unit = {
    val result = cudnnDestroyDropoutDescriptor(dropDesc.ptr)
    check(result)
  }

  @inline
  final def destroyFilterDescriptor(filterDesc: _FilterStruct)
  : Unit = {
    val result = cudnnDestroyFilterDescriptor(filterDesc.ptr)
    check(result)
  }

  @inline
  final def destroyLRNDescriptor(lrnDesc: _LRNStruct)
  : Unit = {
    val result = cudnnDestroyLRNDescriptor(lrnDesc.ptr)
    check(result)
  }

  @inline
  final def destroyOpTensorDescriptor(opTensorDesc: _OpTensorStruct)
  : Unit = {
    val result = cudnnDestroyOpTensorDescriptor(opTensorDesc.ptr)
    check(result)
  }

  @inline
  final def destroyPoolingDescriptor(poolDesc: _PoolingStruct)
  : Unit = {
    val result = cudnnDestroyPoolingDescriptor(poolDesc.ptr)
    check(result)
  }

  @inline
  final def destroyTensorDescriptor(tensorDesc: _TensorStruct)
  : Unit = {
    val result = cudnnDestroyTensorDescriptor(tensorDesc.ptr)
    check(result)
  }

  @inline
  final def divisiveNormalizationBackward(device:  LogicalDevice,
                                          lrnDesc: _LRNStruct,
                                          mode:    Int,
                                          alpha:   _RealTensorNativeReal,
                                          xDesc:   _TensorStruct, xPtr: _RealTensorPointer,
                                          mean:    _RealDeviceBuffer,
                                          dyData:  _RealDeviceBuffer,
                                          tmp1:    _RealDeviceBuffer,
                                          tmp2:    _RealDeviceBuffer,
                                          beta:    _RealTensorNativeReal,
                                          dxDesc:  _TensorStruct, dxPtr: _RealTensorPointer,
                                          dMean:   _RealDeviceBuffer)
  : Unit = {
    val result = cudnnDivisiveNormalizationBackward(
      device.dnnContextPtr,
      lrnDesc.ptr,
      mode,
      alpha.ptr,
      xDesc.ptr, xPtr,
      mean.ptr,
      dyData.ptr,
      tmp1.ptr,
      tmp2.ptr,
      beta.ptr,
      dxDesc.ptr, dxPtr,
      dMean.ptr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def divisiveNormalizationForward(device:  LogicalDevice,
                                         lrnDesc: _LRNStruct,
                                         mode:    Int,
                                         alpha:   _RealTensorNativeReal,
                                         xDesc:   _TensorStruct, xPtr: _RealTensorPointer,
                                         mean:    _RealDeviceBuffer,
                                         tmp1:    _RealDeviceBuffer,
                                         tmp2:    _RealDeviceBuffer,
                                         beta:    _RealTensorNativeReal,
                                         yDesc:   _TensorStruct, yPtr: _RealTensorPointer)
  : Unit = {
    val result = cudnnDivisiveNormalizationForward(
      device.dnnContextPtr,
      lrnDesc.ptr,
      mode,
      alpha.ptr,
      xDesc.ptr, xPtr,
      mean.ptr,
      tmp1.ptr,
      tmp2.ptr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def dropoutBackward(device:   LogicalDevice,
                            dropDesc: _DropoutStruct,
                            dyDesc:   _TensorStruct, dyPtr: _RealTensorPointer,
                            dxDesc:   _TensorStruct, dxPtr: _RealTensorPointer,
                            cache:    _ByteDeviceBuffer)
  : Unit = {
    val cachePtr = cache.ptr
    val result = cudnnDropoutBackward(
      device.dnnContextPtr,
      dropDesc.ptr,
      dyDesc.ptr, dyPtr,
      dxDesc.ptr, dxPtr,
      cachePtr,
      cachePtr.capacity()
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def dropoutForward(device:   LogicalDevice,
                           dropDesc: _DropoutStruct,
                           xDesc:    _TensorStruct, xPtr: _RealTensorPointer,
                           yDesc:    _TensorStruct, yPtr: _RealTensorPointer)
  : _ByteDeviceBuffer = {
    val cache = {
      using(NativeSizeT.allocate(1L))(tmp => {
        val tmpPtr = tmp.ptr
        val result = cudnnDropoutGetReserveSpaceSize(xDesc.ptr, tmpPtr)
        check(result)
        _ByteDeviceBuffer(device, tmpPtr.get())
      })
    }
    val cachePtr = cache.ptr
    val result = cudnnDropoutForward(
      device.dnnContextPtr,
      dropDesc.ptr,
      xDesc.ptr, xPtr,
      yDesc.ptr, yPtr,
      cachePtr,
      cachePtr.capacity()
    )
    check(result)
    device.trySynchronize()
    cache
  }

  @inline
  final def dropoutGetStateSize(device: LogicalDevice)
  : Long = {
    using(NativeSizeT.allocate(1L))(tmp => {
      val tmpPtr = tmp.ptr
      val result = cudnnDropoutGetStatesSize(device.dnnContextPtr, tmpPtr)
      check(result)
      device.trySynchronize()
      tmpPtr.get()
    })
  }

  @inline
  final def getActivationDescriptor(actDesc: _ActivationStruct)
  : (Int, Int, Double) = {
    val mode          = new Array[Int](1)
    val reluNaNOpt    = new Array[Int](1)
    val reluThreshold = new Array[Double](1)
    val result = cudnnGetActivationDescriptor(
      actDesc.ptr,
      mode,
      reluNaNOpt,
      reluThreshold
    )
    check(result)
    (mode(0), reluNaNOpt(0), reluThreshold(0))
  }

  @inline
  final def getConvolutionBackwardDataAlgorithm(device:      LogicalDevice,
                                                wDesc:       _FilterStruct,
                                                dyDesc:      _TensorStruct,
                                                convDesc:    _ConvolutionStruct,
                                                dxDesc:      _TensorStruct,
                                                preference:  Int,
                                                memoryLimit: Long)
  : Int = {
    val algorithm = new Array[Int](1)
    val result = cudnnGetConvolutionBackwardDataAlgorithm(
      device.dnnContextPtr,
      wDesc.ptr,
      dyDesc.ptr,
      convDesc.ptr,
      dxDesc.ptr,
      preference,
      memoryLimit,
      algorithm
    )
    check(result)
    algorithm(0)
  }


  @inline
  final def getConvolutionBackwardFilterAlgorithm(device:      LogicalDevice,
                                                  xDesc:       _TensorStruct,
                                                  dyDesc:      _TensorStruct,
                                                  convDesc:    _ConvolutionStruct,
                                                  dwDesc:      _FilterStruct,
                                                  preference:  Int,
                                                  memoryLimit: Long)
  : Int = {
    val algorithm = new Array[Int](1)
    val result = cudnnGetConvolutionBackwardFilterAlgorithm(
      device.dnnContextPtr,
      xDesc.ptr,
      dyDesc.ptr,
      convDesc.ptr,
      dwDesc.ptr,
      preference,
      memoryLimit,
      algorithm
    )
    check(result)
    algorithm(0)
  }

  @inline
  final def getConvolutionForwardAlgorithm(device:      LogicalDevice,
                                           xDesc:       _TensorStruct,
                                           wDesc:       _FilterStruct,
                                           convDesc:    _ConvolutionStruct,
                                           yDesc:       _TensorStruct,
                                           preference:  Int,
                                           memoryLimit: Long)
  : Int = {
    val algorithm = new Array[Int](1)
    val result = cudnnGetConvolutionForwardAlgorithm(
      device.dnnContextPtr,
      xDesc.ptr,
      wDesc.ptr,
      convDesc.ptr,
      yDesc.ptr,
      preference,
      memoryLimit,
      algorithm
    )
    check(result)
    algorithm(0)
  }

  @inline
  final def getConvolutionNdDescriptor(convDesc: _ConvolutionStruct)
  : (Int, Int, Int) = {
    using(NativeInt.allocate(1L), NativeInt.allocate(1L), NativeInt.allocate(1L))(
      (noDims, convMode, dataType) => {
        val noDimsPtr   = noDims.ptr
        val convModePtr = convMode.ptr
        val dataTypePtr = dataType.ptr
        val result = cudnnGetConvolutionNdDescriptor(
          convDesc.ptr,
          0,
          noDimsPtr,
          NativeInt.NULL.ptr,
          NativeInt.NULL.ptr,
          NativeInt.NULL.ptr,
          convModePtr,
          dataTypePtr
        )
        check(result)
        (noDimsPtr.get(), convModePtr.get(), dataTypePtr.get())
      }
    )
  }

  @inline
  final def getConvolutionNdDescriptor(convDesc: _ConvolutionStruct,
                                       padding:  Array[Int],
                                       stride:   Array[Int],
                                       upScale:  Array[Int])
  : (Int, Int) = {
    require(
      padding.length == stride.length &&
      padding.length == upScale.length
    )
    val noDimsOut = new Array[Int](1)
    val convMode  = new Array[Int](1)
    val dataType  = new Array[Int](1)
    val result = cudnnGetConvolutionNdDescriptor(
      convDesc.ptr,
      padding.length,
      noDimsOut,
      padding,
      stride,
      upScale,
      convMode,
      dataType
    )
    check(result)
    assume(noDimsOut(0) == padding.length)
    (convMode(0), dataType(0))
  }

  @inline
  final def getFilterNdDescriptor(filterDesc: _FilterStruct)
  : (Int, Int, Int) = {
    using(NativeInt.allocate(1L), NativeInt.allocate(1L), NativeInt.allocate(1L))(
      (noDims, dataType, dataFormat) => {
        val noDimsPtr   = noDims.ptr
        val dataTypePtr = dataType.ptr
        val dataFormatPtr = dataFormat.ptr
        val result = cudnnGetFilterNdDescriptor(
          filterDesc.ptr,
          0,
          dataTypePtr,
          dataFormatPtr,
          noDimsPtr,
          NativeInt.NULL.ptr
        )
        check(result)

        // Extract the bits we actually need.
        (noDimsPtr.get(), dataTypePtr.get(), dataFormatPtr.get())
      }
    )
  }

  @inline
  final def getFilterNdDescriptor(filterDesc: _FilterStruct,
                                  dims:       Array[Int])
  : (Int, Int) = {
    val noDimsOut  = new Array[Int](1)
    val dataType   = new Array[Int](1)
    val dataFormat = new Array[Int](1)
    val result = cudnnGetFilterNdDescriptor(
      filterDesc.ptr,
      dims.length,
      dataType,
      dataFormat,
      noDimsOut,
      dims
    )
    check(result)
    assume(noDimsOut(0) == dims.length)
    (dataType(0), dataFormat(0))
  }

  @inline
  final def getLRNDescriptor(lrnDesc: _LRNStruct,
                             n:       Int,
                             alpha:   Double,
                             beta:    Double,
                             k:       Double)
  : (Int, Double, Double, Double) = {
    val n     = new Array[Int](1)
    val alpha = new Array[Double](1)
    val beta  = new Array[Double](1)
    val k     = new Array[Double](1)
    val result = cudnnGetLRNDescriptor(
      lrnDesc.ptr,
      n,
      alpha,
      beta,
      k
    )
    check(result)
    (n(0), alpha(0), beta(0), k(0))
  }

  @inline
  final def getOpTensorDescriptor(opTensorDesc: _OpTensorStruct)
  : (Int, Int, Int) = {
    val op                 = new Array[Int](1)
    val comparisonDataType = new Array[Int](1)
    val nanOpt             = new Array[Int](1)
    val result = cudnnGetOpTensorDescriptor(
      opTensorDesc.ptr,
      op,
      comparisonDataType,
      nanOpt
    )
    check(result)
    (op(0), comparisonDataType(0), nanOpt(0))
  }

  @inline
  final def getTensor4dDescriptor(tensorDesc: _TensorStruct)
  : (Int, Int, Int, Int, Int, Int, Int, Int, Int) = {
    val dataType = new Array[Int](1)
    val n        = new Array[Int](1)
    val c        = new Array[Int](1)
    val h        = new Array[Int](1)
    val w        = new Array[Int](1)
    val nStride  = new Array[Int](1)
    val cStride  = new Array[Int](1)
    val hStride  = new Array[Int](1)
    val wStride  = new Array[Int](1)
    val result = cudnnGetTensor4dDescriptor(
      tensorDesc.ptr,
      dataType,
      n,
      c,
      h,
      w,
      nStride,
      cStride,
      hStride,
      wStride
    )
    check(result)
    (dataType(0), n(0), c(0), h(0), w(0), nStride(0), cStride(0), hStride(0), wStride(0))
  }

  @inline
  final def getTensorNdDescriptor(tensorDesc: _TensorStruct)
  : (Int, Int) = {
    using(NativeInt.allocate(1L), NativeInt.allocate(1L))(
      (noDims, dataType) => {
        val noDimsPtr   = noDims.ptr
        val dataTypePtr = dataType.ptr
        val result = cudnnGetTensorNdDescriptor(
          tensorDesc.ptr,
          0,
          dataTypePtr,
          noDimsPtr,
          NativeInt.NULL.ptr,
          NativeInt.NULL.ptr
        )
        check(result)

        // Extract the bits we actually need.
        (noDimsPtr.get, dataTypePtr.get)
      }
    )
  }

  @inline
  final def getTensorNdDescriptor(tensorDesc: _TensorStruct,
                                  dims:       Array[Int],
                                  stride:     Array[Int])
  : Int = {
    require(dims.length == stride.length)
    val noDims   = new Array[Int](1)
    val dataType = new Array[Int](1)
    val result = cudnnGetTensorNdDescriptor(
      tensorDesc.ptr,
      dims.length,
      dataType,
      noDims,
      dims,
      stride
    )
    check(result)
    assume(noDims(0) == dims.length)
    dataType(0)
  }

  @inline
  final def lrnCrossChannelBackward(device:  LogicalDevice,
                                    lrnDesc: _LRNStruct,
                                    mode:    Int,
                                    alpha:   _RealTensorNativeReal,
                                    yDesc:   _TensorStruct, yPtr:  _RealTensorPointer,
                                    dyDesc:  _TensorStruct, dyPtr: _RealTensorPointer,
                                    xDesc:   _TensorStruct, xPtr:  _RealTensorPointer,
                                    beta:    _RealTensorNativeReal,
                                    dxDesc:  _TensorStruct, dxPtr: _RealTensorPointer)
  : Unit = {
    val result = cudnnLRNCrossChannelBackward(
      device.dnnContextPtr,
      lrnDesc.ptr,
      mode,
      alpha.ptr,
      yDesc.ptr,  yPtr,
      dyDesc.ptr, dyPtr,
      xDesc.ptr,  xPtr,
      beta.ptr,
      dxDesc.ptr, dxPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def lrnCrossChannelForward(device:  LogicalDevice,
                                   lrnDesc: _LRNStruct,
                                   mode:    Int,
                                   alpha:   _RealTensorNativeReal,
                                   xDesc:   _TensorStruct, xPtr: _RealTensorPointer,
                                   beta:    _RealTensorNativeReal,
                                   yDesc:   _TensorStruct, yPtr: _RealTensorPointer)
  : Unit = {
    val result = cudnnLRNCrossChannelForward(
      device.dnnContextPtr,
      lrnDesc.ptr,
      mode,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def poolingBackward(device:  LogicalDevice,
                            poolDesc: _PoolingStruct,
                            alpha:    _RealTensorNativeReal,
                            yDesc:    _TensorStruct, yPtr:  _RealTensorPointer,
                            dyDesc:   _TensorStruct, dyPtr: _RealTensorPointer,
                            xDesc:    _TensorStruct, xPtr:  _RealTensorPointer,
                            beta:     _RealTensorNativeReal,
                            dxDesc:   _TensorStruct, dxPtr: _RealTensorPointer)
  : Unit = {
    val result = cudnnPoolingBackward(
      device.dnnContextPtr,
      poolDesc.ptr,
      alpha.ptr,
      yDesc.ptr,  yPtr,
      dyDesc.ptr, dyPtr,
      xDesc.ptr,  xPtr,
      beta.ptr,
      dxDesc.ptr, dxPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def poolingForward(device:  LogicalDevice,
                           poolDesc: _PoolingStruct,
                           alpha:    _RealTensorNativeReal,
                           xDesc:    _TensorStruct, xPtr: _RealTensorPointer,
                           beta:     _RealTensorNativeReal,
                           yDesc:    _TensorStruct, yPtr: _RealTensorPointer)
  : Unit = {
    val result = cudnnPoolingForward(
      device.dnnContextPtr,
      poolDesc.ptr,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def opTensor(device:       LogicalDevice,
                     opTensorDesc: _OpTensorStruct,
                     alpha0:       _RealTensorNativeReal,
                     x0Desc:       _TensorStruct, x0Ptr: _RealTensorPointer,
                     alpha1:       _RealTensorNativeReal,
                     x1Desc:       _TensorStruct, x1Ptr: _RealTensorPointer,
                     beta:         _RealTensorNativeReal,
                     yDesc:        _TensorStruct, yPtr: _RealTensorPointer)
  : Unit = {
    val result = cudnnOpTensor(
      device.dnnContextPtr,
      opTensorDesc.ptr,
      alpha0.ptr,
      x0Desc.ptr, x0Ptr,
      alpha1.ptr,
      x1Desc.ptr, x1Ptr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def scaleTensor(device: LogicalDevice,
                        alpha:  _RealTensorNativeReal,
                        xDesc:  _TensorStruct, xPtr: _RealTensorPointer)
  : Unit = {
    val result = cudnnScaleTensor(
      device.dnnContextPtr,
      xDesc.ptr, xPtr,
      alpha.ptr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def setActivationDescriptor(actDesc:       _ActivationStruct,
                                    mode:          Int,
                                    reluNaNOpt:    Int,
                                    reluThreshold: Double)
  : Unit = {
    val result = cudnnSetActivationDescriptor(
      actDesc.ptr,
      mode,
      reluNaNOpt,
      reluThreshold
    )
    check(result)
  }

  @inline
  final def setConvolution2dDescriptor(convDesc: _ConvolutionStruct,
                                       paddingY: Int, paddingX: Int,
                                       strideY:  Int, strideX:  Int,
                                       upScaleX: Int, upScaleY: Int,
                                       mode:     Int)
  : Unit = {
    val result = cudnnSetConvolution2dDescriptor(
      convDesc.ptr,
      paddingY, paddingX,
      strideY,  strideX,
      upScaleX, upScaleY,
      mode
    )
    check(result)
  }

  @inline
  final def setConvolutionNdDescriptor(convDesc: _ConvolutionStruct,
                                       padding:  Array[Int],
                                       stride:   Array[Int],
                                       upScale:  Array[Int],
                                       mode:     Int,
                                       dataType: Int)
  : Unit = {
    require(
      padding.length == stride.length &&
      padding.length == upScale.length
    )
    val result = cudnnSetConvolutionNdDescriptor(
      convDesc.ptr,
      padding.length,
      padding,
      stride,
      upScale,
      mode,
      dataType
    )
    check(result)
  }

  @inline
  final def setDropoutDescriptor(dropDesc:    _DropoutStruct,
                                 device:      LogicalDevice,
                                 probability: Real,
                                 state:       _ByteDeviceBuffer,
                                 seed:        Long)
  : Unit = {
    val statePtr = state.ptr
    val result = cudnnSetDropoutDescriptor(
      dropDesc.ptr,
      device.dnnContextPtr,
      FloatEx(probability),
      statePtr,
      statePtr.capacity(),
      seed
    )
    check(result)
  }

  @inline
  final def setFilter4dDescriptor(filterDesc: _FilterStruct,
                                  dataType:   Int,
                                  dataFormat: Int,
                                  k:          Int,
                                  c:          Int,
                                  h:          Int,
                                  w:          Int)
  : Unit = {
    val result = cudnnSetFilter4dDescriptor(
      filterDesc.ptr,
      dataType,
      dataFormat,
      k,
      c,
      h,
      w
    )
    check(result)
  }

  @inline
  final def setFilterNdDescriptor(filterDesc: _FilterStruct,
                                  dataType:   Int,
                                  dataFormat: Int,
                                  dims:       Array[Int])
  : Unit = {
    val result = cudnnSetFilterNdDescriptor(
      filterDesc.ptr,
      dataType,
      dataFormat,
      dims.length,
      dims
    )
    check(result)
  }

  @inline
  final def setLRNDescriptor(lrnDesc: _LRNStruct,
                             n:       Int,
                             alpha:   Double,
                             beta:    Double,
                             k:       Double)
  : Unit = {
    val result = cudnnSetLRNDescriptor(
      lrnDesc.ptr,
      n,
      alpha,
      beta,
      k
    )
    check(result)
  }

  @inline
  final def setOpTensorDescriptor(opTensorDesc:       _OpTensorStruct,
                                  op:                 Int,
                                  comparisonDataType: Int,
                                  nanOpt:             Int)
  : Unit = {
    val result = cudnnSetOpTensorDescriptor(
      opTensorDesc.ptr,
      op,
      comparisonDataType,
      nanOpt
    )
    check(result)
  }

  @inline
  final def setPooling2dDescriptor(poolDesc: _PoolingStruct,
                                   mode:     Int,
                                   nanOpt:   Int,
                                   dimsY:    Int, dimsX:    Int,
                                   paddingY: Int, paddingX: Int,
                                   strideY:  Int, strideX:  Int)
  : Unit = {
    val result = cudnnSetPooling2dDescriptor(
      poolDesc.ptr,
      mode,
      nanOpt,
      dimsY, dimsX,
      paddingY, paddingX,
      strideY, strideX
    )
    check(result)
  }

  @inline
  final def setPoolingNdDescriptor(poolDesc: _PoolingStruct,
                                   mode:     Int,
                                   nanOpt:   Int,
                                   dims:     Array[Int],
                                   padding:  Array[Int],
                                   stride:   Array[Int])
  : Unit = {
    require(
      dims.length == stride.length &&
      dims.length == padding.length
    )
    val result = cudnnSetPoolingNdDescriptor(
      poolDesc.ptr,
      mode,
      nanOpt,
      dims.length,
      dims,
      padding,
      stride
    )
    check(result)
  }

  @inline
  final def setStream(device: LogicalDevice)
  : Unit = {
    val result = cudnnSetStream(device.dnnContextPtr, device.streamPtr)
    check(result)
  }

  @inline
  final def setTensor(device: LogicalDevice,
                      yDesc:  _TensorStruct, yPtr: _RealTensorPointer,
                      value:  _RealTensorNativeReal)
  : Unit = {
    val result = cudnnSetTensor(
      device.dnnContextPtr,
      yDesc.ptr, yPtr,
      value.ptr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def setTensor4dDescriptor(tensorDesc: _TensorStruct,
                                  dataFormat: Int,
                                  dataType:   Int,
                                  n:          Int,
                                  c:          Int,
                                  h:          Int,
                                  w:          Int)
  : Unit = {
    val result = cudnnSetTensor4dDescriptor(
      tensorDesc.ptr,
      dataFormat,
      dataType,
      n,
      c,
      h,
      w
    )
    check(result)
  }

  @inline
  final def setTensor4dDescriptor(tensorDesc: _TensorStruct,
                                  dataType:   Int,
                                  n:          Int,
                                  c:          Int,
                                  h:          Int,
                                  w:          Int,
                                  nStride:    Int,
                                  cStride:    Int,
                                  hStride:    Int,
                                  wStride:    Int)
  : Unit = {
    val result = cudnnSetTensor4dDescriptorEx(
      tensorDesc.ptr,
      dataType,
      n,
      c,
      h,
      w,
      nStride,
      cStride,
      hStride,
      wStride
    )
    check(result)
  }

  @inline
  final def setTensorNdDescriptor(tensorDesc: _TensorStruct,
                                  dataType:   Int,
                                  dims:       Array[Int],
                                  stride:     Array[Int])
  : Unit = {
    require(dims.length == stride.length)
    val result = cudnnSetTensorNdDescriptor(
      tensorDesc.ptr,
      dataType,
      dims.length,
      dims,
      stride
    )
    check(result)
  }

  @inline
  final def softmaxForward(device:    LogicalDevice,
                           algorithm: Int,
                           mode:      Int,
                           alpha:     _RealTensorNativeReal,
                           xDesc:     _TensorStruct, xPtr: _RealTensorPointer,
                           beta:      _RealTensorNativeReal,
                           yDesc:     _TensorStruct, yPtr: _RealTensorPointer)
  : Unit = {
    val result = cudnnSoftmaxForward(
      device.dnnContextPtr,
      algorithm,
      mode,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def softmaxBackward(device:    LogicalDevice,
                            algorithm: Int,
                            mode:      Int,
                            alpha:     _RealTensorNativeReal,
                            yDesc:     _TensorStruct, yPtr:  _RealTensorPointer,
                            dyDesc:    _TensorStruct, dyPtr: _RealTensorPointer,
                            beta:      _RealTensorNativeReal,
                            dxDesc:    _TensorStruct, dxPtr: _RealTensorPointer)
  : Unit = {
    val result = cudnnSoftmaxBackward(
      device.dnnContextPtr,
      algorithm,
      mode,
      alpha.ptr,
      yDesc.ptr,  yPtr,
      dyDesc.ptr, dyPtr,
      beta.ptr,
      dxDesc.ptr, dxPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def transformTensor[TPtr <: Pointer, UPtr <: Pointer](device: LogicalDevice,
                                                              alpha:  NativeDouble,
                                                              xDesc:  _TensorStruct, xPtr: DoublePointer,
                                                              beta:   NativeDouble,
                                                              yDesc:  _TensorStruct, yPtr: DoublePointer)
  : Unit = {
    val result = cudnnTransformTensor(
      device.dnnContextPtr,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def transformTensor[TPtr <: Pointer, UPtr <: Pointer](device: LogicalDevice,
                                                              alpha:  NativeDouble,
                                                              xDesc:  _TensorStruct, xPtr: DoublePointer,
                                                              beta:   NativeFloat,
                                                              yDesc:  _TensorStruct, yPtr: FloatPointer)
  : Unit = {
    val result = cudnnTransformTensor(
      device.dnnContextPtr,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def transformTensor[TPtr <: Pointer, UPtr <: Pointer](device: LogicalDevice,
                                                              alpha:  NativeDouble,
                                                              xDesc:  _TensorStruct, xPtr: DoublePointer,
                                                              beta:   NativeFloat,
                                                              yDesc:  _TensorStruct, yPtr: HalfPointer)
  : Unit = {
    val result = cudnnTransformTensor(
      device.dnnContextPtr,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def transformTensor[TPtr <: Pointer, UPtr <: Pointer](device: LogicalDevice,
                                                              alpha:  NativeFloat,
                                                              xDesc:  _TensorStruct, xPtr: FloatPointer,
                                                              beta:   NativeDouble,
                                                              yDesc:  _TensorStruct, yPtr: DoublePointer)
  : Unit = {
    val result = cudnnTransformTensor(
      device.dnnContextPtr,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def transformTensor[TPtr <: Pointer, UPtr <: Pointer](device: LogicalDevice,
                                                              alpha:  NativeFloat,
                                                              xDesc:  _TensorStruct, xPtr: FloatPointer,
                                                              beta:   NativeFloat,
                                                              yDesc:  _TensorStruct, yPtr: FloatPointer)
  : Unit = {
    val result = cudnnTransformTensor(
      device.dnnContextPtr,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def transformTensor[TPtr <: Pointer, UPtr <: Pointer](device: LogicalDevice,
                                                              alpha:  NativeFloat,
                                                              xDesc:  _TensorStruct, xPtr: FloatPointer,
                                                              beta:   NativeFloat,
                                                              yDesc:  _TensorStruct, yPtr: HalfPointer)
  : Unit = {
    val result = cudnnTransformTensor(
      device.dnnContextPtr,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def transformTensor[TPtr <: Pointer, UPtr <: Pointer](device: LogicalDevice,
                                                              alpha:  NativeFloat,
                                                              xDesc:  _TensorStruct, xPtr: HalfPointer,
                                                              beta:   NativeDouble,
                                                              yDesc:  _TensorStruct, yPtr: DoublePointer)
  : Unit = {
    val result = cudnnTransformTensor(
      device.dnnContextPtr,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def transformTensor[TPtr <: Pointer, UPtr <: Pointer](device: LogicalDevice,
                                                              alpha:  NativeFloat,
                                                              xDesc:  _TensorStruct, xPtr: HalfPointer,
                                                              beta:   NativeFloat,
                                                              yDesc:  _TensorStruct, yPtr: FloatPointer)
  : Unit = {
    val result = cudnnTransformTensor(
      device.dnnContextPtr,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def transformTensor[TPtr <: Pointer, UPtr <: Pointer](device: LogicalDevice,
                                                              alpha:  NativeFloat,
                                                              xDesc:  _TensorStruct, xPtr: HalfPointer,
                                                              beta:   NativeFloat,
                                                              yDesc:  _TensorStruct, yPtr: HalfPointer)
  : Unit = {
    val result = cudnnTransformTensor(
      device.dnnContextPtr,
      alpha.ptr,
      xDesc.ptr, xPtr,
      beta.ptr,
      yDesc.ptr, yPtr
    )
    check(result)
    device.trySynchronize()
  }

}

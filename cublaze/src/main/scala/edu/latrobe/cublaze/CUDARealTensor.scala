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

import breeze.linalg.DenseMatrix
import edu.latrobe._
import edu.latrobe.native._
import org.bytedeco.javacpp.cuda._

/**
  * A tensor that resides in the GPUs memory.
  *
  * This is cuDNNs preferred format. However, our JVM kernel implementations all
  * imply that the channels are a the smallest (unbreakable) dimension of a tensor.
  */
final class CUDARealTensor(val          device: LogicalDevice,
                           override val layout: IndependentTensorLayout)
  extends TensorEx[CUDARealTensor]
    with RealTensor
    with Closable {

  val data
  : _RealTensorDeviceBuffer = _RealTensorDeviceBuffer(device, layout.noValues)

  private var _desc
  : _TensorStruct = _

  @transient
  lazy val desc
  : _TensorStruct = {
    if (_desc == null) {
      _desc = _TensorStruct.nchw(layout, _RealTensorDeviceBuffer.dataType)
    }
    _desc
  }

  private var _size
  : _SizeStruct = _

  @transient
  lazy val size
  : _SizeStruct = {
    if (_size == null) {
      _size = _SizeStruct(layout.size.noValues, layout.noSamples, Real.size)
    }
    _size
  }

  override def repr
  : CUDARealTensor = this

  override def toString
  : String = s"CUDARealTensor[$layout]"

  override protected def doClose()
  : Unit = {
    if (_size != null) {
      _size.close()
    }
    if (_desc != null) {
      _desc.close()
    }
    data.close()
    super.doClose()
  }

  override def platform
  : CUDA.type = CUDA

  override def createSibling(newLayout: TensorLayout)
  : CUDARealTensor = CUDARealTensor(device, newLayout.makeIndependent)

  override def createSiblingAndClear(newLayout: TensorLayout)
  : CUDARealTensor = {
    val result = createSibling(newLayout)
    result.clear()
    result
  }

  override def copy
  : CUDARealTensor = {
    val result = createSibling()
    copyTo(result)
    result
  }

  def copyTo(other: Tensor)
  : Unit = {
    require(other.layout == layout)
    other match {
      case other: CUDARealTensor =>
        if (device.physicalDevice == other.device.physicalDevice) {
          _CUDA.memcpyAsync(
            device,
            other.data.ptr,
            data.ptr,
            data.capacityInBytes,
            cudaMemcpyDeviceToDevice
          )
        }
        else {
          logger.trace("CUDA.memcopy: Moving data between devices!")
          _CUDA.memcpyPeer(
            other.device,
            other.data.ptr,
            device,
            data.ptr,
            data.capacityInBytes
          )
        }

      case _ =>
        other := values
    }
  }

  override def values
  : Array[Real] = {
    using(StagingBufferLock.request())(lock => {
      lock.download(this)
      lock.buffer.toArray(layout.noValues)
    })
  }

  override def valuesMatrix
  : DenseMatrix[Real] = new DenseMatrix(
    layout.size.noValues,
    layout.noSamples,
    values
  )

  override def valuesMatrixEx
  : DenseMatrix[Real] = new DenseMatrix(
    layout.size.noValues,
    layout.noSamples,
    values
  )

  override def get(valueNo: Int)
  : Real = {
    // TODO: Isolate value!
    using(StagingBufferLock.request())(lock => {
      lock.download(this)
      lock.buffer(valueNo)
    })
  }

  override def get(result: Array[Real], offset: Int, stride: Int)
  : Unit = {
    // TODO: Isolate values!
    using(StagingBufferLock.request())(lock => {
      lock.download(this)
      if (stride == 1) {
        lock.buffer.get(result, offset, layout.noValues)
      }
      else {
        val tmp = new Array[Real](layout.noValues)
        lock.buffer.get(tmp)
        ArrayEx.set(
          result, offset, stride,
          tmp,    0,      1,
          tmp.length
        )
      }
    })
  }

  override def put(valueNo: Int, value: Real)
  : Unit = {
    // TODO: Isolate value!
    using(StagingBufferLock.request())(lock => {
      lock.download(this)
      lock.buffer.update(valueNo, value)
      lock.upload(this)
    })
  }

  override def put(array: Array[Real], offset: Int, stride: Int)
  : Unit = {
    // TODO: Isolate values!
    using(StagingBufferLock.request())(lock => {
      if (stride == 1) {
        lock.buffer.put(array, offset, layout.noValues)
      }
      else {
        val tmp = new Array[Real](layout.noValues)
        ArrayEx.set(
          tmp,   0,      1,
          array, offset, stride,
          tmp.length
        )
        lock.buffer.put(tmp)
      }
      lock.upload(this)
    })
  }

  override def clear()
  : Unit = {
    _CUDA.memset(
      device,
      data,
      0,
      data.capacityInBytes
    )
  }

  override def fill(fn:         () => Real,
                    threadSafe: Boolean)
  : Unit = {
    using(RealArrayTensor.zeros(layout))(tmp => {
      tmp.fill(fn, threadSafe)
      :=(tmp)
    })
  }


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  override def reshape(newSize: Size)
  : CUDARealTensor = {
    val result = createSibling(layout.derive(newSize))
    reshapeTo(result)
    result
  }

  @inline
  def reshapeTo(result: CUDARealTensor): Unit = {
    val inpSize = layout.size
    val outSize = result.layout.size
    require(result.layout.noValues == layout.noValues)

    val srcPtr = data.ptr
    val dstPtr = result.data.ptr

    // Convert to 1D.
    if (outSize.noTuples == 1) {
      using(
        _TensorStruct.nhwc(layout, _RealTensorDeviceBuffer.dataType)
      )(descNHWC => {
        _CUDNN.transformTensor(
          device,
          _RealTensorNativeReal.one,  desc,     srcPtr,
          _RealTensorNativeReal.zero, descNHWC, dstPtr
        )
      })
    }
    // Convert from 1D.
    else if (inpSize.noTuples == 1) {
      using(
        _TensorStruct.nhwc(result.layout, _RealTensorDeviceBuffer.dataType)
      )(descNHWC => {
        _CUDNN.transformTensor(
          device,
          _RealTensorNativeReal.one,  descNHWC,    srcPtr,
          _RealTensorNativeReal.zero, result.desc, dstPtr
        )
      })
    }
    // Convert via 1D.
    else {
      val wsPtr = device.scratchBuffer.asRealTensorPtr
      device.burst({
        using(
          _TensorStruct.nhwc(layout, _RealTensorDeviceBuffer.dataType)
        )(descNHWC => {
          _CUDNN.transformTensor(
            device,
            _RealTensorNativeReal.one,  desc,     srcPtr,
            _RealTensorNativeReal.zero, descNHWC, wsPtr
          )
        })
        using(
          _TensorStruct.nhwc(result.layout, _RealTensorDeviceBuffer.dataType)
        )(descNHWC => {
          _CUDNN.transformTensor(
            device,
            _RealTensorNativeReal.one,  descNHWC,    wsPtr,
            _RealTensorNativeReal.zero, result.desc, dstPtr
          )
        })
      })
    }
  }

  override def apply(index: Int)
  : CUDARealTensor = {
    require(index >= 0 && index <= layout.noSamples)

    val result = CUDARealTensor(device, layout.derive(1))

    val n      = result.layout.size.noValues
    val offset = n * index
    _CUBLAS.copy(
      device,
      n,
      data.ptr.withOffset(offset), 1,
      result.data.ptr,             1
    )

    result
    /*
    logger.info("This is a slow dummy implementation. You should not call this!")
    using(toRealTensor)(tmp => {
      using(
        tmp(index)
      )(_.toCUDARealTensor(device))
    })
    */
  }

  override def apply(indices: Range): CUDARealTensor = {
    logger.info("This is a slow dummy implementation. You should not call this!")
    using(toRealArrayTensor)(tmp => {
      using(
        tmp(indices)
      )(_.toCUDARealTensor(device))
    })
  }

  override def splitSamples
  : Array[Tensor] = ArrayEx.tabulate(layout.noSamples)(apply)

  override def concat(other: Tensor)
  : CUDARealTensor = {
    val tmp = other.asOrToCUDARealTensor(device)
    val result = concat(tmp)
    if (tmp ne other) {
      tmp.close()
    }
    result
    /*
    logger.info("This is a slow dummy implementation. You should not call this!")
    using(toRealTensor)(tmp => {
      using(
        tmp.concat(other)
      )(_.toCUDARealTensor(device))
    })
    */
  }

  @inline
  def concat(other: CUDARealTensor)
  : CUDARealTensor = {
    val result = CUDARealTensor(device, layout.concat(other.layout))
    val dstPtr = result.data.ptr

    device.burst({
      val n0 = layout.noValues
      _CUBLAS.copy(
        device,
        n0,
        data.ptr, 1,
        dstPtr, 1
      )

      val n1 = other.layout.noValues
      _CUBLAS.copy(
        device,
        n1,
        other.data.ptr, 1,
        dstPtr.withOffset(n0), 1
      )
    })

    result
  }

  override def concat[T <: Tensor](others: Array[T])
  : CUDARealTensor = {
    val tmp = ArrayEx.map(
      others
    )(_.asOrToCUDARealTensor(device))

    val result = concat(tmp)
    ArrayEx.foreach(tmp, others)((tmp, other) => {
      if (tmp ne other) {
        tmp.close()
      }
    })
    result
    /*
    logger.warn("This is a slow dummy implementation. You should not call this!")
    using(toRealTensor)(tmp => {
      using(
        tmp.concat(others)
      )(_.toCUDARealTensor(device))
    })
    */
  }

  def concat(others: Array[CUDARealTensor])
  : CUDARealTensor = {
    val newLayout = ArrayEx.foldLeft(
      layout,
      others
    )((res, tensor) => res.concat(tensor.layout))

    val result = CUDARealTensor(device, newLayout)
    val dstPtr = result.data.ptr

    device.burst({
      val n = layout.noValues
      _CUBLAS.copy(
        device,
        n,
        data.ptr, 1,
        dstPtr,   1
      )

      var offset = n
      ArrayEx.foreach(others)(other => {
        val n = other.layout.noValues
        _CUBLAS.copy(
          device,
          n,
          other.data.ptr,            1,
          dstPtr.withOffset(offset), 1
        )
        offset += n
      })
    })

    result
  }

  /*
  final override def transformValues(fn: Real => Real): Unit = {
    val tmp = valuesArray
    tmp.fastTransform(fn)
    put(tmp)
  }

  final override def mapValues(fn: Real => Real): RealTensor = {
    //val tmp = valuesPtr.toArray
    // tmp.fastTransform(fn)
    // val result = CUDATensorNCHW(device, size, noSamples)
    // result.valuesPtr.put(tmp)
    // result
    // TODO: Have to make a decision here!
    val result = toRealTensor
    result.transformValues(fn)
    result
  }
  */

  /*
  override def mapValues(fn: Real => Real): CUDATensorNHWC = {
    val tmp = valuesArray
    tmp.fastTransform(fn)
    CUDATensorNHWC(device, size, noSamples).put(tmp)
  }
  */

  override protected def doSet(value: Real)
  : Unit = {
    /*
    Slower!
    _NPP.set(
      device,
      value,
      data,
      layout.noValues
    )*/
    using(
      _RealTensorNativeReal(value)
    )(doSet)
  }

  @inline
  protected def doSet(value: _RealTensorNativeReal)
  : Unit = {
    _CUDNN.setTensor(
      device,
      desc, data.ptr,
      value
    )
    // TODO: Test if this is faster.
    /**
      * _CUBLAS.copy(
      lock.device,
      5,
      lock.device.one.ptr,
      0,
      b.data.ptr,
      1
    )
    */
  }


  override protected def doSet(other: Tensor)
  : Unit = other match {
    case other: CUDARealTensor =>
      other.copyTo(this)
    case _ =>
      put(other.values)
  }

  override protected def doSet(other: Tensor, beta: Real)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    using(
      _RealTensorNativeReal(beta)
    )(doAdd(_RealTensorNativeReal.zero, tmp, _))
    if (tmp ne other) {
      tmp.close()
    }
  }

  override protected def doSet(other0: Tensor, other1: Tensor)
  : Unit = {
    val tmp0 = other0.asOrToCUDARealTensor(device)
    val tmp1 = other1.asOrToCUDARealTensor(device)
    doSet(tmp0, tmp1)
    if (tmp1 ne other1) {
      tmp1.close()
    }
    if (tmp0 ne other0) {
      tmp0.close()
    }
  }

  @inline
  private def doSet(other0: CUDARealTensor, other1: CUDARealTensor)
  : Unit = {
    _NPP.mul(
      device,
      other0.data.ptr,
      other1.data.ptr,
      data.ptr,
      size
    )
  }

  override protected def doSet(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = {
    val tmp0 = other0.asOrToCUDARealTensor(device)
    val tmp1 = other1.asOrToCUDARealTensor(device)
    using(
      _RealTensorNativeReal(beta)
    )(doAdd(tmp0, tmp1, _))
    if (tmp1 ne other1) {
      tmp1.close()
    }
    if (tmp0 ne other0) {
      tmp0.close()
    }
  }

  @inline
  private def doSet(other0: CUDARealTensor, other1: CUDARealTensor, beta: Real)
  : Unit = {
    val dstPtr = data.ptr

    device.burst({
      // Multiply.
      _NPP.mul(
        device,
        other0.data.ptr,
        other1.data.ptr,
        dstPtr,
        size
      )

      // Scale.
      _NPP.mulC_I(
        device,
        beta,
        dstPtr,
        size
      )
    })
  }

  override protected def doAdd(value: Real)
  : Unit = {
    /*
    //Below code looks super scary. But is actually faster!
    _NPP.addC_I(
      device,
      value,
      data.ptr,
      layout.noValues
    )
    */
    using(
      _RealTensorNativeReal(value)
    )(doAdd)
  }

  @inline
  protected def doAdd(value: _RealTensorNativeReal)
  : Unit = {
    _CUBLAS.axpy(
      device,
      layout.noValues,
      value,
      device.one.ptr, 0,
      data.ptr,       1
    )
  }

  override protected def doAdd(other: Tensor)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    doAdd(tmp)
    if (tmp ne other) {
      tmp.close()
    }
  }

  @inline
  protected def doAdd(other: CUDARealTensor)
  : Unit = doAdd(other, _RealTensorNativeReal.one)

  override protected def doAdd(alpha: Real,
                               other: Tensor)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    using(
      _RealTensorNativeReal(alpha)
    )(doAdd(_, tmp))
    if (tmp ne other) {
      tmp.close()
    }
  }

  @inline
  private def doAdd(alpha: _RealTensorNativeReal,
                    other: CUDARealTensor)
  : Unit = doAdd(alpha, other, _RealTensorNativeReal.one)

  override protected def doAdd(other: Tensor, beta: Real)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    using(
      _RealTensorNativeReal(beta)
    )(doAdd(tmp, _))
    if (tmp ne other) {
      tmp.close()
    }
  }

  @inline
  private def doAdd(other: CUDARealTensor, beta: _RealTensorNativeReal)
  : Unit = {
    val srcPtr = other.data.ptr
    val dstPtr = data.ptr
    _CUBLAS.axpy(
      device,
      layout.noValues,
      beta,
      srcPtr, 1,
      dstPtr, 1
    )
  }

  override protected def doAdd(alpha: Real,
                               other: Tensor, beta: Real)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    using(
      _RealTensorNativeReal(alpha),
      _RealTensorNativeReal(beta)
    )(doAdd(_, tmp, _))
    if (tmp ne other) {
      tmp.close()
    }
  }

  @inline
  private def doAdd(alpha: _RealTensorNativeReal,
                    other: CUDARealTensor, beta: _RealTensorNativeReal)
  : Unit = {
    // We are in a dilemma here. This supports half, but BLAS tends to be faster!.
    /*
    _CUDNN.transformTensor(
      device,
      beta,  other.desc, other.data.ptr,
      alpha, desc,       data.ptr
    )
    */
    val noRows = layout.size.noValues
    val noCols = layout.noSamples
    val srcPtr = other.data.ptr
    val dstPtr = data.ptr
    _CUBLAS.geam(
      device,
      alpha,
      dstPtr, noRows, noRows, noCols, aTrans = false,
      beta,
      srcPtr, noRows, noRows, noCols, bTrans = false,
      dstPtr, noRows, noRows, noCols
    )
  }

  override protected def doAdd(other0: Tensor, other1: Tensor)
  : Unit = {
    val tmp0 = other0.asOrToCUDARealTensor(device)
    val tmp1 = other1.asOrToCUDARealTensor(device)
    doAdd(tmp0, tmp1)
    if (tmp1 ne other1) {
      tmp1.close()
    }
    if (tmp0 ne other0) {
      tmp0.close()
    }
  }

  @inline
  private def doAdd(other0: CUDARealTensor, other1: CUDARealTensor)
  : Unit = {
    if (other0 eq other1) {
      _NPP.addSquare(
        device,
        other0.data.ptr,
        data.ptr,
        size
      )
    }
    else {
      _NPP.addProduct(
        device,
        other0.data.ptr,
        other1.data.ptr,
        data.ptr,
        size
      )
    }
  }

  override protected def doAdd(alpha:  Real,
                               other0: Tensor, other1: Tensor)
  : Unit = {
    val tmp0 = other0.asOrToCUDARealTensor(device)
    val tmp1 = other1.asOrToCUDARealTensor(device)
    /*
    using(
      _RealTensorNativeReal(alpha)
    )(doAdd(_, tmp0, tmp1))
    */
    doAdd(alpha, tmp0, tmp1)
    if (tmp1 ne other1) {
      tmp1.close()
    }
    if (tmp0 ne other0) {
      tmp0.close()
    }
  }

  @inline
  private def doAdd(alpha:  Real,
                    other0: CUDARealTensor, other1: CUDARealTensor)
  : Unit = {
    val dstPtr = data.ptr

    device.burst({
      // Scale this.
      _NPP.mulC_I(
        device,
        alpha,
        dstPtr,
        size
      )

      // Add product.
      _NPP.addProduct(
        device,
        other0.data.ptr,
        other1.data.ptr,
        dstPtr,
        size
      )
    })
  }

  override protected def doAdd(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = {
    val tmp0 = other0.asOrToCUDARealTensor(device)
    val tmp1 = other1.asOrToCUDARealTensor(device)
    using(
      _RealTensorNativeReal(beta)
    )(doAdd(tmp0, tmp1, _))
    if (tmp1 ne other1) {
      tmp1.close()
    }
    if (tmp0 ne other0) {
      tmp0.close()
    }
  }

  @inline
  private def doAdd(other0: CUDARealTensor, other1: CUDARealTensor, beta: _RealTensorNativeReal)
  : Unit = {
    val noValues = layout.noValues
    val wsPtr    = device.scratchBuffer.ptr
    val srcPtr0  = other0.data.ptr
    val srcPtr1  = if (other0 eq other1) srcPtr0 else other1.data.ptr
    device.burst({
      _CUBLAS.gmm(
        device,
        srcPtr0, 1, 1, noValues,
        srcPtr1, 1,    noValues,
        wsPtr,   1, 1, noValues
      )
      _CUBLAS.axpy(
        device,
        noValues,
        beta,
        wsPtr,    1,
        data.ptr, 1
      )
    })
  }

  override protected def doAdd(alpha: Real,
                               other0: Tensor, other1: Tensor, beta: Real)
  : Unit = {
    val tmp0 = other0.asOrToCUDARealTensor(device)
    val tmp1 = other1.asOrToCUDARealTensor(device)
    /*
    using(
      _RealTensorNativeReal(alpha),
      _RealTensorNativeReal(beta)
    )(doAdd(_, tmp0, tmp1, _))
    */
    doAdd(alpha, tmp0, tmp1, beta)
    if (tmp1 ne other1) {
      tmp1.close()
    }
    if (tmp0 ne other0) {
      tmp0.close()
    }
  }

  @inline
  private def doAdd(alpha:  Real,
                    other0: CUDARealTensor, other1: CUDARealTensor, beta: Real)
  : Unit = {
    val wsPtr    = device.scratchBuffer.ptr
    val srcPtr0  = other0.data.ptr
    device.burst({
      // Multiply arguments.
      if (other0 eq other1) {
        _NPP.sqr(
          device,
          srcPtr0,
          wsPtr,
          size
        )
      }
      else {
        _NPP.mul(
          device,
          srcPtr0,
          other1.data.ptr,
          wsPtr,
          size
        )
      }

      // Blend.
      val dstPtr = data.ptr
      _NPP.alphaCompC(
        device,
        dstPtr, alpha,
        wsPtr,  beta,
        dstPtr,
        size
      )
    })
  }

  /*
  @inline
  private def doAdd(alpha:  _RealTensorNativeReal,
                    other0: CUDARealTensor, other1: CUDARealTensor, beta: _RealTensorNativeReal)
  : Unit = {
    /*
    Note that opTensor is superslow in cuDNN 5!
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one, other0.desc, other0.data.ptr,
      beta,                      other1.desc, other1.data.ptr,
      alpha,                     desc,        data.ptr
    )
    */
    using(device.requestScratchBuffer())(lock => {
      val noValues = layout.noValues
      val wsPtr    = lock.buffer.ptr
      val srcPtr0  = other0.data.ptr
      val srcPtr1  = if (other0 eq other1) srcPtr0 else other1.data.ptr
      device.burst({
        _CUBLAS.gmm(
          device,
          srcPtr0, 1, 1, noValues,
          srcPtr1, 1,    noValues,
          wsPtr,   1, 1, noValues
        )
        val noRows = layout.size.noValues
        val noCols = layout.noSamples
        val dstPtr = data.ptr
        _CUBLAS.geam(
          device,
          alpha,
          dstPtr, noRows, noRows, noCols, aTrans = false,
          beta,
          wsPtr, noRows, noRows, noCols, bTrans = false,
          dstPtr, noRows, noRows, noCols
        )
      })
    })
  }
  */

  override protected def doSubtract(other: Tensor)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    doSubtract(tmp)
    if (tmp ne other) {
      tmp.close()
    }
  }

  @inline
  private def doSubtract(other: CUDARealTensor)
  : Unit = doAdd(other, _RealTensorNativeReal.minusOne)

  override def subtractR(value: Real)
  : Unit = {
    _NPP.subCRev_I(
      device,
      value,
      data.ptr,
      layout.noValues
    )
  }

  override protected def doMultiply(value: Real)
  : Unit = {
    using(
      _RealTensorNativeReal(value)
    )(doMultiply)
  }

  @inline
  private def doMultiply(value: _RealTensorNativeReal)
  : Unit = {
    // Surprisingly much faster than NPP and CUDNN variants.
    _CUBLAS.scal(
      device,
      layout.noValues,
      value,
      data.ptr, 1
    )
  }

  override protected def doMultiply(other: Tensor)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    doMultiply(tmp)
    if (tmp ne other) {
      tmp.close()
    }
  }

  @inline
  private def doMultiply(other: CUDARealTensor)
  : Unit = {
    /*
    Slower than NPP.mul.
    _CUDNN.opTensor(
      device,
      _OpTensorStruct.multiply,
      _RealTensorNativeReal.one,  desc,       data.ptr,
      _RealTensorNativeReal.one,  other.desc, other.data.ptr,
      _RealTensorNativeReal.zero, desc,       data.ptr
    )
    */
    /*
    Slower than gmm.
    _NPP.mul(
      device,
      other.data,
      data,
      data,
      layout.noValues
    )
    */
    val noValues = layout.noValues
    val srcPtr   = other.data.ptr
    val dstPtr   = data.ptr
    _CUBLAS.gmm(
      device,
      dstPtr, 1, 1, noValues,
      srcPtr, 1,    noValues,
      dstPtr, 1, 1, noValues
    )
  }

  override protected def doMultiply(other: Tensor, beta: Real)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    doMultiply(tmp)
    if (tmp ne other) {
      tmp.close()
    }
  }

  @inline
  private def doMultiply(other: CUDARealTensor, beta: _RealTensorNativeReal)
  : Unit = {
    val noValues = layout.noValues
    val srcPtr   = other.data.ptr
    val dstPtr   = data.ptr
    device.burst({
      _CUBLAS.gmm(
        device,
        dstPtr, 1, 1, noValues,
        srcPtr, 1,    noValues,
        dstPtr, 1, 1, noValues
      )
      _CUBLAS.scal(
        device,
        noValues,
        beta,
        dstPtr, 1
      )
    })
  }

  override protected def doDivide(other: Tensor)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    doDivide(tmp)
    if (tmp ne other) {
      tmp.close()
    }
  }

  @inline
  private def doDivide(other: CUDARealTensor)
  : Unit = {
    // Surprise, even with the overhead of the size object it is faster!
    /*
    _NPP.div_I(
      device,
      other.data.ptr,
      data.ptr,
      layout.noValues
    )
    */
    _NPP.div_I(
      device,
      other.data.ptr,
      data.ptr,
      size
    )
  }

  override protected def doDivide(epsilon0: Real,
                                  other:    Tensor)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    /*
    using(
      _RealTensorNativeReal(epsilon0)
    )(doDivide(_, tmp))
    */
    doDivide(epsilon0, tmp)
    if (tmp ne other) {
      tmp.close()
    }
  }

  @inline
  private def doDivide(epsilon0: Real,
                       other:    CUDARealTensor)

  : Unit = {
    val dstPtr = data.ptr
    device.burst({
      _NPP.addC_I(
        device,
        epsilon0,
        dstPtr,
        size
      )
      _NPP.div_I(
        device,
        other.data.ptr,
        dstPtr,
        size
      )
    })
  }

  /*
  @inline
  private def doDivide(epsilon0: NativeReal,
                       other:    CUDARealTensor)

  : Unit = {
    val dstPtr = data.ptr
    device.burst({
      _CUBLAS.axpy(
        device,
        layout.noValues,
        epsilon0,
        device.one.ptr, 0,
        dstPtr,         1
      )
      _NPP.div_I(
        device,
        other.data.ptr,
        dstPtr,
        size
      )
    })
  }
  */

  override protected def doDivide(other: Tensor, epsilon1: Real)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    /*
    using(
      _RealTensorNativeReal(epsilon1)
    )(doDivide(tmp, _))
    */
    doDivide(tmp, epsilon1)
    if (tmp ne other) {
      tmp.close()
    }
  }

  @inline
  protected def doDivide(other: CUDARealTensor, epsilon1: Real)
  : Unit = {
    val wsPtr = device.scratchBuffer.ptr

    device.burst({
      // Add epsilon.
      _NPP.addC(
        device,
        other.data.ptr,
        epsilon1,
        wsPtr,
        size
      )

      // Do divide.
      _NPP.div_I(
        device,
        wsPtr,
        data.ptr,
        size
      )
    })
  }

  /*
  @inline
  protected def doDivide(other: CUDARealTensor, epsilon1: NativeReal)
  : Unit = {
    using(device.requestScratchBuffer())(lock => {
      val noValues = layout.noValues
      val wsPtr    = lock.buffer.ptr
      device.burst({
        // Add epsilon.
        _CUBLAS.copy(
          device,
          noValues,
          other.data.ptr, 1,
          wsPtr,          1
        )
        _CUBLAS.axpy(
          device,
          noValues,
          epsilon1,
          device.one.ptr, 0,
          wsPtr,          1
        )

        // Do divide.
        _NPP.div_I(
          device,
          wsPtr,
          data.ptr,
          size
        )
      })
    })
  }
  */

  override protected def doDivide(epsilon0: Real,
                                  other:    Tensor, epsilon1: Real)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    /*
    using(
      _RealTensorNativeReal(epsilon0),
      _RealTensorNativeReal(epsilon1)
    )(doDivide(_, tmp, _))
    */
    doDivide(epsilon0, tmp, epsilon1)
    if (tmp ne other) {
      tmp.close()
    }
  }

  @inline
  private def doDivide(epsilon0: Real,
                       other:    CUDARealTensor, epsilon1: Real)
  : Unit = {
    val wsPtr  = device.scratchBuffer.ptr
    val dstPtr = data.ptr

    device.burst({
      // Add epsilon.
      _NPP.addC(
        device,
        other.data.ptr,
        epsilon1,
        wsPtr,
        size
      )

      // Do divide.
      _NPP.addC_I(
        device,
        epsilon0,
        dstPtr,
        size
      )
      _NPP.div_I(
        device,
        wsPtr,
        dstPtr,
        size
      )
    })
  }

  /*
  @inline
  private def doDivide(epsilon0: NativeReal,
                       other:    CUDARealTensor, epsilon1: NativeReal)
  : Unit = {
    using(device.requestScratchBuffer())(lock => {
      val noValues = layout.noValues
      val wsPtr    = lock.buffer.ptr
      val onePtr   = device.one.ptr
      val dstPtr   = data.ptr
      device.burst({
        // Add epsilon.
        _CUBLAS.copy(
          device,
          noValues,
          other.data.ptr, 1,
          wsPtr,          1
        )
        _CUBLAS.axpy(
          device,
          noValues,
          epsilon1,
          onePtr, 0,
          wsPtr,  1
        )

        // Do divide.
        _CUBLAS.axpy(
          device,
          noValues,
          epsilon0,
          onePtr, 0,
          dstPtr, 1
        )
        _NPP.div_I(
          device,
          wsPtr,
          dstPtr,
          size
        )
      })
    })
  }
  */

  override protected def doDivideR(value: Real)
  : Unit = {
    _NPP.divCRev_I(
      device,
      value,
      data.ptr,
      layout.noValues
    )
  }

  override protected def doDot(other: Tensor)
  : Real = {
    val tmp = other.asOrToCUDARealTensor(device)
    val result = doDot(tmp)
    if (tmp ne other) {
      tmp.close()
    }
    Real(result)
  }

  @inline
  private def doDot(other: CUDARealTensor)
  : _RealTensorReal = {
    _CUBLAS.dot(
      device,
      layout.noValues,
      data.ptr,       1,
      other.data.ptr, 1
    )
  }

  override protected def doLerp(other: Tensor, t: Real)
  : Unit = doAdd(Real.one - t, other, t)

  override protected def doLerp(other0: Tensor, other1: Tensor, t: Real)
  : Unit = doAdd(Real.one - t, other0, other1, t)

  override def l1Norm(epsilon: Double)
  : Real = {
    require(epsilon == 0.0)
    val result = _CUBLAS.asum(
      device,
      layout.noValues,
      data.ptr, 1
    )
    Real(result)
  }

  override def l2Norm(epsilon: Double)
  : Real = {
    require(epsilon == 0.0)
    val result = _CUBLAS.nrm2(
      device,
      layout.noValues,
      data.ptr, 1
    )
    Real(result)
  }

  override def l2NormSq
  : Real = {
    val srcPtr = data.ptr
    _CUBLAS.dot(
      device,
      layout.noValues,
      srcPtr, 1,
      srcPtr, 1
    )
  }

  override def sum
  : Real = {
    val result = _NPP.sum(
      device,
      data.ptr,
      size
    )
    Real(result)
  }

  override def min()
  : Real = {
    val result = _NPP.min(
      device,
      data.ptr,
      size
    )
    Real(result)
  }

  override def min(other: Tensor)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    min(tmp)
    if (tmp ne other) {
      tmp.close()
    }
  }

  def min(other: CUDARealTensor)
  : Unit = {
    require(layout == other.layout)
    _NPP.minEvery_I(
      device,
      other.data.ptr,
      data.ptr,
      layout.noValues
    )
  }

  override def max()
  : Real = {
    val result = _NPP.max(
      device,
      data.ptr,
      size
    )
    Real(result)
  }

  override def max(other: Tensor)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    max(tmp)
    if (tmp ne other) {
      tmp.close()
    }
  }

  def max(other: CUDARealTensor)
  : Unit = {
    require(layout == other.layout)
    _NPP.maxEvery_I(
      device,
      other.data.ptr,
      data.ptr,
      size
    )
  }

  override def maxAbs()
  : Real = {
    val index = _CUBLAS.iamax(
      device,
      layout.noValues,
      data.ptr, 1
    )
    get(index)
  }

  override protected def doMaxByAbs(other: Tensor)
  : Unit = {
    val tmp = other.asOrToCUDARealTensor(device)
    maxByAbs(tmp)
    if (tmp ne other) {
      tmp.close()
    }
  }

  @inline
  def maxByAbs(other: CUDARealTensor)
  : Unit = {
    val noValues = layout.noValues
    val wsPtr    = device.scratchBuffer.asRealTensorPtr

    device.burst({
      _NPP.threshold_GT(
        device,
        data.ptr,
        wsPtr,
        noValues,
        Real.zero
      )
      _NPP.threshold_LT_I(
        device,
        data.ptr,
        noValues,
        Real.zero
      )
      // TODO: Use faster function!
      _NPP.add_I(
        device,
        wsPtr,
        data.ptr,
        noValues
      )
    })
  }

  override def abs()
  : Unit = {
    _NPP.abs_I(
      device,
      data.ptr,
      layout.noValues
    )
  }

  override def sign()
  : Unit = {
    // TODO: Forgot to implement this. Dough!!!
    throw new NotImplementedError
  }

  override def sqr()
  : Unit = {
    /*
    Slower than gmm.
    _NPP.sqr(
      device,
      data,
      data,
      layout.noValues
    )
    */
    val noValues = layout.noValues
    val dstPtr   = data.ptr
    _CUBLAS.gmm(
      device,
      dstPtr, 1, 1, noValues,
      dstPtr, 1,    noValues,
      dstPtr, 1, 1, noValues
    )
  }

  override def sqrt()
  : Unit = {
    /*
    _NPP.sqrt_I(
      device,
      data.ptr,
      layout.noValues
    )
    */
    // Surprise, even with the overhead of the size object it is faster!
    _NPP.sqrt_I(
      device,
      data.ptr,
      size
    )
  }

  override def mean
  : Real = {
    val result = _NPP.mean(
      device,
      data.ptr,
      layout.noValues
    )
    Real(result)
  }

  override def stdDev(epsilon: Double)
  : Real = {
    require(epsilon == Real.zero)
    val result = _NPP.stdDev(
      device,
      data.ptr,
      layout.noValues
    )
    Real(result)
  }

  /*
  // Move compatible bias to device and add it.
  override def scaleAddBias(bias: DVec, alpha: RealPtr, beta: RealPtr): Unit = {
    using(TensorDesc.nchw(1, 1, bias.length), DeviceBuffer.derive(buffer.device, bias))(
      (biasDesc, biasBuffer) => scaleAddBias(biasDesc, biasBuffer, alpha, beta)
    )
  }
  */

  // Move compatible bias to device and add it.
  /*
  override def scaleAddBias(bias: DVec, alpha: RealPtr, beta: RealPtr): Unit = {
    using(
      TensorDescPtr.nhwc(1, 1, bias.length),
      DeviceRealPtr.derive(device, bias)
    )(
      (biasDescPtr, biasDataPtr) => {
        scaleAddBias(biasDescPtr, biasDataPtr, alpha, beta)
      }
    )
  }*/

  /*
 final def scaleAddBias(bias: CUDATensor, alpha: RealPtr, beta: RealPtr)
 : Unit = cudnnSynchronized(device, cudnnAddTensor(
   device.cudnnContext,
   CUDNN_ADD_SAME_C,
   alpha, bias.desc.ptr, bias.buffer.ptr,
   beta,  desc.ptr,      buffer.ptr
 ))

 // Adds a bias like tensor to each sample.
 final def scaleAddBias(biasDesc:    TensorDesc,
                        biasBuffer:  DeviceBuffer,
                        alpha:       RealPtr,
                        beta:        RealPtr)
 : Unit = cudnnSynchronized(device, cudnnAddTensor(
   device.cudnnContext,
   CUDNN_ADD_SAME_C,
   alpha, biasDesc.ptr, biasBuffer.ptr,
   beta,  desc.ptr,     buffer.ptr
 ))

 def scaleAddBias(bias: DVec, alpha: RealPtr, beta: RealPtr): Unit
 */

  // TODO: Needs implementation that maintains CUDA vector.
  override def ++(other: Tensor)
  : CUDARealTensor = {
    val tmp = other.asOrToCUDARealTensor(device)
    val result = ++(tmp)
    if (tmp ne other) {
      tmp.close()
    }
    result
    /*
    using(toRealTensor)(a => {
      val b = other.asOrToRealTensor
      val c = a ++ b

      val result = c.toCUDARealTensor(device)

      // Cleanup.
      if (b ne other) {
        b.close()
      }
      result
    })
    */
  }

  @inline
  def ++(other: CUDARealTensor)
  : CUDARealTensor = {
    val result = CUDARealTensor(device, layout ++ other.layout)
    device.burst({
      // TODO: Figure out why this does not work with transformTensor. https://devtalk.nvidia.com/default/topic/939233/gpu-accelerated-libraries/cudnntransformtensor-amp-cudnnaddtensor-interpret-stride-different/
      using(
        _TensorStruct.nchw(layout, result.layout.size, _RealTensorDeviceBuffer.dataType)
      )(dstDescPtr => {
        _CUDNN.addTensor(
          device,
          _RealTensorNativeReal.one,  desc,       data.ptr,
          _RealTensorNativeReal.zero, dstDescPtr, result.data.ptr
        )
      })

      using(
        _TensorStruct.nchw(other.layout, result.layout.size, _RealTensorDeviceBuffer.dataType)
      )(dstDescPtr => {
        _CUDNN.addTensor(
          device,
          _RealTensorNativeReal.one,  other.desc, other.data.ptr,
          _RealTensorNativeReal.zero, dstDescPtr, result.data.ptr.withOffset(layout.size.noTuples)
        )
      })
    })
    result
  }

  override def :++(other: Tensor)
  : CUDARealTensor = {
    val tmp = other.asOrToCUDARealTensor(device)
    val result = :++(tmp)
    if (tmp ne other) {
      tmp.close()
    }
    result
    /*
    using(toRealTensor)(a => {
      val b = other.asOrToRealTensor
      val c = a :++ b

      val result = c.toCUDARealTensor(device)

      // Cleanup.
      if (b ne other) {
        b.close()
      }
      result
    })
    */
  }

  @inline
  def :++(other: CUDARealTensor)
  : CUDARealTensor = {
    val result = CUDARealTensor(device, layout :++ other.layout)
    device.burst({
      // TODO: Figure out why this does not work with transformTensor. https://devtalk.nvidia.com/default/topic/939233/gpu-accelerated-libraries/cudnntransformtensor-amp-cudnnaddtensor-interpret-stride-different/
      using(
        _TensorStruct.nchw(layout, result.layout.size, _RealTensorDeviceBuffer.dataType)
      )(dstDescPtr => {
        _CUDNN.addTensor(
          device,
          _RealTensorNativeReal.one, desc, data.ptr,
          _RealTensorNativeReal.zero, dstDescPtr, result.data.ptr
        )
      })

      using(
        _TensorStruct.nchw(other.layout, result.layout.size, _RealTensorDeviceBuffer.dataType)
      )(dstDescPtr => {
        _CUDNN.addTensor(
          device,
          _RealTensorNativeReal.one, other.desc, other.data.ptr,
          _RealTensorNativeReal.zero, dstDescPtr, result.data.ptr.withOffset(layout.size.noValues)
        )
      })
    })
    result
  }

  override protected def doSlice(tuple0: Int,
                                 result: Tensor)
  : Unit = {
    val tmp = result.asOrToCUDARealTensor(device)
    doSlice(tuple0, tmp)
    if (tmp ne result) {
      tmp.copyTo(result)
      tmp.close()
    }
    /*
    using(toRealTensor)(
      _.slice(result, unit0)
    )
    */
  }

  @inline
  protected def doSlice(tuple0: Int,
                        result: CUDARealTensor)
  : Unit = {
    // TODO: Figure out why this does not work with transformTensor. https://devtalk.nvidia.com/default/topic/939233/gpu-accelerated-libraries/cudnntransformtensor-amp-cudnnaddtensor-interpret-stride-different/
    using(
      _TensorStruct.nchw(result.layout, layout.size, _RealTensorDeviceBuffer.dataType)
    )(srcDesc => {
      _CUDNN.addTensor(
        device,
        _RealTensorNativeReal.one,  srcDesc,     data.ptr.withOffset(tuple0),
        _RealTensorNativeReal.zero, result.desc, result.data.ptr
      )
    })
  }

  override protected def doSliceChannels(channel0: Int,
                                         result:   Tensor)
  : Unit = {
    val tmp = result.asOrToCUDARealTensor(device)
    doSliceChannels(channel0, tmp)
    if (tmp ne result) {
      tmp.copyTo(result)
      tmp.close()
    }
    /*
    using(toRealTensor)(
      _.sliceChannels(result, channel0)
    )
    */
  }

  @inline
  protected def doSliceChannels(channel0: Int,
                                result:   CUDARealTensor)
  : Unit = {
    // TODO: Figure out why this does not work with transformTensor. https://devtalk.nvidia.com/default/topic/939233/gpu-accelerated-libraries/cudnntransformtensor-amp-cudnnaddtensor-interpret-stride-different/
    val offset = layout.size.noTuples * channel0
    using(
      _TensorStruct.nchw(result.layout, layout.size, _RealTensorDeviceBuffer.dataType)
    )(srcDesc => {
      _CUDNN.addTensor(
        device,
        _RealTensorNativeReal.one,  srcDesc,     data.ptr.withOffset(offset),
        _RealTensorNativeReal.zero, result.desc, result.data.ptr
      )
    })
  }

  override def toRealArrayTensor
  : RealArrayTensor = RealArrayTensor(layout, values)

  override def asOrToRealArrayTensor
  : RealArrayTensor = RealArrayTensor(layout, values)

}

object CUDARealTensor {

  final def apply(device: LogicalDevice, layout: IndependentTensorLayout)
  : CUDARealTensor = new CUDARealTensor(device, layout)

  final def derive(device: LogicalDevice, size: Size, noSamples: Int)
  : CUDARealTensor = apply(device, IndependentTensorLayout(size, noSamples))

  final def fill(device: LogicalDevice, layout: IndependentTensorLayout, value: Real)
  : CUDARealTensor = {
    val result = apply(device, layout)
    result := value
    result
  }

  final def ones(device: LogicalDevice, layout: IndependentTensorLayout)
  : CUDARealTensor = fill(device, layout, Real.one)

  final def zeros(device: LogicalDevice, layout: IndependentTensorLayout)
  : CUDARealTensor = {
    val result = apply(device, layout)
    result.clear()
    result
  }

  /*
  /**
    * This is used for filters.
    */
  final def apply(device:     LogicalDevice,
                  kernel:     Kernel,
                  noChannels: Int,
                  noMaps:     Int)
  : CUDARealTensor = apply(device, Size1(kernel.noValues, noChannels), noMaps)

  final def derive(device: LogicalDevice, values: DenseVector[Real])
  : CUDARealTensor = {
    val result = like(device, values)
    result := values
    result
  }

  final def like(device: LogicalDevice, values: DenseVector[Real])
  : CUDARealTensor = apply(device, Size1(1, values.length), 1)

  final def like(other: CUDARealTensor)
  : CUDARealTensor = apply(other.device, other.size, other.noSamples)
  */

}

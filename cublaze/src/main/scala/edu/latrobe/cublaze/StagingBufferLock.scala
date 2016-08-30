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
import org.bytedeco.javacpp.cuda._

final class StagingBufferLock private(val bufferIndex: Int,
                                      val buffer:      _RealHostBuffer)
  extends AutoClosing {
  require(bufferIndex >= 0 && buffer != null)

  override protected def doClose()
  : Unit = {
    StagingBufferLock.unlock(bufferIndex)
    super.doClose()
  }

  def download(src: CUDARealTensor)
  : Unit = {
    require(src.data.capacityInBytes <= buffer.capacityInBytes)

    val device = src.device

    // We can bypass the transform for bias tensors.
    if (
      (_RealDeviceBuffer eq _RealTensorDeviceBuffer) &&
      (src.layout.size.noValues == src.layout.size.noChannels)
    ) {
      _CUDA.memcpyAsync(
        device,
        buffer.ptr,
        src.data.ptr,
        src.data.capacityInBytes,
        cudaMemcpyDeviceToHost
      )
    }
    else {
      using(
        _TensorStruct.nhwc(src.layout, _RealDeviceBuffer.dataType)
      )(dstDesc => {
        val wsPtr = device.scratchBuffer.asRealPtr
        _CUDNN.transformTensor(
          device,
          _RealTensorNativeReal.one, src.desc, src.data.ptr,
          NativeReal.zero,           dstDesc,  wsPtr
        )
        _CUDA.memcpyAsync(
          device,
          buffer.ptr,
          wsPtr,
          src.data.capacityInBytes,
          cudaMemcpyDeviceToHost
        )
      })
    }

    // Make sure the data has been handed over properly in synchronous mode!
    if (CUBLAZE_ASYNCHRONOUS) {
      device.forceSynchronize()
    }
  }

  def upload(dst: CUDARealTensor)
  : Unit = {
    require(dst.data.capacityInBytes <= buffer.capacityInBytes)

    val device = dst.device

    // We can bypass the transform for bias tensors.
    if (
      (_RealDeviceBuffer eq _RealTensorDeviceBuffer) &&
      (dst.layout.size.noValues == dst.layout.size.noChannels)
    ) {
      _CUDA.memcpyAsync(
        device,
        dst.data.ptr,
        buffer.ptr,
        dst.data.capacityInBytes,
        cudaMemcpyHostToDevice
      )
    }
    else {
      using(
        _TensorStruct.nhwc(dst.layout, _RealDeviceBuffer.dataType)
      )(srcDesc => {
        val wsPtr = device.scratchBuffer.asRealPtr
        _CUDA.memcpyAsync(
          device,
          wsPtr,
          buffer.ptr,
          dst.data.capacityInBytes,
          cudaMemcpyHostToDevice
        )
        _CUDNN.transformTensor(
          device,
          NativeReal.one,             srcDesc,  wsPtr,
          _RealTensorNativeReal.zero, dst.desc, dst.data.ptr
        )
      })
    }

    // Make sure the data has been handed over properly in synchronous mode!
    if (CUBLAZE_ASYNCHRONOUS) {
      device.forceSynchronize()
    }
  }

}

object StagingBufferLock {

  /**
    * Staging buffers in the host memory.
    */
  final private val stagingBuffers
  : Array[_RealHostBuffer] = {
    val capacity = CUBLAZE_STAGING_BUFFER_SIZE / Real.size
    Array.fill[_RealHostBuffer](
      CUBLAZE_NO_STAGING_BUFFERS
    )(_RealHostBuffer(capacity))
  }

  final private val locks
  : Array[Boolean] = new Array[Boolean](stagingBuffers.length)

  final private val lockRNG
  : PseudoRNG = PseudoRNG()

  final def request()
  : StagingBufferLock = synchronized {
    while (true) {
      // Try finding a free buffer.
      val offset = lockRNG.nextInt(locks.length)
      var i      = 0
      while (i < locks.length) {
        val j = (i + offset) % locks.length
        if (!locks(j)) {
          locks(j) = true
          return new StagingBufferLock(j, stagingBuffers(j))
        }
        i += 1
      }

      // Wait for monitor signal.
      try {
        wait()
      }
      catch {
        case ex: InterruptedException =>
          logger.debug(s"CUDA.lockStagingBuffer: Caught exception $ex!")
          // Probably the GC has not removed the lock. Let's give it a chance.
          System.gc()
          System.runFinalization()
      }
    }

    // Something went terribly wrong.
    throw new UnknownError
  }

  final private def unlock(index: Int): Unit = synchronized {
    locks(index) = false
    notifyAll()
  }

}

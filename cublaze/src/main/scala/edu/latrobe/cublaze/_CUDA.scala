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
import java.nio.charset._
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.cuda._

/**
 * Enumerates available CUDA devices and manages access to them.
 */
private[cublaze] object _CUDA {

  final def check(resultCode: Int): Unit = {
    if (resultCode != cudaSuccess) {
      using(
        cudaGetErrorName(resultCode),
        cudaGetErrorString(resultCode)
      )(
        (errNamePtr, errStrPtr) => {
          val errName = errNamePtr.getString(StandardCharsets.US_ASCII.name)
          val errStr  = errStrPtr.getString(StandardCharsets.US_ASCII.name)
          throw new InternalError(s"CUDA: $errName - $errStr")
        }
      )
    }
  }

  final val version
  : Int = CUDA_VERSION

  final val driverVersion: Int = {
    val version = new Array[Int](1)
    val result  = cudaDriverGetVersion(version)
    check(result)
    version(0)
  }

  final val runtimeVersion: Int = {
    val version = new Array[Int](1)
    val result  = cudaRuntimeGetVersion(version)
    check(result)
    version(0)
  }

  @inline
  final def deviceGetCacheConfig
  : Int = {
    val config = new Array[Int](1)
    val result = cudaDeviceGetCacheConfig(config)
    check(result)
    config(0)
  }

  @inline
  final def deviceGetLimit(limit: Int)
  : Long = {
    using(NativeSizeT.allocate(1L))(tmp => {
      val result = cudaDeviceGetLimit(tmp.ptr, limit)
      check(result)
      tmp.ptr.get()
    })
  }

  @inline
  final def deviceGetPCIBusID(deviceIndex: Int)
  : String = {
    val buffer = new Array[Byte](16)
    val result = cudaDeviceGetPCIBusId(buffer, buffer.length, deviceIndex)
    check(result)
    StringEx.render(buffer)
  }

  @inline
  final def deviceGetSharedMemConfig
  : Int = {
    val config = new Array[Int](1)
    val result = cudaDeviceGetSharedMemConfig(config)
    check(result)
    config(0)
  }

  @inline
  final def deviceSetCacheConfig(cacheConfig: Int)
  : Unit = {
    val result = cudaDeviceSetCacheConfig(cacheConfig)
    check(result)
  }

  @inline
  final def deviceSetLimit(limit: Int, value: Long)
  : Unit = {
    val result = cudaDeviceSetLimit(limit, value)
    check(result)
  }

  @inline
  final def deviceSetSharedMemConfig(config: Int)
  : Unit = {
    val result = cudaDeviceSetSharedMemConfig(config)
    check(result)
  }

  @inline
  final def free(ptr: Pointer)
  : Unit = {
    val result = cudaFree(ptr)
    check(result)
  }

  @inline
  final def freeHost(ptr: Pointer)
  : Unit = {
    val result = cudaFreeHost(ptr)
    check(result)
  }

  @inline
  final def getDevice
  : Int = {
    val index = new Array[Int](1)
    val result = cudaGetDevice(index)
    check(result)
    index(0)
  }

  @inline
  final def getDeviceCount
  : Int = {
    val noDevices = new Array[Int](1)
    val result    = cudaGetDeviceCount(noDevices)
    check(result)
    noDevices(0)
  }

  @inline
  final def getDeviceFlags
  : Int = {
    val flags  = new Array[Int](1)
    val result = cudaGetDeviceFlags(flags)
    check(result)
    flags(0)
  }

  @inline
  final def getDeviceProperties(device: PhysicalDevice)
  : cudaDeviceProp = {
    val ptr    = new cudaDeviceProp
    val result = cudaGetDeviceProperties(ptr, device.index)
    check(result)
    ptr
  }

  @inline
  final def malloc(ptr: Pointer, noBytes: Long)
  : Unit = {
    require(noBytes >= 0L)

    var result = cudaMalloc(ptr, noBytes)
    if (result == cudaErrorMemoryAllocation) {
      logger.error(s"CUDA.malloc exception: $cudaErrorMemoryAllocation")
      System.gc()
      System.runFinalization()
      result = cudaMalloc(ptr, noBytes)
    }
    check(result)
  }

  @inline
  final def mallocHost(ptr: Pointer, noBytes: Long)
  : Unit = {
    val result = cudaHostAlloc(ptr, noBytes, cudaHostAllocPortable)
    check(result)
  }

  @inline
  final def memcpy(dstPtr:  Pointer,
                   srcPtr:  Pointer,
                   noBytes: Long,
                   kind:    Int)
  : Unit = {
    val result = cudaMemcpy(
      dstPtr,
      srcPtr,
      noBytes,
      kind
    )
    check(result)
  }

  @inline
  final def memcpyAsync(device:    LogicalDevice,
                        dstPtr:    Pointer,
                        srcPtr:    Pointer,
                        noBytes:   Long,
                        kind:      Int)
  : Unit = {
    val result = cudaMemcpyAsync(
      dstPtr,
      srcPtr,
      noBytes,
      kind,
      device.streamPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def memcpyPeer(dstDevice: LogicalDevice,
                       dstPtr:    Pointer,
                       srcDevice: LogicalDevice,
                       srcPtr:    Pointer,
                       noBytes:   Long)
  : Unit = {
    val result = cudaMemcpyPeerAsync(
      dstPtr, dstDevice.index,
      srcPtr, srcDevice.index,
      noBytes,
      srcDevice.streamPtr
    )
    check(result)
    srcDevice.trySynchronize()
  }

  @inline
  final def memset[T <: Pointer](device:  LogicalDevice,
                                 buffer:  _DeviceBuffer[T],
                                 value:   Int,
                                 noBytes: Long)
  : Unit = {
    val result = cudaMemsetAsync(
      buffer.ptr,
      value,
      noBytes,
      device.streamPtr
    )
    check(result)
    device.trySynchronize()
  }

  @inline
  final def memGetInfo
  : (Long, Long) = {
    using(NativeSizeT.allocate(1L), NativeSizeT.allocate(1L))(
      (freeSize, totalSize) => {
        val result = cudaMemGetInfo(
          freeSize.ptr,
          totalSize.ptr
        )
        check(result)
        (freeSize.ptr.get, totalSize.ptr.get)
      }
    )
  }

  @inline
  final def setDevice(device: LogicalDevice)
  : Unit = setDevice(device.physicalDevice)

  @inline
  final def setDevice(device: PhysicalDevice)
  : Unit = {
    val result = cudaSetDevice(device.index)
    check(result)
  }

  @inline
  final def setDeviceFlags(flags: Int)
  : Unit = {
    val result = cudaSetDeviceFlags(flags)
    check(result)
  }

}

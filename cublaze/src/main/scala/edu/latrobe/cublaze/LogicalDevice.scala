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
import org.bytedeco.javacpp.cublas._
import org.bytedeco.javacpp.cuda._
import org.bytedeco.javacpp.cudnn._
import scala.collection._
import scala.util.hashing._

final class LogicalDevice private (val index:          Int,
                                   val physicalDevice: PhysicalDevice)
  extends AutoClosing
    with Equatable {
  _CUDA.setDevice(physicalDevice)

  val streamPtr
  : CUstream_st = new CUstream_st
  _CUDA.check(cudaStreamCreate(streamPtr))

  val blasContextPtr
  : cublasContext = _CUBLAS.create()
  _CUBLAS.setStream(this)

  val dnnContextPtr
  : cudnnContext = _CUDNN.create()
  _CUDNN.setStream(this)

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), index.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LogicalDevice]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: LogicalDevice =>
      index == other.index
    case _ =>
      false
  })

  override protected def doClose()
  : Unit = {
    one.close()
    scratchBuffer.close()
    _CUDA.setDevice(physicalDevice)
    _CUDNN.destroy(this)
    _CUBLAS.destroy(this)
    _CUDA.check(cudaStreamDestroy(streamPtr))
    super.doClose()
  }

  /*
  def requestScratchBuffer()
  : ScratchBufferLock = physicalDevice.requestScratchBuffer()
  */

  private var _burstMode
  : Boolean = false

  def burstMode
  : Boolean = _burstMode

  def trySynchronize()
  : Unit = {
    if (!_burstMode && !CUBLAZE_ASYNCHRONOUS) {
      forceSynchronize()
    }
  }

  def forceSynchronize()
  : Unit = {
    val result = cudaStreamSynchronize(streamPtr)
    _CUDA.check(result)
  }

  def burst(fn: => Unit)
  : Unit = {
    _burstMode = true
    fn
    _burstMode = false
    trySynchronize()
  }

  private[cublaze] val scratchBuffer
  : _RealDeviceBuffer = {
    val capacity = CUBLAZE_SCRATCH_BUFFER_SIZE.toInt / Real.size
    _RealDeviceBuffer(this, capacity)
  }

  // A dummy buffer containing a single one.
  // This is actually slightly redundant. But well... 2-8 bytes... Who cares...
  private[cublaze] val one
  : _RealDeviceBuffer = _RealDeviceBuffer(this, 1)
  _NPP.set(this, Real.one, one.ptr, 1)
  forceSynchronize()

}

object LogicalDevice {

  final private val devices
  : Array[LogicalDevice] = {
    val builder = Array.newBuilder[LogicalDevice]
    val noDevices = CUBLAZE_NO_LOGICAL_DEVICES.map(_.toString.toInt).toArray
    ArrayEx.foreachPair(noDevices)((deviceNo, n) => {
      if (deviceNo < PhysicalDevice.count) {
        var i = 0
        while (i < n) {
          builder += new LogicalDevice(i, PhysicalDevice(deviceNo))
          i       += 1
        }
      }
    })
    builder.result()
  }

  final val count
  : Int = devices.length

  final def apply(index: Int)
  : LogicalDevice = devices(index)

  /**
    * Avoids that we prefer using a specific device.
    */
  final private val deviceRNG
  : PseudoRNG = PseudoRNG()

  /**
    * Quick lookup table to check whether a thread already has claimed a
    * device.
    * ThreadID => DeviceIndex
    */
  final private val locks
  : mutable.Map[Long, Int] = mutable.Map.empty

  final def tryClaim()
  : Option[DeviceClaim] = synchronized {
    val threadID = Thread.currentThread().getId

    // Check for existing device claim.
    val deviceIndex = locks.get(threadID)
    deviceIndex.foreach(i => {
      val device = devices(i)
      return Some(DeviceClaim(threadID, device))
    })

    // Claim a random device.
    val lockedDevices = locks.values.toArray
    val offset = deviceRNG.nextInt(devices.length)
    var i      = 0
    while (i < devices.length) {
      val deviceIndex = (i + offset) % devices.length
      if (!ArrayEx.contains(lockedDevices, deviceIndex)) {
        val device = devices(deviceIndex)
        _CUDA.setDevice(device)
        locks += Tuple2(threadID, deviceIndex)
        return Some(DeviceClaim(threadID, device))
      }
      i += 1
    }

    // Give up. No devices available.
    None
  }

  final def available
  : Boolean = {
    val claim = tryClaim()
    claim.foreach(claim => {
      claim.close()
      return true
    })
    false
  }

  final def claim()
  : DeviceClaim = tryClaim().get

  final private[cublaze] def unlock(claim: DeviceClaim)
  : Unit = synchronized {
    locks.remove(claim.creatorID)
  }

}
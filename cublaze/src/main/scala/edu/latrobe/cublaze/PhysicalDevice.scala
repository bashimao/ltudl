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
import org.bytedeco.javacpp.cuda._
import org.json4s.JsonAST._
import scala.util.hashing._

final class PhysicalDevice private(val index: Int)
  extends Closable
    with Equatable {
  _CUDA.setDevice(this)

  val propertiesPtr
  : cudaDeviceProp = _CUDA.getDeviceProperties(this)

  /*
  private val scratchBuffers
  : Array[_RealDeviceBuffer] = {
    val capacity = CUBLAZE_SCRATCH_BUFFER_SIZE.toInt / Real.size
    val n        = CUBLAZE_NO_SCRATCH_BUFFERS(index).toString.toInt
    ArrayEx.fill(
      n
    )(_RealDeviceBuffer(this, capacity))
  }

  private val locks
  : Array[Boolean] = new Array[Boolean](scratchBuffers.length)

  private val lockRNG
  : PseudoRNG = PseudoRNG()

  def requestScratchBuffer()
  : ScratchBufferLock = synchronized {
    while (true) {
      // Try finding a free buffer.
      val offset = lockRNG.nextInt(scratchBuffers.length)
      var i      = 0
      while (i < locks.length) {
        val j = (i + offset) % locks.length
        if (!locks(j)) {
          locks(j) = true
          return ScratchBufferLock(j, scratchBuffers(j))
        }
        i += 1
      }

      // Wait for monitor signal.
      try {
        wait()
      }
      catch {
        case ex: InterruptedException =>
          logger.debug(s"requestScratchBuffer: Caught exception $ex!")
          // Probably the GC has not removed the lock. Let's give it a chance.
          System.gc()
          System.runFinalization()
      }
    }

    // Something went terribly wrong.
    throw new UnknownError
  }

  protected[cublaze] def unlockScratchBuffer(index: Int)
  : Unit = synchronized {
    locks(index) = false
    notifyAll()
  }
  */

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), index.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[PhysicalDevice]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: PhysicalDevice =>
      index == other.index
    case _ =>
      false
  })

  override protected def doClose()
  : Unit = {
    propertiesPtr.deallocate()
    super.doClose()
  }


  // ---------------------------------------------------------------------------
  //    Information retrieval.
  // ---------------------------------------------------------------------------
  /**
    * cudaDeviceGetPCIBusId
    */
  val busID
  : String = _CUDA.deviceGetPCIBusID(index)

  /**
    * Calls cudaGetDeviceFlags
    */
  def flags
  : Int = _CUDA.getDeviceFlags

  /**
    * Calls cudaSetDeviceFlags
    */
  def flags_=(value: Int)
  : Unit = _CUDA.setDeviceFlags(value)

  /**
    * Calls cudaDeviceGetLimit
    */
  def getLimit(limit: Int)
  : Long = _CUDA.deviceGetLimit(limit)

  /**
    * cudaDeviceSetLimit
    */
  def setLimit(limit: Int, value: Long)
  : Unit = _CUDA.deviceSetLimit(limit, value)

  def stackSizeLimit
  : Long = getLimit(cudaLimitStackSize)

  def stackSizeLimit_=(value: Long)
  : Unit = setLimit(cudaLimitStackSize, value)

  def printfFifoSizeLimit
  : Long = getLimit(cudaLimitPrintfFifoSize)

  def printfFifoSizeLimit_=(value: Long)
  : Unit = setLimit(cudaLimitPrintfFifoSize, value)

  def mallocHeapSizeLimit
  : Long = getLimit(cudaLimitMallocHeapSize)

  def mallocHeapSizeLimit_=(value: Long)
  : Unit = setLimit(cudaLimitMallocHeapSize, value)

  def runtimeSynchronizationDepthLimit
  : Long = getLimit(cudaLimitDevRuntimeSyncDepth)

  def runtimeSynchronizationDepthLimit_=(value: Long)
  : Unit = setLimit(cudaLimitDevRuntimeSyncDepth, value)

  def runtimePendingLaunchCountLimit
  : Long = getLimit(cudaLimitDevRuntimePendingLaunchCount)

  def runtimePendingLaunchCountLimit_=(value: Long)
  : Unit = setLimit(cudaLimitDevRuntimePendingLaunchCount, value)

  /**
    * Calls cudaDeviceGetCacheConfig
    */
  def cacheConfiguration
  : Int = _CUDA.deviceGetCacheConfig

  /**
    * Calls cudaDeviceSetCacheConfig
    */
  def cacheConfiguration_=(value: Int)
  : Unit = _CUDA.deviceSetCacheConfig(value)

  /**
    * Calls cudaDeviceGetSharedMemConfig
    */
  def sharedMemoryConfiguration
  : Int = _CUDA.deviceGetSharedMemConfig

  /**
    * Calls cudaDeviceSetSharedMemConfig
    */
  def sharedMemoryConfiguration_=(value: Int)
  : Unit = _CUDA.deviceSetSharedMemConfig(value)

  /**
    * Calls cudaDeviceReset
    */
  /*
  def reset(): Unit = {
    CUDA.checkResult(cudaSetDevice(physicalIndex))
    CUDA.checkResult(cudaDeviceReset())
  }
  */

  def freeMemorySize
  : Long = _CUDA.memGetInfo._1

  val memorySize
  : Long = _CUDA.memGetInfo._2


  def collectRuntimeStatus()
  : JObject = {
    val fields = List.newBuilder[JField]

    fields += Json.field("index", index)
    fields += Json.field("name", StringEx.render(propertiesPtr.name().getStringBytes))
    fields += Json.field("busID", busID)
    fields += Json.field("memorySize", memorySize)
    fields += Json.field("freeSize", freeMemorySize)

    JObject(fields.result())
  }

}

object PhysicalDevice {

  final val count
  : Int = _CUDA.getDeviceCount

  final private val devices
  : Array[PhysicalDevice] = ArrayEx.tabulate(
    count
  )(i => new PhysicalDevice(i))

  final def apply(index: Int)
  : PhysicalDevice = devices(index)

  final def current()
  : PhysicalDevice = {
    val index = _CUDA.getDevice
    devices(index)
  }

  final def collectRuntimeStatus()
  : JArray = {
    val tmp = ArrayEx.map(
      devices
    )(_.collectRuntimeStatus())
    Json(tmp)
  }

}

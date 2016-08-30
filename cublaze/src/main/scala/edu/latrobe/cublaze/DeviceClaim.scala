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
import scala.collection._
import scala.ref._

final class DeviceClaim(val creatorID: Long,
                        val device:    LogicalDevice)
  extends AutoClosing {
  DeviceClaim.register(this)

  override protected def doClose()
  : Unit = {
    DeviceClaim.unregister(this)
    super.doClose()
  }

}

object DeviceClaim {

  final def apply(creatorID: Long,
                  device:    LogicalDevice)
  : DeviceClaim = new DeviceClaim(creatorID, device)

  /**
    * All existing locks should show up here.
    * (otherwise something is going terribly wrong!)
    */
  final private val claims
  : mutable.Map[Long, mutable.Buffer[WeakReference[DeviceClaim]]] = {
    mutable.Map.empty
  }

  final private def register(claim: DeviceClaim)
  : Unit = synchronized {
    val buffer = claims.getOrElseUpdate(claim.creatorID, mutable.Buffer.empty)
    buffer += WeakReference(claim)
  }

  final private def unregister(claim: DeviceClaim)
  : Unit = synchronized {
    // Remove claim.
    val buffer = claims(claim.creatorID)
    val tmp    = buffer.find(_.get.exists(_ eq claim))
    buffer -= tmp.get

    // If this was the last claim, unlock the device.
    if (buffer.isEmpty) {
      claims.remove(claim.creatorID)
      LogicalDevice.unlock(claim)
    }
  }

/*
  final private val locks
  : Array[Boolean] = Array.ofDim[Boolean](CUDA.logicalDevices.length)

  final private val locksRNG
  : Rand[Int] = PseudoRNG().randInt(locks.length)


  final private val locksByThread = {
    mutable.Map.empty[Long, WeakReference[DeviceClaim]]
  }

  final private def doTryRequest(): DeviceClaim = {
    val threadID = Thread.currentThread().getId

    // Try to find existing lock that fulfills request.
    locksByThread.get(threadID) match {
      case Some(weakLock) =>
        val optLock = weakLock.get
        val lock    = optLock.orNull
        if (lock != null) {
          return lock.claim()
        }
      case None =>
    }

    if (locks.nonEmpty) {
      // We offset the search by a random index to avoid always populating
      // devices front to back.
      val offset = locksRNG.draw()
      var i = 0
      while (i < locks.length) {
        val j = (i + offset) % locks.length
        if (!locks(j)) {
          locks(j) = true
          // Create lock and register it.
          val lock = new DeviceClaim(CUDA.logicalDevices(j))
          locksByThread += threadID -> WeakReference(lock)
          return lock
        }
        i += 1
      }
    }

    // Give up
    null
  }

  final def tryRequest(): DeviceClaim = synchronized {
    doTryRequest()
  }

  final def request(): DeviceClaim = synchronized {
    // Poll for a permit or wait for the next chance of such a permit to become
    // available.
    while (true) {
      val lock = doTryRequest()
      if (lock != null) {
        // Ok, now we have a device lock.
        return lock
      }

      // Wait for monitor signal.
      blazeLogger.info("No CUDA device available. Waiting...")
      try {
        wait()
      }
      catch {
        case ex: InterruptedException =>
          blazeLogger.debug(s"CUDA.lockDevice: Caught exception $ex!")
          // Probably the GC has not removed the lock. Let's give it a chance.
          System.gc()
      }
    }

    // Something went terribly wrong.
    throw new UnknownError
  }

  final def isLockAvailable: Boolean = using(tryRequest())(_ != null)

  final private def unlock(creatorThreadID: Long, device: LogicalDevice)
  : Unit = synchronized {
    locks(device.index) = false
    locksByThread.remove(creatorThreadID)
    notify()
  }
  */
}
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
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.cuda._

/**
  * Will be used to exchange data between CPU and GPU.
  *
  * IMPORTANT: Always instantiate host buffers after all threads have been created to make sure they are mapped into the right memory area.
  */
abstract class _HostBuffer[TPtr <: Pointer]
  extends AutoClosingPointerEx[TPtr]
    with CopyableEx[_HostBuffer[TPtr]] {

  def capacityInBytes
  : Long

  final override protected def doClose()
  : Unit = {
    _CUDA.freeHost(_ptr)
    super.doClose()
  }



  final def copyTo[UPtr <: Pointer](other: _HostBuffer[UPtr])
  : Unit = {
    require(other.capacityInBytes >= capacityInBytes)
    _CUDA.memcpy(
      other.ptr,
      ptr,
      capacityInBytes,
      cudaMemcpyHostToHost
    )
  }

}

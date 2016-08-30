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

/**
  * A buffer that lives within the cuda device.
  */
abstract class _DeviceBuffer[TPtr <: Pointer]
  extends AutoClosingPointerEx[TPtr] {

  def device
  : PhysicalDevice

  def capacityInBytes
  : Long

  final override protected def doClose()
  : Unit = {
    // TODO: Checked if (finalizing) in old code. Is this a performance issue?
    val devicePrev = PhysicalDevice.current()
    if (devicePrev ne device) {
      _CUDA.setDevice(device)
      _CUDA.free(_ptr)
      _CUDA.setDevice(devicePrev)
    }
    else {
      _CUDA.free(_ptr)
    }
    super.doClose()
  }

  final def asBytePtr
  : BytePointer =new BytePointer {
    address = ptr.address()
    capacity(capacityInBytes)
  }

  final def asDoublePtr
  : DoublePointer = new DoublePointer {
    address = ptr.address()
    capacity(capacityInBytes / DoubleEx.size)
  }

  final def asFloatPtr
  : FloatPointer = new FloatPointer {
    address = ptr.address()
    capacity(capacityInBytes / FloatEx.size)
  }

  final def asHalfPtr
  : HalfPointer = new HalfPointer {
    address = ptr.address()
    capacity(capacityInBytes / Half.size)
  }

  // ---------------------------------------------------------------------------
  //    REAL SWITCH DOUBLE
  // ---------------------------------------------------------------------------
  /*
  final def asRealPtr
  : RealPointer = asDoublePtr
  */
  // ---------------------------------------------------------------------------
  //    REAL SWITCH FLOAT
  // ---------------------------------------------------------------------------
  final def asRealPtr
  : RealPointer = asFloatPtr
  // ---------------------------------------------------------------------------
  //    REAL SWITCH HALF
  // ---------------------------------------------------------------------------
  /*
  final def asRealPtr
  : RealPointer = asHalfPtr
  */
  // ---------------------------------------------------------------------------
  //    REAL SWITCH END
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  //    REAL TENSOR SWITCH DOUBLE
  // ---------------------------------------------------------------------------
  /*
  final def asRealTensorPtr
  : _RealTensorPointer = asDoublePtr
  */
  // ---------------------------------------------------------------------------
  //    REAL TENSOR SWITCH FLOAT
  // ---------------------------------------------------------------------------
  final def asRealTensorPtr
  : _RealTensorPointer = asFloatPtr
  // ---------------------------------------------------------------------------
  //    REAL TENSOR SWITCH HALF
  // ---------------------------------------------------------------------------
  /*
  final def asRealTensorPtr
  : _RealTensorPointer = asHalfPtr
  */
  // ---------------------------------------------------------------------------
  //    REAL TENSOR SWITCH END
  // ---------------------------------------------------------------------------

  def getSlicePtr(offset: Long, length: Long)
  : TPtr

}

abstract class DeviceBufferCompanion[T <: _DeviceBuffer[TPtr], TPtr <: Pointer, TValue] {

final def apply(device: LogicalDevice, capacity: Long)
: T = apply(device.physicalDevice, capacity)

def apply(device: PhysicalDevice, capacity: Long)
: T

}

abstract class _DeviceBufferEx[TThis <: _DeviceBufferEx[_, _], TPtr <: Pointer]
extends _DeviceBuffer[TPtr] {

def allocateSibling()
: TThis

}

abstract class DeviceBufferExCompanion[T <: _DeviceBuffer[TPtr], TPtr <: Pointer, TValue]
extends DeviceBufferCompanion[T, TPtr, TValue] {

def dataType
: Int

}
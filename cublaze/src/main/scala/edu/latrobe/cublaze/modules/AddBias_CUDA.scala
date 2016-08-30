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
import edu.latrobe.blaze.modules._
import edu.latrobe.cublaze._

abstract class AddBias_CUDA
  extends AddBias
    with MapLayer_CUDA[AddBiasBuilder] {

  final override val (bias, biasReference) = {
    val ref = builder.biasReference
    val tmp = weightBufferBuilder.get(ref)
    if (tmp.isDefined) {
      val result = tmp.get.asInstanceOf[CUDARealTensor]
      (result, None)
    }
    else {
      val result = CUDARealTensor.zeros(device, biasLayout)
      val newRef = weightBufferBuilder.register(ref, result)
      (result, Some(newRef))
    }
  }

  override protected def doClose(): Unit = {
    if (biasReference.isDefined) {
      bias.close()
    }
    super.doClose()
  }

  override def refresh(): Unit = {}


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredictPerValue(inPlaceAllowed: Boolean,
                                                 input:          Tensor)
  : CUDARealTensor = {
    // Upload weights to device.
    val out = {
      if (inPlaceAllowed) {
        input.asOrToCUDARealTensor(device)
      }
      else {
        input.toCUDARealTensor(device)
      }
    }

    // Add bias and return.
    doPredictPerValue(out)
    out
  }

  protected def doPredictPerValue(output: CUDARealTensor)
  : Unit

  final override protected def doPredictPerUnit(inPlaceAllowed: Boolean,
                                                input:          Tensor)
  : CUDARealTensor = {
    // Upload weights to device.
    val out = {
      if (inPlaceAllowed) {
        input.asOrToCUDARealTensor(device)
      }
      else {
        input.toCUDARealTensor(device)
      }
    }

    // Add bias and return.
    doPredictPerUnit(out)
    out
  }

  protected def doPredictPerUnit(output: CUDARealTensor)
  : Unit

  final override protected def doPredictPerChannel(inPlaceAllowed: Boolean,
                                                   input:          Tensor)
  : CUDARealTensor = {
    // Upload weights to device.
    val out = {
      if (inPlaceAllowed) {
        input.asOrToCUDARealTensor(device)
      }
      else {
        input.toCUDARealTensor(device)
      }
    }

    // Add bias and return.
    doPredictPerChannel(out)
    out
  }

  protected def doPredictPerChannel(output: CUDARealTensor)
  : Unit

  final override protected def doPredictPerSample(inPlaceAllowed: Boolean,
                                                  input:          Tensor)
  : CUDARealTensor = {
    // Upload weights to device.
    val out = {
      if (inPlaceAllowed) {
        input.asOrToCUDARealTensor(device)
      }
      else {
        input.toCUDARealTensor(device)
      }
    }

    // Add bias and return.
    doPredictPerSample(out)
    out
  }

  protected def doPredictPerSample(output: CUDARealTensor)
  : Unit

  final override protected def doPredictPerBatch(inPlaceAllowed: Boolean,
                                                 input:          Tensor)
  : CUDARealTensor = {
    // Upload weights to device.
    val out = {
      if (inPlaceAllowed) {
        input.asOrToCUDARealTensor(device)
      }
      else {
        input.toCUDARealTensor(device)
      }
    }

    // Add bias and return.
    doPredictPerBatch(out)
    out
  }

  protected def doPredictPerBatch(output: CUDARealTensor)
  : Unit

  final override protected def doPredictInvPerValue(output: Tensor)
  : CUDARealTensor = {
    val inp = output.toCUDARealTensor(device)
    doPredictInvPerValue(inp)
    inp
  }

  protected def doPredictInvPerValue(input: CUDARealTensor)
  : Unit

  final override protected def doPredictInvPerUnit(output: Tensor)
  : CUDARealTensor = {
    val inp = output.toCUDARealTensor(device)
    doPredictInvPerUnit(inp)
    inp
  }

  protected def doPredictInvPerUnit(input: CUDARealTensor)
  : Unit

  final override protected def doPredictInvPerChannel(output: Tensor)
  : CUDARealTensor = {
    val inp = output.toCUDARealTensor(device)
    doPredictInvPerChannel(inp)
    inp
  }

  protected def doPredictInvPerChannel(input: CUDARealTensor)
  : Unit

  final override protected def doPredictInvPerSample(output: Tensor)
  : CUDARealTensor = {
    val inp = output.toCUDARealTensor(device)
    doPredictInvPerSample(inp)
    inp
  }

  protected def doPredictInvPerSample(input: CUDARealTensor)
  : Unit

  final override protected def doPredictInvPerBatch(output: Tensor)
  : CUDARealTensor = {
    val inp = output.toCUDARealTensor(device)
    doPredictInvPerBatch(inp)
    inp
  }

  protected def doPredictInvPerBatch(input: CUDARealTensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    Backward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveWeightGradientsPerValue(error: Tensor,
                                                               sink:  ValueTensor)
  : Unit = {
    // Move sink tensor into device.
    val err = error.asOrToCUDARealTensor(device)
    val dst = sink.asOrToCUDARealTensor(device)

    doDeriveWeightGradientsPerValue(err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      dst.copyTo(sink)
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
  }

  protected def doDeriveWeightGradientsPerValue(error: CUDARealTensor,
                                                sink:  CUDARealTensor)
  : Unit

  final override protected def doDeriveWeightGradientsPerUnit(error: Tensor,
                                                              sink:  ValueTensor)
  : Unit = {
    // Move sink tensor into device.
    val err = error.asOrToCUDARealTensor(device)
    val dst = sink.asOrToCUDARealTensor(device)

    doDeriveWeightGradientsPerUnit(err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      dst.copyTo(sink)
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
  }

  protected def doDeriveWeightGradientsPerUnit(error: CUDARealTensor,
                                               sink:  CUDARealTensor)
  : Unit

  final override protected def doDeriveWeightGradientsPerChannel(error: Tensor,
                                                                 sink:  ValueTensor)
  : Unit = {
    // Move sink tensor into device.
    val err = error.asOrToCUDARealTensor(device)
    val dst = sink.asOrToCUDARealTensor(device)

    doDeriveWeightGradientsPerChannel(err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      dst.copyTo(sink)
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
  }

  protected def doDeriveWeightGradientsPerChannel(error: CUDARealTensor,
                                                  sink:  CUDARealTensor)
  : Unit

  final override protected def doDeriveWeightGradientsPerSample(error: Tensor,
                                                                sink:  ValueTensor)
  : Unit = {
    // Move sink tensor into device.
    val err = error.asOrToCUDARealTensor(device)
    val dst = sink.asOrToCUDARealTensor(device)

    doDeriveWeightGradientsPerSample(err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      dst.copyTo(sink)
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
  }

  protected def doDeriveWeightGradientsPerSample(error: CUDARealTensor,
                                                 sink:  CUDARealTensor)
  : Unit

  final override protected def doDeriveWeightGradientsPerBatch(error: Tensor,
                                                               sink:  ValueTensor)
  : Unit = {
    // Move sink tensor into device.
    val err = error.asOrToCUDARealTensor(device)
    val dst = sink.asOrToCUDARealTensor(device)

    doDeriveWeightGradientsPerBatch(err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      dst.copyTo(sink)
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
  }

  protected def doDeriveWeightGradientsPerBatch(error: CUDARealTensor,
                                                sink:  CUDARealTensor)
  : Unit

}

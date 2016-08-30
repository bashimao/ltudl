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

abstract class MultiplyFilter_CUDA
  extends MultiplyFilter
    with Layer_CUDA[MultiplyFilterBuilder] {

  final override lazy val outputPlatform
  : CUDA.type = CUDA

  final override val (filter, filterReference) = {
    val ref = builder.filterReference
    val tmp = weightBufferBuilder.get(ref)
    if (tmp.isDefined) {
      val result = tmp.get.asInstanceOf[CUDARealTensor]
      (result, None)
    }
    else {
      val result = CUDARealTensor.zeros(device, filterLayout)
      val newRef = weightBufferBuilder.register(ref, result)
      (result, Some(newRef))
    }
  }

  override protected def doClose()
  : Unit = {
    if (filterReference.isDefined) {
      filter.close()
    }
    super.doClose()
  }

  final override def refresh(): Unit = {}


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredictPerValue(input: Tensor)
  : CUDARealTensor = {
    val out = input.toCUDARealTensor(device)
    doPredictPerValue(out)
    out
  }

  protected def doPredictPerValue(output: CUDARealTensor)
  : Unit

  final override protected def doPredictPerUnit(input: Tensor)
  : CUDARealTensor = {
    val out = input.toCUDARealTensor(device)
    doPredictPerUnit(out)
    out
  }

  protected def doPredictPerUnit(output: CUDARealTensor)
  : Unit

  final override protected def doPredictPerChannel(input: Tensor)
  : CUDARealTensor = {
    val out = input.toCUDARealTensor(device)
    doPredictPerChannel(out)
    out
  }

  protected def doPredictPerChannel(output: CUDARealTensor)
  : Unit

  final override protected def doPredictPerSample(input: Tensor)
  : CUDARealTensor = {
    val out = input.toCUDARealTensor(device)
    doPredictPerSample(out)
    out
  }

  protected def doPredictPerSample(output: CUDARealTensor)
  : Unit

  final override protected def doPredictPerBatch(input: Tensor)
  : CUDARealTensor = {
    val out = input.toCUDARealTensor(device)
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
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveFilterGradientsPerValue(input: Tensor,
                                                               error: Tensor,
                                                               sink:  ValueTensor)
  : Unit = {
    // Move sink tensor into device.
    val inp = input.asOrToCUDARealTensor(device)
    val err = error.asOrToCUDARealTensor(device)
    val dst = sink.asOrToCUDARealTensor(device)

    doDeriveFilterGradientsPerValue(inp, err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      dst.copyTo(sink)
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
    if (inp ne input) {
      inp.close()
    }
  }

  protected def doDeriveFilterGradientsPerValue(input: CUDARealTensor,
                                                error: CUDARealTensor,
                                                sink:  CUDARealTensor)
  : Unit

  final override protected def doDeriveFilterGradientsPerUnit(input: Tensor,
                                                              error: Tensor,
                                                              sink:  ValueTensor)
  : Unit = {
    // Move sink tensor into device.
    val inp = input.asOrToCUDARealTensor(device)
    val err = error.asOrToCUDARealTensor(device)
    val dst = sink.asOrToCUDARealTensor(device)

    doDeriveFilterGradientsPerUnit(inp, err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      dst.copyTo(sink)
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
    if (inp ne input) {
      inp.close()
    }
  }

  protected def doDeriveFilterGradientsPerUnit(input: CUDARealTensor,
                                               error: CUDARealTensor,
                                               sink:  CUDARealTensor)
  : Unit

  final override protected def doDeriveFilterGradientsPerChannel(input: Tensor,
                                                                 error: Tensor,
                                                                 sink:  ValueTensor)
  : Unit = {
    // Move sink tensor into device.
    val inp = input.asOrToCUDARealTensor(device)
    val err = error.asOrToCUDARealTensor(device)
    val dst = sink.asOrToCUDARealTensor(device)

    doDeriveFilterGradientsPerChannel(inp, err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      dst.copyTo(sink)
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
    if (inp ne input) {
      inp.close()
    }
  }

  protected def doDeriveFilterGradientsPerChannel(input: CUDARealTensor,
                                                  error: CUDARealTensor,
                                                  sink:  CUDARealTensor)
  : Unit

  final override protected def doDeriveFilterGradientsPerSample(input: Tensor,
                                                                error: Tensor,
                                                                sink:  ValueTensor)
  : Unit = {
    // Move sink tensor into device.
    val inp = input.asOrToCUDARealTensor(device)
    val err = error.asOrToCUDARealTensor(device)
    val dst = sink.asOrToCUDARealTensor(device)

    doDeriveFilterGradientsPerSample(inp, err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      dst.copyTo(sink)
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
    if (inp ne input) {
      inp.close()
    }
  }

  protected def doDeriveFilterGradientsPerSample(input: CUDARealTensor,
                                                 error: CUDARealTensor,
                                                 sink:  CUDARealTensor)
  : Unit

  final override protected def doDeriveFilterGradientsPerBatch(input: Tensor,
                                                               error: Tensor,
                                                               sink:  ValueTensor)
  : Unit = {
    // Move sink tensor into device.
    val inp = input.asOrToCUDARealTensor(device)
    val err = error.asOrToCUDARealTensor(device)
    val dst = sink.asOrToCUDARealTensor(device)

    doDeriveFilterGradientsPerBatch(inp, err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      dst.copyTo(sink)
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
    if (inp ne input) {
      inp.close()
    }
  }

  protected def doDeriveFilterGradientsPerBatch(input: CUDARealTensor,
                                                error: CUDARealTensor,
                                                sink:  CUDARealTensor)
  : Unit

  final override protected def doDeriveInputErrorPerValue(error: Tensor)
  : CUDARealTensor = {
    val err = error.asOrToCUDARealTensor(device)
    doDeriveInputErrorPerValue(err)
    err
  }

  protected def doDeriveInputErrorPerValue(error: CUDARealTensor)
  : Unit

  final override protected def doDeriveInputErrorPerUnit(error: Tensor)
  : CUDARealTensor = {
    val err = error.asOrToCUDARealTensor(device)
    doDeriveInputErrorPerUnit(err)
    err
  }

  protected def doDeriveInputErrorPerUnit(error: CUDARealTensor)
  : Unit

  final override protected def doDeriveInputErrorPerChannel(error: Tensor)
  : CUDARealTensor = {
    val err = error.asOrToCUDARealTensor(device)
    doDeriveInputErrorPerChannel(err)
    err
  }

  protected def doDeriveInputErrorPerChannel(error: CUDARealTensor)
  : Unit

  final override protected def doDeriveInputErrorPerSample(error: Tensor)
  : CUDARealTensor = {
    val err = error.asOrToCUDARealTensor(device)
    doDeriveInputErrorPerSample(err)
    err
  }

  protected def doDeriveInputErrorPerSample(error: CUDARealTensor)
  : Unit

  final override protected def doDeriveInputErrorPerBatch(error: Tensor)
  : CUDARealTensor = {
    val err = error.asOrToCUDARealTensor(device)
    doDeriveInputErrorPerBatch(err)
    err
  }

  protected def doDeriveInputErrorPerBatch(error: CUDARealTensor)
  : Unit

}

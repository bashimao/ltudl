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

package edu.latrobe.blaze.modules.jvm

import edu.latrobe._
import edu.latrobe.blaze.modules._

abstract class MultiplyFilter_JVM
  extends MultiplyFilter {

  final override lazy val outputPlatform
  : JVM.type = JVM

  final override val (filter, filterReference) = {
    val ref = builder.filterReference
    val tmp = weightBufferBuilder.get(ref)
    if (tmp.isDefined) {
      val result = tmp.get.asInstanceOf[RealArrayTensor]
      assume(result.layout == filterLayout)
      (result, None)
    }
    else {
      val result = RealArrayTensor.zeros(filterLayout)
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
  : RealArrayTensor = {
    val out = input.toRealArrayTensor
    doPredictPerValue(out)
    out
  }

  protected def doPredictPerValue(output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerUnit(input: Tensor)
  : RealArrayTensor = {
    val out = input.toRealArrayTensor
    doPredictPerUnit(out)
    out
  }

  protected def doPredictPerUnit(output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerChannel(input: Tensor)
  : RealArrayTensor = {
    val out = input.toRealArrayTensor
    doPredictPerChannel(out)
    out
  }

  protected def doPredictPerChannel(output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerSample(input: Tensor)
  : RealArrayTensor = {
    val out = input.toRealArrayTensor
    doPredictPerSample(out)
    out
  }

  protected def doPredictPerSample(output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerBatch(input: Tensor)
  : RealArrayTensor =  {
    val out = input.toRealArrayTensor
    doPredictPerBatch(out)
    out
  }

  protected def doPredictPerBatch(output: RealArrayTensor)
  : Unit

  final override protected def doPredictInvPerValue(output: Tensor)
  : RealArrayTensor = {
    val inp = output.toRealArrayTensor
    doPredictInvPerValue(inp)
    inp
  }

  protected def doPredictInvPerValue(input: RealArrayTensor)
  : Unit

  final override protected def doPredictInvPerUnit(output: Tensor)
  : RealArrayTensor = {
    val inp = output.toRealArrayTensor
    doPredictInvPerUnit(inp)
    inp
  }

  protected def doPredictInvPerUnit(input: RealArrayTensor)
  : Unit

  final override protected def doPredictInvPerChannel(output: Tensor)
  : RealArrayTensor = {
    val inp = output.toRealArrayTensor
    doPredictInvPerChannel(inp)
    inp
  }

  protected def doPredictInvPerChannel(input: RealArrayTensor)
  : Unit

  final override protected def doPredictInvPerSample(output: Tensor)
  : RealArrayTensor = {
    val inp = output.toRealArrayTensor
    doPredictInvPerSample(inp)
    inp
  }

  protected def doPredictInvPerSample(input: RealArrayTensor)
  : Unit

  final override protected def doPredictInvPerBatch(output: Tensor)
  : RealArrayTensor = {
    val inp = output.toRealArrayTensor
    doPredictInvPerBatch(inp)
    inp
  }

  protected def doPredictInvPerBatch(input: RealArrayTensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveFilterGradientsPerValue(input: Tensor,
                                                               error: Tensor,
                                                               sink:  ValueTensor)
  : Unit = {
    val inp = input.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor
    val dst = sink.asOrToRealArrayTensor

    doDeriveFilterGradientsPerValue(inp, err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      sink := dst
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
    if (inp ne input) {
      inp.close()
    }
  }

  protected def doDeriveFilterGradientsPerValue(input: RealArrayTensor,
                                                error: RealArrayTensor,
                                                sink:  RealArrayTensor)
  : Unit

  final override protected def doDeriveFilterGradientsPerUnit(input: Tensor,
                                                              error: Tensor,
                                                              sink:  ValueTensor)
  : Unit = {
    val inp = input.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor
    val dst = sink.asOrToRealArrayTensor

    doDeriveFilterGradientsPerUnit(inp, err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      sink := dst
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
    if (inp ne input) {
      inp.close()
    }
  }

  protected def doDeriveFilterGradientsPerUnit(input: RealArrayTensor,
                                               error: RealArrayTensor,
                                               sink:  RealArrayTensor)
  : Unit

  final override protected def doDeriveFilterGradientsPerChannel(input: Tensor,
                                                                 error: Tensor,
                                                                 sink:  ValueTensor)
  : Unit = {
    val inp = input.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor
    val dst = sink.asOrToRealArrayTensor

    doDeriveFilterGradientsPerChannel(inp, err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      sink := dst
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
    if (inp ne input) {
      inp.close()
    }
  }

  protected def doDeriveFilterGradientsPerChannel(input: RealArrayTensor,
                                                  error: RealArrayTensor,
                                                  sink:  RealArrayTensor)
  : Unit

  final override protected def doDeriveFilterGradientsPerSample(input: Tensor,
                                                                error: Tensor,
                                                                sink:  ValueTensor)
  : Unit = {
    val inp = input.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor
    val dst = sink.asOrToRealArrayTensor

    doDeriveFilterGradientsPerSample(inp, err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      sink := dst
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
    if (inp ne input) {
      inp.close()
    }
  }

  protected def doDeriveFilterGradientsPerSample(input: RealArrayTensor,
                                                 error: RealArrayTensor,
                                                 sink:  RealArrayTensor)
  : Unit

  final override protected def doDeriveFilterGradientsPerBatch(input: Tensor,
                                                               error: Tensor,
                                                               sink:  ValueTensor)
  : Unit = {
    val inp = input.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor
    val dst = sink.asOrToRealArrayTensor

    doDeriveFilterGradientsPerBatch(inp, err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      sink := dst
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
    if (inp ne input) {
      inp.close()
    }
  }

  protected def doDeriveFilterGradientsPerBatch(input: RealArrayTensor,
                                                error: RealArrayTensor,
                                                sink:  RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorPerValue(error: Tensor)
  : RealArrayTensor = {
    val err = error.asOrToRealArrayTensor
    doDeriveInputErrorPerValue(err)
    err
  }

  protected def doDeriveInputErrorPerValue(error: RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorPerUnit(error: Tensor)
  : RealArrayTensor = {
    val err = error.asOrToRealArrayTensor
    doDeriveInputErrorPerUnit(err)
    err
  }

  protected def doDeriveInputErrorPerUnit(error: RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorPerChannel(error: Tensor)
  : RealArrayTensor = {
    val err = error.asOrToRealArrayTensor
    doDeriveInputErrorPerChannel(err)
    err
  }

  protected def doDeriveInputErrorPerChannel(error: RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorPerSample(error: Tensor)
  : RealArrayTensor = {
    val err = error.asOrToRealArrayTensor
    doDeriveInputErrorPerSample(err)
    err
  }

  protected def doDeriveInputErrorPerSample(error: RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorPerBatch(error: Tensor)
  : RealArrayTensor = {
    val err = error.asOrToRealArrayTensor
    doDeriveInputErrorPerBatch(err)
    err
  }

  protected def doDeriveInputErrorPerBatch(error: RealArrayTensor)
  : Unit

}

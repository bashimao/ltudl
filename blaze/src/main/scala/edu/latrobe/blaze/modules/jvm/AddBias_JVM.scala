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
import edu.latrobe.blaze.{Mode, PredictContext}
import edu.latrobe.blaze.modules._

abstract class AddBias_JVM
  extends AddBias
    with MapLayer_JVM[AddBiasBuilder] {

  final override val (bias, biasReference) = {
    val ref = builder.biasReference
    val tmp = weightBufferBuilder.get(ref)
    if (tmp.isDefined) {
      val result = tmp.get.asInstanceOf[RealArrayTensor]
      assume(result.layout == biasLayout)
      (result, None)
    }
    else {
      val result = RealArrayTensor.zeros(biasLayout)
      val newRef = weightBufferBuilder.register(ref, result)
      (result, Some(newRef))
    }
  }

  override protected def doClose()
  : Unit = {
    if (biasReference.isDefined) {
      bias.close()
    }
    super.doClose()
  }

  final override def refresh(): Unit = {}


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredictPerValue(inPlaceAllowed: Boolean,
                                                 input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictPerValue(out)
    out
  }

  protected def doPredictPerValue(output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerUnit(inPlaceAllowed: Boolean,
                                                input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictPerUnit(out)
    out
  }

  protected def doPredictPerUnit(output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerChannel(inPlaceAllowed: Boolean,
                                                   input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictPerChannel(out)
    out
  }

  protected def doPredictPerChannel(output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerSample(inPlaceAllowed: Boolean,
                                                  input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
    doPredictPerSample(out)
    out
  }

  protected def doPredictPerSample(output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerBatch(inPlaceAllowed: Boolean,
                                                 input:          Tensor)
  : RealArrayTensor = {
    val out = {
      if (inPlaceAllowed) {
        input.asOrToRealArrayTensor
      }
      else {
        input.toRealArrayTensor
      }
    }
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
  : Tensor = {
    val inp = output.toRealArrayTensor
    doPredictInvPerChannel(inp)
    inp
  }

  protected def doPredictInvPerChannel(input: RealArrayTensor)
  : Unit

  final override protected def doPredictInvPerSample(output: Tensor)
  : Tensor = {
    val inp = output.toRealArrayTensor
    doPredictInvPerSample(inp)
    inp
  }

  protected def doPredictInvPerSample(input: RealArrayTensor)
  : Unit

  final override protected def doPredictInvPerBatch(output: Tensor)
  : Tensor = {
    val inp = output.toRealArrayTensor
    doPredictInvPerBatch(inp)
    inp
  }

  protected def doPredictInvPerBatch(input: RealArrayTensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveWeightGradientsPerValue(error: Tensor,
                                                               sink:  ValueTensor)
  : Unit = {
    val err = error.asOrToRealArrayTensor
    val dst = sink.asOrToRealArrayTensor

    doDeriveWeightGradientsPerValue(err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      sink := dst
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
  }

  protected def doDeriveWeightGradientsPerValue(error: RealArrayTensor,
                                                sink:  RealArrayTensor)
  : Unit

  final override protected def doDeriveWeightGradientsPerUnit(error: Tensor,
                                                              sink:  ValueTensor)
  : Unit = {
    val err = error.asOrToRealArrayTensor
    val dst = sink.asOrToRealArrayTensor

    doDeriveWeightGradientsPerUnit(err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      sink := dst
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
  }

  protected def doDeriveWeightGradientsPerUnit(error: RealArrayTensor,
                                               sink:  RealArrayTensor)
  : Unit

  final override protected def doDeriveWeightGradientsPerChannel(error: Tensor,
                                                                 sink:  ValueTensor)
  : Unit = {
    val err = error.asOrToRealArrayTensor
    val dst = sink.asOrToRealArrayTensor

    doDeriveWeightGradientsPerChannel(err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      sink := dst
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
  }

  protected def doDeriveWeightGradientsPerChannel(error: RealArrayTensor,
                                                  sink:  RealArrayTensor)
  : Unit

  final override protected def doDeriveWeightGradientsPerSample(error: Tensor,
                                                                sink:  ValueTensor)
  : Unit = {
    val err = error.asOrToRealArrayTensor
    val dst = sink.asOrToRealArrayTensor

    doDeriveWeightGradientsPerSample(err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      sink := dst
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
  }

  protected def doDeriveWeightGradientsPerSample(error: RealArrayTensor,
                                                 sink:  RealArrayTensor)
  : Unit

  final override protected def doDeriveWeightGradientsPerBatch(error: Tensor,
                                                               sink:  ValueTensor)
  : Unit = {
    val err = error.asOrToRealArrayTensor
    val dst = sink.asOrToRealArrayTensor

    doDeriveWeightGradientsPerBatch(err, dst)

    // Deallocate temporaries.
    if (dst ne sink) {
      sink := dst
      dst.close()
    }
    if (err ne error) {
      err.close()
    }
  }

  protected def doDeriveWeightGradientsPerBatch(error: RealArrayTensor,
                                                sink:  RealArrayTensor)
  : Unit

}

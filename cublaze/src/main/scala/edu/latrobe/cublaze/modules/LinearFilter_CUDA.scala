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
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.cublaze._

abstract class LinearFilter_CUDA
  extends LinearFilter
    with Layer_CUDA[LinearFilterBuilder] {

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

  override def refresh(): Unit = {}


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(input: Tensor)
  : CUDARealTensor = {
    val inp       = input.asOrToCUDARealTensor(device)
    val inpLayout = inp.layout
    val outLayout = inpLayout.derive(outputSize)
    val out       = CUDARealTensor(device, outLayout)

    doPredict(inp, out)

    // Deallocate tensors if necessary.
    if (inp ne input) {
      inp.close()
    }
    out
  }

  protected def doPredict(input:  CUDARealTensor,
                          output: CUDARealTensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveFilterGradients(input: Tensor,
                                                       error: Tensor,
                                                       sink:  Tensor)
  : Unit = {
    // Upload data onto device.
    val inp = input.asOrToCUDARealTensor(device)
    val err = error.asOrToCUDARealTensor(device)
    val dst = sink.asOrToCUDARealTensor(device)

    doDeriveFilterGradients(inp, err, dst)

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

  protected def doDeriveFilterGradients(input: CUDARealTensor,
                                        error: CUDARealTensor,
                                        sink:  CUDARealTensor)
  : Unit

  final override protected def doDeriveInputError(error: Tensor)
  : CUDARealTensor = {
    val oldErr       = error.asOrToCUDARealTensor(device)
    val oldErrLayout = oldErr.layout
    val newErrLayout = oldErrLayout.derive(inputSizeHint)
    val newErr       = CUDARealTensor(device, newErrLayout)

    doDeriveInputError(oldErr, newErr)

    // Deallocate temporary memory.
    if (oldErr ne error) {
      oldErr.close()
    }
    newErr
  }

  protected def doDeriveInputError(oldError: CUDARealTensor,
                                   newError: CUDARealTensor)
  : Unit

}

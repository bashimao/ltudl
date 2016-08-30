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

abstract class LinearFilter_JVM
  extends LinearFilter {

  final override lazy val outputPlatform
  : JVM.type = JVM

  final override val (filter, filterReference) = {
    val ref = builder.filterReference
    val tmp = weightBufferBuilder.get(ref)
    if (tmp.isDefined) {
      val result = tmp.get.asInstanceOf[RealArrayTensor]
      (result, None)
    }
    else {
      val result = RealArrayTensor.zeros(filterLayout)
      val newRef = weightBufferBuilder.register(ref, result)
      (result, Some(newRef))
    }
  }

  final protected val w = filter.valuesMatrix

  final protected val w_t = w.t


  // ---------------------------------------------------------------------------
  //    Weights related
  // ---------------------------------------------------------------------------
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
  : RealArrayTensor = {
    val inp       = input.asOrToRealArrayTensor
    val inpLayout = inp.layout
    val outLayout = inpLayout.derive(outputSize)
    val out       = RealArrayTensor.zeros(outLayout)
    doPredict(inp, out)

    // Deallocate tensors if necessary.
    if (inp ne input ) {
      inp.close()
    }
    out
  }

  protected def doPredict(input:  RealArrayTensor,
                          output: RealArrayTensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveFilterGradients(input: Tensor,
                                                 error: Tensor,
                                                 sink:  Tensor)
  : Unit = {
    val inp = input.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor
    val dst = sink.asOrToRealArrayTensor

    doDeriveFilterGradients(inp, err, dst)

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

  protected def doDeriveFilterGradients(input: RealArrayTensor,
                                        error: RealArrayTensor,
                                        sink:  RealArrayTensor)
  : Unit

  final override protected def doDeriveInputError(error: Tensor)
  : RealArrayTensor = {
    val oldErr       = error.asOrToRealArrayTensor
    val oldErrLayout = oldErr.layout
    val newErrLayout = oldErrLayout.derive(inputSizeHint)
    val newErr       = RealArrayTensor.zeros(newErrLayout)
    doDeriveInputError(oldErr, newErr)

    // Deallocate temporary memory.
    if (oldErr ne error) {
      oldErr.close()
    }
    newErr
  }

  protected def doDeriveInputError(oldError: RealArrayTensor,
                                   newError: RealArrayTensor)
  : Unit

}

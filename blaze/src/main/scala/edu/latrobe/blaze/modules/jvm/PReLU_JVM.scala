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
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.{PReLU, PReLUBuilder}

abstract class PReLU_JVM
  extends PReLU
    with MapLayer_JVM[PReLUBuilder] {

  final override val (pReLU, pReLUReference) = {
    val ref = builder.pReLUReference
    val tmp = weightBufferBuilder.get(ref)
    if (tmp.isDefined) {
      val result = tmp.get.asInstanceOf[RealArrayTensor]
      (result, None)
    }
    else {
      val result = RealArrayTensor.zeros(pReLULayout)
      val newRef = weightBufferBuilder.register(ref, result)
      (result, Some(newRef))
    }
  }

  override protected def doClose()
  : Unit = {
    if (pReLUReference.isDefined) {
      pReLU.close()
    }
    super.doClose()
  }

  final override def refresh(): Unit = {}


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(input: Tensor)
  : RealArrayTensor = {
    val out = input.toRealArrayTensor
    doPredict(out)
    out
  }

  protected def doPredict(output: RealArrayTensor): Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDerivePReLUGradients(input: Tensor,
                                                      error: Tensor,
                                                      sink:  ValueTensor)
  : Unit = {
    val inp = input.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor
    val dst = sink.asOrToRealArrayTensor

    doDerivePReLUGradients(inp, err, dst)

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

  protected def doDerivePReLUGradients(input: RealArrayTensor,
                                       error: RealArrayTensor,
                                       sink:  RealArrayTensor)
  : Unit

  final override protected def doDeriveInputError(input: Tensor, error: Tensor)
  : Tensor = {
    val inp = input.asOrToRealArrayTensor
    val err = error.asOrToRealArrayTensor

    doDeriveInputError(inp, err)

    // Deallocate temporaries.
    if (inp ne input) {
      inp.close()
    }
    err
  }

  protected def doDeriveInputError(input: RealArrayTensor,
                                   error: RealArrayTensor)
  : Unit

}

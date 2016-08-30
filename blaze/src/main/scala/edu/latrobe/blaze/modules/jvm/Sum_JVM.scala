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
import edu.latrobe.sizes._

abstract class Sum_JVM
  extends modules.Sum {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredictPerUnit(input: Tensor)
  : Tensor = {
    val inp = input.asOrToRealArrayTensor
    val out = RealArrayTensor.zeros(inp.layout.derive(1))
    doPredictPerUnit(inp, out)
    out
  }

  protected def doPredictPerUnit(input:  RealArrayTensor,
                                 output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerChannel(input: Tensor)
  : Tensor = {
    val inp = input.asOrToRealArrayTensor
    val out = RealArrayTensor.zeros(IndependentTensorLayout.derive(inp.layout.size.noChannels))
    doPredictPerChannel(inp, out)
    out
  }

  protected def doPredictPerChannel(input:  RealArrayTensor,
                                    output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerSample(input: Tensor)
  : Tensor = {
    val inp = input.asOrToRealArrayTensor
    val out = RealArrayTensor.zeros(inp.layout.derive(Size1.one))
    doPredictPerSample(inp, out)
    out
  }

  protected def doPredictPerSample(input:  RealArrayTensor,
                                   output: RealArrayTensor)
  : Unit

  final override protected def doPredictPerBatch(input: Tensor)
  : Tensor = {
    val inp = input.asOrToRealArrayTensor
    val out = RealArrayTensor.zeros(IndependentTensorLayout.one)
    doPredictPerBatch(inp, out)
    out
  }

  protected def doPredictPerBatch(input:  RealArrayTensor,
                                  output: RealArrayTensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveInputErrorPerUnit(inputLayout: TensorLayout,
                                                         error:       Tensor)
  : Tensor = {
    val oldErr = error.asOrToRealArrayTensor
    val newErr = RealArrayTensor.zeros(inputLayout.makeIndependent)
    doDeriveInputErrorPerUnit(oldErr, newErr)
    newErr
  }

  protected def doDeriveInputErrorPerUnit(oldError: RealArrayTensor,
                                          newError: RealArrayTensor)
  : Unit

  final override protected def ddoDeriveInputErrorPerChannel(inputLayout: TensorLayout,
                                                             error:       Tensor)
  : Tensor = {
    val oldErr = error.asOrToRealArrayTensor
    val newErr = RealArrayTensor.zeros(inputLayout.makeIndependent)
    ddoDeriveInputErrorPerChannel(oldErr, newErr)
    newErr
  }

  protected def ddoDeriveInputErrorPerChannel(oldError: RealArrayTensor,
                                              newError: RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorPerSample(inputLayout: TensorLayout,
                                                           error:       Tensor)
  : Tensor = {
    val oldErr = error.asOrToRealArrayTensor
    val newErr = RealArrayTensor.zeros(inputLayout.makeIndependent)
    doDeriveInputErrorPerSample(oldErr, newErr)
    newErr
  }

  protected def doDeriveInputErrorPerSample(oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit

  final override protected def doDeriveInputErrorPerBatch(inputLayout: TensorLayout,
                                                          error:       Tensor)
  : Tensor = {
    val oldErr = error.asOrToRealArrayTensor
    val newErr = RealArrayTensor.zeros(inputLayout.makeIndependent)
    doDeriveInputErrorPerBatch(oldErr, newErr)
    newErr
  }

  protected def doDeriveInputErrorPerBatch(oldError: RealArrayTensor,
                                           newError: RealArrayTensor)
  : Unit

}

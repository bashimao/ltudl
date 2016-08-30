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

package edu.latrobe.blaze.modules.generic

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._

final class Square_Generic_Baseline(override val builder:        SquareBuilder,
                                    override val inputHints:     BuildHints,
                                    override val seed:           InstanceSeed,
                                    override val weightBufferBuilder: ValueTensorBufferBuilder)
    extends Square_Generic {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredictEx(output: Tensor)
  : Unit = output.sqr()

  override protected def doPredictInvEx(input: Tensor)
  : Unit = input.sqrt()


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputErrorEx(input: Tensor,
                                              error: Tensor)
  : Unit = error.multiply(input, Real.two)

}

object Square_Generic_Baseline_Description
  extends GenericModuleVariantDescription[SquareBuilder] {

  override def build(builder:        SquareBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Square_Generic_Baseline = new Square_Generic_Baseline(
    builder, hints, seed, weightsBuilder
  )

}

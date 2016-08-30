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

final class Dropout_Generic_Baseline(override val builder:        DropoutBuilder,
                                     override val inputHints:     BuildHints,
                                     override val seed:           InstanceSeed,
                                     override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Dropout_Generic {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredictForTraining(output: Tensor,
                                              rng:    PseudoRNG)
  : PredictContext = {
    // Build a dropout mask.
    val bernoulli = rng.bernoulliDistribution(
      probability,
      Real.zero,
      if (useOriginalAlgorithm) Real.one else boostFactor
    )
    val mask = output.createSibling()
    mask.fill(bernoulli.sample, threadSafe = false)

    // Apply it.
    output :*= mask

    Dropout_Generic_Baseline_Context(mask)
  }

  override protected def doPredictForInference(output: Tensor)
  : Unit = {
    if (useOriginalAlgorithm) {
      output *= probabilityInv
    }
    else {
      // Do nothing.
    }
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(context: PredictContext,
                                            error:   Tensor)
  : Tensor = context match {
    case Dropout_Generic_Baseline_Context(mask) =>
      error :*= mask
      error
    case _ =>
      throw new MatchError(context)
  }

  /*
  override protected def doDeriveInputErrorForInference(error: Tensor)
  : Tensor = {
    if (useOriginalAlgorithm) {
      error *= probabilityInv
    }
    else {
      // Do nothing.
    }
    error
  }
  */

}

final case class Dropout_Generic_Baseline_Context(mask: Tensor)
  extends PredictContext {

  override protected def doClose()
  : Unit = {
    mask.close()
    super.doClose()
  }

}

object Dropout_Generic_Baseline_Description
  extends GenericModuleVariantDescription[DropoutBuilder] {

  override def build(builder:        DropoutBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Dropout_Generic_Baseline = new Dropout_Generic_Baseline(
    builder, hints, seed, weightsBuilder
  )

}

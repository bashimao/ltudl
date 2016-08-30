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

import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._

object BatchNormalization_Generic_Baseline_Description
  extends GenericModuleVariantDescriptionEx[BatchNormalizationBuilder] {

  override def generateBuilder(builder: BatchNormalizationBuilder,
                               hints:   BuildHints)
  : ModuleBuilder = {
    val a = {
      val tmp = NormalizationBuilder()
      builder.copyTo(tmp)
      tmp.domain                    = TensorDomain.Channel
      tmp.learningRate             = builder.learningRate
      tmp.epsilon                  = builder.epsilon
      tmp.runningMeanReference     = builder.runningMeanReference
      tmp.runningVarianceReference = builder.runningVarianceReference
      tmp
    }
    val b = {
      val tmp = MultiplyFilterBuilder()
      builder.copyTo(tmp)
      tmp.domain           = TensorDomain.Channel
      tmp.filterReference = builder.filterReference
      tmp
    }
    val c = {
      val tmp = AddBiasBuilder()
      builder.copyTo(tmp)
      tmp.domain         = TensorDomain.Channel
      tmp.biasReference = builder.biasReference
      tmp
    }
    SequenceBuilder(a, b, c)
  }

}

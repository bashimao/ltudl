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

object Normalization_Generic_Baseline_Description
  extends GenericModuleVariantDescriptionEx[NormalizationBuilder] {

  override def generateBuilder(builder: NormalizationBuilder, hints: BuildHints)
  : ModuleBuilder = {
    val a = {
      val tmp = ZeroMeanBuilder()
      builder.copyTo(tmp)
      tmp.domain                = builder.domain
      tmp.learningRate         = builder.learningRate
      tmp.runningMeanReference = builder.runningMeanReference
      tmp
    }
    val b = {
      val tmp = UnitVarianceBuilder()
      builder.copyTo(tmp)
      tmp.domain                     = builder.domain
      tmp.learningRate              = builder.learningRate
      tmp.epsilon                   = builder.epsilon
      tmp.runningVarianceDevReference = builder.runningVarianceReference
      tmp
    }
    SequenceBuilder(
      a,
      b
    )
  }

}

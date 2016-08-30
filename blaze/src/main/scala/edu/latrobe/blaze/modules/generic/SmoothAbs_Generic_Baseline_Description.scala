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

object SmoothAbs_Generic_Baseline_Description
  extends GenericModuleVariantDescriptionEx[SmoothAbsBuilder] {

  override def generateBuilder(builder: SmoothAbsBuilder, hints: BuildHints)
  : SequenceBuilder = {
    val a = {
      val tmp = SquareBuilder()
      builder.copyTo(tmp)
      tmp
    }
    val b = {
      val tmp = SquareRootBuilder(builder.epsilon)
      builder.copyTo(tmp)
      tmp
    }
    SequenceBuilder(a, b, ChangePlatformPreferenceBuilder(hints.platform))
  }

}

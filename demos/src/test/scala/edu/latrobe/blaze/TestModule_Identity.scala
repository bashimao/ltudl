/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2015 Matthias Langer (t3l@threelights.de)
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

package edu.latrobe.blaze

import edu.latrobe.RealArrayTensor
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._
import org.scalatest._

final class TestModule_Identity extends FlatSpec with Matchers {

  "Identity" should "behave as exactly as defined" in {
    val x = RealArrayTensor(Size1(2, 1), 2, Array(0.0f, 1.0f, -1.0f, 15.0f))
    val y = x.copy
    val e = RealArrayTensor(Size1(2, 1), 2, Array(1.0f, 1.0f,  1.0f,  1.0f))
    val d = e.copy

    // Should change nothing.
    val layer = IdentityBuilder().build(BuildHints.derive(x))
    val prediction = layer.predict(Training(0L), x, null)
    y shouldEqual prediction.output
    val gHat = layer.weightBuffer.allocateZeroedSibling()
    val dHat = layer.deriveGradients(prediction, e, gHat).compute()
    d shouldEqual dHat
  }

}

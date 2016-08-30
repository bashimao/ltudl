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

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._
import org.scalatest._
import TestUtils._
import edu.latrobe.blaze.modules.jvm.HardTanh_JVM_Baseline_Description

final class TestModule_HardTanh extends FlatSpec with Matchers {

  "All HardTanh variants" should "behave as exactly as defined" in {
    val x = RealArrayTensor(Size1(2, 1), 4, Array(0.0f, -0.0f, 0.5f, -0.5f, -1.0f, 1.0f, -15.0f, 15.0f))
    val y = RealArrayTensor(Size1(2, 1), 4, ArrayEx.map(x.values)(RealRange.minusOneToOne.clip))
    val d = RealArrayTensor(Size1(2, 1), 4, ArrayEx.map(x.values)(x => if (x > -1.0f && x < 1.0f) 3.5f else 0.0f))
    val e = RealArrayTensor(Size1(2, 1), 4, ArrayEx.fill(2 * 4, 3.5f))

    // Isolate Baseline variant.
    if (true) {
      HardTanhBuilder.unregisterAll()
      HardTanhBuilder.register(10, HardTanh_JVM_Baseline_Description)
      val layer = HardTanhBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
    }
  }

}

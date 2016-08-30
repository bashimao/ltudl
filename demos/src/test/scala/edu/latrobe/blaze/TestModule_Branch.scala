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

package edu.latrobe.blaze

import edu.latrobe._
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._
import org.scalatest.{FlatSpec, Matchers}

class TestModule_Branch extends FlatSpec with Matchers {

  "Branch" should "cancel the input" in {

    val x = RealArrayTensor(Size1(1, 2), Array(2.1f, -2.2f))
    val y = RealArrayTensor(Size1(1, 2), Array(1.0f, 1.0f))

    val mb = SequenceBuilder(
      BranchBuilder(
        IdentityBuilder(),
        MultiplyValuesBuilder(TensorDomain.Batch, -Real.one)
      ),
      MergeBuilder()
    )

    val m = mb.build(BuildHints.derive(x))

    val ctx = m.predict(Training(0L), x, y)

    val out = ctx.output

    all(out.values) should be(0.0f)
  }

}

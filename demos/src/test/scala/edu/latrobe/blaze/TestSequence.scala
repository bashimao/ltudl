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

import breeze.linalg.{*, DenseMatrix, DenseVector}
import edu.latrobe._
import edu.latrobe.blaze.gradientchecks._
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._
import org.scalatest._
import TestUtils._

final class TestSequence extends FlatSpec with Matchers {

  "FullyConnected + MSE" should "pass gradient checking" in {
    val x = RealArrayTensor(Size1(1, 3), 3, Array(
      1.0f, 2.0f, 3.0f, // sample 0
      4.0f, 5.0f, 6.0f, // sample 1
      8.0f, 9.0f, 0.0f  // sample 2
    ))
    val b = DenseVector(
      -10.0f, // neuron 0
      -20.0f  // neuron 1
    )
    val w = new DenseMatrix(3, 2, Array(
      0.1f, 0.2f, 0.3f, // neuron 0
      0.4f, 0.5f, 0.6f  // neuron 1
    ))
    val y = RealArrayTensor.derive(Size1(1, 2), {
      val tmp: DenseMatrix[Real] = w.t * x.valuesMatrix
      tmp(::, *) += b
      tmp
    })
    val r = RealArrayTensor.ones(Size1(1, 2), 3)

    // Build sequence and gradient checker.
    val seq = SequenceBuilder(
      LinearFilterBuilder(2),
      AddBiasBuilder(),
      SquaredErrorConstraintBuilder()
    ).build(BuildHints.derive(x))
    MatrixEx.copy(seq.weightBuffer(0, 0).values,            w)
    VectorEx.copy(seq.weightBuffer(0, 0).values, w.size, 1, b)

    val gc = ProbeEntireBufferBuilder().build()

    // Run tests.
    val prediction = seq.predict(Training(0L), x, r)
    similarity(y, prediction.output) should be < tolerance0
    val res = gc(0L, seq, x, r)
    res.rating should be < tolerance1
  }

}

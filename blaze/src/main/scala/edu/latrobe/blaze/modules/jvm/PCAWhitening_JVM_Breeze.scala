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

import breeze.linalg.{*, DenseMatrix, svd}
import breeze.stats.mean
import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.PCAWhiteningBuilder

final class PCAWhitening_JVM_Breeze(override val builder:        PCAWhiteningBuilder,
                                    override val inputHints:     BuildHints,
                                    override val seed:           InstanceSeed,
                                    override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends PCAWhitening_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(output: RealArrayTensor)
  : Unit = {
    val out = output.valuesMatrix

    // 1. Subtract mean.
    // TODO: Update once we have newer breeze support.
    // val mu = mean(out(::, *)).t
    val mu = mean(out(::, *)).toDenseVector
    out(*, ::) -= mu

    // 2. Compute PCA transformation.
    val scaleMatrix: DenseMatrix[Real] = {
      // Compute singular values.
      val (u, s) = {
        val tmp = svd(out * out.t)
        (tmp.U, tmp.∑)
      }

      // Compute transform matrix.
      VectorEx.transform(
        s
      )(x => Real(1.0 / Math.sqrt(x + epsilon)))
      s * u.t
    }

    // Compute PCA whitened values.
    // TODO: Inefficient, check this operation.
    out := scaleMatrix * out
  }

}

object PCAWhitening_JVM_Breeze_Description
  extends ModuleVariant_JVM_Description[PCAWhiteningBuilder] {

  final override def build(builder:        PCAWhiteningBuilder,
                           hints:          BuildHints,
                           seed:           InstanceSeed,
                           weightsBuilder: ValueTensorBufferBuilder)
  : PCAWhitening_JVM_Breeze = new PCAWhitening_JVM_Breeze(
    builder, hints, seed, weightsBuilder
  )

}

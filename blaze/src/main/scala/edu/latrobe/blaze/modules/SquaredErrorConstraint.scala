/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._

/**
 * Sum of squared errors.
 *
 *             m       n
 *            ---     ---                 2
 *         1  \    c  \   ( yHat   - y   )
 * cost = --- /   --- /   (     ij    ij )
 *         m  ---  n  ---
 *            j=0     i=0
 *
 *                                 2 c (              )
 * gradient(w. respect to input) = --- ( yHat   - y   )
 *                                  n  (     ij    ij )
 *
 *
 * Will take a shortcut if scaling by n is disabled and c = 0.5.
 *
 * gradient = yHat  - y
 *                ij   ij
 *
 */
final class SquaredErrorConstraint(override val builder:        SquaredErrorConstraintBuilder,
                                   override val inputHints:     BuildHints,
                                   override val seed:           InstanceSeed,
                                   override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ConstraintEx[SquaredErrorConstraintBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doEvaluate(reference:   Tensor,
                                    output:      Tensor,
                                    scaleFactor: Real)
  : Real = {
    val y    = reference.valuesMatrixEx
    val yHat = output.valuesMatrix

    val sum = MatrixEx.foldLeftEx(0.0, yHat, y)(
      (sum, yHat, y) => {
        val tmp: Double = yHat - y
        sum + tmp * tmp
      },
      (sum, yHat) => {
        val tmp: Double = yHat
        sum + tmp * tmp
      }
    )

    Real(sum * scaleFactor)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(reference:   Tensor,
                                            output:      Tensor,
                                            scaleFactor: Real,
                                            error:       Tensor)
  : Tensor = {
    val scaleFactor2 = scaleFactor + scaleFactor
    error.add(output,    +scaleFactor2)
    error.add(reference, -scaleFactor2)
    error
  }

}

final class SquaredErrorConstraintBuilder
  extends ConstraintExBuilder[SquaredErrorConstraintBuilder] {

  override def repr
  : SquaredErrorConstraintBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SquaredErrorConstraintBuilder]

  override protected def doCopy()
  : SquaredErrorConstraintBuilder = SquaredErrorConstraintBuilder()


  // ---------------------------------------------------------------------------
  //    Weights buffer handling related.
  // ---------------------------------------------------------------------------
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : SquaredErrorConstraint = new SquaredErrorConstraint(
    this, hints, seed, weightsBuilder
  )

}

object SquaredErrorConstraintBuilder {

  final def apply()
  : SquaredErrorConstraintBuilder = new SquaredErrorConstraintBuilder

  final def apply(domain: TensorDomain, scaleCoefficient: Real)
  : SquaredErrorConstraintBuilder = apply().setDomain(
    domain
  ).setScaleCoefficient(scaleCoefficient)

}
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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
 * Some equations taken from UFLDL, others from Wikipedia.
 *
 * From UFLDL:
 * Computes the Kullback-Leibler (KL) divergence between a Bernoulli random
 * variable with mean rho, and a Bernoulli  and random variable with mean
 * rhoHat_i.
 *
 * Also see:
 * https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
 *
 *
 *               ---        ( rho  )               ( 1 - rho  )
 *               \    rho ln( ---  ) + (1 - rho) ln( -------  )
 * cost = beta * /          (  ^^  )               (      ^^  )
 *               ---        ( rho  )               ( 1 - rho  )
 *                i         (    j )               (        j )
 *
 *                                     ( ( 1 - rho  )    rho
 *         (             ^^ )          ( ( -------- ) - ----
 * gradient(w. resp. to rho ) = beta * ( (      ^^  )    ^^
 *         (               i)          ( ( 1 - rho  )    rho
 *                                     ( (        i )       i
 *
 * Remark: Please note that our current implementation allows specifying
 *         different values for rho (since you can inject any kind of tensor)
 *         and also makes sure we do not flood buffers with NaN in numerically
 *         critical situations.
 */
// TODO: This is not the fastest way to do this. Was faster before. Add back variant that takes a single value rho.
final class BinaryKLDivergenceConstraint(override val builder:        BinaryKLDivergenceConstraintBuilder,
                                         override val inputHints:     BuildHints,
                                         override val seed:           InstanceSeed,
                                         override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ConstraintEx[BinaryKLDivergenceConstraintBuilder] {

  val epsilon
  : Double = builder.epsilon


  // ---------------------------------------------------------------------------
  //    Cost/Gradient computation related.
  // ---------------------------------------------------------------------------
  override protected def doEvaluate(reference:   Tensor,
                                    output:      Tensor,
                                    scaleFactor: Real)
  : Real = {
    val p    = reference.valuesMatrixEx
    val pHat = output.valuesMatrix

    val sum = MatrixEx.foldLeftEx(0.0, pHat, p)(
      (sum, pHat, p) => {
        val pDiff     = 1.0 - p
        val pHatDiff  = 1.0 - pHat
        val a: Double = if (pHatDiff > epsilon) pHatDiff else epsilon
        val b: Double = if (pHat     > epsilon) pHat     else epsilon
        sum + pDiff * Math.log(pDiff / a) + p * Math.log(p / b)
      },
      (sum, pHat) => {
        val pHatDiff  = 1.0 - pHat
        val a: Double = if (pHatDiff > epsilon) pHatDiff else epsilon
        sum + Math.log(1.0 / a)
      }
    )

    Real(sum * scaleFactor)
  }


  // TODO: Think about different forward pass depending on ComputeMode.

  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(reference:   Tensor,
                                            output:      Tensor,
                                            scaleFactor: Real,
                                            error:       Tensor)
  : Tensor = {
    val err  = error.valuesMatrix
    val p    = reference.valuesMatrix
    val pHat = output.valuesMatrix

    val factor: Double = scaleFactor

    MatrixEx.transform(err, p, pHat)((err, p, pHat) => {
      val diff      = 1.0 - pHat
      val a: Double = if (diff > epsilon) diff else epsilon
      val b: Double = if (pHat > epsilon) pHat else epsilon
      Real(err + ((1.0 - p) / a - p / b) * factor)
    })

    RealArrayTensor.derive(error.layout.size, err)
  }

}

final class BinaryKLDivergenceConstraintBuilder
  extends ConstraintExBuilder[BinaryKLDivergenceConstraintBuilder] {

  override def repr
  : BinaryKLDivergenceConstraintBuilder = this

  /**
   * Since KL divergence requires us to compute the log of a quotient, we have
   * to make sure this is not getting out hands. (infinity, nan!)
   */
  private var _epsilon
  : Real = Real.minQuotient1

  def epsilon
  : Real = _epsilon

  def epsilon_=(value: Real)
  : Unit = {
    require(value >= Real.minQuotient1)
    _epsilon = value
  }

  def setEpsilon(value: Real)
  : BinaryKLDivergenceConstraintBuilder = {
    epsilon_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"${_epsilon}%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _epsilon.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[BinaryKLDivergenceConstraintBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BinaryKLDivergenceConstraintBuilder =>
      _epsilon == other._epsilon
    case _ =>
      false
  })


  override protected def doCopy()
  : BinaryKLDivergenceConstraintBuilder = BinaryKLDivergenceConstraintBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: BinaryKLDivergenceConstraintBuilder =>
        other._epsilon = _epsilon
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = new BinaryKLDivergenceConstraint(
    this, hints, seed, weightsBuilder
  )

}

object BinaryKLDivergenceConstraintBuilder {

  final def apply(): BinaryKLDivergenceConstraintBuilder = {
    new BinaryKLDivergenceConstraintBuilder
  }

  final def apply(domain: TensorDomain, scaleCoefficient: Real)
  : BinaryKLDivergenceConstraintBuilder = apply().setDomain(
    domain
  ).setScaleCoefficient(scaleCoefficient)

  final def apply(domain: TensorDomain, scaleCoefficient: Real, epsilon: Real)
  : BinaryKLDivergenceConstraintBuilder = apply(
    domain, scaleCoefficient
  ).setEpsilon(epsilon)

}

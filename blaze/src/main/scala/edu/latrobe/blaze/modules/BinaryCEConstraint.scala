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
import scala.util.hashing._

/**
  * Cost function for binary decisions. (Bernoulli cross entropy!)
  *
  * FProp part is identity (like for all constraints):
  *
  * f(x_a) = x_a
  *
  * d f(x_a)
  * -------- = 1
  *   x_a
  *
  *  d f(x_a)
  * ----------- = 0
  * x_b, a != b
  *
  * D f(x_a)
  * -------- = 1
  *  D x_a
  *
  *
  * Cost part:
  *
  *               n
  *              ---
  *              \
  * J(x | y) = c /   -y_i * log(x_i + e) - (1 - y_i) * log(1 - x_i + e)
  *              ---
  *              i=0
  *
  *
  * d J(x_a | y)             1        (         )        1
  * ------------ = -y_a * ------- 1 - ( 1 - y_a ) * ----------- -1
  *    d x_a              x_a + e     (         )   1 - x_a + e
  *
  *                 -y_a       1 - y_a
  *              = ------- + -----------
  *                x_a + e   1 - x_a + e
  *
  *                  1 - y_a       y_a
  *              = ----------- - -------
  *                1 - x_a + e   x_a + e
  *
  *                (1 - y_a) (x_a + e) - y_a (1 - x_a + e)
  *              = ---------------------------------------
  *                       (1 - x_a + e) (x_a + e)
  *
  *                x_a + e - y_a (x_a + e) - (y_a - y_a (x_a + e))
  *              = -----------------------------------------------
  *                            (1 - x_a + e) (x_a + e)
  *
  *                x_a + e - y_a (x_a + e) - y_a + y_a (x_a + e)
  *              = ---------------------------------------------
  *                          (1 - x_a + e) (x_a + e)
  *
  *                     x_a + e - y_a
  *              = -----------------------
  *                (1 - x_a + e) (x_a + e)
  *
  *                         x_a + e - y_a
  *              = -------------------------------
  *                (x_a + e) - (x_a + e) (x_a + e)
  *
  * Note that +e is a joke of a contribution. Normally, it won't change the
  * result at all.
  *
  *
  * d J(x_a | y)             1        (         )         1
  * ------------ = -y_a * ------- 0 - ( 1 - y_a ) * ----------- 0
  *  d x_b, b!=a          x_a + e     (         )   1 - x_a + e
  *
  *                = 0
  *
  * D J(x_a | y)     d J(x_a | y)
  * ------------ = c ------------ da
  *    D x_a             d x_a
  *
  */
// TODO: Add
final class BinaryCEConstraint(override val builder:        BinaryCEConstraintBuilder,
                               override val inputHints:     BuildHints,
                               override val seed:           InstanceSeed,
                               override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ConstraintEx[BinaryCEConstraintBuilder] {

  val epsilon
  : Real = builder.epsilon


  // ---------------------------------------------------------------------------
  //    Cost/Gradient computation related.
  // ---------------------------------------------------------------------------
  override protected def doEvaluate(reference:   Tensor,
                                    output:      Tensor,
                                    scaleFactor: Real)
  : Real ={
    // TODO: Allow computing this entirely in the GPU.
    val x = output.valuesMatrix
    val y = reference.valuesMatrixEx
    val e = DoubleEx(epsilon)

    val sum = MatrixEx.foldLeftEx(0.0, x, y)(
      (sum, x, y) => {
        sum - Math.log(1.0 - x + e) * (1.0 - y) - Math.log(x + e) * y
      },
      (sum, x) => {
        sum - Math.log(1.0 - x + e)
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
    // TODO: Create a CPU and a GPU optimized variant.
    using(
      output + epsilon,
      output.createSibling()
    )((tmp0, tmp1) => {
      // Compute denominator.
      tmp1.set(tmp0, tmp0, -Real.one)
      tmp1 += tmp0

      // Compute nominator.
      tmp0 -= reference
      //tmp0.set(output,    +scaleFactor)
      //tmp0.add(reference, -scaleFactor)

      // TODO: Add weighting.

      // Add quotient to error.
      tmp0 :/= tmp1
      error.add(tmp0, scaleFactor)
    })

    error
  }

}

final class BinaryCEConstraintBuilder
  extends ConstraintExBuilder[BinaryCEConstraintBuilder] {

  override def repr
  : BinaryCEConstraintBuilder = this

  // Anything below Real.epsilon is dangerous because the limes of the log
  // function.
  private var _epsilon
  : Real = Real.epsilon

  def epsilon
  : Real = _epsilon

  def epsilon_=(value: Real)
  : Unit = {
    require(value >= Real.zero)
    _epsilon = value
  }

  def setEpsilon(value: Real)
  : BinaryCEConstraintBuilder = {
    epsilon_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"${_epsilon}%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _epsilon.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[BinaryCEConstraintBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BinaryCEConstraintBuilder =>
      _epsilon == other._epsilon
    case _ =>
      false
  })

  override protected def doCopy()
  : BinaryCEConstraintBuilder = BinaryCEConstraintBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: BinaryCEConstraintBuilder =>
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
  : BinaryCEConstraint = new BinaryCEConstraint(
    this, hints, seed, weightsBuilder
  )

}

object BinaryCEConstraintBuilder {

  final def apply()
  : BinaryCEConstraintBuilder = new BinaryCEConstraintBuilder

  final def apply(epsilon: Real)
  : BinaryCEConstraintBuilder = apply().setEpsilon(epsilon)

}

/*

final class BinaryCEConstraintX(override val builder:       BinaryCEConstraintBuilder,
                                override val inputHints:    ModuleBuildHints,
                                override val weightsLinker: WeightsLinker,
                                override val seed:          InstanceSeed)
  extends ConstraintByMeasure[BinaryCEConstraintBuilder]
  with NonTrainable {

  import builder.{beta, rho}

  protected val invRho = Real.one - rho


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  /*
  override protected def doDeriveInputError(mode:    ComputeMode,
                                            error:   DVec,
                                            measure: DVec)
  : Unit = {
    val diff: DVec = computeDifference(measure)
    axpy(beta, diff, error)
  }
  */

  override protected def doDeriveInputError(mode:    ComputeMode,
                                            error:   DMat,
                                            measure: DVec)
  : Unit = {
    // TODO: Use lerp!
    val diff: DVec = measure.fastMap(
      mes => invRho / (Real.one - mes) - rho / mes
    )
    diff *= beta
    error(::, *) += diff
  }


  // ---------------------------------------------------------------------------
  //    Cost and gradients computation related.
  // ---------------------------------------------------------------------------
  @inline
  protected def computeRawCost(measure: DVec): Real = {
    var a = 0.0
    var b = 0.0
    measure.foreach(mes => {
      a += Math.log(rho / mes)
      b += Math.log(invRho / (Real.one - mes))
    })
    // TODO: Use lerp!
    Real(a * rho + b * invRho)
  }

}
*/
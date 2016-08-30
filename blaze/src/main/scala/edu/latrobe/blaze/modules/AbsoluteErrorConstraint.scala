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
  * Sum of absolute errors.
  *
  *             m       n
  *            ---     ---
  *         1  \    c  \
  * cost = --- /   --- /   smooth_abs(yHat_ij, y_ij)
  *         m  ---  n  ---
  *            j=0     i=0
  *
  *                                /---------------------------
  * smooth_abs(yHat_ij, y_ij) =   /                 2
  *                             \/  (yHat_ij - y_ij)  + epsilon
  *
  * Also note that:
  *
  * smooth_abs(yHat_ij, y_ij, epsilon = 0) = | yHat_ij - y_ij |
  *
  *
  * d smooth_abs(yHat_ij, y_ij)       2 (yHat_ij - y_ij)
  * --------------------------- = ---------------------------
  *         d yHat_ij             2 smooth_abs(yHat_ij, y_ij)
  *
  *
  *                                    yHat_ij - y_ij
  *                             = -------------------------
  *                               smooth_abs(yHat_ij, y_ij)
  *
  * Remark:
  *    d      (                 2           )
  * --------- ( (yHat_ij - y_ij)  + epsilon ) = 2 (yHat_ij - y_ij) * 1 = 2 (yHat_ij - y_ij)
  * d yHat_ij
  *
  */
final class AbsoluteErrorConstraint(override val builder:        AbsoluteErrorConstraintBuilder,
                                    override val inputHints:     BuildHints,
                                    override val seed:           InstanceSeed,
                                    override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ConstraintEx[AbsoluteErrorConstraintBuilder] {

  val epsilon
  : Double = DoubleEx(builder.epsilon)


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doEvaluate(reference:   Tensor,
                                    output:      Tensor,
                                    scaleFactor: Real)
  : Real = {
    val y    = reference.valuesMatrixEx
    val yHat = output.valuesMatrix

    val sum: Real = {
      if (epsilon == 0.0) {
        MatrixEx.foldLeftEx(Real.zero, yHat, y)(
          (sum, yHat, y) => sum + Math.abs(yHat - y),
          (sum, yHat)    => sum + Math.abs(yHat)
        )
      }
      else {
        val tmp = MatrixEx.foldLeftEx(0.0, yHat, y)(
          (sum, yHat, y) => {
            val tmp = yHat - y
            sum + Math.sqrt(tmp * tmp + epsilon)
          },
          (sum, yHat) => {
            sum + Math.sqrt(yHat * yHat + epsilon)
          }
        )
        Real(tmp)
      }
    }

    sum * scaleFactor
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(reference:   Tensor,
                                            output:      Tensor,
                                            scaleFactor: Real,
                                            error:       Tensor)
  : Tensor = {
    val y    = reference.valuesMatrixEx
    val yHat = output.valuesMatrix
    val err  = error.valuesMatrix

    if (epsilon == 0.0) {
      MatrixEx.transformEx(err, yHat, y)(
        (err, yHat, y) => {
          if (yHat - y < Real.zero) -err else err
        },
        (err, yHat) => {
          if (yHat < Real.zero) -err else err
        }
      )
    }
    else {
      MatrixEx.transformEx(err, yHat, y)(
        (err, yHat, y) => {
          val tmp = yHat - y
          val abs = Math.sqrt(tmp * tmp + epsilon)
          Real(err * tmp / abs)
        },
        (err, yHat) => {
          val abs = Math.sqrt(yHat * yHat + epsilon)
          Real(err * yHat / abs)
        }
      )
    }

    RealArrayTensor.derive(error.layout.size, err)
  }

}

final class AbsoluteErrorConstraintBuilder
  extends ConstraintExBuilder[AbsoluteErrorConstraintBuilder] {

  override def repr
  : AbsoluteErrorConstraintBuilder = this

  private var _epsilon
  : Real = Real.zero

  def epsilon
  : Real = _epsilon

  def epsilon_=(value: Real)
  : Unit = {
    require(value >= Real.zero)
    _epsilon = value
  }

  def setEpsilon(value: Real)
  : AbsoluteErrorConstraintBuilder = {
    epsilon_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"${_epsilon}%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _epsilon.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AbsoluteErrorConstraintBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AbsoluteErrorConstraintBuilder =>
      _epsilon == other._epsilon
    case _ =>
      false
  })

  override protected def doCopy()
  : AbsoluteErrorConstraintBuilder = AbsoluteErrorConstraintBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: AbsoluteErrorConstraintBuilder =>
        other._epsilon = _epsilon
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Weights buffer handling related.
  // ---------------------------------------------------------------------------
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : AbsoluteErrorConstraint = new AbsoluteErrorConstraint(
    this, hints, seed, weightsBuilder
  )

}

object AbsoluteErrorConstraintBuilder {

  final def apply()
  : AbsoluteErrorConstraintBuilder = new AbsoluteErrorConstraintBuilder

  final def apply(domain: TensorDomain, scaleCoefficient: Real)
  : AbsoluteErrorConstraintBuilder = apply().setDomain(
    domain
  ).setScaleCoefficient(scaleCoefficient)

}

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

package edu.latrobe.blaze.validators

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
  * Subtracts the reference from the prediction. Then compares it against a
  * predefined range. Scores a true positive if the difference is in that range
  * or a false positive if it is not.
  */
final class SimilarityValidator(override val builder: SimilarityValidatorBuilder,
                                override val seed:    InstanceSeed)
  extends ValidatorEx[SimilarityValidatorBuilder] {

  val tolerance
  : RealRange = builder.tolerance

  override def apply(reference: Tensor, output: Tensor): ValidationScore = {
    val tol = tolerance
    val ref = reference.valuesMatrixEx
    val out = output.valuesMatrix

    var tp = 0L
    var fp = 0L
    MatrixEx.foreach(out, ref)((out, ref) => {
      if (tol.contains(out - ref)) {
        tp += 1L
      }
      else {
        fp += 1L
      }
    })
    ValidationScore(tp, fp)
  }

}

final class SimilarityValidatorBuilder
  extends ValidatorExBuilder[SimilarityValidatorBuilder] {

  override def repr
  : SimilarityValidatorBuilder = this

  private var _tolerance
  : RealRange = RealRange.derive(0.05f)

  def tolerance
  : RealRange = _tolerance

  def tolerance_=(value: RealRange)
  : Unit = {
    require(value != null)
    _tolerance = value
  }

  def setTolerance(value: RealRange)
  : SimilarityValidatorBuilder = {
    tolerance_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _tolerance :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _tolerance.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SimilarityValidatorBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: SimilarityValidatorBuilder =>
      _tolerance == other._tolerance
    case _ =>
      false
  })

  override protected def doCopy()
  : SimilarityValidatorBuilder = SimilarityValidatorBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: SimilarityValidatorBuilder =>
        other._tolerance = _tolerance
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : SimilarityValidator = new SimilarityValidator(this, seed)

}

object SimilarityValidatorBuilder {

  final def apply()
  : SimilarityValidatorBuilder = new SimilarityValidatorBuilder

  final def apply(tolerance: RealRange)
  : SimilarityValidatorBuilder = apply().setTolerance(tolerance)

  final def derive(tolerance: Real)
  : SimilarityValidatorBuilder = apply(RealRange(-tolerance, tolerance))

}


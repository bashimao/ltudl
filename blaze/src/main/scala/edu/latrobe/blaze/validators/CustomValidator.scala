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
  * Compares predicted values against reference. Then checks iwhether they fall
  * into a certain positive range to quantify them as positive or negative
  * values. Produces a
  * very expressive validation result that can cope with multi-label references
  * and shows how well the model approximates a set of binary labels.
  */
final class CustomValidator(override val builder: CustomValidatorBuilder,
                            override val seed:    InstanceSeed)
  extends ValidatorEx[CustomValidatorBuilder] {

  val yPredicateFn
  : Real => Boolean = builder.yPredicateFn

  val yHatPredicateFn
  : Real => Boolean = builder.yPredicateFn

  override def apply(reference: Tensor, output: Tensor): ValidationScore = {
    val ref = reference.valuesMatrixEx
    val out = output.valuesMatrix

    var tp = 0L
    var fp = 0L
    var tn = 0L
    var fn = 0L
    MatrixEx.foreach(out, ref)((out, ref) => {
      val yHat = yHatPredicateFn(out)
      val y    = yPredicateFn(ref)
      if (yHat) {
        // We believe its positive.
        if (y) {
          // True - Confirmed!
          tp += 1L
        }
        else {
          // False - Not confirmed!
          fp += 1L
        }
      }
      else {
        // We believe it's negative.
        if (y) {
          // False - But the label is negative!
          fn += 1L
        }
        else {
          // True - The label is negative as well. So everything is fine.
          tn += 1L
        }
      }
    })
    ValidationScore(tp, fp, tn, fn)
  }

}

final class CustomValidatorBuilder
  extends ValidatorExBuilder[CustomValidatorBuilder] {


  override def repr
  : CustomValidatorBuilder = this

  /**
    * Predicate function to determine whether the outcome is positive.
    */
  private var _yPredicateFn
  : Real => Boolean = x => x >= Real.pointFive

  def yPredicateFn
  : Real => Boolean = _yPredicateFn

  def yPredicateFn_=(value: Real => Boolean)
  : Unit = {
    require(value != null)
    _yPredicateFn = value
  }

  def setYPredicateFn(value: Real => Boolean)
  : CustomValidatorBuilder = {
    yPredicateFn_=(value)
    repr
  }

  /**
    * Predicate function to determine whether the outcome is positive.
    */
  private var _yHatPredicateFn
  : Real => Boolean = x => x >= Real.pointFive

  def yHatPredicateFn
  : Real => Boolean = _yHatPredicateFn

  def yHatPredicateFn_=(value: Real => Boolean)
  : Unit = {
    require(value != null)
    _yHatPredicateFn = value
  }

  def setYHatPredicateFn(value: Real => Boolean)
  : CustomValidatorBuilder = {
    yHatPredicateFn_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _yPredicateFn :: _yHatPredicateFn :: super.doToString()

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _yPredicateFn.hashCode())
    tmp = MurmurHash3.mix(tmp, _yHatPredicateFn.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[CustomValidatorBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: CustomValidatorBuilder =>
      _yPredicateFn    == other._yPredicateFn &&
      _yHatPredicateFn == other._yHatPredicateFn
    case _ =>
      false
  })

  override protected def doCopy()
  : CustomValidatorBuilder = CustomValidatorBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: CustomValidatorBuilder =>
        other._yPredicateFn    = _yPredicateFn
        other._yHatPredicateFn = _yHatPredicateFn
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : CustomValidator = new CustomValidator(this, seed)

}

object CustomValidatorBuilder {

  final def apply()
  : CustomValidatorBuilder = new CustomValidatorBuilder

  final def apply(predicateFn: Real => Boolean)
  : CustomValidatorBuilder = apply(predicateFn, predicateFn)

  final def apply(yPredicateFn:    Real => Boolean,
                  yHatPredicateFn: Real => Boolean)
  : CustomValidatorBuilder = apply().setYPredicateFn(
    yPredicateFn
  ).setYHatPredicateFn(yHatPredicateFn)

  final def derive(threshold: Real)
  : CustomValidatorBuilder = apply(_ >= threshold)

  final def derive(positiveRange: RealRange)
  : CustomValidatorBuilder = apply(positiveRange.contains)

}

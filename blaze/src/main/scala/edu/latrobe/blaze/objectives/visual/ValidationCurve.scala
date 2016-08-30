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

package edu.latrobe.blaze.objectives.visual

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.validators._
import edu.latrobe.io.vega._
import edu.latrobe.time._

final class ValidationCurve(override val builder:    ValidationCurveBuilder,
                            override val dataSeries: DataSeries2D,
                            override val seed:       InstanceSeed)
  extends CurveEx[ValidationCurveBuilder] {

  val validator
  : Validator = builder.validator.build(seed)

  val scoreTransformFn
  : ValidationScore => Real = builder.scoreTransformFn

  override protected def doClose()
  : Unit = {
    validator.close()
    super.doClose()
  }

  override def yValueFor(optimizer:           OptimizerLike,
                         runBeginIterationNo: Long,
                         runBeginTime:        Timestamp,
                         runNoSamples:        Long,
                         model:               Module,
                         batch:               Batch,
                         output:              Tensor,
                         value:               Real)
  : Real = {
    val score = validator(batch.output, output)
    val y = scoreTransformFn(score)
    y
  }

}

final class ValidationCurveBuilder
  extends CurveExBuilder[ValidationCurveBuilder] {

  override def repr
  : ValidationCurveBuilder = this

  private var _validator
  : ValidatorBuilder = Top1LabelValidatorBuilder()

  def validator
  : ValidatorBuilder = _validator

  def validator_=(value: ValidatorBuilder)
  : Unit = {
    require(value != null)
    _validator = value
  }

  def setValidator(value: ValidatorBuilder)
  : ValidationCurveBuilder = {
    validator_=(value)
    this
  }

  private var _scoreTransformFn
  : ValidationScore => Real = score => score.accuracy * 100

  def scoreTransformFn
  : ValidationScore => Real = _scoreTransformFn

  def scoreTransformFn_=(value: ValidationScore => Real)
  : Unit = {
    require(value != null)
    _scoreTransformFn = value
  }

  def setScoreTransformFn(value: ValidationScore => Real)
  : ValidationCurveBuilder = {
    scoreTransformFn_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] =  _validator :: _scoreTransformFn :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ValidationCurveBuilder]

  override def hashCode(): Int = super.hashCode()

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ValidationCurveBuilder =>
      _validator        == other._validator &&
      _scoreTransformFn == other._scoreTransformFn
    case _ =>
      false
  })

  override protected def doCopy()
  : ValidationCurveBuilder = ValidationCurveBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ValidationCurveBuilder =>
        other._validator        = _validator.copy
        other._scoreTransformFn = _scoreTransformFn
      case _ =>
    }
  }

  override protected[visual] def doBuild(dataSeries: DataSeries2D, seed: InstanceSeed)
  : ValidationCurve = new ValidationCurve(this, dataSeries, seed)

  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _validator.permuteSeeds(fn)
  }

}


object ValidationCurveBuilder {

  final def apply()
  : ValidationCurveBuilder = new ValidationCurveBuilder

  final def apply(validator: ValidatorBuilder)
  : ValidationCurveBuilder = apply().setValidator(validator)

}
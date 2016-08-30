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

package edu.latrobe.blaze.objectives

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.validators._
import edu.latrobe.time._
import scala.util.hashing._

/**
  * Validates current output and injects it into the children as value.
  */
final class ValidateOutput(override val builder: ValidateOutputBuilder,
                           override val seed:    InstanceSeed)
  extends DependentObjectiveEx[ValidateOutputBuilder] {

  val validator
  : Validator = builder.validator.build(seed)

  val scoreTransformFn
  : ValidationScore => Real = builder.scoreTransformFn

  override protected def doClose()
  : Unit = {
    validator.close()
    super.doClose()
  }

  override protected def doEvaluate(sink:                Sink,
                                    optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Option[ObjectiveEvaluationResult] = {
    val score = validator(batch.output, output)
    val value = scoreTransformFn(score)
    super.doEvaluate(
      sink,
      optimizer, runBeginIterationNo, runBeginTime, runNoSamples,
      model,
      batch, output, value
    )
  }

}

final class ValidateOutputBuilder
  extends DependentObjectiveExBuilder[ValidateOutputBuilder] {

  override def repr
  : ValidateOutputBuilder = this

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
  : ValidateOutputBuilder = {
    validator_=(value)
    repr
  }

  private var _scoreTransformFn
  : ValidationScore => Real = score => score.accuracy

  def scoreTransformFn
  : ValidationScore => Real = _scoreTransformFn

  def scoreTransformFn_=(value: ValidationScore => Real)
  : Unit = {
    require(value != null)
    _scoreTransformFn = value
  }

  def setScoreTransformFn(value: ValidationScore => Real)
  : ValidateOutputBuilder = {
    scoreTransformFn_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _validator :: _scoreTransformFn :: super.doToString()

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _validator.hashCode())
    tmp = MurmurHash3.mix(tmp, _scoreTransformFn.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ValidateOutputBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ValidateOutputBuilder =>
      _validator        == other._validator &&
      _scoreTransformFn == other._scoreTransformFn
    case _ =>
      false
  })

  override protected def doCopy()
  : ValidateOutputBuilder = ValidateOutputBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ValidateOutputBuilder =>
        other._validator        = _validator.copy
        other._scoreTransformFn = _scoreTransformFn
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : ValidateOutput = new ValidateOutput(this, seed)


  // ---------------------------------------------------------------------------
  //   Cascading mutable state.
  // ---------------------------------------------------------------------------
  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _validator.permuteSeeds(fn)
  }

}

object ValidateOutputBuilder {

  final def apply()
  : ValidateOutputBuilder = new ValidateOutputBuilder

  final def apply(validator: ValidatorBuilder)
  : ValidateOutputBuilder = apply().setValidator(validator)

  final def apply(validator:        ValidatorBuilder,
                  scoreTransformFn: ValidationScore => Real)
  : ValidateOutputBuilder = apply(validator).setScoreTransformFn(scoreTransformFn)

  final def top1Label
  : ValidateOutputBuilder = apply(Top1LabelValidatorBuilder())

  final def top5Label
  : ValidateOutputBuilder = topKLabel(5)

  final def topKLabel(k: Int)
  : ValidateOutputBuilder = apply(TopKLabelsValidatorBuilder(k))

}

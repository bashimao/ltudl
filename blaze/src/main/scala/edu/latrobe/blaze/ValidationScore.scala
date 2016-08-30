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

package edu.latrobe.blaze

import edu.latrobe._
import scala.util.hashing._

/**
  * Metrics derived from wikipage: https://en.wikipedia.org/wiki/F1_score
  */
// TODO: Make this abstract and add confusion matrices and other scoring systems as alternatives.
final class ValidationScore(val truePositives: Long, val falsePositives: Long,
                            val trueNegatives: Long, val falseNegatives: Long)
  extends Serializable
    with Equatable {
  require(
    truePositives  >= 0L &&
    falsePositives >= 0L &&
    trueNegatives  >= 0L &&
    falseNegatives >= 0L
  )

  override def toString
  : String = {
    s"TP=$truePositives FP=$falsePositives TN=$trueNegatives FN=$falseNegatives"
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, truePositives.hashCode())
    tmp = MurmurHash3.mix(tmp, falsePositives.hashCode())
    tmp = MurmurHash3.mix(tmp, trueNegatives.hashCode())
    tmp = MurmurHash3.mix(tmp, falseNegatives.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ValidationScore]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ValidationScore =>
      truePositives  == other.truePositives  &&
      falsePositives == other.falsePositives &&
      trueNegatives  == other.trueNegatives  &&
      falseNegatives == other.falseNegatives
    case _ =>
      false
  })

  // Simple sums
  def conditionNegativeCount
  : Long = trueNegatives + falsePositives

  def conditionPositiveCount
  : Long = truePositives + falseNegatives

  def predictionNegativeCount
  : Long = trueNegatives + falseNegatives

  def predictionPositiveCount
  : Long = truePositives + falsePositives

  def totalPopulation
  : Long = truePositives + falsePositives + trueNegatives + falseNegatives

  // Based on total population
  def prevalence
  : Real = conditionPositiveCount / Real(totalPopulation)

  // Based on condition
  def truePositiveRate
  : Real = truePositives / Real(conditionPositiveCount)

  def falseNegativeRate
  : Real = falseNegatives / Real(conditionPositiveCount)

  def falsePositiveRate
  : Real = falsePositives / Real(conditionNegativeCount)

  def trueNegativeRate
  : Real = trueNegatives / Real(conditionNegativeCount)

  def recall
  : Real = truePositiveRate

  def sensitivity
  : Real = truePositiveRate

  def hitRate
  : Real = truePositiveRate

  def fallOut
  : Real = falsePositiveRate

  def specificity
  : Real = trueNegativeRate

  // Based on prediction outcome
  def positivePredictiveValue
  : Real = truePositives / Real(predictionPositiveCount)

  def falseDiscoveryRate
  : Real = falsePositives / Real(predictionPositiveCount)

  def falseOmissionRate
  : Real = falseNegatives / Real(predictionNegativeCount)

  def negativePredictiveValue
  : Real = trueNegatives / Real(predictionNegativeCount)

  def precision
  : Real = positivePredictiveValue


  // Likelihoods
  def positiveLikelihoodRatio
  : Real = truePositiveRate / falsePositiveRate

  def negativeLikelihoodRatio
  : Real = trueNegativeRate / falseNegativeRate

  def diagnosticOddsRatio
  : Real = falsePositiveRate / negativeLikelihoodRatio


  // Quality measures.
  def accuracy
  : Real = (truePositives + trueNegatives) / Real(totalPopulation)

  def f1Score
  : Real = (2 * precision * recall) / (precision + recall)

  def fScore(degree: Real)
  : Unit = {
    val degree2 = degree * degree
    ((Real.one + degree2) * precision * recall) / (degree2 * precision + recall)
  }

  def matthewsCorrelationCoefficient
  : Double = {
    var tmp = 0.0
    tmp += truePositives + falsePositives
    tmp *= truePositives + falseNegatives
    tmp *= trueNegatives + falsePositives
    tmp *= trueNegatives + falseNegatives
    tmp = Math.sqrt(tmp)
    (truePositives * trueNegatives - falsePositives * falseNegatives) / tmp
  }


  def +(other: ValidationScore)
  : ValidationScore = ValidationScore(
    truePositives  + other.truePositives,
    falsePositives + other.falsePositives,
    trueNegatives  + other.trueNegatives,
    falseNegatives + other.falseNegatives
  )

  def -(other: ValidationScore)
  : ValidationScore = ValidationScore(
    truePositives  - other.truePositives,
    falsePositives - other.falsePositives,
    trueNegatives  - other.trueNegatives,
    falseNegatives - other.falseNegatives
  )

}

object ValidationScore {

  final def apply(truePositives: Long, falsePositives: Long)
  : ValidationScore = apply(
    truePositives, falsePositives,
    0L,            0L
  )

  final def apply(truePositives: Long, falsePositives: Long,
                  trueNegatives: Long, falseNegatives: Long)
  : ValidationScore = new ValidationScore(
    truePositives, falsePositives,
    trueNegatives, falseNegatives
  )

  final val oneTruePositive
  : ValidationScore = apply(1L, 0L, 0L, 0L)

  final val oneFalsePositive
  : ValidationScore = apply(0L, 1L, 0L, 0L)

  final val oneTrueNegative
  : ValidationScore = apply(0L, 0L, 1L, 0L)

  final val oneFalseNegative
  : ValidationScore = apply(0L, 0L, 0L, 1L)

  final val zero
  : ValidationScore = apply(0L, 0L)

}

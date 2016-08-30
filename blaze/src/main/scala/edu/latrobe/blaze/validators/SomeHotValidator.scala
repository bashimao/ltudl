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

package edu.latrobe.blaze.validators

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
  * A validator designed to test outputs against some-hot encoded
  * ground truths.
  */
abstract class SomeHotValidator[TBuilder <: SomeHotValidatorBuilder[TBuilder]]
  extends ValidatorEx[TBuilder] {

  final val isHotThreshold
  : Real = builder.isHotThreshold

}

abstract class SomeHotValidatorBuilder[TThis <: SomeHotValidatorBuilder[TThis]]
  extends ValidatorExBuilder[TThis] {

  /**
    * Threshold above which a value is considered as being hot.
    */
  final var isHotThreshold
  : Real = Real.pointFive

  final def setIsHotThreshold(value: Real)
  : TThis = {
    isHotThreshold_=(value)
    repr
  }

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), isHotThreshold.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: SomeHotValidatorBuilder[_] =>
      isHotThreshold == other.isHotThreshold
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: SomeHotValidatorBuilder[_] =>
        other.isHotThreshold = isHotThreshold
      case _ =>
    }
  }

}

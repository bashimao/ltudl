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

package edu.latrobe

trait MutableAccumulatorLike
  extends Copyable {

  def update(value: Real)
  : Unit

  @inline
  final def update(values: Array[Real])
  : Unit = {
    ArrayEx.foreach(
      values
    )(update)
  }

  @inline
  final def update(values:       Array[Real],
                   rng:          PseudoRNG,
                   noSamplesMax: Int)
  : Unit = {
    ArrayEx.foreach(
      values,
      rng, noSamplesMax
    )(update)
  }

  def reset()
  : Unit

}

trait MutableAccumulatorLikeCompanion {

  def apply()
  : MutableAccumulatorLike

  def derive(values: Array[Real])
  : MutableAccumulatorLike

  def derive(values:       Array[Real],
             rng:          PseudoRNG,
             noSamplesMax: Int)
  : MutableAccumulatorLike

}

trait MutableAccumulatorLikeEx[TThis <: MutableAccumulatorLikeEx[_]]
  extends MutableAccumulatorLike
    with CopyableEx[TThis] {

  def :=(other: TThis)
  : Unit

  def +=(other: TThis)
  : Unit

}

trait MutableAccumulatorLikeExCompanion[T <: MutableAccumulatorLikeEx[_]]
  extends MutableAccumulatorLikeCompanion {

  override def apply()
  : T

  final override def derive(values: Array[Real])
  : T = {
    val result = apply()
    result.update(values)
    result
  }

  final override def derive(values:       Array[Real],
                            rng:          PseudoRNG,
                            noSamplesMax: Int)
  : T = {
    val result = apply()
    result.update(values, rng, noSamplesMax)
    result
  }

}

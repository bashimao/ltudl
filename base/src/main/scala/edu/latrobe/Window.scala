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

import scala.util.hashing._

/**
 * Weighted kernels wrap around normal kernels and assign a weight to each
 * indexed item.
 *
 * Base class for all weighting windows. Please note that the weights of a
 * weighting window should always sum to 1.
 */
abstract class Window
  extends Serializable
    with Equatable {

  /**
    * Frequently used. Best override this with a val or even a constructor arg!
    */
  def noWeights
  : Int

  override def hashCode()
  : Int = foldLeftWeights(
    super.hashCode()
  )((res, value) => MurmurHash3.mix(res, value.hashCode()))

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Window]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Window =>
      require(noWeights == other.noWeights)
      val n = noWeights
      var i = 0
      while (i < n) {
        if (apply(i) != other.apply(i)) {
          return false
        }
        i += 1
      }
      true
    case _ =>
      false
  })


  // ---------------------------------------------------------------------------
  //    Weights related.
  // ---------------------------------------------------------------------------
  def apply(index: Int)
  : Real

  @transient
  final lazy val sum
  : Real = Real(foldLeftWeights(0.0)(_ + _))

  @transient
  final lazy val mean
  : Real = Real(foldLeftWeights(0.0)(_ + _) / noWeights)

  @transient
  final lazy val isNormalized
  : Boolean = Math.abs(sum - Real.one) < 1e-5f


  // ---------------------------------------------------------------------------
  //    Iteration methods.
  // ---------------------------------------------------------------------------
  final def foldLeftWeights[T](z0: T)
                              (fn: (T, Real) => T)
  : T = {
    var z = z0
    val n = noWeights
    var i = 0
    while (i < n) {
      z = fn(z, apply(i))
      i += 1
    }
    z
  }

  final def foreachWeight(fn: Real => Unit)
  : Unit = {
    val n = noWeights
    var i = 0
    while (i < n) {
      fn(apply(i))
      i += 1
    }
  }

  final def foreachWeightPair(fn: (Int, Real) => Unit)
  : Unit = {
    val n = noWeights
    var i = 0
    while (i < n) {
      fn(i, apply(i))
      i += 1
    }
  }

  final def toArray
  : Array[Real] = ArrayEx.tabulate(
    noWeights
  )(apply)

}

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
import scala.collection._
import scala.util.hashing._

/**
 * @param rating Overall rating of the gradient test.
 * @param differences Individual differences for each tested gradient.
 */
final class GradientDeviation(val rating:      Real,
                              val differences: SortedMap[(Int, Int, Int), (Real, Long)])
  extends Serializable
    with Equatable
    with CopyableEx[GradientDeviation] {
  require(differences != null)

  override def toString
  : String = f"Rating: $rating%.4g, Differences: $differences%s"

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, rating.hashCode())
    tmp = MurmurHash3.mix(tmp, differences.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[GradientDeviation]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: GradientDeviation =>
      rating      == other.rating &&
      differences == other.differences
    case _ =>
      false
  })

  override def copy
  : GradientDeviation = GradientDeviation(rating, differences)


  /**
   * Weight of this instance.
   */
  def relevance: Long = MapEx.foldLeftValues(0L, differences)(_ + _._2)

  /**
   * Workaround for current breeze version. Vector[Real].mapActiveValues causes
   * unintended tabulate.
   */
  //def differenceValues: SMat = differences.mapActiveValues(_._1)

  def minDifference
  : ((Int, Int, Int), (Real, Long)) = differences.minBy(_._2._1)

  def maxDifference
  : ((Int, Int, Int), (Real, Long)) = differences.maxBy(_._2._1)

  def +(other: GradientDeviation): GradientDeviation = {
    var aSum = 0L
    var bSum = 0L
    val newDiffs = MapEx.zipValuesEx(differences, other.differences)(
      (a, b) => {
        aSum += a._2
        bSum += b._2
        val w = a._2 + b._2
        Tuple2(MathMacros.lerp(a._1, b._1, b._2 / Real(w)), w)
      },
      a => {
        aSum += a._2
        a
      },
      b => {
        bSum += b._2
        b
      }
    )

    val newRating = {
      val sum = aSum + bSum
      if (sum > 0L) {
        MathMacros.lerp(rating, other.rating, bSum / Real(sum))
      }
      else {
        Real.zero
      }
    }
    GradientDeviation(newRating, newDiffs)
  }

}

object GradientDeviation {

  final def zero
  : GradientDeviation = apply(Real.zero, SortedMap.empty)

  final def apply(rating:      Real,
                  differences: SortedMap[(Int, Int, Int), (Real, Long)])
  : GradientDeviation = new GradientDeviation(rating, differences)

}

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

import breeze.linalg.{*, argtopk}
import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
  * Treats the highest k predictions as the label and compares them against the
  * reference. If any of the corresponding k references contains the true label
  * (>= threshold), we have a true positive. Otherwise, we are adding
  * a false positive. Hence, this presumes that there is only one true label in
  * the reference tensor for each sample. Otherwise, results maybe misleading.
  */
final class TopKLabelsValidator(override val builder: TopKLabelsValidatorBuilder,
                                override val seed:    InstanceSeed)
  extends OneHotValidator[TopKLabelsValidatorBuilder] {

  val k
  : Int = builder.k

  override def apply(reference: Tensor, output: Tensor): ValidationScore = {
    val ref  = reference.valuesMatrixEx
    val out  = output.valuesMatrix
    // TODO: Update once we have newer breeze support.
    // val rows = argtopk(out(::, *), k)
    val rows = argtopk(out(::, *), k).toArray
    val thr  = isHotThreshold

    var tp = 0L
    var fp = 0L
    // TODO: Update once we have newer breeze support.
    // rows.foreachPair((col, rows) => {
    ArrayEx.foreachPair(rows)((col, rows) => {
      var tmp = 0
      for (row <- rows) {
        val y = ref(row, col)
        if (y >= thr) {
          tmp += 1
        }
      }
      if (tmp > 0) {
        tp += 1L
      }
      else {
        fp += 1L
      }
    })
    ValidationScore(tp, fp)
  }

}

final class TopKLabelsValidatorBuilder
  extends OneHotValidatorBuilder[TopKLabelsValidatorBuilder] {

  override def repr
  : TopKLabelsValidatorBuilder = this

  private var _k
  : Int = 5

  def k
  : Int = _k

  def k_=(value: Int)
  : Unit = {
    require(value > 0)
    _k = value
  }

  def setK(value: Int)
  : TopKLabelsValidatorBuilder = {
    k_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _k :: f"$isHotThreshold%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _k.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[TopKLabelsValidatorBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: TopKLabelsValidatorBuilder =>
      _k == other._k
    case _ =>
      false
  })

  override protected def doCopy()
  : TopKLabelsValidatorBuilder = TopKLabelsValidatorBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: TopKLabelsValidatorBuilder =>
        other._k = _k
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : TopKLabelsValidator = new TopKLabelsValidator(this, seed)

}

object TopKLabelsValidatorBuilder {

  final def apply()
  : TopKLabelsValidatorBuilder = new TopKLabelsValidatorBuilder

  final def apply(k: Int)
  : TopKLabelsValidatorBuilder = apply().setK(k)

  final def apply(k: Int, isHotThreshold: Real)
  : TopKLabelsValidatorBuilder = apply(k).setIsHotThreshold(isHotThreshold)

}

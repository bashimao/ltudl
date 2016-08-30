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

import breeze.linalg.{*, argmax}
import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
  * Treats the highest prediction as the label and checks whether the
  * corresponding reference is the true label (>= threshold). So it will either
  * be true positive or a false positive. Hence, this presumes that there is
  * only one true label in the reference tensor for each sample. Otherwise,
  * results maybe misleading.
  */
final class Top1LabelValidator(override val builder: Top1LabelValidatorBuilder,
                               override val seed:    InstanceSeed)
  extends OneHotValidator[Top1LabelValidatorBuilder] {

  override def apply(reference: Tensor, output: Tensor): ValidationScore = {
    val ref  = reference.valuesMatrixEx
    val out  = output.valuesMatrix
    // TODO: Update once we have newer breeze support.
    //val rows = argmax(out(::, *)).t
    val rows = argmax(out(::, *)).toArray

    var tp = 0L
    var fp = 0L
    // TODO: Update once we have newer breeze support.
    // rows.foreachPair((col, row) => {
    ArrayEx.foreachPair(rows)((col, row) => {
      val y = ref(row, col)
      if (y >= isHotThreshold) {
        tp += 1L
      }
      else {
        fp += 1L
      }
    })
    ValidationScore(tp, fp)
  }

}

final class Top1LabelValidatorBuilder
  extends OneHotValidatorBuilder[Top1LabelValidatorBuilder] {

  override def repr
  : Top1LabelValidatorBuilder = this

  override protected def doToString()
  : List[Any] = f"$isHotThreshold%.4g" :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Top1LabelValidatorBuilder]

  override protected def doCopy()
  : Top1LabelValidatorBuilder = Top1LabelValidatorBuilder()

  override def build(seed: InstanceSeed)
  : Top1LabelValidator = new Top1LabelValidator(this, seed)

}

object Top1LabelValidatorBuilder {

  final def apply()
  : Top1LabelValidatorBuilder = new Top1LabelValidatorBuilder

  final def apply(isHotThreshold: Real)
  : Top1LabelValidatorBuilder = apply().setIsHotThreshold(isHotThreshold)

}

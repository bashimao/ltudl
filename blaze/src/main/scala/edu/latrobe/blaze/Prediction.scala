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

/**
  * @param input The input of the prediction.
  * @param reference The ground truth given as input.
  * @param output The actual prediction result.
  */
final class Prediction(val input:     Tensor,
                       val reference: Tensor,
                       val output:    Tensor,
                       val value:     Real)
  extends AutoCloseable
    with Serializable {
  require(input != null && output != null)

  override def toString
  : String = s"Prediction[$output, $value]"

  def cost
  : Cost = Cost(value, output.layout.noSamples)

  /**
    * Drops the prediction output.
    */
  override def close()
  : Unit = {
    if (output ne input) {
      if (output ne reference) {
        output.close()
      }
    }
  }

}

object Prediction {

  final def apply(input:     Tensor,
                  reference: Tensor,
                  output:    Tensor,
                  cost:      Real)
  : Prediction = new Prediction(input, reference, output, cost)

}
/*
final class PredictionSet(val predictions: IndexedSeq[Prediction])
  extends PredictionLike {
  require(predictions != null)

  override def close(): Unit = dropOutput()

  override def input: TensorSet = TensorSet(predictions.map(_.input))

  override def reference: TensorSet = TensorSet(predictions.map(_.reference))

  override def output: TensorSet = TensorSet(predictions.map(_.output))

  override def dropOutput(): Unit = predictions.foreach(_.dropOutput())

}

object PredictionSet {

  final def apply(predictions: IndexedSeq[Prediction])
  : PredictionSet = new PredictionSet(predictions)

}
*/

/*
final class PredictionEx(val prediction: Prediction,
                         val tensors:    List[Tensor],
                         val contexts:   List[PredictContext])
  extends PredictionLike {
  require(
    prediction != null &&
    tensors    != null &&
    contexts   != null
  )

  override def close()
  : Unit = dropIntermediates().close()

  override def input: Tensor = prediction.input

  override def reference: Tensor = prediction.reference

  override def output: Tensor = prediction.output

  override def dropOutput(): Unit = prediction.dropOutput()

  /**
    * Drops backprop related data. And returns a new object. Do not use this
    * object after ditching the backprop data.
    */
  def dropIntermediates(): Prediction = {
    tensors.foreach(tensor => {
      if (tensor != null && (tensor ne prediction.input) && (tensor ne prediction.reference) && (tensor ne prediction.output)) {
        tensor.dispose()
      }
    })

    prediction
  }

}

object PredictionEx {

  final def apply(prediction: Prediction,
                  tensors:    List[Tensor],
                  contexts:   List[PredictContext])
  : PredictionEx = new PredictionEx(prediction, tensors, contexts)

}
*/
/*
final class PredictionExSet(val predictions: IndexedSeq[PredictionEx])
  extends PredictionExLike {
  require(predictions != null)

  override def close(): Unit = dropIntermediates().dropOutput()

  override def input: TensorSet = TensorSet(predictions.map(_.input))

  override def reference: TensorSet = TensorSet(predictions.map(_.reference))

  override def output: TensorSet = TensorSet(predictions.map(_.output))

  def tensors: IndexedSeq[List[Tensor]] = predictions.map(_.tensors)

  def contexts: IndexedSeq[List[PredictContext]] = predictions.map(_.contexts)

  override def dropOutput(): Unit = predictions.foreach(_.dropOutput())

  override def dropIntermediates()
  : PredictionSet = PredictionSet(predictions.map(_.dropIntermediates()))

}

object PredictionExSet {

  final def apply(predictions: IndexedSeq[PredictionEx])
  : PredictionExSet = new PredictionExSet(predictions)

}
*/


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

abstract class TensorDomain
  extends Serializable

/**
  * Tensor Model
  *
  * |               BATCH                     |
  * | Sample 0    | Sample 1    | Sample 2    |
  * | RGB RGB RGB | RGB RGB RGB | RGB RGB RGB |
  *
  * The entire thing is a batch.
  * Each segment is a sample.
  * Each block of RGB is a tuple.
  * R, G and B are channels. R's are equidistant from each other.
  * Each sample in the above example has 9 units. Hence the batch consists of 9 unit vectors each having 3 elements.
  * Since there are 3 samples in the batch.the batch consists of 3 * 9 values.
  *
  */
object TensorDomain {

  /**
    * The operation is performed with the entire batch in mind.
    *
    * For constraints:
    * Performs no scaling. Costs add up, gradients are added as well.
    */
  case object Batch
    extends TensorDomain

  /**
    * The operation is performed separately for each sample.
    *
    * For constraints:
    * Average per sample of a batch. This is the default mode of Torch and other
    * frameworks.
    */
  case object Sample
    extends TensorDomain

  /**
    * The operation is performed separately for each channel per sample.
    *
    * Only supported in a few locations.
    */
  case object SampleChannel
    extends TensorDomain

  /**
    * The operation is performed across values belonging to the same channel.
    *
    * For constraints:
    * Average per channel.
    */
  case object Channel
    extends TensorDomain

  /**
    * The operation is applied separately across activation dimensions.
    *
    * For constraints:
    * Per neuron is the desirable measure if you want to introduce constraints
    * that like "avg. activation of each neuron in the layer". See ULFLDL
    * examples to see how such constraints can become useful.
    */
  case object Unit
    extends TensorDomain

  /**
    * This goes even further. Not supported in a lot locations. This will apply
    * the action on each instance of a value. Hence, each cell in the tensor
    * is considered as a separate context..
    */
  case object Value
    extends TensorDomain

}

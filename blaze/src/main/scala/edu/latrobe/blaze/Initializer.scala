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

package edu.latrobe.blaze

import edu.latrobe._

/**
 * Allows lazy weights initialization at a later time.
 */
abstract class Initializer
  extends InstanceEx[InitializerBuilder] {

  /**
    * Executes the weights initializer, and uses the provided buffer variable
    * for storing results.
    * @param module The context in which the weights are used. Use the handle field to get a human readable name that describes the nature of the weights.
    * @param weights The weights.
    * @param inputFanSize The average size of the input fan of each neuron.
    * @param outputFanSize The average size of the output fan of each neuron.
    *                      Note that this only represents the correct input fan
    *                      of the next layer if it has equivalent parameters.
    */
  def apply(layer:         Module,
            reference:     LabeledBufferReference,
            weights:       ValueTensor,
            inputFanSize:  Int,
            outputFanSize: Int)
  : Unit

}

abstract class InitializerBuilder
  extends InstanceExBuilder0[InitializerBuilder, Initializer] {
}

abstract class InitializerEx[TBuilder <: InitializerExBuilder[_]]
  extends Initializer {

  override def builder
  : TBuilder

}

abstract class InitializerExBuilder[TThis <: InitializerExBuilder[_]]
  extends InitializerBuilder {

  def repr
  : TThis

  override protected def doCopy()
  : TThis

}

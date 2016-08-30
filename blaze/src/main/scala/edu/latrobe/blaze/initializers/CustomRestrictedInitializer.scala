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

package edu.latrobe.blaze.initializers

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
  * A special initializer that allows restricting the initialization to certain
  * subsets of the parameter buffer. Use this if you intend to initialize
  * only a part of the
  */
final class CustomRestrictedInitializer(override val builder: CustomRestrictedInitializerBuilder,
                                        override val seed:    InstanceSeed)
  extends DependentInitializer[CustomRestrictedInitializerBuilder] {

  val filterFn
  : (Module, LabeledBufferReference) => Boolean = builder.filterFn

  override def apply(module:        Module,
                     reference:     LabeledBufferReference,
                     weights:       ValueTensor,
                     inputFanSize:  Int,
                     outputFanSize: Int)
  : Unit = {
    if (filterFn(module, reference)) {
      super.apply(
        module,
        reference,
        weights,
        inputFanSize,
        outputFanSize
      )
    }
  }

}

/**
  * Module handle, usage handle, parameter group no, weight group no.
  */
final class CustomRestrictedInitializerBuilder
  extends DependentInitializerBuilder[CustomRestrictedInitializerBuilder] {

  override def repr
  : CustomRestrictedInitializerBuilder = this

  private var _filterFn
  : (Module, LabeledBufferReference) => Boolean = {
    (moduleHandle, reference) => true
  }

  def filterFn
  : (Module, LabeledBufferReference) => Boolean = _filterFn

  def filterFn_=(value: (Module, LabeledBufferReference) => Boolean)
  : Unit = {
    require(value != null)
    _filterFn = value
  }

  def setFilterFn(value: (Module, LabeledBufferReference) => Boolean)
  : CustomRestrictedInitializerBuilder = {
    filterFn_=(value)
    this
  }

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _filterFn.hashCode())

  override protected def doToString()
  : List[Any] = _filterFn :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[CustomRestrictedInitializerBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: CustomRestrictedInitializerBuilder =>
      _filterFn == other._filterFn
    case _ =>
      false
  })

  override protected def doCopy()
  : CustomRestrictedInitializerBuilder = CustomRestrictedInitializerBuilder()

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: CustomRestrictedInitializerBuilder =>
        other._filterFn = _filterFn
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : CustomRestrictedInitializer = new CustomRestrictedInitializer(this, seed)

}

object CustomRestrictedInitializerBuilder {

  final def apply()
  : CustomRestrictedInitializerBuilder = new CustomRestrictedInitializerBuilder

  final def apply(source: InitializerBuilder)
  : CustomRestrictedInitializerBuilder = apply().setSource(source)

  final def apply(source:   InitializerBuilder,
                  filterFn: (Module, LabeledBufferReference) => Boolean)
  : CustomRestrictedInitializerBuilder = apply(
    source
  ).setFilterFn(filterFn)

}

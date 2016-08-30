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

final class RestrictedInitializer(override val builder: RestrictedInitializerBuilder,
                                  override val seed:    InstanceSeed)
  extends DependentInitializer[RestrictedInitializerBuilder] {

  val moduleHandle = builder.moduleHandle

  val referenceHandle = builder.referenceHandle

  override def apply(module:        Module,
                     reference:     LabeledBufferReference,
                     weights:       ValueTensor,
                     inputFanSize:  Int,
                     outputFanSize: Int)
  : Unit = {
    if (moduleHandle.isDefined && moduleHandle.get != module.handle) {
      return
    }
    if (referenceHandle.isDefined && referenceHandle.get != reference.handle) {
      return
    }
    super.apply(
      module,
      reference,
      weights,
      inputFanSize,
      outputFanSize
    )
  }

}

/**
  * Module handle, usage handle, parameter group no, weight group no.
  */
final class RestrictedInitializerBuilder
  extends DependentInitializerBuilder[RestrictedInitializerBuilder] {

  override def repr
  : RestrictedInitializerBuilder = this

  private var _moduleHandle
  : Option[String] = None

  def moduleHandle
  : Option[String] = _moduleHandle

  def moduleHandle_=(value: Option[String])
  : Unit = {
    require(value != null)
    _moduleHandle = value
  }

  def setModuleHandle(value: Option[String])
  : RestrictedInitializerBuilder = {
    moduleHandle_=(value)
    this
  }

  def setModuleHandle(value: String)
  : RestrictedInitializerBuilder = setModuleHandle(Option(value))

  private var _referenceHandle
  : Option[String] = None

  def referenceHandle
  : Option[String] = _referenceHandle

  def referenceHandle_=(value: Option[String])
  : Unit = {
    require(value != null)
    _referenceHandle = value
  }

  def setReferenceHandle(value: Option[String])
  : RestrictedInitializerBuilder = {
    referenceHandle_=(value)
    this
  }

  def setReferenceHandle(value: String)
  : RestrictedInitializerBuilder = setReferenceHandle(Option(value))

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _moduleHandle.hashCode())
    tmp = MurmurHash3.mix(tmp, _referenceHandle.hashCode())
    tmp
  }

  override protected def doToString()
  : List[Any] = _moduleHandle :: _referenceHandle :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RestrictedInitializerBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: RestrictedInitializerBuilder =>
      _moduleHandle    == other._moduleHandle &&
      _referenceHandle == other._referenceHandle
    case _ =>
      false
  })

  override protected def doCopy()
  : RestrictedInitializerBuilder = RestrictedInitializerBuilder()

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: RestrictedInitializerBuilder =>
        other._moduleHandle    = _moduleHandle
        other._referenceHandle = _referenceHandle
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : RestrictedInitializer = new RestrictedInitializer(this, seed)

}

object RestrictedInitializerBuilder {

  final def apply()
  : RestrictedInitializerBuilder = new RestrictedInitializerBuilder

  final def apply(source: InitializerBuilder)
  : RestrictedInitializerBuilder = apply().setSource(source)

  final def apply(source:          InitializerBuilder,
                  moduleHandle:    Option[String],
                  referenceHandle: Option[String])
  : RestrictedInitializerBuilder = apply(source).setModuleHandle(
    moduleHandle
  ).setReferenceHandle(referenceHandle)

}

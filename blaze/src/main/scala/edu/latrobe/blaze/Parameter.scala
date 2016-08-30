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

import java.io.OutputStreamWriter
import java.util.UUID
import edu.latrobe._
import edu.latrobe.blaze.parameters._

/**
 * Base class for infinite series' of parameter values. Each subsequent value
 * depends on update the notifications received. The actual state of a series
 * can be saved in and restored from an immutable state object.
 */
abstract class Parameter
  extends InstanceEx[ParameterBuilder] {

  /**
    * Must override with constructor argument.
    */
  def name
  : String

  final def render(phaseNo: Long)
  : String = {
    val sb = StringBuilder.newBuilder
    render(phaseNo, sb)
    sb.result()
  }

  final def render(phaseNo: Long, result: StringBuilder)
  : Unit = {
    result.append(name)
    result.append(" = ")
    result.append(f"${get(phaseNo)}%.4g")
  }

  final def render(phaseNo: Long, writer: OutputStreamWriter)
  : Unit = {
    writer.append(name)
    writer.append(" = ")
    writer.append(f"${get(phaseNo)}%.4g")
  }

  def get(phaseNo: Long)
  : Real

  final def get(phaseNo: Long, range: RealRange)
  : Real = range.clipAndWarn(get(phaseNo), name)

  def update(phaseNo: Long, value: Real)
  : Unit


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  final def toTuple
  : (UUID, Parameter) = (uniqueID, this)

}

/**
 * A parameter schedule is an immutable object that describes a infinite
 * value stream. This can be used in optimizer descriptions.
 */
abstract class ParameterBuilder
  extends InstanceExBuilder1[ParameterBuilder, Parameter, String] {

  override def build(name: String, seed: InstanceSeed)
  : Parameter

}

/*
object ParameterBuilder {

  /**
   * I know this is an exception of our rule to not access specifc items in sub
   * namespaces. Reasons: We need access to the constant value series for
   * default arguments. If you have a better idea to do this, go ahead implement
   * it and contribute back to the project! ~Thanks!
   */
  final def derive(value: Real)
  : ParameterBuilder = ConstantValueBuilder(value)

  final val zeros: ParameterBuilder = derive(Real.zero)

  final val ones: ParameterBuilder = derive(Real.one)
  
  final val defaultLearningRate: ParameterBuilder = derive(0.05f)

}
*/
abstract class ParameterEx[TBuilder <: ParameterExBuilder[_]]
  extends Parameter {

  override def builder
  : TBuilder

}

abstract class ParameterExBuilder[TThis <: ParameterExBuilder[_]]
  extends ParameterBuilder {

  override def repr
  : TThis

  override protected def doCopy()
  : TThis

}
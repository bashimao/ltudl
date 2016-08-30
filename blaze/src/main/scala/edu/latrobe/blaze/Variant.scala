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

trait VariantBuilder
  extends InstanceBuilder {

  // ---------------------------------------------------------------------------
  //   Variant preferences.
  // ---------------------------------------------------------------------------
  final private var _preferredPlatform
  : Option[Platform] = None

  final def preferredPlatform
  : Option[Platform] = _preferredPlatform

  final def preferredPlatform_=(value: Platform)
  : Unit = preferredPlatform_=(Option(value))

  final def preferredPlatform_=(value: Option[Platform])
  : Unit = {
    require(value != null)
    _preferredPlatform = value
  }

  def setPreferredPlatform(value: Platform)
  : VariantBuilder

  def setPreferredPlatform(value: Option[Platform])
  : VariantBuilder

  final private var _preferredLibrary
  : Option[String] = None

  final def preferredLibrary
  : Option[String] = _preferredLibrary

  final def preferredLibrary_=(value: String)
  : Unit = preferredLibrary_=(Option(value))

  final def preferredLibrary_=(value: Option[String])
  : Unit = _preferredLibrary = value.map(_.toUpperCase)

  def setPreferredLibrary(value: String)
  : VariantBuilder

  def setPreferredLibrary(value: Option[String])
  : VariantBuilder

  final private var _preferredMethod
  : Option[String] = None

  final def preferredMethod
  : Option[String] = _preferredMethod

  final def preferredMethod_=(value: String)
  : Unit = preferredMethod_=(Option(value))

  final def preferredMethod_=(value: Option[String])
  : Unit = _preferredMethod = value.map(_.toUpperCase)

  def setPreferredMethod(value: String)
  : VariantBuilder

  def setPreferredMethod(value: Option[String])
  : VariantBuilder

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _preferredPlatform.hashCode())
    tmp = MurmurHash3.mix(tmp, _preferredLibrary.hashCode())
    tmp = MurmurHash3.mix(tmp, _preferredMethod.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: VariantBuilderEx[_] =>
      _preferredPlatform == other._preferredPlatform &&
      _preferredLibrary  == other._preferredLibrary  &&
      _preferredMethod   == other._preferredMethod
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: VariantBuilderEx[_] =>
        other._preferredPlatform = _preferredPlatform
        other._preferredLibrary  = _preferredLibrary
        other._preferredMethod   = _preferredMethod
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Recursive mutable variables permutation.
  // ---------------------------------------------------------------------------
  /**
    * Recursive!
    */
  def permutePreferredPlatforms(fn: Option[Platform] => Option[Platform])
  : VariantBuilder

  protected def doPermutePreferredPlatforms(fn: Option[Platform] => Option[Platform])
  : Unit = preferredPlatform_=(fn(_preferredPlatform))

  /**
    * Recursive!
    */
  def permutePreferredLibraries(fn: Option[String] => Option[String])
  : VariantBuilder

  protected def doPermutePreferredLibraries(fn: Option[String] => Option[String])
  : Unit = preferredLibrary_=(fn(_preferredLibrary))

  /**
    * Recursive!
    */
  def permutePreferredMethods(fn: Option[String] => Option[String])
  : VariantBuilder

  protected def doPermutePreferredMethods(fn: Option[String] => Option[String])
  : Unit = preferredMethod_=(fn(_preferredMethod))

}

trait VariantBuilderEx[TThis <: VariantBuilderEx[_]]
  extends InstanceBuilder
    with VariantBuilder {

  override def repr
  : TThis


  // ---------------------------------------------------------------------------
  //   Variant preferences.
  // ---------------------------------------------------------------------------
  final override def setPreferredPlatform(value: Platform)
  : TThis = {
    preferredPlatform_=(value)
    repr
  }

  final override def setPreferredPlatform(value: Option[Platform])
  : TThis = {
    preferredPlatform_=(value)
    repr
  }

  final override def setPreferredLibrary(value: String)
  : TThis = {
    preferredLibrary_=(value)
    repr
  }

  final override def setPreferredLibrary(value: Option[String])
  : TThis = {
    preferredLibrary_=(value)
    repr
  }

  final override def setPreferredMethod(value: String)
  : TThis = {
    preferredMethod_=(value)
    repr
  }

  final override def setPreferredMethod(value: Option[String])
  : TThis = {
    preferredMethod_=(value)
    repr
  }


  // ---------------------------------------------------------------------------
  //    Recursive mutable variables permutation.
  // ---------------------------------------------------------------------------
  final override def permutePreferredPlatforms(fn: Option[Platform] => Option[Platform])
  : TThis = {
    doPermutePreferredPlatforms(fn)
    repr
  }

  final override def permutePreferredLibraries(fn: Option[String] => Option[String])
  : TThis = {
    preferredLibrary_=(fn(preferredLibrary))
    doPermutePreferredLibraries(fn)
    repr
  }


  final override def permutePreferredMethods(fn: Option[String] => Option[String])
  : TThis = {
    preferredMethod_=(fn(preferredMethod))
    doPermutePreferredMethods(fn)
    repr
  }


}

abstract class VariantDescription[TBuilder <: VariantBuilder] {

  final val (platform: Option[IndependentPlatform], libraryName: String, methodName: String) = {
    val parts = getClass.getSimpleName.split("_")
    val p     = IndependentPlatform.derive(parts(1))
    val ln    = parts(Math.min(2, parts.length - 1)) match {
      case "Description$" =>
        ""
      case _ =>
        parts(2).toUpperCase
    }
    val mn = parts(Math.min(3, parts.length - 1)) match {
      case "Description$" =>
        ""
      case _ =>
        parts(3).toUpperCase
    }
    (p, ln, mn)
  }

  final override def toString
  : String = {
    val builder = StringBuilder.newBuilder
    builder ++= platform.map(
      _.toString
    ).getOrElse("Generic")
    if (libraryName != "") {
      builder ++= s"_$libraryName"
    }
    if (methodName != "") {
      builder ++= s"_$methodName"
    }
    builder.result()
  }

  /**
    * Score is currently organized as as bitmask:
    *
    *  3            2             1            0
    * 1098 7654  3210 9876  5432 1098  7654 3210
    * free                       free
    *
    * Variant selected by affinity due to input preferences setup. (platform)
    * 27 = ?
    * 26 = Forced override by setup
    * 25 = prefer because local setup
    * 24 = prefer because build hints
    *
    * Variant selected by affinity due to input preferences setup. (library)
    * 23 = ?
    * 22 = Forced override by setup
    * 21 = prefer because local setup
    * 20 = prefer because build hints
    *
    * Variant selected by affinity due to input preferences setup. (method)
    * 19 = ?
    * 18 = Forced override by setup
    * 17 = prefer because local setup
    * 16 = prefer because build hints
    *
    * Variant selected by affinity due to actual data type observed.
    * 15 = prefer because input data type matches
    * 13 = prefer because JVM platform by default
    *
    * Basic priority:
    * 00 thru 07 = Base priority with which the implementation was registered.
    */
  final def baseScore(builder:  TBuilder,
                      priority: Byte,
                      reasons:  mutable.ArrayBuilder[String])
  : Int = {
    var result = priority.toInt

    // Platform.
    if (builder.preferredPlatform.exists(_ == platform)) {
      result |= 1 << 25
      reasons += "platform preference from builder"
    }

    // Library name.
    if (builder.preferredLibrary.exists(_ == libraryName)) {
      result |= 1 << 21
      reasons += "library preference from builder"
    }
    /*
    if (hints.preferredLibrary == libraryName) {
      result |= 1 << 20
      reason ++= "library preference from hints, "
    }
    */

    // Method name.
    if (builder.preferredMethod.exists(_ == methodName)) {
      result |= 1 << 17
      reasons += "method preference from builder"
    }
    /*
    if (hints.preferredMethod == methodName) {
      result |= 1 << 16
      reason ++= "method preference from hints, "
    }
    */

    /*
    if (platform.isEmpty) {
      result |= 1 << 13
      reason ++= "generic implementation, "
    }
    */

    result
  }

}

abstract class VariantTable[TBuilder <: VariantBuilder, TDescription <: VariantDescription[TBuilder]] {

  final private val _variants
  : mutable.Map[TDescription, Byte] = mutable.Map.empty

  final def variants
  : Map[TDescription, Byte] = _variants

  /**
    * Registers a constructor with in the look up table. The first variant seen
    * will automatically become the default for the respective platform/library.
    */
  final def register(priority: Int, description: TDescription)
  : Unit = {
    require(priority >= 0 && priority <= Byte.MaxValue)
    _variants += Tuple2(description, priority.toByte)
  }

  final def unregister(description: TDescription)
  : Unit = _variants -= description

  final def unregisterAll()
  : Unit = _variants.clear()

}

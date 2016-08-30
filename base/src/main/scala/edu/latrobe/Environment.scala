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

package edu.latrobe

/**
  * Environment variables.
  */
object Environment {

  final val values: Map[String, String] = sys.env

  final def parseBoolean(name:    String,
                         default: Boolean,
                         format:  Boolean => String = if (_) "yes" else "no")
  : Boolean = {
    val value = values.get(name).map({
      case "1" =>
        true
      case "true" =>
        true
      case "yes" =>
        true
      case "enable" =>
        true
      case "enabled" =>
        true
      case _ =>
        false
    }).getOrElse(default)
    if (value == default) {
      logger.info(s"$name = ${format(value)} (default)")
    }
    else {
      logger.info(s"$name = ${format(value)}")
    }
    value
  }

  final def parseInt(name:    String,
                     default: Int,
                     testFn:  Int => Boolean,
                     format:  Int => String = _.toString)
  : Int = {
    val value = values.get(name).map(_.toInt).getOrElse(default)
    val logStr = {
      if (value == default) {
        s"$name = ${format(value)} (default)"
      }
      else {
        s"$name = ${format(value)}"
      }
    }
    if (testFn(value)) {
      logger.info(logStr)
    }
    else {
      logger.error(logStr)
      throw new EnvironmentVariableOutOfRangeError(name, value, default)
    }
    value
  }

  final def parseLong(name:    String,
                      default: Long,
                      testFn:  Long => Boolean,
                      format:  Long => String = _.toString)
  : Long = {
    val value = values.get(name).map(_.toLong).getOrElse(default)
    val logStr = {
      if (value == default) {
        s"$name = ${format(value)} (default)"
      }
      else {
        s"$name = ${format(value)}"
      }
    }
    if (testFn(value)) {
      logger.info(logStr)
    }
    else {
      logger.error(logStr)
      throw new EnvironmentVariableOutOfRangeError(name, value, default)
    }
    value
  }

  final def parseReal(name:    String,
                      default: Real,
                      testFn:  Real => Boolean,
                      format:  Real => String = x => f"$x%.4g")
  : Real = {
    val value = values.get(name).map(Real(_)).getOrElse(default)
    val logStr = {
      if (value == default) {
        s"$name = ${format(value)} (default)"
      }
      else {
        s"$name = ${format(value)}"
      }
    }
    if (testFn(value)) {
      logger.info(logStr)
    }
    else {
      logger.error(logStr)
      throw new EnvironmentVariableOutOfRangeError(name, value, default)
    }
    value
  }


  final def get(name:    String,
                default: String)
  : String = get(name, default, x => true)

  final def get(name:    String,
                default: String,
                testFn:  String => Boolean)
  : String = {
    val value = values.getOrElse(name, default)
    val logStr = {
      if (value == default) {
        s"$name = '$value' (default)"
      }
      else {
        s"$name = '$value'"
      }
    }
    if (testFn(value)) {
      logger.info(logStr)
    }
    else {
      logger.error(logStr)
      throw new EnvironmentVariableOutOfRangeError(name, value, default)
    }
    value
  }

}

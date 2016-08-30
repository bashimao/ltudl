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

import java.nio.charset._

object StringEx {

  @inline
  final def merge(parts: TraversableOnce[String])
  : String = {
    val builder = StringBuilder.newBuilder
    parts.foreach(
      builder ++= _
    )
    builder.result()
  }

  @inline
  final def render(array: Array[Byte])
  : String = render(array, StandardCharsets.US_ASCII)

  @inline
  final def render(array: Array[Byte], charset: Charset)
  : String = new String(array, charset)

  @inline
  final def render(value: Int)
  : String = render(value, 1000)

  @inline
  final def render(value: Int, base: Int)
  : String = render(value, base, "%.1f %s")

  @inline
  final def render(value: Int, base: Int, format: String)
  : String = {
    require(value >= 0)

    val postfix = Array('G', 'M', 'k')
    val base2   = base / 2L

    var limit = base * base
    var i     = 0
    while (i < postfix.length) {
      if (value >= limit * base2) {
        val m = value.toDouble / (limit * base)
        return format.format(m, postfix(i))
      }
      limit /= base
      i     += 1
    }
    value.toString
  }

  @inline
  final def render(value: Long)
  : String = render(value, 1000L)

  @inline
  final def render(value: Long, base: Long)
  : String = render(value, base, "%.1f %s")

  @inline
  final def render(value: Long, base: Long, format: String)
  : String = {
    require(value >= 0)

    val postfix = Array('E', 'P', 'T', 'G', 'M', 'k')
    val base2   = base / 2L

    var limit = base * base * base * base * base
    var i     = 0
    while (i < postfix.length) {
      if (value >= limit * base2) {
        val m = value.toDouble / (limit * base)
        return format.format(m, postfix(i))
      }
      limit /= base
      i     += 1
    }
    value.toString
  }

}

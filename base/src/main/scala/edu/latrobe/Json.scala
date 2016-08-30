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

import org.apache.commons.codec.binary._
import org.json4s.JsonAST._
import scala.collection._

object Json {

  @inline
  final def apply(value: Boolean)
  : JBool = JBool(value)

  @inline
  final def apply(value: Byte)
  : JInt = JInt(BigInt(value))

  @inline
  final def apply(value: Double)
  : JDouble = JDouble(value)

  @inline
  final def apply(value: Float)
  : JDouble = JDouble(value.toDouble)

  @inline
  final def apply(value: Int)
  : JInt = JInt(BigInt(value))

  @inline
  final def apply(value: Long)
  : JInt = JInt(BigInt(value))

  @inline
  final def apply(value: Null)
  : JNull.type = JNull

  @inline
  final def apply(value: Short)
  : JInt = JInt(BigInt(value))

  @inline
  final def apply(value: String)
  : JString = JString(value)

  @inline
  final def apply(src0: Array[Boolean])
  : JArray = {
    val result = List.newBuilder[JBool]
    ArrayEx.foreach(
      src0
    )(result += apply(_))
    apply(result.result())
  }

  @inline
  final def apply(src0: Array[Byte])
  : JString = apply(Base64.encodeBase64String(src0))

  @inline
  final def apply(src0: Array[Array[Byte]])
  : JArray = {
    val result = List.newBuilder[JString]
    result.sizeHint(src0.length)
    ArrayEx.foreach(
      src0
    )(result += apply(_))
    apply(result.result())
  }

  @inline
  final def apply(src0: Array[Double])
  : JArray = {
    val result = List.newBuilder[JDouble]
    ArrayEx.foreach(
      src0
    )(result += apply(_))
    apply(result.result())
  }

  @inline
  final def apply(src0: Array[Float])
  : JArray = {
    val result = List.newBuilder[JDouble]
    ArrayEx.foreach(
      src0
    )(result += apply(_))
    apply(result.result())
  }

  @inline
  final def apply(src0: Array[Int])
  : JArray = {
    val result = List.newBuilder[JInt]
    ArrayEx.foreach(
      src0
    )(result += apply(_))
    apply(result.result())
  }

  @inline
  final def apply(src0: Array[Long])
  : JArray = {
    val result = List.newBuilder[JInt]
    ArrayEx.foreach(
      src0
    )(result += apply(_))
    apply(result.result())
  }

  @inline
  final def apply(src0: Array[Short])
  : JArray = {
    val result = List.newBuilder[JInt]
    ArrayEx.foreach(
      src0
    )(result += apply(_))
    apply(result.result())
  }

  @inline
  final def apply(src0: Array[String])
  : JArray = {
    val result = List.newBuilder[JString]
    ArrayEx.foreach(
      src0
    )(result += apply(_))
    apply(result.result())
  }

  @inline
  final def apply(src0: Array[JArray])
  : JArray = apply(src0.toList)

  @inline
  final def apply(src0: Array[JBool])
  : JArray = apply(src0.toList)

  @inline
  final def apply(src0: Array[JInt])
  : JArray = apply(src0.toList)

  @inline
  final def apply(src0: Array[JDouble])
  : JArray = apply(src0.toList)

  @inline
  final def apply(src0: Array[JObject])
  : JArray = apply(src0.toList)

  @inline
  final def apply(src0: Array[JString])
  : JArray = apply(src0.toList)

  @inline
  final def apply(src0: Array[JValue])
  : JArray = apply(src0.toList)

  @inline
  final def apply[T <: JsonSerializable](src0: Array[T])
  : JArray = {
    val result = List.newBuilder[JValue]
    ArrayEx.foreach(
      src0
    )(result += _.toJson)
    apply(result.result())
  }

  @inline
  final def apply(src0: List[JValue])
  : JArray = JArray(src0)

  @inline
  final def derive(src0: Map[Int, JValue])
  : JArray = {
    val builder = List.newBuilder[JObject]
    MapEx.foreach(src0)((k, v) => {
      builder += JObject(
        Json.field("k", apply(k)),
        Json.field("v", v)
      )
    })
    apply(builder.result())
  }

  /*
  @inline
  final def apply(value: List[JValue])
  : JArray = JArray(value)
  */

  @inline
  final def field(name: String, value: Boolean)
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: Byte)
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: Double)
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: Float)
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: Int)
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: Long)
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: Short)
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: String)
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: List[JValue])
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: Array[Byte])
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: Array[Array[Byte]])
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: Array[Double])
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: Array[Float])
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: Array[Int])
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: Array[Long])
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, value: Array[String])
  : JField = JField(name, apply(value))

  /*
  @inline
  final def field[T <: JValue](name: String, fields: Array[T])
  : JField = field(name, fields.toList)
  */

  @inline
  final def field[T <: JsonSerializable](name: String, value: Array[T])
  : JField = JField(name, apply(value))

  @inline
  final def field(name: String, obj0: JsonSerializable)
  : JField = JField(name, obj0.toJson)

  @inline
  final def field(name: String, obj0: JValue)
  : JField = JField(name, obj0)

  @inline
  final def field(name: String, field: JField)
  : JField = JField(name, JObject(field))

  @inline
  final def field(name: String, field0: JField, fields: JField*)
  : JField = field(name, JObject(field0 :: fields.toList))

  @inline
  final def toBoolean(json: JValue)
  : Boolean = toBoolean(json.asInstanceOf[JBool])

  @inline
  final def toBoolean(json: JBool)
  : Boolean = json.value

  @inline
  final def toByte(json: JValue)
  : Byte = toByte(json.asInstanceOf[JInt])

  @inline
  final def toByte(json: JInt)
  : Byte = json.num.toByte

  @inline
  final def toDouble(json: JValue)
  : Double = toDouble(json.asInstanceOf[JDouble])

  @inline
  final def toDouble(json: JDouble)
  : Double = json.num

  @inline
  final def toFloat(json: JValue)
  : Float = toFloat(json.asInstanceOf[JDouble])

  @inline
  final def toFloat(json: JDouble)
  : Float = json.num.toFloat

  @inline
  final def toInt(json: JValue)
  : Int = toInt(json.asInstanceOf[JInt])

  @inline
  final def toInt(json: JInt)
  : Int = json.num.toInt

  @inline
  final def toLong(json: JValue)
  : Long = toLong(json.asInstanceOf[JInt])

  @inline
  final def toLong(json: JInt)
  : Long = json.num.toLong

  @inline
  final def toNull(json: JValue)
  : Null = toNull(json.asInstanceOf[JNull.type])

  @inline
  final def toNull(json: JNull.type)
  : Null = null

  @inline
  final def toReal(json: JValue)
  : Real = toReal(json.asInstanceOf[JDouble])

  @inline
  final def toReal(json: JDouble)
  : Real = Real(json.num)

  @inline
  final def toShort(json: JValue)
  : Short = toShort(json.asInstanceOf[JInt])

  @inline
  final def toShort(json: JInt)
  : Short = json.num.toShort

  @inline
  final def toString(json: JValue)
  : String = toString(json.asInstanceOf[JString])

  @inline
  final def toString(json: JString)
  : String = json.s

  @inline
  final def toByteArray(json: JValue)
  : Array[Byte] = toByteArray(json.asInstanceOf[JString])

  @inline
  final def toByteArray(json: JString)
  : Array[Byte] = Base64.decodeBase64(json.s)

  @inline
  final def toDoubleArray(json: JValue)
  : Array[Double] = toDoubleArray(json.asInstanceOf[JArray])

  @inline
  final def toDoubleArray(json: JArray)
  : Array[Double] = {
    val builder = Array.newBuilder[Double]
    json.arr.foreach(
      builder += toDouble(_)
    )
    builder.result()
  }

  @inline
  final def toFloatArray(json: JValue)
  : Array[Float] = toFloatArray(json.asInstanceOf[JArray])

  @inline
  final def toFloatArray(json: JArray)
  : Array[Float] = {
    val builder = Array.newBuilder[Float]
    json.arr.foreach(
      builder += toFloat(_)
    )
    builder.result()
  }

  @inline
  final def toRealArray(json: JValue)
  : Array[Real] = toRealArray(json.asInstanceOf[JArray])

  @inline
  final def toRealArray(json: JArray)
  : Array[Real] = {
    val builder = Array.newBuilder[Real]
    json.arr.foreach(
      builder += toReal(_)
    )
    builder.result()
  }

  @inline
  final def toSortedMap[T, U](json: JValue, kFn: JValue => T, vFn: JValue => U)
                             (implicit orderT: Ordering[T])
  : SortedMap[T, U] = toSortedMap(json.asInstanceOf[JArray], kFn, vFn)

  @inline
  final def toSortedMap[T, U](json: JArray, kFn: JValue => T, vFn: JValue => U)
                             (implicit orderT: Ordering[T])
  : SortedMap[T, U] = {
    val builder = SortedMap.newBuilder[T, U]
    json.arr.foreach(kv => {
      val fields = kv.asInstanceOf[JObject].obj.toMap
      val k = kFn(fields("k"))
      val v = vFn(fields("v"))
      builder += Tuple2(k, v)
    })
    builder.result()
  }

}

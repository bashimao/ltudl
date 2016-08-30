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

import edu.latrobe.sizes._
import org.json4s.JsonAST._
import scala.collection._
import scala.reflect._
import scala.util.hashing._

final class TensorLayoutTable(private val layouts: Array[TensorLayout])
  extends DependentTensorLayout
    with TableLike[TensorLayout] {
  require(!ArrayEx.contains(layouts, null))

  override def toString
  : String = s"Table[${layouts.length}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), ArrayEx.hashCode(layouts))

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[TensorLayoutTable]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: TensorLayoutTable =>
      ArrayEx.compare(
        layouts,
        other.layouts
      )
    case _ =>
      false
  })

  override def length
  : Int = layouts.length

  override def getEntry(index: Int)
  : TensorLayout = {
    val arrayIndex = {
      if (index >= 0) {
        index
      }
      else {
        layouts.length + index
      }
    }
    require(arrayIndex >= 0 && arrayIndex < layouts.length)
    layouts(arrayIndex)
  }

  override def iterator
  : Iterator[TensorLayout] = layouts.iterator


  override def size
  : Size1 = Size1(1, noValues)

  override def noSamples
  : Int = 1

  override def noTuples
  : Int = foldLeft(0)(_ + _.noTuples)

  override def noValues
  : Int = foldLeft(0)(_ + _.noValues)

  def foreach(fn: TensorLayout => Unit)
  : Unit = ArrayEx.foreach(
    layouts
  )(fn)

  def foreachPair(fn: (Int, TensorLayout) => Unit)
  : Unit = ArrayEx.foreachPair(
    layouts
  )(fn)

  def foldLeft[T](z0: T)(fn: (T, TensorLayout) => T)
  : T = ArrayEx.foldLeft(
    z0,
    layouts
  )(fn)

  def foldLeftPair[T](z0: T)(fn: (T, Int, TensorLayout) => T)
  : T = ArrayEx.foldLeftPairs(
    z0,
    layouts
  )(fn)

  def map[T](fn: TensorLayout => T)
            (implicit tagT: ClassTag[T])
  : Array[T] = ArrayEx.map(
    layouts
  )(fn)

  def mapPairs[T](fn: (Int, TensorLayout) => T)
                 (implicit tagT: ClassTag[T])
  : Array[T] = ArrayEx.mapPairs(
    layouts
  )(fn)

  def zip[T](other: TensorLayout,
             fn:    (TensorLayout, TensorLayout) => T)
            (implicit tagT: ClassTag[T])
  : Array[T] = other match {
    case other: TensorLayoutTable =>
      ArrayEx.zip(
        layouts,
        other.layouts
      )(fn)
    case _ =>
      ArrayEx.map(
        layouts
      )(fn(_, other))
  }

  def zipPairs[T](other: TensorLayout,
                  fn:    (Int, TensorLayout, TensorLayout) => T)
                 (implicit tagT: ClassTag[T])
  : Array[T] = other match {
    case other: TensorLayoutTable =>
      ArrayEx.zipPairs(
        layouts,
        other.layouts
      )(fn)
    case _ =>
      ArrayEx.mapPairs(
        layouts
      )(fn(_, _, other))
  }

  override def concat(other: TensorLayout)
  : TensorLayoutTable = TensorLayoutTable(zip(other, _.concat(_)))

  override def concat(others: Array[TensorLayout])
  : TensorLayoutTable = ArrayEx.foldLeft(
    this,
    others
  )(_.concat(_))

  override def concat(others: Traversable[TensorLayout])
  : TensorLayoutTable = others.foldLeft(
    this
  )(_.concat(_))

  override def sampleFor(offset: Int)
  : Int = {
    require(offset >= 0 && offset < noValues)
    0
  }

  override def offsetFor(sampleNo: Int)
  : Int = {
    require(sampleNo == 0)
    0
  }

  override def offsetFor(sampleNo: Int, valueNo: Int)
  : Int = {
    require(sampleNo == 0 && valueNo >= 0 && valueNo < noValues)
    valueNo
  }

  override def ++(other: TensorLayout)
  : TensorLayoutTable = TensorLayoutTable(zip(other, _ ++ _))

  override def :++(other: TensorLayout)
  : TensorLayoutTable = TensorLayoutTable(zip(other, _ :++ _))

  override def makeIndependent
  : IndependentTensorLayout = IndependentTensorLayout(size, noSamples)


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  override protected def doToJson()
  : List[JField] = List(
    Json.field("layouts", layouts)
  )

  override def toEdgeLabel
  : String = {
    val builder = StringBuilder.newBuilder
    ArrayEx.foreach(
      layouts
    )(layout => builder ++= "  " ++= layout.toEdgeLabel ++= "\n")
    if (builder.nonEmpty) {
      builder.length = builder.length - 1
    }
    builder.result()
  }

}

object TensorLayoutTable {

  final def apply(layouts: Array[TensorLayout])
  : TensorLayoutTable = new TensorLayoutTable(layouts)

  final def derive(layout0: TensorLayout)
  : TensorLayoutTable = apply(Array(layout0))

  final def derive(layout0: TensorLayout,
                   layouts: TensorLayout*)
  : TensorLayoutTable = apply((layout0 :: layouts.toList).toArray)

  final def derive(layouts: Seq[TensorLayout])
  : TensorLayoutTable = apply(layouts.toArray)

}

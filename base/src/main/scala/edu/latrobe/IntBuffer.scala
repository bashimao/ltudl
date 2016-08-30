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

import scala.collection._
import scala.util.hashing._

final class IntBuffer(override val banks: SortedMap[Int, IntBank])
  extends BufferEx[IntBuffer, IntBank, Int] {

  override def toString
  : String = s"IntBuffer[${banks.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), banks.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[IntBuffer]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: IntBuffer =>
      banks == other.banks
    case _ =>
      false
  })

  def sum
  : Long = {
    MapEx.foldLeftValues(
      0L,
      banks
    )(_ + _.sum)
  }

  def +(value: Int)
  : IntBuffer = {
    val result = mapBanks(
      _ + value
    )
    IntBuffer(result)
  }

  def +(other: IntBuffer)
  : IntBuffer = {
    val result = zipBanksEx(
      other
    )((b0, b1) => b0 + b1, b0 => b0, b1 => b1)
    IntBuffer(result)
  }

  def -(value: Int)
  : IntBuffer = {
    val result = mapBanks(
      _ - value
    )
    IntBuffer(result)
  }

  def -(other: IntBuffer)
  : IntBuffer = {
    val result = zipBanksEx(
      other
    )((b0, b1) => b0 - b1, b0 => b0, b1 => b1)
    IntBuffer(result)
  }

  def partitionSegments(noBucketsMax: Int)
  : Seq[Seq[((Int, Int), Int)]] = {
    val buckets     = mutable.Buffer.empty[mutable.Buffer[((Int, Int), Int)]]
    val segmentList = segments.toSeq.sortWith(_._2 > _._2)
    for(segment <- segmentList) {
      // If have not used all buckets yet.
      if (buckets.length < noBucketsMax) {
        val bucket = mutable.Buffer(segment)
        buckets += bucket
      }
      else {
        // Find bucket with lowest amount and insert there.
        val bucket = buckets.minBy(_.map(_._2).sum)
        bucket += segment
      }
    }
    buckets
  }

  def partitionSegmentsIntoLimitedSizeBuckets(maxBucketSize: Int)
  : Seq[Seq[((Int, Int), Int)]] = {
    val buckets     = mutable.Buffer.empty[mutable.Buffer[((Int, Int), Int)]]
    val segmentList = segments.toSeq.sortWith(_._2 > _._2)
    for(segment <- segmentList) {
      // Try to insert into existing bucket.
      var inserted = false
      val iter = buckets.iterator
      while(iter.hasNext && !inserted) {
        val bucket = iter.next()
        val size = bucket.map(_._2).sum
        if (size + segment._2 <= maxBucketSize) {
          bucket += segment
          inserted = true
        }
      }

      // No Bucket fits. Let's create a new bucket.
      if (!inserted) {
        val bucket = mutable.Buffer(segment)
        buckets += bucket
      }
    }
    buckets
  }


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, IntBank])
  : IntBuffer = IntBuffer(banks)

}

object IntBuffer {

  final def apply(banks: SortedMap[Int, IntBank])
  : IntBuffer = new IntBuffer(banks)

  final val empty
  : IntBuffer = apply(SortedMap.empty)

  final def fillLike(buffer: BufferLike,
                     value:  Int)
  : IntBuffer = {
    val result = MapEx.mapValues(
      buffer.banks
    )(IntBank.fillLike(_, value))
    apply(result)
  }

  final def zeroLike(buffer: BufferLike)
  : IntBuffer = fillLike(buffer, 0)

}

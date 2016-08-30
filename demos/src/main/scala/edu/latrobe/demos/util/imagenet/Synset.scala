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

package edu.latrobe.demos.util.imagenet

import edu.latrobe._
import scala.util.hashing._

/**
 * Describes a synonym ring in wordnet. In ImageNet 2014, IDs below [1, 1000]
 * are main sets. Others, are subsets.
 *
 * IDs are one based!
 */
final class Synset(val imageNetID:    Int,
                   val wordNetID:     String,
                   val words:         Array[String],
                   val gloss:         Array[String],
                   val childIDs:      Array[Int],
                   val wordNetHeight: Int,
                   val noTrainImages: Int)
  extends Serializable
    with Equatable {
  require(
    imageNetID >= 0 &&
    wordNetID.length > 0 &&
    words.length > 0 && !ArrayEx.contains(words, null) &&
    gloss != null &&
    childIDs != null &&
    wordNetHeight >= 0 &&
    noTrainImages >= 0
  )

  override def toString
  : String = {
    s"Synset[$imageNetID, $wordNetID, ${words.length}, ${gloss.length}, ${childIDs.length}, $wordNetHeight, $noTrainImages]"
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Annotation]

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, imageNetID.hashCode())
    tmp = MurmurHash3.mix(tmp, wordNetID.hashCode())
    tmp = MurmurHash3.mix(tmp, ArrayEx.hashCode(words))
    tmp = MurmurHash3.mix(tmp, ArrayEx.hashCode(gloss))
    tmp = MurmurHash3.mix(tmp, ArrayEx.hashCode(childIDs))
    tmp = MurmurHash3.mix(tmp, wordNetHeight.hashCode())
    tmp = MurmurHash3.mix(tmp, noTrainImages.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Synset =>
      imageNetID == other.imageNetID            &&
      wordNetID == other.wordNetID              &&
      ArrayEx.compare(words, other.words)       &&
      ArrayEx.compare(gloss, other.gloss)       &&
      ArrayEx.compare(childIDs, other.childIDs) &&
      wordNetHeight == other.wordNetHeight      &&
      noTrainImages == other.noTrainImages
    case _ =>
      false
  })

}

object Synset {

  final def apply(imageNetID:    Int,
                  wordNetID:     String,
                  words:         Array[String],
                  gloss:         Array[String],
                  childIDs:      Array[Int],
                  wordNetHeight: Int,
                  noTrainImages: Int)
  : Synset = new Synset(
    imageNetID,
    wordNetID,
    words,
    gloss,
    childIDs,
    wordNetHeight,
    noTrainImages
  )

}

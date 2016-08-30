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

import edu.latrobe.time._

/**
  * Minor extension of the equals interface. Will ensure that the compiler
  * reminds us to override canEqual and doEquals and replaces the default hash
  * value.
  */
trait Equatable extends Equals {

  /**
    * Remark: Make sure this one executes fast!
    *
    * Detaches the hash code from object identity, which is usually desirable.
    * I prefer this kind of a abstract implementation for hash code. Firstly of
    * the hash code is already detached from the object identity this way.
    *
    * @return Returns hashPrime
   */
  override def hashCode()
  : Int = hashSeed

  final override def equals(obj: Any)
  : Boolean = obj match {
    case obj: Equatable =>
      if (obj eq this) {
        true
      }
      else {
        canEqual(obj) && doEquals(obj)
      }
    case _ =>
      false
  }

  /**
    * Remark: Make sure this one is versatile!
    */
  protected def doEquals(other: Equatable)
  : Boolean = true

}

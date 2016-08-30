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

trait Closable
  extends AutoCloseable {

  final override def close()
  : Unit = synchronized {
    doClose(false)
  }

  @transient
  final private var _closed
  : Boolean = false

  final def closed
  : Boolean = _closed

  /**
    * Deallocates claimed resources. Implement this in a way so that multiple
    * calls to deallocate have no negative side effects, since it may be called
    * manually and by a finalizer.
    *
    * The behavior of of an object after calling deallocate is unspecified.
    */
  protected def doClose()
  : Unit = {}

  final protected def doClose(finalizing: Boolean)
  : Unit = {
    if (_closed) {
      if (!finalizing) {
        if (LTU_REDUNDANT_CALL_TO_CLOSE_WARNING) {
          logger.warn("Tried to explicitly close an already closed Closable!")
        }
      }
    }
    else {
      doClose()
      // In .NET we would take the object off the finalization queue here. In the JVM... Well... I guess we are screwed...
      _closed = true
    }
  }

  final def tryClose()
  : Boolean = synchronized {
    if (!_closed) {
      doClose(false)
      true
    }
    else {
      false
    }
  }

}

trait ClosableEx
  extends Closable {

  @transient
  final private var _noReferences
  : Long = 0L

  final def noReferences
  : Long = _noReferences

  /**
    * Allows transferring the ownership of this object, by increasing the ref-counter.
    */
  final private[latrobe] def incrementReferenceCount()
  : Unit = synchronized {
    require(!closed)
    _noReferences += 1L
  }

  final private[latrobe] def decrementReferenceCount()
  : Unit = synchronized {
    _noReferences -= 1L
    assume(_noReferences >= 0L)
    if (_noReferences == 0L) {
      doClose(false)
    }
  }

}

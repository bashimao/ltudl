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
  * An object that supports deallocating internal structures. Some objects may
  * require this in order to not leak memory, others do not.
  *
  * Any object that allocates native resources must inherit from Deallocatable.
  *
  * Any object containing other deallocatable objects should also implement
  * Deallocatable.
  *
  * However, only the objects that actually did the allocation should also call
  * Deallocate in its destructor.
  */
trait AutoClosing
  extends Closable {

  final override def finalize()
  : Unit = {
    doClose(true)
    super.finalize()
  }

}

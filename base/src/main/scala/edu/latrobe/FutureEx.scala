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
import scala.concurrent._
import scala.concurrent.duration._

/**
  * This way we can avoid littering the namespace with duration that anyway
  * conflicts with Joda.
  */
object FutureEx {

  @inline
  final def finish[T](future: Future[T])
  : Unit = Await.ready(future, Duration.Inf)

  @inline
  final def get[T](future: Future[T])
  : T = Await.result(future, Duration.Inf)

  @inline
  final def getEx[T](future: Future[T],
                     time:   TimeSpan)
  : Option[T] = {
    val duration = Duration.apply(time.getMillis, MILLISECONDS)
    try {
      val result = Await.result(future, duration)
      Some(result)
    }
    catch {
      case e: TimeoutException =>
        // Intentionally do nothing.
        None
    }
  }

  @inline
  final def getEx[T](future:     Future[T],
                     time:       TimeSpan,
                     callbackFn: => Unit)
  : T = {
    var tmp = getEx(future, time)
    while (tmp.isEmpty) {
      callbackFn
      tmp = getEx(future, time)
    }
    tmp.get
  }

}

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

package edu.latrobe.io

import edu.latrobe._
import org.apache.http._
import org.apache.http.client.methods._
import org.apache.http.impl.client._
import org.apache.http.impl.conn._

final class HttpClient
  extends AutoClosing {

  /*
  For HttpClient >= 4.3

  private val connectionManager
  : PoolingHttpClientConnectionManager = {
    new PoolingHttpClientConnectionManager()
  }

  private val httpClient
  : CloseableHttpClient = {
    val builder = HttpClientBuilder.create()
    builder.setConnectionManager(connectionManager)
    builder.build()
  }

  override protected def doClose(reason: CloseReason)
  : Unit = {
    httpClient.close()
    connectionManager.close()
    super.doClose(reason)
  }

  def execute(request: HttpUriRequest)
  : CloseableHttpResponse = httpClient.execute(request)
  */

  private val connectionManager
  : PoolingClientConnectionManager = {
    new PoolingClientConnectionManager()
  }

  private val httpClient
  : DefaultHttpClient = new DefaultHttpClient(connectionManager)
  //httpClient.getParams.setParameter("http.connection-manager.timeout", 2000)

  override protected def doClose()
  : Unit = {
    connectionManager.shutdown()
    super.doClose()
  }

  def execute(request: HttpUriRequest)
  : HttpResponse = httpClient.execute(request)

}

object HttpClient {

  final def apply(): HttpClient = new HttpClient

  final val default: HttpClient = apply()

}
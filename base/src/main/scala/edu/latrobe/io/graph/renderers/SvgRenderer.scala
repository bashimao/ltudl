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

package edu.latrobe.io.graph.renderers

import edu.latrobe._
import edu.latrobe.io.graph._
import java.io._

object SvgRenderer
  extends GraphRenderer {

  override def render(graph: Graph, stream: OutputStream)
  : Unit = render(graph, new OutputStreamWriter(stream))

  override def render(graph: Graph, writer: Writer)
  : Unit = {
    val dot = DotRenderer.render(graph)
    try {
      val p = Runtime.getRuntime.exec("dot -Tsvg")

      using(
        new BufferedWriter(new OutputStreamWriter(p.getOutputStream)),
        new BufferedReader(new InputStreamReader(p.getInputStream))
      )((stdIn, stdOut) => {
        stdIn.write(dot)
        stdIn.flush()
        stdIn.close()

        var line = ""
        while(line != null) {
          writer.write(line)
          line = stdOut.readLine()
        }
        writer.flush()
      })
    }
    catch {
      case e: Exception =>
        logger.error("Error in SvgRenderer", e)
    }
  }

}

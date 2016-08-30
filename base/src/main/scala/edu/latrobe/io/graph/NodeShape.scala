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

package edu.latrobe.io.graph

abstract class NodeShape
  extends Serializable {
}

object NodeShape {

  case object Box
    extends NodeShape {
  }

  case object Circle
    extends NodeShape {
  }

  case object Diamond
    extends NodeShape {
  }

  case object Ellipse
    extends NodeShape {
  }

  case object Hexagon
    extends NodeShape {
  }

  case object Octagon
    extends NodeShape {
  }

  case object Parallelogram
    extends NodeShape {
  }

  case object Point
    extends NodeShape {
  }

  case object RoundedBox
    extends NodeShape {
  }

  case object Triangle
    extends NodeShape {
  }

}

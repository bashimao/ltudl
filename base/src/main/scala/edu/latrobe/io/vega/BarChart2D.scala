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

package edu.latrobe.io.vega

import edu.latrobe._
import org.json4s.JsonAST._

final class BarChart2D
  extends Chart2D[BarChart2D] {

  override def repr
  : BarChart2D = this

  var xAxisOffset
  : Int = 5

  def setXAxisOffset(value: Int)
  : BarChart2D = {
    xAxisOffset_=(value)
    this
  }

  private var _spaceBetweenBars
  : Int = 12

  def spaceBetweenBars
  : Int = _spaceBetweenBars

  def spaceBetweenBars_=(value: Int)
  : Unit = {
    require(value >= 0)
    _spaceBetweenBars = value
  }

  def setSpaceBetweenBars(value: Int)
  : BarChart2D = {
    spaceBetweenBars_=(value)
    this
  }

  private var _spaceBetweenCharts
  : Int = 12

  def spaceBetweenCharts
  : Int = _spaceBetweenCharts

  def spaceBetweenCharts_=(value: Int)
  : Unit = {
    require(value >= 0)
    _spaceBetweenCharts = value
  }

  def setSpaceBetweenCharts(value: Int)
  : BarChart2D = {
    spaceBetweenCharts_=(value)
    this
  }

  private var _yAxisFormat
  : String = "linear"

  def yAxisFormat
  : String = _yAxisFormat

  def yAxisFormat_=(value: String)
  : Unit = {
    require(value != null)
    _yAxisFormat = value
  }

  def setYAxisFormat(value: String)
  : BarChart2D = {
    yAxisFormat_=(value)
    this
  }

  /**
    * Set this to % for percentages; .3g for more precision.
    */
  private var _labelFormat
  : (String, String) = ("g", "g")

  def labelFormat
  : (String, String) = _labelFormat

  def labelFormat_=(value: (String, String))
  : Unit = {
    require(value._1 != null && value._2 != null)
    _labelFormat = value
  }

  def setLabelFormat(value: (String, String))
  : BarChart2D = {
    labelFormat_=(value)
    this
  }

  def setLabelFormat(x: String, y: String)
  : BarChart2D = setLabelFormat((x, y))

  /**
    * -1 = Automatic
    */
  private var _noMajorTicksX
  : Int = -1

  def noMajorTicksX
  : Int = _noMajorTicksX

  def noMajorTicksX_=(value: Int)
  : Unit = {
    require(value >= -1)
    _noMajorTicksX = value
  }

  def setNoMajorTicksX(value: Int)
  : BarChart2D = {
    noMajorTicksX_=(value)
    this
  }

  private var _majorTickSize
  : (Int, Int) = (7, 7)

  def majorTickSize
  : (Int, Int) = _majorTickSize

  def majorTickSize_=(value: (Int, Int))
  : Unit = {
    require(value != null)
    _majorTickSize = value
  }

  def setMajorTickSize(value: (Int, Int))
  : BarChart2D = {
    majorTickSize_=(value)
    this
  }

  private var _noMinorTicks
  : (Int, Int) = (0, 1)

  def noMinorTicks
  : (Int, Int) = _noMinorTicks

  def noMinorTicks_=(value: (Int, Int))
  : Unit = {
    require(value._1 >= 0 && value._2 >= 0)
    _noMinorTicks = value
  }

  def setNoMinorTicks(value: (Int, Int))
  : BarChart2D = {
    noMinorTicks_=(value)
    this
  }

  private var _minorTickSize
  : (Int, Int) = (3, 3)

  def minorTickSize
  : (Int, Int) = _minorTickSize

  def minorTickSize_=(value: (Int, Int))
  : Unit = {
    require(value != null)
    _minorTickSize = value
  }

  def setMinorTickSize(value: (Int, Int))
  : BarChart2D = {
    minorTickSize_=(value)
    this
  }

  private var _tickLabelsX
  : Option[List[Real]] = None

  def tickLabelsX
  : Option[List[Real]] = _tickLabelsX

  def tickLabelsX_=(value: Option[List[Real]])
  : Unit = {
    require(value != null)
    _tickLabelsX = value
  }

  def tickLabelsX_=(value: List[Real])
  : Unit = tickLabelsX_=(Option(value))

  def setTickLabelsX(value: Option[List[Real]])
  : BarChart2D = {
    tickLabelsX_=(value)
    this
  }

  def setTickLabelsX(value: List[Real])
  : BarChart2D = setTickLabelsX(Option(value))

  private var _fillOpacity
  : Real = 0.7f

  def fillOpacity
  : Real = _fillOpacity

  def fillOpacity_=(value: Real)
  : Unit = {
    require(value >= Real.zero && value <= Real.one)
    _fillOpacity = value
  }

  def setFillOpacity(value: Real)
  : BarChart2D = {
    fillOpacity_=(value)
    this
  }

  override protected def doCopy()
  : BarChart2D = BarChart2D()

  override protected def doCopyTo(other: Chart)
  : Unit = other match {
    case other: BarChart2D =>
      other.xAxisOffset         = xAxisOffset
      other._spaceBetweenBars   = _spaceBetweenBars
      other._spaceBetweenCharts = _spaceBetweenCharts
      other._yAxisFormat        = _yAxisFormat
      other._labelFormat        = _labelFormat
      other._noMajorTicksX      = _noMajorTicksX
      other._majorTickSize      = _majorTickSize
      other._noMinorTicks       = _noMinorTicks
      other._minorTickSize      = _minorTickSize
      other._tickLabelsX        = _tickLabelsX
      other._fillOpacity        = _fillOpacity
    case _ =>
      throw new MatchError(other)
  }

  override protected def doToJson()
  : List[JField] = {
    val data = {
      val builder = List.newBuilder[JObject]
      SeqEx.foreachPair(series)((i, s) => {
        val gJson = Json.field("g", i)
        s.points.foreach(p => {
          // TODO: Change to tuple type with "per series" colors.
          builder += JObject(
            Json.field("x", p.x),
            Json.field("y", p.y),
            gJson
          )
        })
      })
      Json.field(
        "data",
        JObject(
          Json.field("name", "points"),
          Json.field("values", builder.result())
        ) :: Nil
      )
    }

    val scales = {
      val yScale = JObject(
        Json.field("name", "y"),
        Json.field("type", _yAxisFormat),
        Json.field("range", "height"),
        Json.field(
          "domain",
          Json.field("data", "points"),
          Json.field("field", "y")
        ),
        Json.field("zero", value = true)
      )
      val colorScale = JObject(
        Json.field("name", "color"),
        Json.field("type", "ordinal"),
        Json.field(
          "range", {
            val builder = List.newBuilder[JString]
            series.foreach(s => {
              val c   = s.color
              val tmp = f"#${c.getRed}%02x${c.getGreen}%02x${c.getBlue}%02x"
              builder += Json(tmp)
            })
            builder.result()
          }
        ),
        Json.field(
          "domain",
          Json.field("data", "points"),
          Json.field("field", "g")
        )
      )
      val chartScale = JObject(
        Json.field("name", "chart"),
        Json.field("type", "ordinal"),
        Json.field("range", "width"),
        Json.field(
          "domain", {
            val builder = List.newBuilder[JInt]
            SeqEx.foreachPair(series)((i, s) => {
              builder += Json(i)
            })
            builder.result()
          }
        )
      )
      Json.field("scales", yScale :: colorScale :: chartScale :: Nil)
    }

    val axes = {
      val yAxis = JObject(
        Json.field("type", "y"),
        Json.field("scale", "y"),
        Json.field("grid", value = true),
        Json.field("offset", _spaceBetweenCharts),
        Json.field("tickSizeMajor", _majorTickSize._2),
        Json.field("tickSizeMinor", _minorTickSize._2),
        Json.field("subdivide", _noMinorTicks._2),
        Json.field("formatType", "number"),
        Json.field("format", _labelFormat._2)
      )
      Json.field("axes", yAxis :: Nil)
    }

    val legends = {
      val legend0 = JObject(
        Json.field("fill", "color"),
        Json.field(
          "values", {
            val builder = List.newBuilder[JString]
            series.foreach(s => {
              builder += Json(s.label)
            })
            builder.result()
          }
        ),
        Json.field(
          "properties",
          Json.field(
            "symbols",
            Json.field(
              "stroke",
              Json.field("value", "transparent")
            )
          ),
          Json.field(
            "legend",
            Json.field(
              "stroke",
              Json.field("value", "#dddddd")
            )
          )
        )
      )
      Json.field("legends", legend0 :: Nil)
    }

    val marks = {
      val transform0 = JObject(
        Json.field("type", "facet"),
        Json.field("groupby", Json("g") :: Nil)
      )

      // For drawing rectangles.
      val xOrdinalScale = JObject(
        Json.field("name", "x_ordinal"),
        Json.field("type", "ordinal"),
        Json.field("range", "width"),
        Json.field(
          "domain",
          Json.field("field", "x"),
          Json.field(
            "sort",
            // Correct the sorting of the bins. Also see: https://github.com/vega/vega/issues/553
            Json.field("field", "x"),
            Json.field("op", "max")
          )
        ),
        Json.field("zero", value = false),
        Json.field("nice", value = false)
      )


      // TODO: Depends if I can get an answer for: https://github.com/vega/vega/issues/557
      /*
      // For drawing ticks.
      val xLinearScale = JObject(
        Json.field("name", "x_linear"),
        Json.field("type", "linear"),
        Json.field("range", "width"),
        Json.field(
          "domain",
          Json.field("field", "x")
        ),
        Json.field("zero", JBool.False),
        Json.field("nice", JBool.False)
      )
      */

      val xAxis = {
        val fields = List.newBuilder[JField]
        fields += Json.field("type", "x")
        fields += Json.field("scale", "x_ordinal")
        fields += Json.field("offset", xAxisOffset)
        fields += Json.field("tickSizeEnd", 0)
        fields += Json.field("tickSizeMajor", _majorTickSize._1)
        fields += Json.field("tickSizeMinor", _minorTickSize._1)
        fields += Json.field("subdivide", _noMinorTicks._1)
        fields += Json.field("formatType", "number")
        fields += Json.field("format", _labelFormat._1)
        fields += Json.field(
          "properties",
          Json.field(
            "labels",
            Json.field(
              "angle",
              Json.field("value", 45.0)
            ),
            Json.field(
              "align",
              Json.field("value", "left")
            )
          )
        )
        _tickLabelsX.foreach(tickLabelsX => {
          fields += Json.field("values", tickLabelsX.map(Json.apply))
        })
        JObject(fields.result())
      }

      val mark0 = JObject(
        Json.field("type", "rect"),
        Json.field(
          "properties",
          Json.field(
            "enter",
            Json.field(
              "x",
              Json.field("scale", "x_ordinal"),
              Json.field("field", "x"),
              Json.field("offset", _spaceBetweenBars / 2)
            ),
            Json.field(
              "width",
              Json.field("scale", "x_ordinal"),
              Json.field("band", value = true),
              Json.field("offset", -_spaceBetweenBars)
            ),
            Json.field(
              "y",
              Json.field("scale", "y"),
              Json.field("field", "y")
            ),
            Json.field(
              "y2",
              Json.field("scale", "y"),
              Json.field("value", 0)
            ),
            Json.field(
              "fill",
              Json.field("scale", "color"),
              Json.field("field", "g")
            )
          ),
          Json.field(
            "update",
            Json.field(
              "fillOpacity",
              Json.field("value", _fillOpacity)
            )
          ),
          Json.field(
            "hover",
            Json.field(
              "fillOpacity",
              Json.field("value", Real.one)
            )
          )
        )
      )

      val groupMark0 = JObject(
        Json.field("type", "group"),
        Json.field(
          "from",
          Json.field("data", "points"),
          Json.field("transform", transform0 :: Nil)
        ),
        Json.field("scales", xOrdinalScale /*:: xLinearScale*/ :: Nil),
        Json.field("axes", xAxis :: Nil),
        Json.field(
          "properties",
          Json.field(
            "enter",
            Json.field(
              "x",
              Json.field("scale", "chart"),
              Json.field("field", "g")
            ),
            Json.field(
              "y",
              Json.field("value", 0)
            ),
            Json.field(
              "width",
              Json.field("scale", "chart"),
              Json.field("band", value = true),
              Json.field("offset", -_spaceBetweenCharts)
            ),
            Json.field(
              "height",
              Json.field(
                "field",
                Json.field("group", "height")
              )
            )
          )
        ),
        Json.field("marks", mark0 :: Nil)
      )
      Json.field("marks", groupMark0 :: Nil)
    }

    List(
      Json.field("width", width),
      Json.field("height", height),
      data,
      scales,
      axes,
      legends,
      marks
    )
  }

}

object BarChart2D {

  final def apply()
  : BarChart2D = new BarChart2D

}

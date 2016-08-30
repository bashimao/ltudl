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

final class ScatterPlot2D
  extends Chart2D[ScatterPlot2D] {

  override def repr
  : ScatterPlot2D = this

  private var _xAxisFormat
  : String = "linear"

  def xAxisFormat
  : String = _xAxisFormat

  def xAxisFormat_=(value: String)
  : Unit = {
    require(value != null)
    _xAxisFormat = value
  }

  def setXAxisFormat(value: String)
  : ScatterPlot2D = {
    xAxisFormat_=(value)
    this
  }

  private var _xAxisLabel
  : Option[String] = None

  def xAxisLabel
  : Option[String] = _xAxisLabel

  def xAxisLabel_=(value: String)
  : Unit = xAxisLabel_=(Option(value))

  def xAxisLabel_=(value: Option[String])
  : Unit = {
    require(value != null)
    _xAxisLabel = value
  }

  def setXAxisLabel(value: String)
  : ScatterPlot2D = {
    xAxisLabel_=(value)
    this
  }

  def setXAxisLabel(value: Option[String])
  : ScatterPlot2D = {
    xAxisLabel_=(value)
    this
  }

  private var _xAxisMinValue
  : Option[Real] = Some(Real.zero)

  def xAxisMinValue
  : Option[Real] = _xAxisMinValue

  def xAxisMinValue_=(value: Option[Real])
  : Unit = {
    require(value != null)
    _xAxisMinValue = value
  }

  def setXAxisMinValue(value: Option[Real])
  : ScatterPlot2D = {
    xAxisMinValue_=(value)
    this
  }

  private var _xAxisMaxValue
  : Option[Real] = None

  def xAxisMaxValue
  : Option[Real] = _xAxisMaxValue

  def xAxisMaxValue_=(value: Option[Real])
  : Unit = {
    require(value != null)
    _xAxisMaxValue = value
  }

  def setXAxisMaxValue(value: Option[Real])
  : ScatterPlot2D = {
    xAxisMaxValue_=(value)
    this
  }

  private var _y0AxisFormat
  : String = "linear"

  def y0AxisFormat
  : String = _y0AxisFormat

  def y0AxisFormat_=(value: String)
  : Unit = {
    require(value != null)
    _y0AxisFormat = value
  }

  def setY0AxisFormat(value: String)
  : ScatterPlot2D = {
    y0AxisFormat_=(value)
    this
  }

  private var _y0AxisLabel
  : Option[String] = None

  def y0AxisLabel
  : Option[String] = _y0AxisLabel

  def y0AxisLabel_=(value: String)
  : Unit = y0AxisLabel_=(Option(value))

  def y0AxisLabel_=(value: Option[String])
  : Unit = {
    require(value != null)
    _y0AxisLabel = value
  }

  def setY0AxisLabel(value: String)
  : ScatterPlot2D = {
    y0AxisLabel_=(value)
    this
  }

  def setY0AxisLabel(value: Option[String])
  : ScatterPlot2D = {
    y0AxisLabel_=(value)
    this
  }

  private var _y0AxisMinValue
  : Option[Real] = None

  def y0AxisMinValue
  : Option[Real] = _y0AxisMinValue

  def y0AxisMinValue_=(value: Option[Real])
  : Unit = {
    require(value != null)
    _y0AxisMinValue = value
  }

  def setY0AxisMinValue(value: Option[Real])
  : ScatterPlot2D = {
    y0AxisMinValue_=(value)
    this
  }

  private var _y0AxisMaxValue
  : Option[Real] = None

  def y0AxisMaxValue
  : Option[Real] = _y0AxisMaxValue

  def y0AxisMaxValue_=(value: Option[Real])
  : Unit = {
    require(value != null)
    _y0AxisMaxValue = value
  }

  def setY0AxisMaxValue(value: Option[Real])
  : ScatterPlot2D = {
    y0AxisMaxValue_=(value)
    this
  }

  private var _y1AxisFormat
  : String = "linear"

  def y1AxisFormat
  : String = _y1AxisFormat

  def y1AxisFormat_=(value: String)
  : Unit = {
    require(value != null)
    _y1AxisFormat = value
  }

  def setY1AxisFormat(value: String)
  : ScatterPlot2D = {
    y0AxisFormat_=(value)
    this
  }

  private var _y1AxisLabel
  : Option[String] = None

  def y1AxisLabel
  : Option[String] = _y1AxisLabel

  def y1AxisLabel_=(value: String)
  : Unit = y1AxisLabel_=(Option(value))

  def y1AxisLabel_=(value: Option[String])
  : Unit = {
    require(value != null)
    _y1AxisLabel = value
  }

  def setY1AxisLabel(value: String)
  : ScatterPlot2D = {
    y1AxisLabel_=(value)
    this
  }

  def setY1AxisLabel(value: Option[String])
  : ScatterPlot2D = {
    y1AxisLabel_=(value)
    this
  }

  private var _y1AxisMinValue
  : Option[Real] = None

  def y1AxisMinValue
  : Option[Real] = _y1AxisMinValue

  def y1AxisMinValue_=(value: Option[Real])
  : Unit = {
    require(value != null)
    _y1AxisMinValue = value
  }

  def setY1AxisMinValue(value: Option[Real])
  : ScatterPlot2D = {
    y1AxisMinValue_=(value)
    this
  }

  private var _y1AxisMaxValue
  : Option[Real] = None

  def y1AxisMaxValue
  : Option[Real] = _y1AxisMaxValue

  def y1AxisMaxValue_=(value: Option[Real])
  : Unit = {
    require(value != null)
    _y1AxisMaxValue = value
  }

  def setY1AxisMaxValue(value: Option[Real])
  : ScatterPlot2D = {
    y1AxisMaxValue_=(value)
    this
  }

  override protected def doToJson()
  : List[JField] = {
    val y0Used = series.exists(_.yAxisNo == 0)
    val y1Used = series.exists(_.yAxisNo == 1)

    val data = Json.field("data", {
      val builder = List.newBuilder[JObject]
      series.foreach(
        builder += _.toJson
      )
      builder.result()
    })

    val scales = {
      val builder = List.newBuilder[JObject]
      if (true) {
        val fields = List.newBuilder[JField]
        fields += Json.field("name", "x")
        fields += Json.field("type", _xAxisFormat)
        fields += Json.field("range", "width")
        fields += Json.field(
          "domain",
          Json.field(
            "fields", {
              val builder = List.newBuilder[JObject]
              series.foreach(s => {
                builder += JObject(
                  Json.field("data", s.label),
                  Json.field("field", "x")
                )
              })
              builder.result()
            }
          )
        )
        fields += Json.field("zero", value = false)
        _xAxisMinValue.foreach(
          fields += Json.field("domainMin", _)
        )
        _xAxisMaxValue.foreach(
          fields += Json.field("domainMax", _)
        )
        builder += JObject(fields.result())
      }
      if (y0Used) {
        val fields = List.newBuilder[JField]
        fields += Json.field("name", "y0")
        fields += Json.field("type", _y0AxisFormat)
        fields += Json.field("range", "height")
        fields += Json.field(
          "domain",
          Json.field(
            "fields", {
              val builder = List.newBuilder[JObject]
              series.foreach(s => {
                if (s.yAxisNo == 0) {
                  builder += JObject(
                    Json.field("data", s.label),
                    Json.field("field", "y")
                  )
                }
              })
              builder.result()
            }
          )
        )
        fields += Json.field("zero", value = false)
        _y0AxisMinValue.foreach(
          fields += Json.field("domainMin", _)
        )
        _y0AxisMaxValue.foreach(
          fields += Json.field("domainMax", _)
        )
        builder += JObject(fields.result())
      }
      if (y1Used) {
        val fields = List.newBuilder[JField]
        fields += Json.field("name", "y1")
        fields += Json.field("type", _y1AxisFormat)
        fields += Json.field("range", "height")
        fields += Json.field(
          "domain",
          Json.field(
            "fields", {
              val builder = List.newBuilder[JObject]
              series.foreach(s => {
                if (s.yAxisNo == 1) {
                  builder += JObject(
                    Json.field("data", s.label),
                    Json.field("field", "y")
                  )
                }
              })
              builder.result()
            }
          )
        )
        fields += Json.field("zero", value = false)
        fields += Json.field("zero", value = false)
        _y1AxisMinValue.foreach(
          fields += Json.field("domainMin", _)
        )
        _y1AxisMaxValue.foreach(
          fields += Json.field("domainMax", _)
        )
        builder += JObject(fields.result())
      }
      if (true) {
        val fields = List.newBuilder[JField]
        fields += Json.field("name", "g")
        fields += Json.field("type", "ordinal")
        fields += Json.field(
          "range", {
            val builder = List.newBuilder[JString]
            series.foreach(s => {
              val c = s.color
              builder += Json(
                f"#${c.getRed}%02x${c.getGreen}%02x${c.getBlue}%02x"
              )
            })
            builder.result()
          }
        )
        fields += Json.field(
          "domain", {
            val builder = List.newBuilder[JString]
            series.foreach(s => {
              builder += Json(s.label)
            })
            builder.result()
          }
        )
        builder += JObject(fields.result())
      }
      Json.field("scales", builder.result())
    }

    val axes = {
      val builder = List.newBuilder[JObject]
      if (true) {
        builder += JObject(
          Json.field("type", "x"),
          Json.field("scale", "x"),
          Json.field("title", _xAxisLabel.getOrElse("")),
          Json.field("grid", value = true)
        )
      }
      if (y0Used) {
        builder += JObject(
          Json.field("type", "y"),
          Json.field("scale", "y0"),
          Json.field("title", _y0AxisLabel.getOrElse("")),
          Json.field("grid", value = true)
        )
      }
      if (y1Used) {
        builder += JObject(
          Json.field("type", "y"),
          Json.field("scale", "y1"),
          Json.field("title", _y1AxisLabel.getOrElse("")),
          Json.field("grid", value = true),
          Json.field("orient", "right")
        )
      }
      Json.field("axes", builder.result())
    }

    val legends = {
      val properties = Json.field(
        "properties",
        Json.field(
          "symbols",
          Json.field(
            "stroke",
            Json.field("value", "transparent")
          )
        )/*,
        Json.field(
          "legend",
          Json.field(
            "stroke",
            Json.field("value", "#dddddd")
          )
        )*/
      )
      val builder = List.newBuilder[JObject]
      if (y0Used) {
        builder += JObject(
          Json.field("fill", "g"),
          Json.field("title", _y0AxisLabel.getOrElse("Left")),
          Json.field(
            "values", {
              val builder = List.newBuilder[JString]
              series.foreach(s => {
                if (s.yAxisNo == 0) {
                  builder += Json(s.label)
                }
              })
              builder.result()
            }
          ),
          properties
        )
      }
      if (y1Used) {
        builder += JObject(
          Json.field("fill", "g"),
          Json.field("title", _y1AxisLabel.getOrElse("Right")),
          Json.field(
            "values", {
              val builder = List.newBuilder[JString]
              series.foreach(s => {
                if (s.yAxisNo == 1) {
                  builder += Json(s.label)
                }
              })
              builder.result()
            }
          ),
          properties
        )
      }
      Json.field("legends", builder.result())
    }

    val marks = {
      val builder = List.newBuilder[JObject]
      series.foreach(s => {
        builder += JObject(
          Json.field("type", if (s.symbolSize > 0) "symbol" else "line"),
          Json.field(
            "from",
            Json.field("data", s.label)
          ),
          Json.field(
            "properties",
            Json.field(
              "enter", {
                val fields = List.newBuilder[JField]
                fields += Json.field(
                  "x",
                  Json.field("scale", "x"),
                  Json.field("field", "x")
                )
                fields += Json.field(
                  "y",
                  Json.field("scale", s"y${s.yAxisNo}"),
                  Json.field("field", "y")
                )
                fields += Json.field(
                  "stroke",
                  Json.field("scale", "g"),
                  Json.field("value", s.label)
                )
                fields += Json.field(
                  "strokeWidth",
                  Json.field("value", s.lineWidth)
                )
                if (s.symbolSize > 0) {
                  fields += Json.field(
                    "fill",
                    Json.field("scale", "g"),
                    Json.field("value", s.label)
                  )
                }
                JObject(fields.result())
              }
            ),
            Json.field("update", JObject()),
            Json.field("hover", JObject())
          )
        )
      })
      Json.field("marks", builder.result())
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

  override protected def doCopy()
  : ScatterPlot2D = ScatterPlot2D()

  override protected def doCopyTo(other: Chart)
  : Unit = other match {
    case other: ScatterPlot2D =>
      other._xAxisFormat    = _xAxisFormat
      other._xAxisLabel     = _xAxisLabel
      other._xAxisMinValue  = _xAxisMinValue
      other._xAxisMaxValue  = _xAxisMaxValue
      other._y0AxisFormat   = _y0AxisFormat
      other._y0AxisLabel    = _y0AxisLabel
      other._y0AxisMinValue = _y0AxisMinValue
      other._y0AxisMaxValue = _y0AxisMaxValue
      other._y1AxisFormat   = _y1AxisFormat
      other._y1AxisLabel    = _y1AxisLabel
      other._y1AxisMinValue = _y1AxisMinValue
      other._y1AxisMaxValue = _y1AxisMaxValue
    case _ =>
      throw new MatchError(other)
  }

}

object ScatterPlot2D {

  final def apply()
  : ScatterPlot2D = new ScatterPlot2D

}

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

// TODO: I already proposed adding this to spire. Have to wait until they have it and remove this then. https://github.com/non/spire/issues/501
/**
 * This is a mix of non's (https://gist.github.com/non/29f8d66036afca402f96#file-half-scala-L15)
 * implementation and http://stackoverflow.com/questions/6162651/half-precision-floating-point-in-java/6162687#6162687
 *
 * Float16 represents 16-bit floating-point values.
 *
 * This type does not actually support arithmetic directly. The
 * expected use case is to convert to Float to perform any actual
 * arithmetic, then convert back to a Float16 if needed.
 *
 * Binary representation:
 *
 *     sign (1 bit)
 *     |
 *     | exponent (5 bits)
 *     | |
 *     | |     mantissa (10 bits)
 *     | |     |
 *     . ..... ..........
 *
 * Value interpretation (in order of precedence, with _ wild):
 *
 *     0 00000 0000000000  (positive) zero
 *     1 00000 0000000000  negative zero
 *     . 00000 ..........  subnormal number
 *     . 11111 0000000000  +/- infinity
 *     . 11111 ..........  not-a-number
 *     . ..... ..........  normal number
 *
 * For non-zero exponents, the mantissa has an implied leading 1 bit,
 * so 10 bits of data provide 11 bits of precision for normal numbers.
 */
final class Half(val raw: Short)
  extends AnyVal {

  /**
   * String representation of this Float16 value.
   */
  override def toString: String = toFloat.toString


  def isNaN: Boolean = (raw & 0x7FFF) > 0x7C00

  def nonNaN: Boolean = (raw & 0x7FFF) <= 0x7C00

  /**
   * Returns if this is a zero value (positive or negative).
   */
  def isZero: Boolean = (raw & 0x7FFF) == 0

  def nonZero: Boolean = (raw & 0x7FFF) != 0

  def isPositiveZero: Boolean = raw == -0x8000

  def isNegativeZero: Boolean = raw == 0

  def isInfinite: Boolean = (raw & 0x7FFF) == 0x7C00

  def isPositiveInfinity: Boolean = raw == 0x7C00

  def isNegativeInfinity: Boolean = raw == 0xFC00

  /**
   * Whether this Float16 value is finite or not.
   *
   * For the purposes of this method, infinities and NaNs are
   * considered non-finite. For those values it returns false and for
   * all other values it returns true.
   */
  def isFinite: Boolean = (raw & 0x7c00) != 0x7c00

  /**
   * Return the sign of a Float16 value as a Float.
   *
   * There are five possible return values:
   *
   *  * NaN: the value is Float16.NaN (and has no sign)
   *  * -1F: the value is a non-zero negative number
   *  * -0F: the value is Float16.NegativeZero
   *  *  0F: the value is Float16.Zero
   *  *  1F: the value is a non-zero positive number
   *
   * PositiveInfinity and NegativeInfinity return their expected
   * signs.
   */
  def signum: Float = {
    if (raw == -0x8000) {
      0F
    }
    else if (raw == 0) {
      -0F
    }
    else if ((raw & 0x7FFF) > 0x7C00) {
      Float.NaN
    }
    else {
      ((raw >>> 14) & 2) - 1F
    }
  }

  // ignores the higher 16 bits
  def toFloat: Float = {
    // 10 bits mantissa, 5 bits exponent
    var man = raw & Half.MantissaMask
    var exp = raw & Half.ExponentMask

    // If NaN/Inf, set exp to Nan/Inf
    if (exp == Half.ExponentMask) {
      exp = 0x3fc00
    }
    else if (exp != 0)                   // normalized value
    {
      exp += 0x1C000                    // exp - 15 + 127

      // smooth transition
      if (man == 0 && exp > 0x1C400) {
        return java.lang.Float.intBitsToFloat(
          (raw & Half.SignMask) << Half.SignShift | exp << Half.ExponentShift | Half.MantissaMask
        )
      }
    }
    else if (man != 0)                  // && exp==0 -> subnormal
    {
      exp = 0x1C400                     // make it normal
      do {
        man <<= 1                       // mantissa * 2
        exp -= 0x400                    // decrease exp by 1
      } while((man & 0x400) == 0)       // while not normal
      man &= 0x3FF                      // discard subnormal bit
    }
    // else +/-0 -> +/-0

    // combine all parts
    java.lang.Float.intBitsToFloat(
      (raw & Half.SignMask) << Half.SignShift | (exp | man) << Half.ExponentShift
    )
  }


  /**
   * Reverse the sign of this Float16 value.
   *
   * This just involves toggling the sign bit with XOR.
   *
   * -Float16.NaN has no meaningful effect.
   * -Float16.Zero returns Float16.NegativeZero.
   */
  def unary_-(): Half = Half.fromLowBits(raw ^ Half.SignMask)

  def +(other: Half): Half = Half(toFloat + other.toFloat)
  
  def -(other: Half): Half = Half(toFloat - other.toFloat)

  def *(other: Half): Half = Half(toFloat * other.toFloat)

  def /(other: Half): Half = Half(toFloat / other.toFloat)

  def ^(other: Int): Half = Half(Math.pow(toFloat, other))

  def ==(other: Half): Boolean = {
    if (isNaN || other.isNaN) {
      false
    }
    else if (isZero && other.isZero) {
      true
    }
    else {
      raw == other.raw
    }
  }

  def <(other: Half): Boolean = {
    if (raw == other.raw || isNaN || other.isNaN || (isZero && other.isZero)) {
      return false
    }

    val ls = (raw       >>> 15) & 1
    val rs = (other.raw >>> 15) & 1
    if (ls < rs) {
      return true
    }
    if (ls > rs) {
      return false
    }

    val le = (raw       >>> 10) & 31
    val re = (other.raw >>> 10) & 31
    if (le < re) {
      return ls == 1
    }
    if (le > re) {
      return ls == 0
    }
    val lm = raw       & 1023
    val rm = other.raw & 1023

    if (ls == 1) lm < rm else rm < lm
  }

  def <=(other: Half): Boolean = {
    if (isNaN || other.isNaN) {
      return false
    }
    if (isZero && other.isZero) {
      return true
    }

    val ls = (raw       >>> 15) & 1
    val rs = (other.raw >>> 15) & 1
    if (ls < rs) {
      return true
    }
    if (ls > rs) {
      return false
    }

    val le = (raw       >>> 10) & 31
    val re = (other.raw >>> 10) & 31
    if (le < re) {
      return ls == 1
    }
    if (le > re) {
      return ls == 0
    }
    val lm = raw       & 1023
    val rm = other.raw & 1023

    if (ls == 1) lm <= rm else rm <= lm
  }

  def >(other: Half): Boolean = !(isNaN || other.isNaN || this <= other)

  def >=(other: Half): Boolean = !(isNaN || other.isNaN || this < other)

}

object Half {

  // interesting Float16 constants
  // with the exception of NaN, values go from smallest to largest
  final val NaN: Half               = fromLowBits(0x7C01)

  final val NegativeInfinity: Half  = fromLowBits(0x7C00)

  final val MinValue: Half          = fromLowBits(0x7BFF)

  final val MinusOne: Half          = fromLowBits(0x3C00)

  final val MaxNegativeNormal: Half = fromLowBits(0x0400)

  final val MaxNegative: Half       = fromLowBits(0x0001)

  final val NegativeZero: Half      = fromLowBits(0x0000)

  final val Zero: Half              = fromLowBits(0x8000)

  final val MinPositive: Half       = fromLowBits(0x8001)

  final val MinPositiveNormal: Half = fromLowBits(0x8400)

  final val One: Half               = fromLowBits(0xBC00)

  final val MaxValue: Half          = fromLowBits(0xFBFF)

  final val PositiveInfinity: Half  = fromLowBits(0xFC00)

  final val SignMask: Int = 0x8000

  // sign  << ( 31 - 15 )
  final val SignShift: Int = 16

  final val ExponentMask: Int = 0x7C00

  // value << ( 23 - 10 )
  final val ExponentShift: Int = 13

  final val MantissaMask: Int = 0x03FF

  @inline
  final def apply(value: Double): Half = apply(value.toFloat)

  @inline
  final def apply(value: Float): Half = fromLowBits(halfBits(value))

  @inline
  final def halfBits(value: Float): Int = {
    val bits: Int  = java.lang.Float.floatToIntBits(value)
    val sign: Int  = bits >>> 16 & 0x8000;        // sign only
    var round: Int = (bits & 0x7FFFFFFF) + 0x1000; // rounded value

    // might be or become NaN/Inf
    if (round >= 0x47800000) {
      // avoid Inf due to rounding
      if ((bits & 0x7FFFFFFF) >= 0x47800000) {
        // is or must become NaN/Inf
        if (round < 0x7F800000) {
          // was value but too large
          // make it +/-Inf
          sign | ExponentMask
        }
        else {
          // remains +/-Inf or NaN
          // keep NaN (and Inf) bits
          sign | ExponentMask | (bits & 0x007FFFFF) >>> ExponentShift
        }
      }
      else {
        // unrounded not quite Inf
        sign | 0x7BFF
      }
    }
    // remains normalized value
    else if (round >= 0x38800000) {
      // exp - 127 + 15
      sign | round - 0x38000000 >>> ExponentShift
    }
    else if(round < 0x33000000) {
      // too small for subnormal
      // becomes +/-0
      sign
    }
    else {
      round = (bits & 0x7FFFFFFF) >>> 23 // tmp exp for subnormal calc

      // add subnormal bit
      // round depending on cut off
      // div by 2^(1-(exp-127+15)) and >> 13 | exp=0
      sign | ((bits & 0x7FFFFF | 0x800000) + (0x800000 >>> round - 102) >>> 126 - round)
    }
  }

  @inline
  final def fromLowBits(raw: Int): Half = new Half(raw.toShort)


  final val size: Int = java.lang.Short.SIZE / 8

}

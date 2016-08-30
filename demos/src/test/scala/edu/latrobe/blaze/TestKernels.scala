/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.blaze

import breeze.linalg._
import edu.latrobe._
import edu.latrobe.kernels._
import edu.latrobe.sizes._
import org.scalatest._

class TestKernels extends FlatSpec with Matchers {

  /**
   * 012345
   * abbccd a
   * aabbcc b
   * 0aabb0 c
   * 00aa00 d
   * 000000 e
   */
  "TemporalKernel" should "should infer the correct number of values when moving origin (1)" in {
    val kernels = Array(
      (Kernel1(2, 2, +1), 4), // a
      (Kernel1(2, 2,  0), 3), // a
      (Kernel1(2, 2, -1), 2), // b
      (Kernel1(2, 2, -2), 1), // c
      (Kernel1(2, 2, -3), 0)  // d
    )
    for (noChannels <- 1 to 5; noMaps <- 1 to 5) {
      val inpSize = Size1(6, noChannels)
      for ((kernel, ref) <- kernels) {
        print(s"NoChannels: $noChannels, Kernel: $kernel, NoMaps: $noMaps => ")
        var cnt = 0
        kernel.foreachOutput(inpSize, noMaps, (i0, i1, offset0) => {
          cnt += 1
          print(s"$i0/$offset0, ")
        })
        println(s" => $cnt")
        cnt should be(ref)
      }
    }
  }

  /**
   * 0123456789
   * a0bbbb0ccc a
   * aa0bbbb0cc b
   * aaa0bbbb0c c
   * aaaa0bbbb0 d
   * 0aaaa00000 e
   * 00aaaa0000 f
   * 000aaaa000 g
   * 0000000000 h
   */
  it should "should infer the correct number of values when moving origin (2)" in {
    val kernels = Array(
      (Kernel1(4, 5, +3), 3), // a
      (Kernel1(4 ,5, +2), 3), // b
      (Kernel1(4, 5, +1), 2), // c
      (Kernel1(4, 5,  0), 2), // d
      (Kernel1(4, 5, -1), 1), // e
      (Kernel1(4, 5, -2), 1), // f
      (Kernel1(4, 5, -3), 1), // g
      (Kernel1(4, 5, -4), 0)  // h
    )
    for (noChannels <- 1 to 5; noMaps <- 1 to 5) {
      val inpSize = Size1(10, noChannels)
      for ((kernel, ref) <- kernels) {
        print(s"NoChannels: $noChannels, Kernel: $kernel, NoMaps: $noMaps => ")
        var cnt = 0
        kernel.foreachOutput(inpSize, noMaps, (i0, i1, offset0) => {
          cnt += 1
          print(s"$i0/$offset0, ")
        })
        println(s" => $cnt")
        cnt should be(ref)
      }
    }
  }

  /**
   * 0123456789
   * a00b00c00d a
   * aa0bb0cc00 b
   * aaabbbccc0 c
   * aaabbbcccc d
   * aaabbbbb00 e
   * aaabbbbbb0 f
   * aaabbbbbbb g
   * aaaaaaaa00 h
   * aaaaaaaaa0 i
   * aaaaaaaaaa j
   * 0000000000 k
   */
  it should "should infer the correct number of values when changing dims" in {
    val kernels = Array(
      (Kernel1( 1, 3, 0), 4), // a
      (Kernel1( 2, 3, 0), 3), // b
      (Kernel1( 3, 3, 0), 3), // c
      (Kernel1( 4, 3, 0), 3), // d
      (Kernel1( 5, 3, 0), 2), // e
      (Kernel1( 6, 3, 0), 2), // f
      (Kernel1( 7, 3, 0), 2), // g
      (Kernel1( 8, 3, 0), 1), // h
      (Kernel1( 9, 3, 0), 1), // i
      (Kernel1(10, 3, 0), 1), // j
      (Kernel1(11, 3, 0), 0)  // k
    )
    for (noChannels <- 1 to 5; noMaps <- 1 to 5) {
      val inpSize = Size1(10, noChannels)
      for ((kernel, ref) <- kernels) {
        print(s"NoChannels: $noChannels, Kernel: $kernel, NoMaps: $noMaps => ")
        var cnt = 0
        kernel.foreachOutput(inpSize, noMaps, (i0, i1, offset0) => {
          cnt += 1
          print(s"$i0/$offset0, ")
        })
        println(s" => $cnt")
        cnt should be(ref)
      }
    }
  }

  /**
   * 0123456789A
   * 00100100100
   */
  it should "should visit fields in a specific order (1)" in {
    val inpSize = Size1(11, 1)
    val kernel = Kernel1(2, 3, -2)
    val ref = Array(2, 5, 8)
    kernel.foreachOutput(inpSize, 1, (i0, i1, offset0) => {
      println((i0, i1, offset0))
      offset0 should be(ref(i0))
    })
  }

  /**
   * 0123456789
   * 1010101000
   */
  it should "should visit fields in a specific order (2)" in {
    val inpSize = Size1(10, 1)
    val kernel = Kernel1(4, 2, 0)
    val ref = Array(0, 2, 4, 6)
    kernel.foreachOutput(inpSize, 1, (i0, i1, offset0) => {
      println((i0, i1, offset0))
      offset0 should be (ref(i0))
    })
  }

  /**
   * 0123456789
   * 1234
   *   1234
   *     1234
   *       1234
   */
  it should "should visit pairs in a specific order" in {
    val inpSize = Size1(10, 1)
    val kernel = Kernel1(4, 2, 0)
    val ref = DenseMatrix.create(4, 4, Array(
      0, 1, 2, 3,
      2, 3, 4, 5,
      4, 5, 6, 7,
      6, 7, 8, 9
    ))
    val ofs = DenseMatrix.zeros[Int](4, 4)
    kernel.foreachValidPair(inpSize, 1, (i0, i1, offset0) => {
      i0 should be(i1 - 1)
      (j0, j1, offset0, offset1) => {
        j0 should be(j1 - 1)
        offset0 should be(offset1 - 1)
        //println(s"$j0 -> $j1, $offset0 -> $offset1")
        //offset0 should be(ref(j0, i0))
        ofs(j0, i0) = offset0
      }
    })
    all((ref - ofs).data) should be(0)
  }


  /**
   * 0 1 2 3 4 5    x x x x x x
   * 6 7 8 9 0 1    x 1 2 x x x
   * 2 3 4 5 6 7 => 0 3 4 x x x
   * 8 9 0 1 2 3    x x x x x x
   * 4 5 6 7 8 9    x x x x x x
   * 0 1 2 3 4 5    x x x x x x
   *
   * 00 . . . . 0
   * 01 . . . . 1
   * 02 . . . . 2
   * 03 . . . . 3
   * 04 . . . . 4
   * 05 . . . . 5
   * 10 . . . . 6
   * 11 1 . . . 7
   * 12 . 2 . . 8
   * 13 . . . . 9
   * 14 . . . . 10
   * 15 . . . . 11
   * 20 . . . . 12
   * 21 . . 3 . 13
   * 22 . . . 4 14
   * 23 . . . . 15
   * 24 . . . . 16
   * 25 . . . . 17
   * 30 . . . . 18
   * 31 . . . . 19
   * 32 . . . . 20
   * 33 . . . . 21
   * 34 . . . . 22
   * 35 . . . . 23
   * 40 . . . . 24
   * 41 . . . . 25
   * 42 . . . . 26
   * 43 . . . . 27
   * 44 . . . . 28
   * 45 . . . . 29
   * 50 . . . . 30
   * 51 . . . . 31
   * 52 . . . . 32
   * 53 . . . . 33
   * 54 . . . . 34
   * 55 . . . . 35
   */
  "PlanarKernel" should "should visit fields in specific order (1, 1)" in {
    val inpSize = Size2((6, 6), 1)
    val kernel = Kernel2((3, 3), (1, 1), (-1, -1))
    val matrix = DenseMatrix.zeros[Int](inpSize.noValues, 4)
    kernel.foreachOutput(inpSize, 1, (i0, i1, offset0) => {
      matrix.update(offset0, i0, i1)
      println(s"$offset0, $i0, $i1")
    })
    val ref = DenseMatrix.zeros[Int](inpSize.noValues, 4)
    ref.update(7, 0, 1)
    ref.update(8, 1, 2)
    ref.update(13, 2, 3)
    ref.update(14, 3, 4)
    println(matrix)
    matrix should be (ref)
  }

  /**
   * 0 1 2 3 4 5 6    x x x x x x x
   * 7 8 9 0 1 2 3    x 1 x 2 x x x
   * 4 5 6 7 8 9 0 => x 3 x 4 x x x
   * 1 2 3 4 5 6 7    x x x x x x x
   * 8 9 0 1 2 3 4    x x x x x x x
   * 5 6 7 8 9 0 1    x x x x x x x
   *
   * 00 . . . . 0
   * 01 . . . . 1
   * 02 . . . . 2
   * 03 . . . . 3
   * 04 . . . . 4
   * 05 . . . . 5
   * 05 . . . . 6
   * 10 . . . . 7
   * 11 1 . . . 8
   * 12 . . . . 9
   * 13 . 2 . . 10
   * 14 . . . . 11
   * 15 . . . . 12
   * 15 . . . . 13
   * 20 . . . . 14
   * 21 . . 3 . 15
   * 22 . . . . 16
   * 23 . . . 4 17
   * 24 . . . . 18
   * 25 . . . . 19
   * 25 . . . . 20
   * 30 . . . . 21
   * 31 . . . . 22
   * 32 . . . . 23
   * 33 . . . . 24
   * 34 . . . . 25
   * 35 . . . . 26
   * 35 . . . . 27
   * 40 . . . . 28
   * 41 . . . . 29
   * 42 . . . . 30
   * 43 . . . . 31
   * 44 . . . . 32
   * 45 . . . . 33
   * 46 . . . . 34
   * 50 . . . . 35
   * 51 . . . . 36
   * 52 . . . . 37
   * 53 . . . . 38
   * 54 . . . . 39
   * 55 . . . . 40
   * 56 . . . . 41
   */
  it should "should visit fields in specific order (2, 1)" in {
    val inpSize = Size2((7, 6), 1)
    val kernel = Kernel2((3, 3), (2, 1), (-1, -1))
    val matrix = DenseMatrix.zeros[Int](inpSize.noValues, 4)
    kernel.foreachOutput(inpSize, 1, (i0, i1, offset0) => {
      matrix.update(offset0, i0, i1)
      println(s"$offset0, $i0, $i1")
    })
    val ref = DenseMatrix.zeros[Int](inpSize.noValues, 4)
    ref.update( 8, 0, 1)
    ref.update(10, 1, 2)
    ref.update(15, 2, 3)
    ref.update(17, 3, 4)
    println(matrix)
    matrix should be (ref)
  }

  /**
   * 0 1 2 3 4 5 6    x x x x x x x
   * 7 8 9 0 1 2 3    x 1 2 3 x x x
   * 4 5 6 7 8 9 0 => x x x x x x x
   * 1 2 3 4 5 6 7    x 4 5 6 x x x
   * 8 9 0 1 2 3 4    x x x x x x x
   * 5 6 7 8 9 0 1    x x x x x x x
   * 2 3 4 5 6 7 8    x x x x x x x
   *
   * 00 . . . . . . 0
   * 01 . . . . . . 1
   * 02 . . . . . . 2
   * 03 . . . . . . 3
   * 04 . . . . . . 4
   * 05 . . . . . . 5
   * 05 . . . . . . 6
   * 10 . . . . . . 7
   * 11 1 . . . . . 8
   * 12 . 2 . . . . 9
   * 13 . . 3 . . . 10
   * 14 . . . . . . 11
   * 15 . . . . . . 12
   * 15 . . . . . . 13
   * 20 . . . . . . 14
   * 21 . . . . . . 15
   * 22 . . . . . . 16
   * 23 . . . . . . 17
   * 24 . . . . . . 18
   * 25 . . . . . . 19
   * 25 . . . . . . 20
   * 30 . . . . . . 21
   * 31 . . . 4 . . 22
   * 32 . . . . 5 . 23
   * 33 . . . . . 6 24
   * 34 . . . . . . 25
   * 35 . . . . . . 26
   * 35 . . . . . . 27
   * 40 . . . . . . 28
   * 41 . . . . . . 29
   * 42 . . . . . . 30
   * 43 . . . . . . 31
   * 44 . . . . . . 32
   * 45 . . . . . . 33
   * 46 . . . . . . 34
   * 50 . . . . . . 35
   * 51 . . . . . . 36
   * 52 . . . . . . 37
   * 53 . . . . . . 38
   * 54 . . . . . . 39
   * 55 . . . . . . 40
   * 56 . . . . . . 41
   * 60 . . . . . . 42
   * 61 . . . . . . 43
   * 62 . . . . . . 44
   * 63 . . . . . . 45
   * 64 . . . . . . 46
   * 65 . . . . . . 47
   * 66 . . . . . . 48
   */
  it should "should visit fields in specific order (1, 2)" in {
    val inpSize = Size2((7, 7), 1)
    val kernel = Kernel2((3, 3), (1, 2), (-1, -1))
    val matrix = DenseMatrix.zeros[Int](inpSize.noValues, 6)
    kernel.foreachOutput(inpSize, 1, (i0, i1, offset0) => {
      matrix.update(offset0, i0, i1)
      println(s"$offset0, $i0, $i1")
    })
    val ref = DenseMatrix.zeros[Int](inpSize.noValues, 6)
    ref.update( 8, 0, 1)
    ref.update( 9, 1, 2)
    ref.update(10, 2, 3)
    ref.update(22, 3, 4)
    ref.update(23, 4, 5)
    ref.update(24, 5, 6)
    println(matrix)
    matrix should be (ref)
  }

  /**
   * 0 1 2 3 4 5 6    x x x x x x x
   * 7 8 9 0 1 2 3    x 1 x 2 x x x
   * 4 5 6 7 8 9 0 => x x x x x x x
   * 1 2 3 4 5 6 7    x 3 x 4 x x x
   * 8 9 0 1 2 3 4    x x x x x x x
   * 5 6 7 8 9 0 1    x x x x x x x
   * 2 3 4 5 6 7 8    x x x x x x x
   *
   * 00 . . . . 0
   * 01 . . . . 1
   * 02 . . . . 2
   * 03 . . . . 3
   * 04 . . . . 4
   * 05 . . . . 5
   * 05 . . . . 6
   * 10 . . . . 7
   * 11 1 . . . 8
   * 12 . . . . 9
   * 13 . 2 . . 10
   * 14 . . . . 11
   * 15 . . . . 12
   * 15 . . . . 13
   * 20 . . . . 14
   * 21 . . . . 15
   * 22 . . . . 16
   * 23 . . . . 17
   * 24 . . . . 18
   * 25 . . . . 19
   * 25 . . . . 20
   * 30 . . . . 21
   * 31 . . 3 . 22
   * 32 . . . . 23
   * 33 . . . 4 24
   * 34 . . . . 25
   * 35 . . . . 26
   * 35 . . . . 27
   * 40 . . . . 28
   * 41 . . . . 29
   * 42 . . . . 30
   * 43 . . . . 31
   * 44 . . . . 32
   * 45 . . . . 33
   * 46 . . . . 34
   * 50 . . . . 35
   * 51 . . . . 36
   * 52 . . . . 37
   * 53 . . . . 38
   * 54 . . . . 39
   * 55 . . . . 40
   * 56 . . . . 41
   * 60 . . . . 42
   * 61 . . . . 43
   * 62 . . . . 44
   * 63 . . . . 45
   * 64 . . . . 46
   * 65 . . . . 47
   * 66 . . . . 48
   */
  it should "should visit fields in specific order (2, 2)" in {
    val inpSize = Size2((7, 7), 1)
    val kernel = Kernel2((3, 3), (2, 2), (-1, -1))
    val matrix = DenseMatrix.zeros[Int](inpSize.noValues, 4)
    kernel.foreachOutput(inpSize, 1, (i0, i1, offset0) => {
      matrix.update(offset0, i0, i1)
    })
    val ref = DenseMatrix.zeros[Int](inpSize.noValues, 4)
    ref.update(8, 0, 1)
    ref.update(10, 1, 2)
    ref.update(22, 2, 3)
    ref.update(24, 3, 4)
    matrix should be (ref)
  }

  /**
   * 1 2 x x x     x x 1 2 x    x x x x x    x x x x x
   * 3 4 x x x     x x 3 4 x    x x x x x    x x x x x
   * 5 6 x x x     x x 5 6 x    1 2 x x x    x x 1 2 x
   * x x x x x     x x x x x    3 4 x x x    x x 3 4 x
   * x x x x x     x x x x x    5 6 x x x    x x 5 6 x
   *
   * 00  11   .   .   .   0
   * 01  12   .   .   .   1
   * 02   .  21   .   .   2
   * 03   .  22   .   .   3
   * 04   .   .   .   .   4
   * 10  13   .   .   .   5
   * 11  14   .   .   .   6
   * 12   .  23   .   .   7
   * 13   .  24   .   .   8
   * 14   .   .   .   .   9
   * 20  15   .  31   .   10
   * 21  16   .  32   .   11
   * 22   .  25   .  41   12
   * 23   .  26   .  42   13
   * 24   .   .   .   .   14
   * 30   .   .  33   .   15
   * 31   .   .  34   .   16
   * 32   .   .   .  43   17
   * 33   .   .   .  44   18
   * 34   .   .   .   .   19
   * 40   .   .  35   .   20
   * 41   .   .  36   .   21
   * 42   .   .   .  45   22
   * 43   .   .   .  46   23
   * 44   .   .   .   .   24
   */
  it should "visit pair fields in the specified order (2 x 3)" in {
    val inpSize = Size2((5, 5), 1)
    val kernel = Kernel2((2, 3), (2, 2), (0, 0))
    val matrix = DenseMatrix.zeros[Int](inpSize.noValues, 4)
    kernel.foreachValidPair(inpSize, 1, (i0, i1, offset0) => {
      (j0, j1, offset0, offset1) => {
        println(s"$j0 -> $j1, $offset0 -> $offset1")
        matrix.update(offset0, i0, i1 * 10 + j1)
      }
    })

    val ref = DenseMatrix.zeros[Int](inpSize.noValues, 4)
    ref.update( 0, 0, 11); ref.update( 2, 1, 21); ref.update(10, 2, 31); ref.update(12, 3, 41)
    ref.update( 1, 0, 12); ref.update( 3, 1, 22); ref.update(11, 2, 32); ref.update(13, 3, 42)
    ref.update( 5, 0, 13); ref.update( 7, 1, 23); ref.update(15, 2, 33); ref.update(17, 3, 43)
    ref.update( 6, 0, 14); ref.update( 8, 1, 24); ref.update(16, 2, 34); ref.update(18, 3, 44)
    ref.update(10, 0, 15); ref.update(12, 1, 25); ref.update(20, 2, 35); ref.update(22, 3, 45)
    ref.update(11, 0, 16); ref.update(13, 1, 26); ref.update(21, 2, 36); ref.update(23, 3, 46)

    println(matrix)
    println(ref)
    matrix should be (ref)
  }

  /**
   * 1 2 3 x x     x x 1 2 3     x x x x x     x x x x x
   * 4 5 6 x x     x x 4 5 6     x x x x x     x x x x x
   * 7 8 9 x x     x x 7 8 9     1 2 3 x x     x x 1 2 3
   * x x x x x     x x x x x     4 5 6 x x     x x 4 5 6
   * x x x x x     x x x x x     7 8 9 x x     x x 7 8 9
   *
   * 00  11   .   .   .   0
   * 01  12   .   .   .   1
   * 02  13  21   .   .   2
   * 03   .  22   .   .   3
   * 04   .  23   .   .   4
   * 10  14   .   .   .   5
   * 11  15   .   .   .   6
   * 12  16  24   .   .   7
   * 13   .  25   .   .   8
   * 14   .  26   .   .   9
   * 20  17   .  31   .   10
   * 21  18   .  32   .   11
   * 22  19  27  33  41   12
   * 23   .  28   .  42   13
   * 24   .  29   .  43   14
   * 30   .   .  34   .   15
   * 31   .   .  35   .   16
   * 32   .   .  36  44   17
   * 33   .   .   .  45   18
   * 34   .   .   .  46   19
   * 40   .   .  37   .   20
   * 41   .   .  38   .   21
   * 42   .   .  39  47   22
   * 43   .   .   .  48   23
   * 44   .   .   .  49   24
   */
  it should "visit pair fields in the specified order (3 x 3)" in {
    val inpSize = Size2((5, 5), 1)
    val kernel = Kernel2((3, 3), (2, 2), (0, 0))
    val matrix = DenseMatrix.zeros[Real](50, 4)
    kernel.foreachValidPair(inpSize, 1, (i0, i1, offset0) => {
      (j0, j1, offset0, offset1) => {
        matrix.update(offset0, i0, (i0 + 1) * 10 + j0 + 1)
      }
    })

    val ref = DenseMatrix.zeros[Real](50, 4)
    ref.update( 0, 0, 11); ref.update( 2, 1, 21); ref.update(10, 2, 31); ref.update(12, 3, 41)
    ref.update( 1, 0, 12); ref.update( 3, 1, 22); ref.update(11, 2, 32); ref.update(13, 3, 42)
    ref.update( 2, 0, 13); ref.update( 4, 1, 23); ref.update(12, 2, 33); ref.update(14, 3, 43)
    ref.update( 5, 0, 14); ref.update( 7, 1, 24); ref.update(15, 2, 34); ref.update(17, 3, 44)
    ref.update( 6, 0, 15); ref.update( 8, 1, 25); ref.update(16, 2, 35); ref.update(18, 3, 45)
    ref.update( 7, 0, 16); ref.update( 9, 1, 26); ref.update(17, 2, 36); ref.update(19, 3, 46)
    ref.update(10, 0, 17); ref.update(12, 1, 27); ref.update(20, 2, 37); ref.update(22, 3, 47)
    ref.update(11, 0, 18); ref.update(13, 1, 28); ref.update(21, 2, 38); ref.update(23, 3, 48)
    ref.update(12, 0, 19); ref.update(14, 1, 29); ref.update(22, 2, 39); ref.update(24, 3, 49)

    println(matrix)
    println(ref)
    matrix should be (ref)
  }

  /**
   * First 3x3 tile (0 1 2 3 6 not visible):
   *
   *    -2 -1  0  1  2 |  3  4
   * -2                |
   * -1     0  1  2    |
   *  0     3  4  5    |
   *  1     6  7  8    |
   *  2                |
   *  3                |
   *  ----------------- ioDims
   *  4
   *
   *  Second 3x3 tile (0 1 2 not visible):
   *
   *    -2 -1  0  1  2 |  3  4
   * -2                |
   * -1        0  1  2 |
   *  0        3  4  5 |
   *  1        6  7  8 |
   *  2                |
   *  3                |
   *  ----------------- ioDims
   *  4
   *
   *  Third 3x3 tile (0 1 2 5 8 not visible):
   *
   *    -2 -1  0  1  2 |  3  4
   * -2                |
   * -1           0  1 | 2
   *  0           3  4 | 5
   *  1           6  7 | 8
   *  2                |
   *  3                |
   *  ----------------- ioDims
   *  4
   */
  it should "if mapping kernel, visit all pair fields in a specific order (manual test)" in {

    val ioSize = Size2((3, 4), 1)

    val k = Kernel2.centered((3, 3))

    val buf = Array(
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue),
      DenseMatrix.fill[Int](ioSize.dims._1, ioSize.dims._2)(Int.MinValue)
    )

    k.foreachValidPair(ioSize, ioSize.noChannels, (i0, i1, offset0) => {
      println(s"($i0)")
      (j0, j1, offset0, offset1) => {
        println(s"i: $i0, j: $j0, o: $offset0")
        buf(i0).data(offset0) = j1
      }
    })

    var i = 0
    for (b <- buf) {
      println(s"b $i -->")
      println(b.t)
      i += 1
    }
  }


  /**
   * Data:
   * X1 X2 X3 X4, Y1 Y2 Y3 Y4, Z1 Z2 Z3 Z4
   * X5 X6 X7 X8, Y5 Y6 Y7 Y8, Z5 Z6 Z7 Z8
   * X9 XA XB XC, Y9 YA YB YC, Z9 ZA ZB ZC
   *
   * In memory:
   * X1 X2 X3 X4 X5 X6 X7 X8 X9 XA XB XC, Y1 Y2 Y3 Y4 Y5 Y6 Y7 Y8 Y9 YA YB YC, Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 ZA ZB ZC
   *
   * Kernel:
   * 3 x 3, delta 2x1, origin -1, -1, convolutions 3 x 2
   *
   * Visit order per tile:
   * [0] = 1 2 5 6
   * [1] = 2 3 4 6 7 8
   * [2] = 1 2 5 6 9 A
   * [3] = 2 3 4 6 7 8 A B C
   * [4] = 5 6 9 A
   * [5] = 6 7 8 A B C
   */
  it should "visit all fields as predicted" in {
    val vec = DenseVector(
      //"junk", "junk", "junk", "junk", "junk",
      "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "XA", "XB", "XC",
      "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "YA", "YB", "YC",
      "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "ZA", "ZB", "ZC"
    )

    val inpSize = Size3((4, 3, 3), 1)
    val kernel = Kernel2((3, 3), (2, 1), (1, 1))

    val result = DenseVector(
      "X1", "X2", "X5", "X6",
      "X2", "X3", "X4", "X6", "X7", "X8",
      "X1", "X2", "X5", "X6", "X9", "XA",
      "X2", "X3", "X4", "X6", "X7", "X8", "XA", "XB", "XC",
      "X5", "X6", "X9", "XA",
      "X6", "X7", "X8", "XA", "XB", "XC",
      "Y1", "Y2", "Y5", "Y6",
      "Y2", "Y3", "Y4", "Y6", "Y7", "Y8",
      "Y1", "Y2", "Y5", "Y6", "Y9", "YA",
      "Y2", "Y3", "Y4", "Y6", "Y7", "Y8", "YA", "YB", "YC",
      "Y5", "Y6", "Y9", "YA",
      "Y6", "Y7", "Y8", "YA", "YB", "YC",
      "Z1", "Z2", "Z5", "Z6",
      "Z2", "Z3", "Z4", "Z6", "Z7", "Z8",
      "Z1", "Z2", "Z5", "Z6", "Z9", "ZA",
      "Z2", "Z3", "Z4", "Z6", "Z7", "Z8", "ZA", "ZB", "ZC",
      "Z5", "Z6", "Z9", "ZA",
      "Z6", "Z7", "Z8", "ZA", "ZB", "ZC"
    )

    var i = 0
    kernel.foreachValidPair(inpSize, 1, (i0, i1, offset0) => {
      //i0 should be(i1)
      (j0, j1, offset0, offset1) => {
        //j0 should be(j1)
        //offset0 should be(offset1)
        println(s"($i) Instance: $i0, Index: $j0, Offset: $offset0, Content: ${vec(offset0)} =!= ${result(i)}")
        vec(offset0) should be(result(i))
        i += 1
      }
    })

  }

}

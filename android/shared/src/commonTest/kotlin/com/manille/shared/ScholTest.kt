package com.manille.shared

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ScholTest {
    // card id = suit * 8 + rank; ranks 7 8 9 J Q K A 10
    private fun c(suit: Int, rank: Int) = suit * 8 + rank

    @Test fun aSevenEightNineTenTrickToasts() {
        assertTrue(isScholTrick(listOf(c(0, 0), c(1, 1), c(2, 2), c(3, 7))))
        assertTrue(isScholTrick(listOf(c(2, 7), c(2, 2), c(2, 1), c(2, 0))))
    }

    @Test fun anyOtherTrickDoesNot() {
        assertFalse(isScholTrick(listOf(c(0, 0), c(1, 1), c(2, 2), c(3, 6))))   // ace instead of the 10
        assertFalse(isScholTrick(listOf(c(0, 0), c(1, 0), c(2, 1), c(3, 7))))   // two sevens, no nine
        assertFalse(isScholTrick(listOf(c(0, 0), c(1, 1), c(2, 2))))           // unfinished trick
    }
}

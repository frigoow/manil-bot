package com.manille.shared

import com.cardtable.platform.*

import com.cardtable.platform.Stats
import com.cardtable.platform.StatsStore

// Profiles, saved games and the statistics' files are the platform's
// (com.cardtable.platform.Storage); what a Manille hand adds to them is here.

/** Everything one finished hand contributes to the statistics. */
data class HandRecord(
    val pointsUs: Int, val pointsThem: Int,      // card points
    val scoreUs: Int, val scoreThem: Int,        // match points scored (after the multiplier)
    val tricksUs: Int, val tricksMine: Int, val manillesUs: Int,
    val weDeclared: Boolean, val iDeclared: Boolean, val mull: Boolean,
    val multiplier: Int,
    val meegaanUs: Boolean, val meegaanThem: Boolean, val meegaanMine: Boolean,
    val terugUs: Boolean, val terugThem: Boolean, val terugMine: Boolean,
)

fun StatsStore.recordHand(r: HandRecord): Stats = update { c ->
    val won = r.pointsUs > r.pointsThem; val lost = r.pointsUs < r.pointsThem
    c.add("hands"); if (won) c.add("hands_won"); if (lost) c.add("hands_lost"); if (!won && !lost) c.add("hands_tied")
    c.add("points_for", r.pointsUs); c.add("points_against", r.pointsThem)
    c.add("score_for", r.scoreUs); c.add("score_against", r.scoreThem)
    c.add("tricks_team", r.tricksUs); c.add("tricks_mine", r.tricksMine); c.add("manilles_team", r.manillesUs)
    if (r.tricksUs == 8) c.add("slams"); if (r.tricksUs == 0) c.add("slams_against")
    c.max("best_hand_points", r.pointsUs)
    if (r.weDeclared) { c.add("decl_hands"); if (won) c.add("decl_won") } else { c.add("def_hands"); if (won) c.add("def_won") }
    if (r.iDeclared) {
        c.add("my_decl"); if (won) c.add("my_decl_won")
        if (r.mull) { c.add("my_mull"); if (won) c.add("my_mull_won") }
    }
    if (r.mull) { c.add("mull_hands"); if (won) c.add("mull_won") }
    if (r.multiplier >= 4) { c.add("big_hands"); if (won) c.add("big_won") }
    if (r.meegaanUs) { c.add("mee_us"); if (won) c.add("mee_us_won") }
    if (r.meegaanMine) { c.add("mee_mine"); if (won) c.add("mee_mine_won") }
    if (r.meegaanThem) { c.add("mee_them"); if (won) c.add("mee_them_won") }
    if (r.terugUs) { c.add("ter_us"); if (won) c.add("ter_us_won") }
    if (r.terugMine) { c.add("ter_mine"); if (won) c.add("ter_mine_won") }
    if (r.terugThem) { c.add("ter_them"); if (won) c.add("ter_them_won") }
    c.streak("hand_streak", "best_hand_streak", won)
}

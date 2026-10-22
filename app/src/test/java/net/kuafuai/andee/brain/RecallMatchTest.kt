package net.kuafuai.andee.brain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The retriever that decides whether a stored route is handed to the model.
 *
 * Both failure directions are expensive and they are not symmetric. A missed
 * route costs one slow run — the assistant works the task out again, which is
 * what it did before this existed. A *wrong* route is injected as a suggestion
 * into the turn, and a model that follows it spends the turn walking a plan for
 * a different app while the user watches. So most of what is pinned here is the
 * negative side: entries that share only the words every task shares ("open",
 * "search", 「打开」) must not surface.
 *
 * The scoring is deliberately read as a weighted average of field matches, so
 * these assert *shape* — who ranks first, who is absent — rather than exact
 * numbers. A test that pins a float makes every weight change a test change and
 * teaches nobody why the weights are what they are.
 */
class RecallMatchTest {

    private fun doc(name: String, desc: String = "", content: String = "") =
        RecallMatch.Doc(
            key = name.lowercase(),
            fields = listOf(
                RecallMatch.Field(name, 3.0),
                RecallMatch.Field(desc, 1.6),
                RecallMatch.Field(content, 0.7),
            ),
        )

    /**
     * The case the feature exists for: the song changes, the plan does not.
     *
     * The request shares no word with the stored entry except the app's name —
     * which is exactly the point. `String.contains`, the old lookup, could only
     * have found this if the model typed the stored entry's keyword back at it.
     */
    @Test
    fun matchesThePurposeWhenTheWordsDiffer() {
        val route = doc(
            "YouTube Music 搜歌播放",
            "在 YouTube Music 里按名字搜一首歌并播放",
            "1. 打开 YouTube Music\n2. 点顶部的搜索框\n3. 输入歌名\n4. 点第一条结果",
        )
        val hits = RecallMatch.rank("open youtube music and search for a song", listOf(route))
        assertEquals(1, hits.size)
        assertEquals(route.key, hits.first().key)
    }

    /**
     * Chinese has no spaces, so 「搜歌播放」 and 「搜索歌曲」 share no substring
     * and would never match as strings. Bigrams are what make them overlap at
     * all — this is the half of the feature a Latin-only tokenizer loses.
     */
    @Test
    fun matchesAcrossChineseWordChoice() {
        val route = doc("网易云音乐 搜歌播放", "按歌名搜一首歌并播放")
        val hits = RecallMatch.rank("打开网易云音乐播放歌曲", listOf(route))
        assertEquals(1, hits.size)
        assertEquals(route.key, hits.first().key)
    }

    /**
     * Two entries that share only the generic action words are two different
     * tasks, and suggesting one for the other is the failure that costs a turn.
     * Without the down-weight this scores ~0.5 and gets injected.
     */
    @Test
    fun actionWordsAloneDoNotMatch() {
        val anything = doc("open app and search", "open the app and search")
        val hits = RecallMatch.rank("open youtube music and search for a song", listOf(anything))
        assertTrue("generic words alone must not clear MIN_SCORE: $hits", hits.isEmpty())
    }

    /** A note about a different app must not be reached through 「打开」. */
    @Test
    fun unrelatedChineseEntryIsNotSuggested() {
        val wechat = doc("微信发消息", "在微信里给联系人发一条消息")
        val hits = RecallMatch.rank("open youtube music and search for a song", listOf(wechat))
        assertTrue(hits.isEmpty())
    }

    /** The same query in both: what a route is *called* outranks a passing mention. */
    @Test
    fun nameOutranksTheBody() {
        val named = doc("YouTube 打开方式")
        val mentioned = doc("常用应用", "平时听歌用 youtube")
        val hits = RecallMatch.rank("open youtube", listOf(named, mentioned))
        assertEquals(2, hits.size)
        assertEquals(named.key, hits.first().key)
        assertTrue(hits[1].score < hits[0].score)
    }

    /**
     * The old substring behaviour, kept where it is correct: the whole query
     * written verbatim in an entry is not a guess.
     */
    @Test
    fun verbatimQueryScoresAtLeastOne() {
        val route = doc("YouTube Music 搜歌播放", "按歌名搜歌")
        val hits = RecallMatch.rank("youtube music 搜歌播放", listOf(route))
        assertEquals(1, hits.size)
        // At least 1, not exactly: the name field is weighted 3×, so a word-for-
        // word name match already clears 1 on its own and must not be demoted.
        assertTrue("verbatim must not score below a perfect token match", hits.first().score >= 1.0)
    }

    @Test
    fun blankInputsReturnNothing() {
        val route = doc("YouTube Music 搜歌播放")
        assertTrue(RecallMatch.rank("", listOf(route)).isEmpty())
        assertTrue(RecallMatch.rank("   ", listOf(route)).isEmpty())
        assertTrue(RecallMatch.rank("open youtube", emptyList()).isEmpty())
    }

    /** `limit` is the caller's, and the caller already bounded it. */
    @Test
    fun respectsLimit() {
        val docs = (1..5).map { doc("youtube music route $it", "search youtube music") }
        assertEquals(2, RecallMatch.rank("open youtube music", docs, limit = 2).size)
    }

    @Test
    fun tokensKeepContentAndDropFurniture() {
        val t = RecallMatch.tokens("Open the file 42 and play it")
        assertTrue(t.contains("open"))
        assertTrue(t.contains("file"))
        assertTrue("a number on its own is content here", t.contains("42"))
        assertTrue("stop words carry no subject", !t.contains("the"))
        assertTrue("a two-letter word is noise", !t.contains("it"))
    }

    /** One-character CJK runs survive; longer ones are indexed as bigrams. */
    @Test
    fun tokensSplitCjkIntoBigrams() {
        assertEquals(listOf("搜歌", "歌播", "播放"), RecallMatch.tokens("搜歌播放"))
        assertEquals(listOf("在"), RecallMatch.tokens("在"))
    }

    /** The payload rides through, so the caller does not have to re-look it up. */
    @Test
    fun payloadSurvivesRanking() {
        val marker = Any()
        val d = RecallMatch.Doc("x", listOf(RecallMatch.Field("youtube music", 3.0)), marker)
        val hits = RecallMatch.rank("youtube music", listOf(d))
        assertEquals(1, hits.size)
        assertNotNull(hits.first().payload)
        assertTrue(hits.first().payload === marker)
    }
}

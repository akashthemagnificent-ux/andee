package net.kuafuai.andee.brain

import kotlin.math.ln

/**
 * Purpose-level matching for the notebook, without a model or an embedding.
 *
 * The old lookup was `String.contains`: ask for "open YouTube Music and search
 * for a song" and an entry named 「YouTube Music 搜歌播放」 was invisible unless
 * the model happened to type the exact stored keyword. That is not a search, it
 * is a spelling test — and it failed precisely on the requests worth caching,
 * where the song title changes every time and the *plan* does not.
 *
 * What this does instead is the cheap half of information retrieval:
 *
 *  * **Tokens, not substrings.** A request and an entry are compared as sets of
 *    words, so word order, punctuation and the parts that always differ (a song
 *    title, a city) stop mattering.
 *  * **CJK bigrams.** Chinese has no spaces, so 「搜歌播放」 is indexed as 搜歌,
 *    歌播, 播放. A request containing 「搜索歌曲」 shares no substring with it and
 *    still shares the 搜 token-run shape that IR over bigrams is good at.
 *  * **Inverse document frequency.** A token that appears in every entry
 *    ("open", "app", 「打开」) proves nothing; a token in exactly one entry is
 *    most of the answer. Without this, every request matches every note that
 *    mentions opening something.
 *  * **Field weights.** A hit in the name counts more than the same word buried
 *    in a body of text.
 *  * **Generic action words are down-weighted, not removed.** "search" and
 *    「打开」 do carry intent, so throwing them away loses "what kind of task is
 *    this"; but left at full weight they let any two stepping-stone notes
 *    collide. A quarter weight keeps the signal and drops the noise.
 *
 * Deliberately no stemming and no synonyms. Stemming would be a second language
 * problem (there is no Chinese stemmer here and pretending otherwise is how
 * 「播放」 stops matching 「播放器」), and a synonym table is a maintenance bill
 * nobody reads. What is left is enough for the case this exists for: the same
 * app, the same kind of job, different words.
 *
 * Pure Kotlin, no Android imports — [RecallMatchTest] runs it on the host.
 */
object RecallMatch {

    /** One searchable record. [fields] are weighted, highest first. */
    class Doc(val key: String, val fields: List<Field>, val payload: Any? = null)

    /** A stretch of text and how much a match in it is worth. */
    class Field(val text: String, val weight: Double)

    class Hit(val key: String, val score: Double, val payload: Any? = null)

    /**
     * Below this a "match" is two notes that happen to share a word. Chosen
     * against the two failures that matter: a missed route costs one slow run
     * (recoverable), while a wrong route injected as a suggestion costs a turn
     * spent following the wrong plan (and the user watching it happen).
     */
    const val MIN_SCORE = 0.30

    /** An action word, weighted down rather than dropped. See the class KDoc. */
    private const val GENERIC_WEIGHT = 0.22

    /**
     * Words that say *what kind of act* this is, not *what it is about*. Kept
     * as tokens so "open X and search" still resembles "open X", just less
     * strongly than a real subject-word hit.
     */
    private val GENERIC = setOf(
        "open", "launch", "start", "run", "go", "goto", "tap", "click", "press",
        "play", "search", "find", "look", "use", "app", "application", "screen",
        "打开", "启动", "进入", "点击", "点开", "播放", "搜索", "查找", "找到", "应用", "打开应用",
    )

    /**
     * Function words only. Deliberately short: an aggressive stoplist in a
     * bilingual matcher is how 「音乐」 ends up dropped because it was
     * mis-guessed as filler.
     */
    private val STOP = setOf(
        "a", "an", "the", "and", "or", "but", "for", "to", "of", "in", "on", "at",
        "by", "with", "from", "as", "up", "out", "off", "about", "into", "over",
        "is", "are", "was", "were", "be", "been", "being", "am",
        "it", "its", "this", "that", "these", "those", "there", "here",
        "i", "me", "my", "mine", "you", "your", "yours", "we", "us", "our",
        "he", "she", "him", "her", "they", "them", "their",
        "do", "does", "did", "can", "could", "would", "should", "will", "shall",
        "may", "might", "must", "just", "please", "now", "then", "some", "any",
        "something", "anything", "thing", "things", "let", "lets", "want", "need",
    )

    /**
     * Rank [docs] against [query], best first.
     *
     * @param limit how many to return; the caller decides what to do with them.
     * @param minScore defaults to [MIN_SCORE]; the tests move it to check the
     *   shape of the curve rather than one magic number.
     */
    fun rank(query: String, docs: List<Doc>, limit: Int = 8, minScore: Double = MIN_SCORE): List<Hit> {
        if (docs.isEmpty()) return emptyList()
        val q = tokens(query).distinct()
        if (q.isEmpty()) return emptyList()

        // Index every doc once: token -> best field weight for that token.
        val index = docs.map { d -> d to fieldWeights(d) }

        // Document frequency per query token. Cheap because only query tokens
        // are counted — the corpus is capped at Notebook.MAX_MEMORIES anyway.
        val df = HashMap<String, Int>(q.size)
        for (t in q) df[t] = index.count { (_, m) -> m.containsKey(t) }

        val totalWeight = q.sumOf { weightOf(it) }
        if (totalWeight <= 0.0) return emptyList()

        val hits = ArrayList<Hit>()
        for ((doc, weights) in index) {
            var got = 0.0
            var matched = 0
            for (t in q) {
                val fieldWeight = weights[t] ?: continue
                got += idf(df[t] ?: 0, index.size) * weightOf(t) * fieldWeight
                matched++
            }
            if (matched == 0) continue

            // Not clamped to 1: the field weights are boosts, so a name hit can
            // push this above 1, and that headroom is the whole ordering — a
            // name match has to outrank a passing mention in the body. Only
            // [MIN_SCORE] is a fixed point, and it is a floor, not a range.
            var score = got / totalWeight
            // The whole query written verbatim is not a guess, it is a hit —
            // this is the old substring behaviour, kept where it is correct.
            // `maxOf` rather than assignment: verbatim means "at least as good
            // as a perfect token match", and a stronger score should not be
            // demoted by it.
            if (verbatim(query, doc)) score = maxOf(score, 1.0)

            if (score >= minScore) hits.add(Hit(doc.key, score, doc.payload))
        }
        return hits.sortedByDescending { it.score }.take(limit)
    }

    /**
     * Smoothed IDF. `ln(1 + N/(1+df))`: a token in every doc scores ~0.4 and a
     * token in one doc of fifty scores ~3.2, so rarity — not frequency — is
     * what carries the match.
     */
    private fun idf(df: Int, n: Int): Double = ln(1.0 + n.toDouble() / (1 + df))

    private fun weightOf(token: String): Double = if (token in GENERIC) GENERIC_WEIGHT else 1.0

    /**
     * Best (highest) field weight for each token in a doc — **without** the
     * generic-action discount, which is applied once, in [rank]. Applying it in
     * both places would square it and make "open X" match nothing at all.
     */
    private fun fieldWeights(doc: Doc): Map<String, Double> {
        val out = HashMap<String, Double>()
        for (f in doc.fields) {
            if (f.text.isBlank()) continue
            for (t in tokens(f.text)) {
                val old = out[t]
                if (old == null || f.weight > old) out[t] = f.weight
            }
        }
        return out
    }

    private fun verbatim(query: String, doc: Doc): Boolean {
        val q = query.trim().lowercase()
        if (q.length < 3) return false
        return doc.fields.any { it.text.lowercase().contains(q) }
    }

    /**
     * Split into matchable tokens.
     *
     * Latin and digits: runs of letters/digits, lowercased, length ≥ 3 (≥ 1 for
     * a number, because 「9」 and 「12」 are content in this app — a clock, a
     * floor, a version).
     *
     * CJK: **bigrams**, plus the whole run when it is short enough to be one
     * word. Bigrams are what make the token sets of 「搜歌播放」 and 「搜索歌曲」
     * overlap at all; single characters would collide across unrelated words,
     * and whole runs would never match.
     *
     * Everything else (punctuation, emoji, the apostrophe in "weren't") is a
     * boundary. `weren't` becoming `weren` is correct for this purpose: it is
     * not a word that will ever be in a route note.
     */
    fun tokens(s: String): List<String> {
        if (s.isBlank()) return emptyList()
        val out = ArrayList<String>()
        val run = StringBuilder()
        var cjkRun = StringBuilder()

        fun flushLatin() {
            if (run.isEmpty()) return
            val t = run.toString().lowercase()
            run.setLength(0)
            if (t in STOP) return
            if (t.length >= 3) out.add(t) else if (t[0].isDigit()) out.add(t)
        }

        fun flushCjk() {
            val r = cjkRun.toString()
            cjkRun = StringBuilder()
            if (r.isEmpty()) return
            if (r.length == 1) {
                if (r !in STOP) out.add(r)
                return
            }
            for (i in 0 until r.length - 1) {
                val bg = r.substring(i, i + 2)
                if (bg !in STOP) out.add(bg)
            }
        }

        for (ch in s) {
            when {
                isCjk(ch) -> {
                    flushLatin()
                    cjkRun.append(ch)
                }
                ch.isLetterOrDigit() -> {
                    flushCjk()
                    run.append(ch)
                }
                else -> {
                    flushLatin()
                    flushCjk()
                }
            }
        }
        flushLatin()
        flushCjk()
        return out
    }

    /**
     * Han, plus the kana and hangul blocks a Chinese UI on this device actually
     * shows. Ranges are the BMP ones; nothing in this app writes astral-plane
     * CJK into a note.
     */
    private fun isCjk(ch: Char): Boolean {
        val c = ch.code
        return (c in 0x3040..0x30FF) ||   // hiragana / katakana
            (c in 0x3400..0x4DBF) ||      // CJK ext A
            (c in 0x4E00..0x9FFF) ||      // CJK unified
            (c in 0xF900..0xFAFF) ||      // compatibility ideographs
            (c in 0xAC00..0xD7AF)         // hangul syllables
    }
}

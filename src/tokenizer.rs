//! Tokenizer — loads a HuggingFace `tokenizer.json` (byte-level BPE) using
//! `asmjson` and provides `encode` / `decode`.

use std::{collections::HashMap, fs, path::Path, sync::OnceLock};

use fancy_regex::Regex;
use asmjson::{parse_to_dom, JsonRef};

// ─── Public API ───────────────────────────────────────────────────────────────

/// Converts between raw text and the integer token ids the model understands.
///
/// ## What tokenisation does
///
/// Neural language models can only work with numbers, not letters.  A
/// tokenizer is the bridge:
/// * **Encode**: split text into *tokens* (words, sub-words, or individual
///   bytes) and map each one to an integer id.
/// * **Decode**: map ids back to their text representations and join them.
///
/// This tokenizer uses **byte-level BPE** (byte-pair encoding):
/// * Every byte of input (0–255) starts as its own token.
/// * Pairs of adjacent tokens are repeatedly merged, highest-priority first,
///   until no more merge rules match.  Common words end up as a single token;
///   rare words split into smaller pieces.
///
/// ## Special tokens
///
/// Some ids have fixed meanings that bypass BPE entirely, e.g.:
/// * `<|im_start|>` marks the beginning of a chat turn.
/// * `<|endoftext|>` tells the model (and the generation loop) to stop.
///
/// ## Example
///
/// ```text
/// let tok = Tokenizer::from_file("models/Qwen2.5-0.5B/tokenizer.json")?;
/// let ids = tok.encode("Hello!");        // e.g. [9707, 0]
/// let text = tok.decode(&ids);           // "Hello!"
/// ```
pub struct Tokenizer {
    /// token text → token id
    pub vocab: HashMap<Box<str>, u32>,
    /// token id → token text (indexed by id)
    pub id_to_token: Vec<Box<str>>,
    /// BPE merge rules as `"left right"` strings, in priority order.
    /// Index 0 = highest priority (applied first).
    pub merges: Vec<Box<str>>,
    /// `"left right"` → priority rank, for O(1) pair lookup during BPE.
    merge_rank: HashMap<Box<str>, usize>,
    /// Special / added tokens, e.g. `<|endoftext|>` → id.
    pub special_tokens: HashMap<Box<str>, u32>,
    /// Inverse of special_tokens: id → raw content string, for fast decode.
    special_id_to_content: HashMap<u32, Box<str>>,
    /// For each byte value 0–255, the vocabulary key string for that byte.
    byte_to_tok: Vec<Box<str>>,
    /// Inverse: vocab-character → original byte.
    char_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    /// Load a tokenizer from a HuggingFace `tokenizer.json` file on disk.
    ///
    /// Reads and parses the JSON, builds the vocabulary hash-map and BPE
    /// merge rank index, and initialises the byte ↔ Unicode character map
    /// that Qwen2 uses to represent raw bytes as printable characters.
    ///
    /// ## Example
    ///
    /// ```text
    /// let tok = Tokenizer::from_file("models/Qwen2.5-0.5B/tokenizer.json").unwrap();
    /// println!("{} vocab entries", tok.vocab.len()); // 151643
    /// ```
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let src = fs::read_to_string(path).map_err(|e| e.to_string())?;

        let tape = parse_to_dom(&src, None).ok_or("tokenizer JSON: parse failed")?;
        let root = tape.root().ok_or("tokenizer JSON: empty document")?;
        let model = root.get("model").ok_or("missing 'model' key")?;

        // vocab: {token: id}
        let vocab_pairs: Vec<(String, u32)> = model
            .get("vocab")
            .and_then(|v| v.object_iter())
            .ok_or("missing model.vocab")?
            .filter_map(|(k, v)| v.as_u64().map(|id| (k.to_owned(), id as u32)))
            .collect();

        // merges: ["left right", ...]
        let merges: Vec<String> = model
            .get("merges")
            .and_then(|v| v.array_iter())
            .ok_or("missing model.merges")?
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();

        // added_tokens: [{id, content}, ...]
        let special_pairs: Vec<(u32, String)> =
            if let Some(added) = root.get("added_tokens").and_then(|v| v.array_iter()) {
                added
                    .filter_map(|tok| {
                        let id = tok.get("id")?.as_u64()? as u32;
                        let content = tok.get("content")?.as_str()?.to_owned();
                        Some((id, content))
                    })
                    .collect()
            } else {
                vec![]
            };

        RawTokenizer {
            vocab: vocab_pairs,
            merges,
            special_tokens: special_pairs,
        }
        .into_tokenizer()
    }

    /// Turn a text string into a list of integer token ids.
    ///
    /// ## Steps
    ///
    /// 1. Scan left-to-right for **special tokens** (like `<|im_start|>`);
    ///    emit their ids directly without BPE.
    /// 2. For the text between special tokens, apply the **pre-tokenize
    ///    regex** (Qwen's GPT-2-style pattern) to split into chunks such as
    ///    words, punctuation, and whitespace prefixes.
    /// 3. Convert each chunk to its byte sequence, then run **BPE merging**
    ///    to collapse adjacent byte-tokens according to the merge priority list.
    /// 4. Collect all resulting token ids in order.
    ///
    /// ## Example
    ///
    /// ```text
    /// let ids = tok.encode("Hello!");
    /// // "Hello" is a single high-frequency token; "!" is another → e.g. [9707, 0]
    /// let ids2 = tok.encode("<|im_start|>Hello");
    /// // special token id first, then BPE ids for "Hello"
    /// ```
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut pos = 0;

        while pos < text.len() {
            let remaining = &text[pos..];

            // 1. Try to match a special token at the current position.
            //    If multiple match, take the longest.
            if let Some((len, id)) = self
                .special_tokens
                .iter()
                .filter_map(|(tok, &id)| {
                    if remaining.starts_with(tok.as_ref()) {
                        Some((tok.len(), id))
                    } else {
                        None
                    }
                })
                .max_by_key(|&(len, _)| len)
            {
                ids.push(id);
                pos += len;
                continue;
            }

            // 2. Find where the next special token begins (always > 0 here).
            let seg_len = self
                .special_tokens
                .keys()
                .filter_map(|tok| remaining.find(tok.as_ref()).filter(|&p| p > 0))
                .min()
                .unwrap_or(remaining.len());

            // 3. BPE-encode the regular segment.
            for chunk in pre_tokenize(&text[pos..pos + seg_len]) {
                let byte_toks: Vec<&str> = chunk
                    .bytes()
                    .map(|b| self.byte_to_tok[b as usize].as_ref())
                    .collect();
                let merged = bpe_merge(byte_toks, &self.merge_rank);
                for tok in &merged {
                    if let Some(&id) = self.vocab.get(tok.as_str()) {
                        ids.push(id);
                    }
                }
            }
            pos += seg_len;
        }
        ids
    }

    /// Turn a list of token ids back into a UTF-8 text string.
    ///
    /// ## Steps
    ///
    /// 1. **Special token ids** (e.g. `<|im_end|>`) are emitted as their
    ///    literal content strings.
    /// 2. **Regular token ids** map to a vocabulary string whose characters
    ///    represent individual bytes via Qwen's GPT-2 byte-encoding.  These
    ///    bytes are collected into a buffer.
    /// 3. The byte buffer is converted to UTF-8 with lossy replacement for
    ///    any incomplete multi-byte sequences that span token boundaries.
    ///
    /// ## Example
    ///
    /// ```text
    /// let text = tok.decode(&[9707, 0]); // → "Hello!"
    /// // Multi-byte characters (e.g. emoji) are reassembled correctly even
    /// // if the UTF-8 bytes were split across token boundaries.
    /// ```
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut output = String::new();
        let mut pending: Vec<u8> = Vec::new();

        for &id in ids {
            if let Some(content) = self.special_id_to_content.get(&id) {
                // Flush accumulated bytes before emitting the special token.
                if !pending.is_empty() {
                    output.push_str(&String::from_utf8_lossy(&pending));
                    pending.clear();
                }
                output.push_str(content);
            } else if let Some(tok) = self.id_to_token.get(id as usize) {
                for ch in tok.chars() {
                    if let Some(&b) = self.char_to_byte.get(&ch) {
                        pending.push(b);
                    }
                }
            }
        }
        if !pending.is_empty() {
            output.push_str(&String::from_utf8_lossy(&pending));
        }
        output
    }
}

// ─── SAX state machine ────────────────────────────────────────────────────────

/// Intermediate output produced by `TokenizerSax` and consumed by
/// `RawTokenizer::into_tokenizer`.
#[derive(Default)]
struct RawTokenizer {
    vocab: Vec<(String, u32)>,
    merges: Vec<String>,
    special_tokens: Vec<(u32, String)>,
}

impl RawTokenizer {
    fn into_tokenizer(self) -> Result<Tokenizer, String> {
        // Build vocab and reversed id→token table.
        let max_id = self.vocab.iter().map(|(_, id)| *id).max().unwrap_or(0) as usize;
        let mut id_to_token: Vec<Box<str>> = vec![Box::from(""); max_id + 1];
        let mut vocab: HashMap<Box<str>, u32> = HashMap::with_capacity(self.vocab.len());

        for (tok, id) in self.vocab {
            let tok: Box<str> = tok.into_boxed_str();
            if (id as usize) < id_to_token.len() {
                id_to_token[id as usize] = tok.clone();
            }
            vocab.insert(tok, id);
        }

        // Build merge index.
        let merge_rank: HashMap<Box<str>, usize> = self
            .merges
            .iter()
            .enumerate()
            .map(|(rank, s)| (Box::from(s.as_str()), rank))
            .collect();
        let merges: Vec<Box<str>> = self
            .merges
            .into_iter()
            .map(|s| s.into_boxed_str())
            .collect();

        // Special tokens — also extend id_to_token to cover their ids.
        let special_max = self
            .special_tokens
            .iter()
            .map(|(id, _)| *id as usize)
            .max()
            .unwrap_or(0);
        if special_max >= id_to_token.len() {
            id_to_token.resize(special_max + 1, Box::from(""));
        }
        let mut special_id_to_content: HashMap<u32, Box<str>> =
            HashMap::with_capacity(self.special_tokens.len());
        let special_tokens: HashMap<Box<str>, u32> = self
            .special_tokens
            .into_iter()
            .map(|(id, content)| {
                let content: Box<str> = content.into_boxed_str();
                id_to_token[id as usize] = content.clone();
                special_id_to_content.insert(id, content.clone());
                (content, id)
            })
            .collect();

        // byte ↔ Unicode-character mappings used by the vocabulary.
        let byte_map = build_byte_to_char();
        let byte_to_tok: Vec<Box<str>> = (0usize..256)
            .map(|b| byte_map[b].to_string().into_boxed_str())
            .collect();
        let char_to_byte: HashMap<char, u8> =
            (0usize..256).map(|b| (byte_map[b], b as u8)).collect();

        Ok(Tokenizer {
            vocab,
            id_to_token,
            merges,
            merge_rank,
            special_tokens,
            special_id_to_content,
            byte_to_tok,
            char_to_byte,
        })
    }
}

// ─── BPE merge ────────────────────────────────────────────────────────────────

/// Greedy BPE merge: repeatedly find the highest-priority adjacent pair and
/// apply it everywhere, until no merge applies.
fn bpe_merge(tokens: Vec<&str>, merge_rank: &HashMap<Box<str>, usize>) -> Vec<String> {
    let mut pieces: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();

    loop {
        if pieces.len() < 2 {
            break;
        }

        // Find the pair with the lowest (= highest priority) rank.
        let mut best_rank = usize::MAX;
        let mut best_pair: Option<(String, String)> = None;
        let mut key_buf = String::new();

        for i in 0..pieces.len() - 1 {
            key_buf.clear();
            key_buf.push_str(&pieces[i]);
            key_buf.push(' ');
            key_buf.push_str(&pieces[i + 1]);
            if let Some(&rank) = merge_rank.get(key_buf.as_str())
                && rank < best_rank
            {
                best_rank = rank;
                best_pair = Some((pieces[i].clone(), pieces[i + 1].clone()));
            }
        }

        let Some((left, right)) = best_pair else {
            break;
        };

        // Apply this merge at every occurrence.
        let mut new: Vec<String> = Vec::with_capacity(pieces.len());
        let mut i = 0;
        while i < pieces.len() {
            if i + 1 < pieces.len() && pieces[i] == left && pieces[i + 1] == right {
                let mut merged = left.clone();
                merged.push_str(&right);
                new.push(merged);
                i += 2;
            } else {
                new.push(pieces[i].clone());
                i += 1;
            }
        }
        pieces = new;
    }

    pieces
}

// ─── Pre-tokeniser ────────────────────────────────────────────────────────────

/// Simplified pre-tokeniser: splits on whitespace keeping the first character
/// of each non-space run attached to the preceding space (GPT-2 convention).
///
/// Split `text` into pre-tokenisation chunks using the Qwen2 / GPT-2 regex.
///
/// The pattern matches contractions, words, numbers, punctuation runs, and
/// whitespace segments, which are then individually BPE-encoded.
fn pre_tokenize(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return vec![];
    }
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        )
        .expect("pre_tokenize regex is valid")
    });
    re.find_iter(text)
        .filter_map(|m| m.ok())
        .map(|m| m.as_str())
        .collect()
}

// ─── Byte ↔ character mapping (GPT-2 / Qwen convention) ─────────────────────

/// Build the GPT-2 byte-to-Unicode mapping used by byte-level BPE vocabularies.
///
/// Bytes that have a "nice" Unicode rendition (printable ASCII, ¡–¬, ®–ÿ) map
/// to themselves. The remaining 68 bytes (control characters, DEL, 128–160,
/// soft-hyphen) map to U+0100–U+0143 (Ā…Ń) so they are representable as
/// single Unicode code points and can appear as vocabulary keys.
fn build_byte_to_char() -> [char; 256] {
    let mut map = ['\0'; 256];

    // Direct-mapped ranges: readable Latin characters.
    for b in 33u8..=126 {
        map[b as usize] = b as char;
    }
    for b in 161u8..=172 {
        map[b as usize] = b as char;
    }
    for b in 174u8..=255 {
        map[b as usize] = b as char;
    }

    // Remaining bytes get sequential codepoints from U+0100 onward.
    let mut n = 0u32;
    for cell in map.iter_mut() {
        if *cell == '\0' {
            *cell = char::from_u32(256 + n).expect("codepoint in Basic Multilingual Plane");
            n += 1;
        }
    }

    map
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_map_round_trips() {
        let map = build_byte_to_char();
        // Every byte must be representable (non-NUL mapping).
        for b in 0usize..256 {
            assert_ne!(map[b], '\0', "byte {b} maps to NUL");
        }
        // All 256 characters must be distinct.
        let mut seen = std::collections::HashSet::new();
        for b in 0usize..256 {
            assert!(seen.insert(map[b]), "duplicate char for byte {b}");
        }
    }

    #[test]
    fn pre_tokenize_splits() {
        // Should split contractions and punctuation separately.
        // Leading spaces are absorbed into the following word chunk.
        let chunks = pre_tokenize("Hello, world! It's fine.");
        // "It" gets the preceding space: " It"
        assert!(
            chunks.contains(&" It"),
            "expected ' It' chunk, got {chunks:?}"
        );
        // "'s" is a separate contraction chunk.
        assert!(
            chunks.contains(&"'s"),
            "expected \"'s\" chunk, got {chunks:?}"
        );
        // Punctuation is separate.
        assert!(chunks.contains(&","), "expected ',' chunk, got {chunks:?}");
    }

    #[test]
    fn bpe_merge_simple() {
        // "ab" should merge if there is a merge rule "a b".
        let mut rank: HashMap<Box<str>, usize> = HashMap::new();
        rank.insert(Box::from("a b"), 0);
        let result = bpe_merge(vec!["a", "b", "c"], &rank);
        assert_eq!(result, vec!["ab", "c"]);
    }

    #[test]
    fn load_tokenizer() {
        // Adjust the path if you placed the model elsewhere.
        let path = "models/Qwen2.5-0.5B/tokenizer.json";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping load_tokenizer: model not found at {path}");
            return;
        }
        let tok = Tokenizer::from_file(path).expect("tokenizer load failed");
        assert!(!tok.vocab.is_empty(), "vocab should not be empty");
        assert!(!tok.merges.is_empty(), "merges should not be empty");
        assert!(
            tok.special_tokens.contains_key("<|endoftext|>"),
            "should have <|endoftext|> special token"
        );
        // Basic encode/decode round-trip.
        let text = "Hello";
        let ids = tok.encode(text);
        assert!(!ids.is_empty(), "encode should produce tokens");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "decode should recover original text");

        // Special token encode.
        let special_id = *tok.special_tokens.get("<|endoftext|>").unwrap();
        let ids2 = tok.encode("<|endoftext|>");
        assert_eq!(
            ids2,
            vec![special_id],
            "special token should encode as its id"
        );

        // Special token decode.
        assert_eq!(
            tok.decode(&[special_id]),
            "<|endoftext|>",
            "special token should decode verbatim"
        );

        // Mixed encode: text + special token + text.
        let ids3 = tok.encode("Hi<|endoftext|>Hi");
        let decoded3 = tok.decode(&ids3);
        assert_eq!(decoded3, "Hi<|endoftext|>Hi");

        println!(
            "vocab={}, merges={}, specials={}",
            tok.vocab.len(),
            tok.merges.len(),
            tok.special_tokens.len()
        );
    }
}

use crate::trie::Trie;
use std::hash::Hash;

impl<T: Clone + Default + PartialEq + Eq + Hash> Trie<T> {
    pub fn calculate_jaro_winkler_score(&self, query: &str, word: &str) -> f32 {
        // Calculate Jaro distance
        let jaro_dist = self.jaro_distance(query, word);

        // Calculate prefix length for Winkler adjustment
        let prefix_length = self.common_prefix_length(query, word);

        // Define the prefix scale (this is typically set to 0.1)
        let prefix_scale = 0.1;

        // Calculate Jaro-Winkler distance
        jaro_dist + (prefix_length as f32 * prefix_scale * (1.0 - jaro_dist))
    }

    pub fn jaro_distance(&self, s1: &str, s2: &str) -> f32 {
        let s1_len = s1.len();
        let s2_len = s2.len();

        if s1_len == 0 {
            return if s2_len == 0 { 1.0 } else { 0.0 };
        }

        let match_distance = (s1_len.max(s2_len) / 2) - 1;

        let mut s1_matches = vec![false; s1_len];
        let mut s2_matches = vec![false; s2_len];

        let mut matches = 0.0;
        let mut transpositions = 0.0;

        for i in 0..s1_len {
            let start = i.saturating_sub(match_distance);
            let end = (i + match_distance + 1).min(s2_len);

            for j in start..end {
                if s2_matches[j] || s1.chars().nth(i) != s2.chars().nth(j) {
                    continue;
                }
                s1_matches[i] = true;
                s2_matches[j] = true;
                matches += 1.0;
                break;
            }
        }

        if matches == 0.0 {
            return 0.0;
        }

        let mut k = 0;
        for i in 0..s1_len {
            if !s1_matches[i] {
                continue;
            }
            while !s2_matches[k] {
                k += 1;
            }
            if s1.chars().nth(i) != s2.chars().nth(k) {
                transpositions += 1.0;
            }
            k += 1;
        }

        let jaro = (matches / s1_len as f32
            + matches / s2_len as f32
            + (matches - transpositions / 2.0) / matches)
            / 3.0;

        jaro
    }

    fn common_prefix_length(&self, s1: &str, s2: &str) -> usize {
        let max_prefix_length = 4;
        let mut length = 0;

        for (c1, c2) in s1.chars().zip(s2.chars()) {
            if c1 != c2 || length >= max_prefix_length {
                break;
            }
            length += 1;
        }

        length
    }
}

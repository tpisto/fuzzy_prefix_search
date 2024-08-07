#![allow(dead_code)]

/// Import necessary modules from the standard library
use std::cell::RefCell; // Allows interior mutability, crucial for our Trie implementation
use std::cmp::min; // For finding the minimum of two values in edit distance calculation
use std::collections::HashMap; // For efficient storage and retrieval of children in TrieNode and data_map in Trie
use std::hash::Hash; // Trait bound for generic type T, allowing it to be used as a key in HashMap
use std::rc::{Rc, Weak}; // For reference counting (Rc) and weak references (Weak) to avoid memory leaks in cyclic structures

/// A node within a Trie structure. Represents a single character in a word.
///
/// # Type Parameters
///
/// - `T`: The type of data associated with each word in the trie.
///   Must implement `Default`, `PartialEq`, and `Clone` for equality checks and cloning.
///
/// # Fields
///
/// - `children`: A map from characters to child `TrieNode`s.
/// - `word`: An optional `String` that holds a complete word if the node represents the end of a word.
/// - `data`: A vector of associated data for this node, allowing multiple data items per word.
/// - `is_end`: A flag indicating if this node represents the end of a word.
/// - `parent`: An optional weak reference to the parent node.
/// - `node_char`: The character used to reach this node from its parent.
#[derive(Default, Debug)]
struct TrieNode<T: Default + PartialEq> {
    children: HashMap<char, Rc<RefCell<TrieNode<T>>>>,
    word: Option<String>,
    data: Vec<T>,
    is_end: bool,
    parent: Option<Weak<RefCell<TrieNode<T>>>>,
    node_char: char,
}

/// A Trie data structure for efficient word storage and retrieval.
///
/// # Type Parameters
///
/// - `T`: The type of data associated with each word in the trie.
///   Must implement `Clone`, `Default`, `PartialEq`, `Eq`, and `Hash` for operations
///   like cloning, equality checks, and hashing.
///
/// # Fields
///
/// - `root`: The root node of the trie, wrapped in `Rc` and `RefCell` for shared ownership and interior mutability.
/// - `data_map`: A map that associates data with trie nodes for quick access during removal operations.
///
/// # Examples
///
/// ```
/// use fuzzy_prefix_search::Trie;
///
/// let mut trie = Trie::new();
/// trie.insert("apple", 1);
/// trie.insert("application", 2);
/// trie.insert("app", 3);
///
/// let results = trie.search_within_distance("appl", 1);
/// for result in results {
///     println!("Found word: {}, with data: {:?}", result.word, result.data);
/// }
/// ```
#[derive(Debug)]
pub struct Trie<T: Clone + Default + PartialEq + Eq + Hash> {
    root: Rc<RefCell<TrieNode<T>>>,
    data_map: HashMap<T, Vec<Weak<RefCell<TrieNode<T>>>>>,
}

/// Represents the result of a search in the trie.
///
/// # Type Parameters
///
/// - `T`: The type of data associated with each word in the trie.
///
/// # Fields
///
/// - `word`: The word found in the trie.
/// - `data`: A vector of associated data for the word.
///
/// # Examples
///
/// ```
/// use fuzzy_prefix_search::{Trie, SearchResult};
///
/// let mut trie = Trie::new();
/// trie.insert("apple", 1);
///
/// let results = trie.search_within_distance("apple", 0);
/// assert_eq!(results[0], SearchResult { word: "apple".to_string(), data: vec![1] });
/// ```
#[derive(Debug)]
pub struct SearchResult<T> {
    pub word: String,
    pub data: Vec<T>,
}

/// Represents the result of a search in the trie with an additional score for similarity.
///
/// # Type Parameters
///
/// - `T`: The type of data associated with each word in the trie.
///
/// # Fields
///
/// - `word`: The word found in the trie.
/// - `data`: A vector of associated data for the word.
/// - `score`: The similarity score of the word to the searched term.
///
/// # Examples
///
/// ```
/// use fuzzy_prefix_search::{Trie, SearchResultWithScore};
///
/// let mut trie = Trie::new();
/// trie.insert("apple", 1);
///
/// let results = trie.search_within_distance_scored("appl", 1);
/// for result in results {
///     println!("Found word: {}, score: {}", result.word, result.score);
/// }
/// ```
#[derive(Debug)]
pub struct SearchResultWithScore<T> {
    pub word: String,
    pub data: Vec<T>,
    pub score: f32,
}

/// Implements `PartialEq` trait for `SearchResult` to enable comparison.
///
/// # Type Parameters
///
/// - `T`: The type of data associated with each word in the trie.
impl<T: PartialEq> PartialEq for SearchResult<T> {
    fn eq(&self, other: &Self) -> bool {
        self.word == other.word && self.data == other.data
    }
}

/// Implements methods for the `Trie` struct.
///
/// # Type Parameters
///
/// - `T`: The type of data associated with each word in the trie.
impl<T: Clone + Default + PartialEq + Eq + Hash> Trie<T> {
    /// Creates a new `Trie` with an empty root node and data map.
    ///
    /// # Returns
    ///
    /// A new `Trie` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use fuzzy_prefix_search::Trie;
    ///
    /// let trie: Trie<i32> = Trie::new();
    /// ```
    pub fn new() -> Self {
        Trie {
            root: Rc::new(RefCell::new(TrieNode {
                node_char: '$', // Root node has no parent character
                ..Default::default()
            })),
            data_map: HashMap::new(),
        }
    }

    /// Inserts a word and associated data into the trie.
    ///
    /// # Parameters
    ///
    /// - `word`: The word to insert into the trie.
    /// - `data`: The data associated with the word.
    ///
    /// # Examples
    ///
    /// ```
    /// use fuzzy_prefix_search::Trie;
    ///
    /// let mut trie = Trie::new();
    /// trie.insert("hello", 1);
    /// trie.insert("world", 2);
    /// ```
    pub fn insert(&mut self, word: &str, data: T) {
        // Start at the root node
        let mut current = Rc::clone(&self.root);
        // Add $ prefix to handle empty strings and provide a common starting point
        let augmented_word = format!("${}", word);

        // Traverse/create the trie structure for each character
        for c in augmented_word.chars() {
            let next = {
                // Borrow the current node mutably
                let mut current_ref = current.borrow_mut();
                // Get or insert the child node for the current character
                current_ref
                    .children
                    .entry(c)
                    .or_insert_with(|| {
                        // Create a new node if it doesn't exist, setting its parent to the current node
                        Rc::new(RefCell::new(TrieNode {
                            parent: Some(Rc::downgrade(&current)),
                            node_char: c,
                            ..Default::default()
                        }))
                    })
                    .clone()
            };
            // Move to the next node
            current = next;
        }

        // Update the final node
        let mut current_ref = current.borrow_mut();
        // Set the word if it hasn't been set (handles cases where one word is a prefix of another)
        if current_ref.word.is_none() {
            current_ref.word = Some(word.to_string());
        }
        // Add the data to this node
        current_ref.data.push(data.clone());
        // Mark this node as the end of a word
        current_ref.is_end = true;

        // Update the data_map for quick access during removal operations
        self.data_map
            .entry(data)
            .or_default()
            .push(Rc::downgrade(&current));
    }

    /// Searches for words within a given edit distance or starting with the given prefix.
    ///
    /// # Parameters
    ///
    /// - `word`: The word to search for in the trie.
    /// - `max_distance`: The maximum edit distance allowed for the search.
    ///
    /// # Returns
    ///
    /// A vector of `SearchResult` containing the words and associated data found within the given distance.
    ///
    /// # Examples
    ///
    /// ```
    /// use fuzzy_prefix_search::Trie;
    ///
    /// let mut trie = Trie::new();
    /// trie.insert("apple", 1);
    /// trie.insert("applet", 2);
    ///
    /// let results = trie.search_within_distance("apple", 1);
    /// assert_eq!(results.len(), 2);
    /// ```
    pub fn search_within_distance(&self, word: &str, max_distance: usize) -> Vec<SearchResult<T>> {
        // Add $ prefix to the search word for consistency with stored words
        let augmented_word = format!("${}", word);
        let augmented_word_length = augmented_word.len();
        // Initialize the rows vector with the first row
        let mut rows = vec![vec![0; augmented_word_length + 1]];
        for i in 0..=augmented_word_length {
            rows[0][i] = i;
        }
        let mut results = Vec::new();

        // Start the recursive search from the root node
        self.search_impl(
            &self.root.borrow(),
            '$',
            &mut rows,
            &augmented_word,
            max_distance,
            &mut results,
            true,
        );

        results
    }

    /// Searches for words within a given edit distance or starting with the given prefix and returns results with a similarity score.
    ///
    /// # Parameters
    ///
    /// - `word`: The word to search for in the trie.
    /// - `max_distance`: The maximum edit distance allowed for the search.
    ///
    /// # Returns
    ///
    /// A vector of `SearchResultWithScore` containing the words, associated data, and similarity scores found within the given distance.
    ///
    /// # Examples
    ///
    /// ```
    /// use fuzzy_prefix_search::Trie;
    ///
    /// let mut trie = Trie::new();
    /// trie.insert("apple", 1);
    ///
    /// let results = trie.search_within_distance_scored("appl", 1);
    /// assert!(!results.is_empty());
    /// for result in results {
    ///     println!("Found word: {}, with score: {}", result.word, result.score);
    /// }
    /// ```
    pub fn search_within_distance_scored(
        &self,
        word: &str,
        max_distance: usize,
    ) -> Vec<SearchResultWithScore<T>> {
        self.search_within_distance(word, max_distance)
            .into_iter()
            .map(|result| {
                let score = self.calculate_jaro_winkler_score(word, &result.word);
                SearchResultWithScore {
                    word: result.word.clone(),
                    data: result.data,
                    score,
                }
            })
            .collect()
    }

    /// Recursive implementation of the search algorithm.
    ///
    /// # Parameters
    ///
    /// - `node`: The current node in the trie.
    /// - `ch`: The character of the current node.
    /// - `rows`: The edit distance matrix.
    /// - `word`: The word to search for in the trie.
    /// - `max_distance`: The maximum edit distance allowed for the search.
    /// - `results`: A vector to store the search results.
    /// - `is_root`: A boolean indicating if the current node is the root node.
    fn search_impl(
        &self,
        node: &TrieNode<T>,
        ch: char,
        rows: &mut Vec<Vec<usize>>,
        word: &str,
        max_distance: usize,
        results: &mut Vec<SearchResult<T>>,
        is_root: bool,
    ) {
        let row_length = word.len() + 1;
        let mut current_row = vec![0; row_length];

        // Initialize the first element of the current row
        current_row[0] = if is_root {
            0
        } else {
            rows.last().unwrap()[0] + 1
        };

        // Calculate edit distances for the current row using dynamic programming
        for i in 1..row_length {
            let insert_or_del = min(current_row[i - 1] + 1, rows.last().unwrap()[i] + 1);
            let replace = if word.chars().nth(i - 1) == Some(ch) {
                rows.last().unwrap()[i - 1] // No change needed
            } else {
                rows.last().unwrap()[i - 1] + 1 // Replacement needed
            };
            current_row[i] = min(insert_or_del, replace);
        }

        // Add the current row to the rows vector
        let should_search_childs = *current_row.iter().min().unwrap() <= max_distance;

        // Check if the current node satisfies the search criteria
        // NOTE: This is the "normal" matching case, where we are looking for a word
        if node.word.is_some() {
            if current_row[row_length - 1] <= max_distance {
                collect_all_words_from_this_node(node, results);
                return;
            }
        }

        rows.push(current_row);

        // Recursively search child nodes if within max_distance
        if should_search_childs {
            for (next_ch, child) in &node.children {
                self.search_impl(
                    &child.borrow(),
                    *next_ch,
                    rows,
                    word,
                    max_distance,
                    results,
                    false,
                );
            }
        }
        // If we are not anymore looking for childs, we can check if we have some insertions or deletions to check
        // NOTE: Here is the magic sauce for the prefix search in addition to the "normal" case
        else if rows.len() > max_distance && rows.len() - max_distance >= word.len() + 1 {
            if rows.len() > word.len() {
                // Scan backward (prefix deletions)
                for i in word.len() - max_distance + 1..word.len() + 2 {
                    if rows[i].last().unwrap() <= &max_distance {
                        collect_all_words_from_this_node(node, results);
                        rows.pop();
                        return;
                    }
                }

                // Scan forward (insertions)
                for i in word.len() + 2..word.len() + 2 + max_distance + 1 {
                    if rows.len() > i {
                        if rows[i].last().unwrap() <= &max_distance {
                            collect_all_words_from_this_node(node, results);
                            rows.pop();
                            return;
                        }
                    }
                }
            }
        }

        // Remove the current row before returning
        rows.pop();
    }

    /// Removes all occurrences of a given data value from the trie.
    ///
    /// # Parameters
    ///
    /// - `data`: The data value to remove from the trie.
    ///
    /// # Examples
    ///
    /// ```
    /// use fuzzy_prefix_search::Trie;
    ///
    /// let mut trie = Trie::new();
    /// trie.insert("apple", 1);
    /// trie.insert("application", 2);
    ///
    /// trie.remove_all(&1);
    /// let results = trie.search_within_distance("apple", 0);
    /// assert!(results.is_empty());
    /// ```
    pub fn remove_all(&mut self, data: &T) {
        if let Some(nodes) = self.data_map.get_mut(data) {
            let mut empty_nodes = Vec::new();
            // Retain only nodes that still have data after removal
            nodes.retain(|node_weak| {
                if let Some(node) = node_weak.upgrade() {
                    let mut node_ref = node.borrow_mut();
                    // Remove the specific data from the node
                    node_ref.data.retain(|d| d != data);

                    // If the node has no data and is a word node, mark it for removal
                    if node_ref.data.is_empty() && node_ref.word.is_some() {
                        node_ref.word = None;
                        node_ref.is_end = false;
                    }

                    if node_ref.data.is_empty() {
                        empty_nodes.push(Rc::clone(&node));
                    }
                    !node_ref.data.is_empty()
                } else {
                    false // Remove weak references that can't be upgraded
                }
            });
            // Remove empty nodes from the trie
            for node in empty_nodes {
                self.remove_node(node);
            }
        }
        // Remove the data entry from the data_map
        self.data_map.remove(data);
    }

    /// Removes a node and its parents if they become empty.
    ///
    /// # Parameters
    ///
    /// - `node`: The node to remove from the trie.
    fn remove_node(&mut self, node: Rc<RefCell<TrieNode<T>>>) {
        let mut current = node;
        loop {
            let parent = {
                let current_ref = current.borrow();
                // If the node has children, a word, or data, stop removal
                if !current_ref.children.is_empty()
                    || current_ref.word.is_some()
                    || !current_ref.data.is_empty()
                {
                    break;
                }
                current_ref.parent.as_ref().and_then(Weak::upgrade)
            };

            if let Some(parent_node) = parent {
                let node_char = current.borrow().node_char;
                // Remove the current node from its parent's children
                parent_node.borrow_mut().children.remove(&node_char);

                // Move up to the parent node
                current = parent_node;
            } else {
                // Reached the root, stop removal
                break;
            }
        }
    }
}

/// Helper function to recursively collect all words and data from a node and its descendants.
///
/// # Parameters
///
/// - `node`: The current node in the trie.
/// - `results`: A vector to store the collected search results.
fn collect_all_words_from_this_node<T: Clone + Default + PartialEq>(
    node: &TrieNode<T>,
    results: &mut Vec<SearchResult<T>>,
) {
    // If this node represents a word, add it to the results
    if let Some(ref node_word) = node.word {
        results.push(SearchResult {
            word: node_word.clone(),
            data: node.data.clone(),
        });
    }

    // Recursively process all child nodes
    for (_, child) in &node.children {
        collect_all_words_from_this_node(&child.borrow(), results);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let mut trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("app", 2);
        trie.insert("application", 3);

        let results = trie.search_within_distance("app", 0);
        assert_eq!(results.len(), 3);
        assert!(results.contains(&SearchResult {
            word: "app".to_string(),
            data: vec![2]
        }));
        assert!(results.contains(&SearchResult {
            word: "apple".to_string(),
            data: vec![1]
        }));
        assert!(results.contains(&SearchResult {
            word: "application".to_string(),
            data: vec![3]
        }));
    }

    #[test]
    fn test_search_with_distance() {
        let mut trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("appl", 2);
        trie.insert("aple", 3);
        trie.insert("applet", 4);

        let results = trie.search_within_distance("apple", 1);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_multiple_data_per_word() {
        let mut trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("apple", 2);
        trie.insert("apple", 3);

        let results = trie.search_within_distance("apple", 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data, vec![1, 2, 3]);
    }

    #[test]
    fn test_remove_all() {
        let mut trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("app", 2);
        trie.insert("application", 2);
        trie.insert("apple", 2);

        trie.remove_all(&2);

        let results = trie.search_within_distance("app", 0);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0],
            SearchResult {
                word: "apple".to_string(),
                data: vec![1]
            }
        );
    }

    #[test]
    fn test_empty_string() {
        let mut trie = Trie::new();
        trie.insert("", 1);
        trie.insert("a", 2);

        let results = trie.search_within_distance("", 0);
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0],
            SearchResult {
                word: "".to_string(),
                data: vec![1]
            }
        );
        // The empty string should be considered a prefix of all words
        assert_eq!(
            results[1],
            SearchResult {
                word: "a".to_string(),
                data: vec![2]
            }
        );
    }

    #[test]
    fn test_long_words() {
        let mut trie = Trie::new();
        let long_word = "supercalifragilisticexpialidocious";
        trie.insert(long_word, 1);

        let results = trie.search_within_distance(long_word, 0);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0],
            SearchResult {
                word: long_word.to_string(),
                data: vec![1]
            }
        );
    }

    #[test]
    fn test_prefix_search() {
        let mut trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("application", 2);
        trie.insert("appreciate", 3);

        let results = trie.search_within_distance("app", 0);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_case_sensitivity() {
        let mut trie = Trie::new();
        trie.insert("Apple", 1);
        trie.insert("apple", 2);

        let results = trie.search_within_distance("Apple", 0);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0],
            SearchResult {
                word: "Apple".to_string(),
                data: vec![1]
            }
        );
    }

    #[test]
    fn test_remove_and_reinsert() {
        let mut trie = Trie::new();
        trie.insert("apple", 1);
        trie.remove_all(&1);
        trie.insert("apple", 2);

        let results = trie.search_within_distance("apple", 0);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0],
            SearchResult {
                word: "apple".to_string(),
                data: vec![2]
            }
        );
    }

    #[test]
    fn test_large_distance_search() {
        let mut trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("banana", 2);
        trie.insert("cherry", 3);

        let results = trie.search_within_distance("grape", 5);
        println!("{:?}", results);
        assert_eq!(results.len(), 2);
        // Check that it matches with apple and banana
        assert!(results.contains(&SearchResult {
            word: "apple".to_string(),
            data: vec![1]
        }));
        assert!(results.contains(&SearchResult {
            word: "banana".to_string(),
            data: vec![2]
        }));
    }

    #[test]
    fn test_prefix_additions_with_distance() {
        let mut trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("app", 2);
        trie.insert("application", 3);
        trie.insert("shouldnotbefound", 4);

        let results = trie.search_within_distance("capp", 1);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_prefix_deletions_with_distance() {
        let mut trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("application", 3);
        // Should not be found
        trie.insert("app", 2);
        trie.insert("shouldnotbefound", 4);

        let results = trie.search_within_distance("ppl", 1);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_prefix_deletions_with_distance_2() {
        let mut trie = Trie::new();
        trie.insert("appleton", 1);
        trie.insert("apple", 2);
        trie.insert("matrix", 3);
        trie.insert("appleton", 2);
        trie.insert("applitix", 3);
        trie.insert("applutux", 4);
        trie.insert("applet", 5);
        trie.insert("capplenex", 6);
        trie.insert("capplunix", 7);
        let results = trie.search_within_distance("applu", 1);
        assert_eq!(results.len(), 6);
        // Includes capplunix
        assert!(results.contains(&SearchResult {
            word: "capplunix".to_string(),
            data: vec![7]
        }));
        // Includes applitix
        assert!(results.contains(&SearchResult {
            word: "applitix".to_string(),
            data: vec![3]
        }));
    }
}

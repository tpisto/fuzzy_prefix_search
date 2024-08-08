use std::cmp::min; // For finding the minimum of two values in edit distance calculation
use std::collections::HashMap; // For efficient storage and retrieval of children in TrieNode and data_map in Trie
use std::hash::Hash; // Trait bound for generic type T, allowing it to be used as a key in HashMap
use std::sync::{Arc, RwLock};

// This should be safe because we only access Trie through Arc<RwLock<>> and TrieData is not directly accessed (not public)
unsafe impl<T: Clone + Default + PartialEq + Eq + Hash> Send for TrieData<T> {}
unsafe impl<T: Clone + Default + PartialEq + Eq + Hash> Sync for TrieData<T> {}

/// A node within a Trie structure. Represents a single character in a word.
///
/// # Type Parameters
///
/// - `T`: The type of data associated with each word in the trie.
///   Must implement `Default`, `PartialEq`, and `Clone` for equality checks and cloning.
#[derive(Default, Debug)]
struct TrieNode<T: Default + PartialEq> {
    children: HashMap<char, TrieNode<T>>,
    word: Option<String>,
    data: Vec<T>,
    is_end: bool,
}

/// A Trie data structure for efficient word storage and retrieval.
///
/// # Type Parameters
///
/// - `T`: The type of data associated with each word in the trie.
///   Must implement `Clone`, `Default`, `PartialEq`, `Eq`, and `Hash` for operations
///   like cloning, equality checks, and hashing.
#[derive(Debug)]
pub(crate) struct TrieData<T: Clone + Default + PartialEq + Eq + Hash> {
    root: TrieNode<T>,
    data_map: HashMap<T, Vec<*mut TrieNode<T>>>,
}

/// A thread-safe wrapper for the Trie data structure using RwLock for synchronization.
///
/// # Type Parameters
///
/// - `T`: The type of data associated with each word in the trie.
#[derive(Debug)]
pub struct Trie<T: Clone + Default + PartialEq + Eq + Hash> {
    trie_data: Arc<RwLock<TrieData<T>>>,
}

impl<T: Clone + Default + PartialEq + Eq + Hash> Trie<T> {
    /// Creates a new thread-safe `Trie` with an empty root node and data map.
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
            trie_data: Arc::new(RwLock::new(TrieData {
                root: TrieNode {
                    ..Default::default()
                },
                data_map: HashMap::default(),
            })),
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
    pub fn insert(&self, word: &str, data: T) {
        let mut trie_data = self.trie_data.write().unwrap();
        trie_data.insert(word, data);
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
        let trie_data = self.trie_data.read().unwrap();
        trie_data.search_within_distance(word, max_distance)
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
        let trie_data = self.trie_data.read().unwrap();
        trie_data.search_within_distance_scored(word, max_distance)
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
    pub fn remove_all(&self, data: &T) {
        let mut trie_data = self.trie_data.write().unwrap();
        trie_data.remove_all(data);
    }
}

/// Represents the result of a search in the trie.
///
/// # Type Parameters
///
/// - `T`: The type of data associated with each word in the trie.
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

/// Implements methods for the `TrieData` struct.
///
/// # Type Parameters
///
/// - `T`: The type of data associated with each word in the trie.
impl<T: Clone + Default + PartialEq + Eq + Hash> TrieData<T> {
    /// Inserts a word and associated data into the trie.
    ///
    /// # Parameters
    ///
    /// - `word`: The word to insert into the trie.
    /// - `data`: The data associated with the word.
    fn insert(&mut self, word: &str, data: T) {
        // Start at the root node
        let mut current = &mut self.root;
        // Add $ prefix to handle empty strings and provide a common starting point
        let augmented_word = format!("${}", word);

        // Traverse/create the trie structure for each character
        for c in augmented_word.chars() {
            current = current.children.entry(c).or_insert_with(|| TrieNode {
                ..Default::default()
            });
        }

        // Update the final node
        if current.word.is_none() {
            current.word = Some(word.to_string());
        }
        current.data.push(data.clone());
        current.is_end = true;

        // Update the data_map for quick access during removal operations
        self.data_map
            .entry(data)
            .or_default()
            .push(current as *mut _);
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
    fn search_within_distance(&self, word: &str, max_distance: usize) -> Vec<SearchResult<T>> {
        // Add $ prefix to the search word for consistency with stored words
        let augmented_word = format!("${}", word);
        let last_row: Vec<usize> = (0..=augmented_word.len()).collect();
        let mut results = Vec::new();

        // Start the recursive search from the root node
        self.search_recursive(
            &self.root,
            '$',
            &last_row,
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
    fn search_within_distance_scored(
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
    fn search_recursive(
        &self,
        node: &TrieNode<T>,
        ch: char,
        last_row: &Vec<usize>,
        word: &str,
        max_distance: usize,
        results: &mut Vec<SearchResult<T>>,
        is_root: bool,
    ) {
        let row_length = word.len() + 1;
        let mut current_row = vec![0; row_length];

        // Initialize the first element of the current row
        current_row[0] = if is_root { 0 } else { last_row[0] + 1 };

        // Calculate Levenshtein edit distances for the current row
        // You can debug, by printing current row and checking from here:
        // https://phiresky.github.io/levenshtein-demo/
        for i in 1..row_length {
            let insert_or_del = min(current_row[i - 1] + 1, last_row[i] + 1);
            let replace = if word.chars().nth(i - 1) == Some(ch) {
                last_row[i - 1] // No change needed
            } else {
                last_row[i - 1] + 1 // Replacement needed
            };
            current_row[i] = min(insert_or_del, replace);
        }

        // Check if the current node satisfies the search criteria
        if node.word.is_some() {
            if current_row[row_length - 1] <= max_distance {
                collect_all_words_from_this_node(node, results);
                return;
            }
        }
        // Prefix match, also taking into account the max_distance (insertions or deletions before the word)
        else if current_row[0] >= word.len() - max_distance
            && current_row.last().unwrap() <= &max_distance
        {
            collect_all_words_from_this_node(node, results);
            return;
        }

        // Recursively search child nodes if within max_distance
        if *current_row.iter().min().unwrap() <= max_distance {
            for (next_ch, child) in &node.children {
                self.search_recursive(
                    child,
                    *next_ch,
                    &current_row,
                    word,
                    max_distance,
                    results,
                    false,
                );
            }
        }
    }

    /// Removes all occurrences of a given data value from the trie.
    ///
    /// # Parameters
    ///
    /// - `data`: The data value to remove from the trie.
    fn remove_all(&mut self, data: &T) {
        if let Some(nodes) = self.data_map.get_mut(data) {
            let mut empty_nodes = Vec::new();
            // Retain only nodes that still have data after removal
            nodes.retain(|&node_ptr| {
                let node = unsafe { &mut *node_ptr };
                // Remove the specific data from the node
                node.data.retain(|d| d != data);

                // If the node has no data and is a word node, mark it for removal
                if node.data.is_empty() && node.word.is_some() {
                    node.word = None;
                    node.is_end = false;
                }

                if node.data.is_empty() {
                    empty_nodes.push(node_ptr);
                }
                !node.data.is_empty()
            });
            // Remove empty nodes from the trie
            for &node_ptr in &empty_nodes {
                self.remove_node(unsafe { &mut *node_ptr });
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
    fn remove_node(&mut self, node: &mut TrieNode<T>) {
        let mut current = node as *mut TrieNode<T>;
        loop {
            let parent_ptr = {
                let current_ref = unsafe { &mut *current };
                // If the node has children, a word, or data, stop removal
                if !current_ref.children.is_empty()
                    || current_ref.word.is_some()
                    || !current_ref.data.is_empty()
                {
                    break;
                }
                // Find the parent node by looking up in data_map
                self.data_map
                    .iter()
                    .find_map(|(_, nodes)| nodes.iter().find(|&&ptr| ptr == current).copied())
            };

            if let Some(parent_ptr) = parent_ptr {
                let parent = unsafe { &mut *parent_ptr };
                let node_char = unsafe { &*current }
                    .word
                    .as_ref()
                    .unwrap()
                    .chars()
                    .next()
                    .unwrap();
                // Remove the current node from its parent's children
                parent.children.remove(&node_char);

                // Move up to the parent node
                current = parent_ptr;
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
        collect_all_words_from_this_node(child, results);
    }
}

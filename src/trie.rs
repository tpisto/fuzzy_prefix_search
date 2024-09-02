use core::fmt;
use std::alloc::{alloc, dealloc, Layout};
use std::cmp::min;
use std::collections::HashMap; // For efficient storage and retrieval of children in TrieNode and data_map in Trie
use std::fmt::Debug;
use std::hash::Hash; // Trait bound for generic type T, allowing it to be used as a key in HashMap
use std::ptr;
use std::sync::{Arc, RwLock};

unsafe impl<T: Clone + Default + PartialEq + Eq + Hash + Debug> Send for TrieData<T> {}
unsafe impl<T: Clone + Default + PartialEq + Eq + Hash + Debug> Sync for TrieData<T> {}

/// A node within a Trie structure. Represents a single character in a word.
///
/// # Type Parameters
///
/// - `T`: The type of data associated with each word in the trie.
///   Must implement `PartialEq` for equality checks and cloning.
struct TrieNode<T: Default + PartialEq> {
    children: HashMap<char, *mut TrieNode<T>>,
    parent: *mut TrieNode<T>,
    word: Option<String>,
    data: Vec<T>,
    is_end: bool,
}

impl<T: Default + PartialEq> TrieNode<T> {
    /// Creates a new TrieNode and returns a raw pointer to it.
    ///
    /// # Safety
    ///
    /// This function uses unsafe code to allocate memory and initialize the TrieNode.
    /// The caller is responsible for properly managing the returned pointer.
    fn new() -> *mut Self {
        let layout = Layout::new::<Self>();
        let ptr = unsafe { alloc(layout) as *mut Self };
        unsafe {
            ptr::write(
                ptr,
                TrieNode {
                    children: HashMap::new(),
                    parent: ptr::null_mut(),
                    word: None,
                    data: Vec::new(),
                    is_end: false,
                },
            );
        }
        ptr
    }

    /// Drops a TrieNode, deallocating its memory.
    ///
    /// # Safety
    ///
    /// This function uses unsafe code to deallocate memory.
    /// The caller must ensure that the pointer is valid and that this node is no longer in use.
    unsafe fn drop(ptr: *mut Self) {
        ptr::drop_in_place(ptr);
        dealloc(ptr as *mut u8, Layout::new::<Self>());
    }
}

pub(crate) struct TrieData<T: Clone + Default + PartialEq + Eq + Hash + Debug> {
    root: *mut TrieNode<T>,
    data_map: HashMap<T, Vec<*mut TrieNode<T>>>,
}

impl<T: Clone + Default + PartialEq + Eq + Hash + Debug> Drop for TrieData<T> {
    fn drop(&mut self) {
        self.drop_node(self.root);
    }
}

impl<T: Clone + Default + PartialEq + Eq + Hash + Debug> TrieData<T> {
    fn drop_node(&mut self, node: *mut TrieNode<T>) {
        if !node.is_null() {
            let node = unsafe { &mut *node };
            for child in node.children.values() {
                self.drop_node(*child);
            }
            unsafe { TrieNode::drop(node) };
        }
    }

    /// Inserts a word and associated data into the trie.
    ///
    /// # Parameters
    ///
    /// - `word`: The word to insert into the trie.
    /// - `data`: The data associated with the word.
    fn insert(&mut self, word: &str, data: T) {
        let mut current = self.root;
        let augmented_word = format!("${}", word);

        for c in augmented_word.chars() {
            let node = unsafe { &mut *current };
            current = *node.children.entry(c).or_insert_with(|| {
                let new_node = TrieNode::new();
                unsafe { (*new_node).parent = current };
                new_node
            });
        }

        let node = unsafe { &mut *current };
        node.word = Some(word.to_string());
        node.data.push(data.clone());
        node.is_end = true;

        self.data_map.entry(data).or_default().push(current);
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
        let augmented_word = format!("${}", word);
        let last_row: Vec<usize> = (0..=augmented_word.len()).collect();
        let mut results = Vec::new();

        self.search_recursive(
            self.root,
            '$',
            &last_row,
            &augmented_word,
            augmented_word.chars().count() as u8,
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
                    word: result.word,
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
    /// - `last_row`: The previous row of the edit distance matrix.
    /// - `word`: The word to search for in the trie.
    /// - `word_char_count`: The number of characters in the search word.
    /// - `max_distance`: The maximum edit distance allowed for the search.
    /// - `results`: A mutable vector to store the search results.
    /// - `is_root`: A boolean indicating if the current node is the root node.
    fn search_recursive(
        &self,
        node: *mut TrieNode<T>,
        ch: char,
        last_row: &Vec<usize>,
        word: &str,
        word_char_count: u8,
        max_distance: usize,
        results: &mut Vec<SearchResult<T>>,
        is_root: bool,
    ) {
        let row_length = (word_char_count + 1) as usize;
        let mut current_row = vec![0; row_length];

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

        let node = unsafe { &*node };

        // Check if the current node satisfies the search criteria
        if node.word.is_some() {
            if current_row[row_length - 1] <= max_distance {
                self.collect_all_words_from_this_node(node, results);
                return;
            }
        }
        // Prefix match, also taking into account the max_distance (insertions or deletions before the word)
        else if current_row[0] >= word.len() - max_distance
            && current_row.last().unwrap() <= &max_distance
        {
            self.collect_all_words_from_this_node(node, results);
            return;
        }

        if *current_row.iter().min().unwrap() <= max_distance {
            for (next_ch, child) in &node.children {
                self.search_recursive(
                    *child,
                    *next_ch,
                    &current_row,
                    word,
                    word_char_count,
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
        if let Some(nodes) = self.data_map.get(data) {
            let nodes_to_remove: Vec<_> = nodes.clone();
            for &node_ptr in nodes_to_remove.iter() {
                let node = unsafe { &mut *node_ptr };
                node.data.retain(|d| d != data);

                if node.data.is_empty() {
                    if node.word.is_some() {
                        node.word = None;
                        node.is_end = false;
                    }
                    self.remove_node(node_ptr);
                }
            }
        }
        self.data_map.remove(data);
    }

    /// Removes a node and its parents if they become empty.
    ///
    /// # Parameters
    ///
    /// - `node_ptr`: A raw pointer to the node to be removed.
    fn remove_node(&mut self, mut node_ptr: *mut TrieNode<T>) {
        while !node_ptr.is_null() {
            let node = unsafe { &mut *node_ptr };

            // If the node still has data or is the end of a word, we stop here
            if !node.data.is_empty() || node.is_end {
                break;
            }

            // If the node has children, we can't remove it
            if !node.children.is_empty() {
                break;
            }

            // At this point, we know we can remove this node
            let parent_ptr = node.parent;

            // If there's no parent, this must be the root node, so we stop
            if parent_ptr.is_null() {
                break;
            }

            let parent = unsafe { &mut *parent_ptr };

            // Find and remove this node from its parent's children
            parent.children.retain(|_, &mut child| child != node_ptr);

            // Drop the current node
            unsafe { TrieNode::drop(node_ptr) };

            // Move up to the parent for the next iteration
            node_ptr = parent_ptr;
        }
    }

    /// Collects all words and associated data from a node and its descendants.
    ///
    /// # Parameters
    ///
    /// - `node`: The node to start collecting from.
    /// - `results`: A mutable vector to store the collected results.
    fn collect_all_words_from_this_node(
        &self,
        node: &TrieNode<T>,
        results: &mut Vec<SearchResult<T>>,
    ) {
        if let Some(node_word) = &node.word {
            results.push(SearchResult {
                word: node_word.clone(),
                data: node.data.clone(),
            });
        }

        for child in node.children.values() {
            let child_node = unsafe { &**child };
            self.collect_all_words_from_this_node(child_node, results);
        }
    }
}

/// A thread-safe wrapper for the Trie data structure.
///
/// # Type Parameters
///
/// - `T`: The type of data associated with each word in the trie.
pub struct Trie<T: Clone + Default + PartialEq + Eq + Hash + Debug> {
    trie_data: Arc<RwLock<TrieData<T>>>,
}

impl<T: Clone + Default + PartialEq + Eq + Hash + Debug> Trie<T> {
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
                root: TrieNode::new(),
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
    /// let trie = Trie::new();
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
    /// let trie = Trie::new();
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
    /// let trie = Trie::new();
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
    /// let trie = Trie::new();
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
#[derive(Debug, PartialEq)]
pub struct SearchResultWithScore<T> {
    pub word: String,
    pub data: Vec<T>,
    pub score: f32,
}

impl<T: PartialEq> PartialEq for SearchResult<T> {
    fn eq(&self, other: &Self) -> bool {
        self.word == other.word && self.data == other.data
    }
}

// Custom debuggers and formatters so that we will be able to see the 
// Trie data structure in a more readable way (not just pointer addresses)

impl<T: Clone + Default + PartialEq + Eq + Hash + Debug> fmt::Debug for Trie<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let trie_data = self.trie_data.read().unwrap();
        f.debug_struct("Trie")
            .field("trie_data", &*trie_data)
            .finish()
    }
}

impl<T: Clone + Default + PartialEq + Eq + Hash + Debug> fmt::Debug for TrieData<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrieData")
            .field("root", &unsafe { &*self.root })
            .field("data_map", &self.data_map)
            .finish()
    }
}

impl<T: Default + PartialEq + Debug> fmt::Debug for TrieNode<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrieNode")
            .field(
                "children",
                &self
                    .children
                    .iter()
                    .map(|(k, v)| (k, unsafe { &**v }))
                    .collect::<HashMap<_, _>>(),
            )
            .field("word", &self.word)
            .field("data", &self.data)
            .field("is_end", &self.is_end)
            .finish()
    }
}

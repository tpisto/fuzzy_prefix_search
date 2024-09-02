use core::fmt;
use std::alloc::{alloc, dealloc, Layout};
use std::cmp::min;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::ptr;
use std::sync::{Arc, RwLock};

unsafe impl<T: Clone + Default + PartialEq + Eq + Hash + Debug> Send for TrieData<T> {}
unsafe impl<T: Clone + Default + PartialEq + Eq + Hash + Debug> Sync for TrieData<T> {}

struct TrieNode<T: Default + PartialEq> {
    children: HashMap<char, *mut TrieNode<T>>,
    parent: *mut TrieNode<T>,
    word: Option<String>,
    data: Vec<T>,
    is_end: bool,
}

impl<T: Default + PartialEq> TrieNode<T> {
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

        for i in 1..row_length {
            let insert_or_del = min(current_row[i - 1] + 1, last_row[i] + 1);
            let replace = if word.chars().nth(i - 1) == Some(ch) {
                last_row[i - 1]
            } else {
                last_row[i - 1] + 1
            };
            current_row[i] = min(insert_or_del, replace);
        }

        let node = unsafe { &*node };
        let has_word = node.word.is_some();
        let is_end_of_word = current_row[row_length - 1] <= max_distance;
        let is_potential_match = current_row[0] >= word.len() - max_distance
            && current_row.last().unwrap() <= &max_distance;

        if (has_word && is_end_of_word) || (!has_word && is_potential_match) {
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

pub struct Trie<T: Clone + Default + PartialEq + Eq + Hash + Debug> {
    trie_data: Arc<RwLock<TrieData<T>>>,
}

impl<T: Clone + Default + PartialEq + Eq + Hash + Debug> Trie<T> {
    pub fn new() -> Self {
        Trie {
            trie_data: Arc::new(RwLock::new(TrieData {
                root: TrieNode::new(),
                data_map: HashMap::default(),
            })),
        }
    }

    pub fn insert(&self, word: &str, data: T) {
        let mut trie_data = self.trie_data.write().unwrap();
        trie_data.insert(word, data);
    }

    pub fn search_within_distance(&self, word: &str, max_distance: usize) -> Vec<SearchResult<T>> {
        let trie_data = self.trie_data.read().unwrap();
        trie_data.search_within_distance(word, max_distance)
    }

    pub fn search_within_distance_scored(
        &self,
        word: &str,
        max_distance: usize,
    ) -> Vec<SearchResultWithScore<T>> {
        let trie_data = self.trie_data.read().unwrap();
        trie_data.search_within_distance_scored(word, max_distance)
    }

    pub fn remove_all(&self, data: &T) {
        let mut trie_data = self.trie_data.write().unwrap();
        trie_data.remove_all(data);
    }
}

#[derive(Debug)]
pub struct SearchResult<T> {
    pub word: String,
    pub data: Vec<T>,
}

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

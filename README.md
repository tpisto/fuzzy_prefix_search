[<img alt="github" src="https://img.shields.io/badge/github-tpisto/fuzzy_prefix_search-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/tpisto/fuzzy_prefix_search)
[<img alt="crates.io" src="https://img.shields.io/crates/v/fuzzy_prefix_search.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/fuzzy_prefix_search)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-fuzzy_prefix_search-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/fuzzy_prefix_search)
[<img alt="tests" src="https://img.shields.io/github/actions/workflow/status/tpisto/fuzzy_prefix_search/rust.yml?branch=main&style=for-the-badge" height="20">](https://github.com/tpisto/fuzzy_prefix_search/actions?query=branch%3Amain)

<img src="https://github.com/user-attachments/assets/94bfcebc-4ecd-4911-9eb9-13d0288e3e5b" width="100px">

# Fuzzy Prefix Search

Flexible Trie implementation in Rust for fuzzy prefix string searching and auto-completion.

Documentation:
-   [API reference (docs.rs)](https://docs.rs/fuzzy_prefix_search)

## Features

- Prefix-based fuzzy searching (Levenshtein distance)
- Fuzzy search with customizable edit distance
- Multiple data associations per word
- Jaro-Winkler similarity scoring for search results
- Thread safe
- No dependencies

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
fuzzy_prefix_search = "0.1"
```

## Usage

Here's a quick example of how to use the Fuzzy Prefix Search:

```rust
use fuzzy_prefix_search::Trie;

fn main() {
    let mut trie = Trie::new();

    // Insert words with associated data
    trie.insert("apple", 1);
    trie.insert("application", 2);
    trie.insert("appetite", 3);

    // Perform a fuzzy search
    let results = trie.search_within_distance("appl", 1);

    for result in results {
        println!("Word: {}, Data: {:?}", result.word, result.data);
    }

    // Search with similarity scoring
    let scored_results = trie.search_within_distance_scored("aple", 2);

    for result in scored_results {
        println!("Word: {}, Data: {:?}, Score: {}", result.word, result.data, result.score);
    }
}
```

## Advanced Usage

### Custom Data Types

The Trie supports any data type that implements `Clone`, `Default`, `PartialEq`, `Eq`, and `Hash`:

```rust
#[derive(Clone, Default, PartialEq, Eq, Hash)]
struct CustomData {
    id: u32,
    value: String,
}

let mut trie = Trie::new();
trie.insert("example", CustomData { id: 1, value: "Test".to_string() });
```

### Removing Data

You can remove all occurrences of a specific data value:

```rust
trie.remove_all(&2);
```

## Performance

- O(k) time complexity for insertion, where k is the length of the word
- Space-efficient storage using a tree structure with shared prefixes
- For thread safe we use Arc + RwLock that would need further optimizations
- TODO: Benchmarks, optimizations, algorithm selection...

Caveat Emptor: we use *unsafe* in deletes for 2x read performance compared to Rc/RefCell approach.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Resources

- https://phiresky.github.io/levenshtein-demo/
- https://www.geeksforgeeks.org/trie-insert-and-search/
- http://stevehanov.ca/blog/?id=114
- https://murilo.wordpress.com/2011/02/01/fast-and-easy-levenshtein-distance-using-a-trie-in-c/
- https://blog.vjeux.com/2011/c/c-fuzzy-search-with-trie.html
  
## License

This project is licensed under the MIT or Apache 2.0 License - see the [LICENSE](LICENSE) files for details.

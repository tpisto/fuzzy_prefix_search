# Fuzzy Prefix Search

Flexible Trie implementation in Rust for fuzzy prefix string searching and auto-completion.

## Features

- Fast prefix-based fuzzy searching (Levenshtein distance)
- Fuzzy search with customizable edit distance
- Multiple data associations per word
- Jaro-Winkler similarity scoring for search results
- No unsafe
- No dependencies

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
fuzzy_prefix_search = "0.1.0"
```

## Usage

Here's a quick example of how to use the Fuzzy Search Trie:

```rust
use fuzzy_search_trie::trie::Trie;

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
- TODO: Benchmarks, optimizations, algorithm selection...

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT or Apache 2.0 License - see the [LICENSE](LICENSE) files for details.

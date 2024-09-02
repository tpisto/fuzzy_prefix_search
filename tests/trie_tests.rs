#[cfg(test)]
mod tests {
    use fuzzy_prefix_search::*;

    #[test]
    fn test_insert_and_search() {
        let trie = Trie::new();
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
        let trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("appl", 2);
        trie.insert("aple", 3);
        trie.insert("applet", 4);

        let results = trie.search_within_distance("apple", 1);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_multiple_data_per_word() {
        let trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("apple", 2);
        trie.insert("apple", 3);

        let results = trie.search_within_distance("apple", 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data, vec![1, 2, 3]);
    }

    #[test]
    fn test_remove_all() {
        let trie = Trie::new();
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
    fn test_remove_long_words() {
        let trie = Trie::new();
        trie.insert("aava", "100");
        trie.insert("auto", "101");
        trie.insert("aarrekartta", "101");
        trie.insert("aamu", "102");
        trie.insert("auto", "103");
        trie.insert("aarremartta", "104");

        trie.remove_all(&"101");

        let results = trie.search_within_distance("aa", 1);
        assert_eq!(results.len(), 4);

        assert!(results.contains(&SearchResult {
            word: "aava".to_string(),
            data: vec!["100"]
        }));
        assert!(results.contains(&SearchResult {
            word: "aarremartta".to_string(),
            data: vec!["104"]
        }));
        assert!(results.contains(&SearchResult {
            word: "aamu".to_string(),
            data: vec!["102"]
        }));
        assert!(results.contains(&SearchResult {
            word: "auto".to_string(),
            data: vec!["103"]
        }));
    }

    #[test]
    fn test_remove_with_more_varying_words() {
        let trie = Trie::new();
        trie.insert("abc", 0);
        trie.insert("cde", 1);
        trie.insert("def", 2);
        trie.insert("efg", 3);
        trie.insert("fgh", 4);
        trie.insert("ghi", 5);
        trie.remove_all(&0);

        assert_eq!(trie.search_within_distance("abc", 0).len(), 0);
    }

    #[test]
    fn test_empty_string() {
        let trie = Trie::new();
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
        let trie = Trie::new();
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
        let trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("application", 2);
        trie.insert("appreciate", 3);

        let results = trie.search_within_distance("app", 0);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_case_sensitivity() {
        let trie = Trie::new();
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
        let trie = Trie::new();
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
        let trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("banana", 2);
        trie.insert("cherry", 3);
        trie.insert("churry", 4);

        let results = trie.search_within_distance("grape", 4);
        assert_eq!(results.len(), 3);

        // Check that it matches with apple and banana
        assert!(results.contains(&SearchResult {
            word: "apple".to_string(),
            data: vec![1]
        }));
        assert!(results.contains(&SearchResult {
            word: "banana".to_string(),
            data: vec![2]
        }));
        assert!(results.contains(&SearchResult {
            word: "cherry".to_string(),
            data: vec![3]
        }));
    }

    #[test]
    fn test_prefix_additions_with_distance() {
        let trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("app", 2);
        trie.insert("application", 3);
        trie.insert("shouldnotbefound", 4);

        let results = trie.search_within_distance("capp", 1);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_prefix_deletions_with_distance() {
        let trie = Trie::new();
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
        let trie = Trie::new();
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

    #[test]
    fn test_search_within_distance_scored() {
        let trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("appl", 2);
        trie.insert("ale", 3);
        trie.insert("applet", 4);

        let results = trie.search_within_distance_scored("apple", 1);
        assert_eq!(results.len(), 3);

        // Check that the results match
        assert!(results.contains(&SearchResultWithScore {
            word: "appl".to_string(),
            data: vec![2],
            score: 0.96
        }));
        assert!(results.contains(&SearchResultWithScore {
            word: "apple".to_string(),
            data: vec![1],
            score: 1.0
        }));
        assert!(results.contains(&SearchResultWithScore {
            word: "applet".to_string(),
            data: vec![4],
            score: 0.96666664
        }));
    }

    #[test]
    fn test_search_within_distance_scored_prefix() {
        let trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("application", 2);
        trie.insert("appreciation", 3);

        let results = trie.search_within_distance_scored("app", 0);
        assert_eq!(results.len(), 3);

        // Check that it matches with the scores
        assert!(results.contains(&SearchResultWithScore {
            word: "apple".to_string(),
            data: vec![1],
            score: 0.90666664
        }));
        assert!(results.contains(&SearchResultWithScore {
            word: "application".to_string(),
            data: vec![2],
            score: 0.830303
        }));
        assert!(results.contains(&SearchResultWithScore {
            word: "appreciation".to_string(),
            data: vec![3],
            score: 0.825
        }));
    }

    #[test]
    fn test_search_within_distance_scored_empty_string() {
        let trie = Trie::new();
        trie.insert("", 1);
        trie.insert("a", 2);

        let results = trie.search_within_distance_scored("", 0);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].word, "");
        assert_eq!(results[0].score, 1.0);
        assert!(results[1].score < 1.0);
    }

    #[test]
    fn test_search_within_distance_scored_case_sensitivity() {
        let trie = Trie::new();
        trie.insert("Apple", 1);
        trie.insert("apple", 2);

        let results = trie.search_within_distance_scored("Apple", 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].word, "Apple");
        assert_eq!(results[0].score, 1.0);
    }

    #[test]
    fn test_search_within_distance_scored_large_distance() {
        let trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("banana", 2);
        trie.insert("cherry", 3);
        trie.insert("date", 4);

        let results = trie.search_within_distance_scored("grape", 4);
        assert!(!results.is_empty());

        // Check that the results match
        assert!(results.contains(&SearchResultWithScore {
            word: "apple".to_string(),
            data: vec![1],
            score: 0.59999996
        }));
        assert!(results.contains(&SearchResultWithScore {
            word: "banana".to_string(),
            data: vec![2],
            score: 0.45555556
        }));
        assert!(results.contains(&SearchResultWithScore {
            word: "cherry".to_string(),
            data: vec![3],
            score: 0.41111112
        }));
        assert!(results.contains(&SearchResultWithScore {
            word: "date".to_string(),
            data: vec![4],
            score: 0.6333333
        }));
    }

    #[test]
    fn test_search_within_distance_scored_multiple_data() {
        let trie = Trie::new();
        trie.insert("apple", 1);
        trie.insert("apple", 2);
        trie.insert("applet", 3);

        let results = trie.search_within_distance_scored("apple", 1);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].word, "apple");
        assert_eq!(results[0].data, vec![1, 2]);
        assert_eq!(results[0].score, 1.0);
    }

    #[cfg(test)]
    mod multithreading_tests {
        use super::*;
        use std::sync::Arc;
        use std::thread;

        #[test]
        fn test_concurrent_insertions() {
            let trie = Arc::new(Trie::new());
            let mut handles = vec![];

            for i in 100..200 {
                let trie_clone = Arc::clone(&trie);
                let handle = thread::spawn(move || {
                    let word = format!("word{}", i);
                    trie_clone.insert(&word, i);
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }

            for i in 100..200 {
                let word = format!("word{}", i);
                let results = trie.search_within_distance(&word, 0);
                println!("{:?}", results);
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].data[0], i);
            }
        }

        #[test]
        fn test_concurrent_searches() {
            let trie = Trie::new();
            for i in 0..1000 {
                trie.insert(&format!("word{}", i), i);
            }

            let trie = Arc::new(trie);
            let mut handles = vec![];

            for i in 0..100 {
                let trie_clone = Arc::clone(&trie);
                let handle = thread::spawn(move || {
                    let word = format!("word{}", i * 10);
                    let results = trie_clone.search_within_distance(&word, 1);
                    assert!(!results.is_empty());
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        }
    }
}

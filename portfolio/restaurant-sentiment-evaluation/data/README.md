# Dataset

The public reconstruction targets the Yelp subset of:

**Kotzias, D. (2015). Sentiment Labelled Sentences. UCI Machine Learning Repository.**

DOI: https://doi.org/10.24432/C57604

Dataset page:

https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences

UCI documents:

- 3,000 total labelled sentences,
- 1,000 from Yelp,
- 1,000 from Amazon,
- 1,000 from IMDB,
- 500 positive and 500 negative for each source,
- CC BY 4.0 licensing.

After downloading the UCI archive, use:

`sentiment labelled sentences/yelp_labelled.txt`

The parser in this project expects one review sentence followed by a tab and binary label `0` or `1`.

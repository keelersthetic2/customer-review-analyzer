# customer-review-analyzer
Customer Review Analyzer that uses both data science and deep learning to classify customer reviews by sentiment and extract insights like trends, key words, and themes.

I will be using the IMDB movie review dataset at the following link: https://ai.stanford.edu/~amaas/data/sentiment/

## Project Progress

### Phase 1: Data Preparation (done)
- Loaded raw IMDB dataset (not in repo).
- Combined reviews into a single DataFrame along with their label of pos/neg and their trian/test split (50,000 rows).
- Verified the class and split balances.
- Analyzed the review character lengths (min=32, max=13,704, mean=~1309, median=~970).
- Saved dataset locally as data/imdb_reviews.csv for future phases.

Next: Phase 2 (Cleaning + EDA)

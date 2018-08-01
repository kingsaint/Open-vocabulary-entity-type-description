## About New Dataset
To decrease the percentage of 'Human' entities in our dataset, we draw this new sample from Wikidata. However, note that 'Human' is the most dominant type in Wikidata that has a long-tail distribution of entity types. We downsampled 'Human' instances in this new dataset to accommodate a variety of other types. Using this new dataset, we obtained the following results, which is still better than the baselines across all metrics.

| Model| B-1| B-2 | B-3 | B-4 | ROUGE-L | METEOR | CIDERr |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Our Model | 0.578 | 0.507 | 0.436 | 0.397 | 0.676 | 0.349 | 3.557 |

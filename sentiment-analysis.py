from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
classifier = pipeline('sentiment-analysis')
result = classifier('We are very happy to introduce pipeline to the transformers repository.')
print(result)

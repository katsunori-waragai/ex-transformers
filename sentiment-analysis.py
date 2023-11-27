from transformers import pipeline

if __name__ == "__main__":
    # Allocate a pipeline for sentiment-analysis
    classifier = pipeline('sentiment-analysis')
    text = 'We are very happy to introduce pipeline to the transformers repository.'
    print(text)
    result = classifier(text)
    print(result)
    while True:
        text = input("English text: ")
        if len(text) == 0:
            break
        print(text)
        result = classifier(text)
        print(result)

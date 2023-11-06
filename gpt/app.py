import os
import openai
import argparse
import csv
import re

from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score, pair_confusion_matrix, precision_score, recall_score, accuracy_score
def generate_data():
    answers = []
    for _ in range(10):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """
                    You are a helpful assistant. That is specialised in generating offensive and 
                    unharmful tweets. When generating the tweet end it with OFF if the tweet is offensive and with NOT if the tweet is not offensive.
                """},
                {"role": "user", "content": "Generate 10 offensive and 10 non-offensive tweets."},
            ]
        )
        answers.append(response.choices[0]["message"]["content"])
    
    for answer in answers:
        print(answer)

   
def clean_data():
    # splits = ["NOT", "OFF"]
    list_data = []
    with open("data.txt", "r") as f:
        data = f.readlines()
        for line in data:
            line = line.strip()
            line = line.replace("\"", "")
            if len(line) <= 0:
                continue
            label = line[-3:]
            text = line[:-3].strip()
            list_data.append((text,label))
        
    with open("data_csv.csv", "w", newline="") as f:
        writer = csv.writer(f)
        columns = ["text", "label"]
        writer.writerow(columns)
        for (text, label) in list_data:
            writer.writerow([text,label])

def read_corpus(corpus_file):
    """
    Read the file by the given file name and parse the necessary fields. Can use sentiment or topic classes.
    @param corpus_file: File name of the corpus file (TSV format).
    @return: documents (list of text) and labels (list of labels).
    """
    documents = []
    labels = []
    with open(corpus_file, encoding="utf-8") as in_file:
        for line in in_file:
            columns = line.strip().split('\t')  # Assuming columns are separated by tabs
            tokens = columns[0].strip().split()
            label = columns[1]
            documents.append(tokens)
            labels.append(label)
    return documents, labels


def remove_emojis(x):
  emoji = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
  "]+", re.UNICODE)
  new_docs = []
  for doc in x:
    new_docs.append([re.sub(emoji, '', w) for w in doc])

  return new_docs

def get_predictions(start = 0):
    X, _ = read_corpus('test.tsv')
    X = remove_emojis(X)
    # print(X, y)
    print(len(X))
    y_pred = []
    with open("predictions.txt", "a") as f:
        for index, x in enumerate(X[start:]):
            print(f"Sample {index + start} / {len(X)}")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                        You are a helpful assistant detecting offensive and non-offensive tweets. You will be presented with a tweet and your task is to either label OFF for an offensive tweet and NOT for a non-offensive tweet.
                        ONLY RETURN THE OFF OR NOT
                    """},
                    {"role": "user", "content": f"{x}"},
                ]
            )
            print(response)
            pred = response.choices[0]["message"]["content"] 
            y_pred.append(pred)
            f.write(pred)
            f.write("\n")

def get_scores(key: str, report_dict):
    """
    Get the scores of each topic from the classification report dictionary
    @param key: the key like "camera" or "dvd"
    @param report_dict: The classification report dictionary
    @return: The values of the classification report dictionary of the given key
    """
    dict_values = report_dict[key]
    return dict_values.values()

def save_confusion_matrix(Y_test, Y_pred, classes, name):
    """
    Save the confusion matrix of the specified classifier
    @param Y_test: The ground-truth Y-values
    @param Y_pred: Predicted Y-values
    @param classifier: Classifier used to get the classes
    @param name: Name of the classifier
    """
    cm = confusion_matrix(Y_test, Y_pred)
    # Plot the confusion matrix, given the confusion matrix and the classifier classes
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=classes
    )
    disp.plot()
    disp.ax_.set_title(name)
    # Save the plot cm (confusion matrix) directory
    # Saved with name given, should be a descriptive name
    if not os.path.exists("cm"):
        os.makedirs("cm")
    disp.figure_.savefig(f"cm/{name}.png", dpi=300)
    print(f"Confusion matrix of classifier {name} is saved to cm/{name}.png")

def test_performance(Y_test, Y_pred, classes, name):
    # Get the performance metrics.
    acc = accuracy_score(Y_test, Y_pred)
    macro_precision = precision_score(Y_test, Y_pred, average="macro")
    macro_recall = recall_score(Y_test, Y_pred, average="macro")
    macro_f1 = f1_score(Y_test, Y_pred, average="macro")
    # Compute performance per category by using classification report of sklearn
    report_dict = classification_report(Y_test, Y_pred, output_dict=True)

    print(classification_report(Y_test, Y_pred))
    # Save confusion matrix to image

    save_confusion_matrix(Y_test, Y_pred, classes, name)
    # Load score per category in coresponding variable
    b_p, b_r, b_f1, _ = get_scores("NOT", report_dict)
    c_p, c_r, c_f1, _ = get_scores("OFF", report_dict)

    # Append all results, to return, such that it can be analyzed
    return [
            name,
            acc,
            macro_precision,
            macro_recall,
            macro_f1,
            b_p,
            b_r,
            b_f1,
            c_p,
            c_r,
            c_f1
    ]

def results():
    predictions = []
    with open("predictions.txt", 'r') as f:
        for line in f.readlines():
            line = line.strip()
            predictions.append(line)
    _, y = read_corpus("test.tsv")

    results = test_performance(y, predictions, ["NOT", "OFF"], "GPT")
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='OpenAI generated data')
    parser.add_argument('-g', '--generate', action='store_true')  # on/off flag
    parser.add_argument('-c', '--clean', action='store_true')  # on/off flag
    parser.add_argument('-gpt', '--gpt_predictions', action='store_true')  # on/off flag
    parser.add_argument('-r', '--results', action='store_true')  # on/off flag
    args = parser.parse_args()
    if args.generate:
        generate_data()
    if args.clean:
        clean_data()
    if args.gpt_predictions:
        get_predictions()
    if args.results:
        results()

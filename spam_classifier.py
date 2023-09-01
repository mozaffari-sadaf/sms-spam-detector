import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import pandas as pd
import logging

# Configure logging
logging.basicConfig(filename='spam_classifier.log', level=logging.ERROR)


def load_data(csv_path):
    """
    Load data from a CSV file.
    :param str csv_path: The path to the CSV file.
    :return: A DataFrame containing the loaded data.
    :rtype: pd.DataFrame
    """
    try:
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
        df.columns = ['labels', 'data']
        return df
    except FileNotFoundError:
        logging.error(f"Error: The file '{csv_path}' does not exist.")
    except pd.errors.ParserError:
        logging.error(f"Error: Invalid CSV format in '{csv_path}'.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")


def preprocess_data(df):
    """
    Preprocess the data, including label encoding and TF-IDF transformation.

    :param pd.DataFrame df: The DataFrame containing the data.
    :return: A tuple containing the following elements:
        - scipy.sparse.csr_matrix: The TF-IDF matrix.
        - numpy.ndarray: The label-encoded labels.
    """
    df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
    Y = df['b_labels'].values

    # Apply CountVectorizer for term frequency
    count_vectorizer = CountVectorizer(decode_error='ignore')
    X = count_vectorizer.fit_transform(df['data'])

    return X, Y


def train_model(X, Y):
    """
    Train a Multinomial Naive Bayes model.

    :param scipy.sparse.csr_matrix X: The TF-IDF matrix.
    :param numpy.ndarray Y: The label-encoded labels.
    :return: A tuple containing the following elements:
        - sklearn.naive_bayes.MultinomialNB: The trained model.
        - scipy.sparse.csr_matrix: The training data.
        - numpy.ndarray: The training labels.
        - scipy.sparse.csr_matrix: The testing data.
        - numpy.ndarray: The testing labels.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
        model = MultinomialNB()
        model.fit(X_train, y_train)
        return model, X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")


def visualize(label, data):
    """
    Create and display a WordCloud visualization for a given label.

    :param str label: The label ('spam' or 'ham') for which to create the WordCloud.
    :param pd.DataFrame data: The DataFrame containing the data.
    :return: None
    """
    try:
        words = ''
        for msg in data[data['labels'] == label]['data']:
            msg = msg.lower()
            words += msg + ' '
        wordcloud = WordCloud(width=600, height=400).generate(words)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.savefig(f'./wordcloud_img/{label}_wordcloud.png')  # Save the visualization
        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {str(e)}")


def main():
    """
    Main function for spam classification project.

    This function loads data, preprocesses it, trains a model, generates WordCloud visualizations,
    and prints misclassified messages.
    """
    csv_path = './archive/spam.csv'
    df = load_data(csv_path)
    X, Y = preprocess_data(df)
    model, X_train, y_train, X_test, y_test = train_model(X, Y)

    print("train score:", model.score(X_train, y_train))
    print("test score:", model.score(X_test, y_test))

    visualize('spam', df)
    visualize('ham', df)

    df['predictions'] = model.predict(X)
    sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
    for msg in sneaky_spam:
        print(msg)

    not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
    for msg in not_actually_spam:
        print(msg)


if __name__ == "__main__":
    main()

# SMS Spam Detector
A Python-based machine learning project that classifies SMS messages as either spam or not spam (ham) using a Multinomial Naive Bayes classifier. It also generates WordCloud visualizations of the most common words in spam and ham messages.

## Goals
The goals of this project are to:

  - Build a spam classifier that can accurately classify spam messages.
  - Analyze the results of the spam classifier and identify areas for improvement.

## Methodology
The methodology used in this project is as follows:

  - The spam data was preprocessed using the CountVectorizer.
  - The preprocessed data was then used to train a Multinomial Naive Bayes classifier.
  - The performance of the classifier was evaluated using a test set.

## Results
The results of the spam classifier are as follows:

  - The classifier achieved an accuracy of 97% on the test set.
  - The most common words in spam messages are::
    ![spam_wordcloud](https://github.com/mozaffari-sadaf/sms-spam-detector/assets/49075210/d061b72c-35ff-40e0-ac36-827e2a7cda23)
  - The most common words in ham messages are:
    ![ham_wordcloud](https://github.com/mozaffari-sadaf/sms-spam-detector/assets/49075210/277e872a-ee1a-40b0-af50-734f14dec846)


## Installation and Usage

To run the project, you will need to have Python 3 installed. You can then install the necessary packages by running the following command:
```
pip install -r requirements.txt
```
Once the packages are installed, you can run the project by running the following command:
```
python spam_classifier.py
```

## Project Structure

- spam_classifier.py: The main code for the SMS Spam Detector project contains loading data, preprocessing, training the model, and visualizing results.
- requirements.txt: Lists the Python packages required for the project.
- archive/: Directory containing the SMS dataset file (spam.csv).
- wordcloud_img/: Directory containing the outputs of the cloudword.
- .gitignore: Specifies files and directories to be ignored by Git.
- LICENSE: The license for the project (e.g., MIT License).
- README.md: This README file.

## Contributing
If you would like to contribute to the project, you can do so by submitting a pull request on GitHub.

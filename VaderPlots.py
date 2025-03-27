dataset = ['control','government','nonprofit','privateorg']


import csv  # Import csv module for quoting options
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

for data in dataset:
    analyzer = SentimentIntensityAnalyzer()

    # Function to get sentiment scores
    def get_sentiment_scores(text):
        if isinstance(text, str):  # Ensure input is a string
            scores = analyzer.polarity_scores(text)
            return scores['compound']
        else:
            return None  # Return None for non-text values

    # Read the CSV file with proper quoting to handle commas inside strings
    input_csv = 'datasets/combined/'+data+'_combined.csv'  # Replace with your input CSV file path
    df = pd.read_csv(input_csv, quotechar='"')  # Ensure proper reading of quoted text

    # Apply the sentiment analysis to each line in the 'post_body_text' column
    df['sentiment'] = df['post_body_text'].apply(get_sentiment_scores)

    # Verify the contents of the DataFrame
    print(df.head())

    # Write the results to a new CSV file with proper quoting for text fields
    output_csv = 'datasets/combined/'+data+'_combined_output.csv'  # Replace with your desired output CSV file path
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)  # Correct quoting usage

    # Confirm the file was written
    print(f"Output file saved to: {output_csv}")



    # Read CSV file
    df = pd.read_csv('datasets/combined/'+data+'_combined_output.csv')

    # Plotting the distribution of sentiment scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment'], bins=30, kde=True)
    plt.title('Distribution of Sentiment Scores ('+data+')')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')

    # Save the figure before showing it
    plt.savefig("graphs/"+data+"Sentiment.jpeg", dpi=300)  # Save with high resolution

    # Display the plot
    plt.show()


    # Read data from CSV file
    df = pd.read_csv(output_csv)



    # Group by 'Platform' and calculate the average sentiment score for each platform
    average_sentiment = df.groupby('Platform')['sentiment'].mean()

    # Sort the Series in descending order
    average_sentiment = average_sentiment.sort_values(ascending=False)

    # Print the sorted Series
    print('average sentiment for '+data+':'+str(average_sentiment))


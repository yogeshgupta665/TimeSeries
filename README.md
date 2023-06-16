# Weather Forecasting with Time Series Data

This notebook demonstrates how to use vector search for time series forecasting on climate data using Pinecone. We will be using the Jena Climate dataset, which consists of various climate measurements recorded every 10 minutes over several years.

## Installation

First, we need to install the required dependencies, including Pinecone and other relevant libraries:

```python
!pip install pinecone-client matplotlib==3.2.2 tensorflow==2.9.2 scikit-learn==1.0.2 pandas==1.3.5 tqdm
import pinecone
import os
```

To use Pinecone, you will need an API key. If you don't have one, you can obtain it from the [Pinecone website](https://www.pinecone.io/start/).

## Loading the Dataset

We will use the Jena Climate dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/stytch16/jena-climate-2009-2016). This dataset contains climate measurements such as air temperature, atmospheric pressure, humidity, wind direction, etc., recorded at 10-minute intervals.

After downloading the dataset, we load it into a Pandas DataFrame:

```python
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

original_data_for_insert = pd.read_csv(csv_path)
original_data_for_insert = original_data_for_insert[5::6]

original_data_for_insert['Date Time'] = pd.to_datetime(original_data_for_insert['Date Time'], format='%d.%m.%Y %H:%M:%S')
```

The dataset contains measurements every 10 minutes, but we will use hourly data for our forecasting.

## Setting up the Index

To perform similarity search, we need to create an index in Pinecone. The index will store the feature vectors associated with each timestamp.

```python
# Set up Pinecone index
api_key = os.getenv('PINECONE_API_KEY') or 'YOUR_API_KEY'
env = os.getenv('PINECONE_ENVIRONMENT') or 'YOUR_ENVIRONMENT'
pinecone.init(api_key=api_key, environment=env)

index_name = 'time-series-weather'

if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

pinecone.create_index(name=index_name, dimension=14)
index = pinecone.Index(index_name=index_name)
```

The dimension of the index is set to 14, as we have 14 features in the dataset. We will store the vectors associated with each timestamp in the index.

## Uploading Data to Pinecone

Next, we prepare the data for uploading to Pinecone. We split the data into training and testing sets, and convert the training data into a list of (key, value) tuples.

```python
n = len(original_data_for_insert)
train_data = original_data_for_insert[:int(n*0.9)]
test_data = original_data_for_insert[int(n*0.9):]

items_to_upload = []
for row in train_data.values.tolist():
    key = str(row[0])
    values = row[1:]
    items_to_upload.append((key, values))
```

We will use the training data to build the index.

```python
for batch in chunks(items_to_upload, 500):
    index.upsert(vectors=batch)
```

## Querying for Weather Forecasts

Once the index is set up and populated with data, we can

 perform similarity search to make weather forecasts. We will query the index with a given vector and retrieve the most similar vector from the index. Based on this similarity, we will predict the weather for the next hour.

```python
all_query_results = []
for xq in tqdm(query_data):
    res = index.query(xq, top_k=1)
    all_query_results.append(res)
```

## Evaluating the Results

To evaluate the accuracy of our weather predictions, we compare the predicted values with the true values from the test dataset. We calculate metrics such as mean squared error (MSE), root mean squared error (RMSE), and mean absolute error (MAE) to assess the performance.

```python
true_values, predicted_values = get_predictions(feature)
plot_results(true_values, predicted_values)
print_results(true_values, predicted_values)
```

We repeat this process for each feature in the dataset to analyze the predictions.

## Summary

This notebook demonstrates how to use Pinecone for time series forecasting using climate data. We showed that even with a basic similarity search method, we can achieve reasonably accurate predictions for some features. However, for more complex and accurate predictions, advanced methods like LSTMs may be more suitable.

Remember to delete the index once you no longer need it:

```python
pinecone.delete_index(index_name)
```

Please note that index deletion is permanent, so use caution when deleting an index.

# Regression Week 2: Multiple Regression (Interpretation)

The goal of this first notebook is to explore multiple regression and feature engineering with existing Turi Create functions.

In this notebook you will use data on house sales in King County to predict prices using multiple regression. You will:
* Use SFrames to do some feature engineering
* Use built-in Turi Create functions to compute the regression weights (coefficients/parameters)
* Given the regression weights, predictors and outcome write a function to compute the Residual Sum of Squares
* Look at coefficients and interpret their meanings
* Evaluate multiple models via RSS

# Fire up Turi Create


```python
import turicreate
```

# Load in house sales data

Dataset is from house sales in King County, the region where the city of Seattle, WA is located.


```python
sales = turicreate.SFrame('home_data.sframe/')
```

# Split data into training and testing.
We use seed=0 so that everyone running this notebook gets the same results.  In practice, you may set a random seed (or let Turi Create pick a random seed for you).  


```python
train_data,test_data = sales.random_split(.8,seed=0)
```

# Learning a multiple regression model

Recall we can use the following code to learn a multiple regression model predicting 'price' based on the following features:
example_features = ['sqft_living', 'bedrooms', 'bathrooms'] on training data with the following code:

(Aside: We set validation_set = None to ensure that the results are always the same)


```python
example_features = ['sqft_living', 'bedrooms', 'bathrooms']
example_model = turicreate.linear_regression.create(train_data, target = 'price', features = example_features, 
                                                    validation_set = None)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17384</pre>



<pre>Number of features          : 3</pre>



<pre>Number of unpacked features : 3</pre>



<pre>Number of coefficients    : 4</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 1.018871     | 4146407.600631     | 258679.804477                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


Now that we have fitted the model we can extract the regression weights (coefficients) as an SFrame as follows:


```python
example_weight_summary = example_model.coefficients
print(example_weight_summary)
```

    +-------------+-------+---------------------+--------------------+
    |     name    | index |        value        |       stderr       |
    +-------------+-------+---------------------+--------------------+
    | (intercept) |  None |   87910.0724923957  | 7873.338143401634  |
    | sqft_living |  None |  315.40344055210005 | 3.4557003258547296 |
    |   bedrooms  |  None | -65080.215552827525 | 2717.4568544207045 |
    |  bathrooms  |  None |  6944.020192638836  | 3923.114931441481  |
    +-------------+-------+---------------------+--------------------+
    [4 rows x 4 columns]
    


# Making Predictions

In the gradient descent notebook we use numpy to do our regression. In this book we will use existing Turi Create functions to analyze multiple regressions. 

Recall that once a model is built we can use the .predict() function to find the predicted values for data we pass. For example using the example model above:


```python
example_predictions = example_model.predict(train_data)
print(example_predictions[0]) # should be 271789.505878
```

    271789.5058780301


# Compute RSS

Now that we can make predictions given the model, let's write a function to compute the RSS of the model. Complete the function below to calculate RSS given the model, data, and the outcome.


```python
def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    predicted = model.predict(data);
    # Then compute the residuals/errors
    errors = outcome-predicted;
    # Then square and add them up    
    RSS = (errors*errors).sum();
    return(RSS)    
```

Test your function by computing the RSS on TEST data for the example model:


```python
rss_example_train = get_residual_sum_of_squares(example_model, test_data, test_data['price'])
print(rss_example_train) # should be 2.7376153833e+14
```

    273761538330193.0


# Create some new features

Although we often think of multiple regression as including multiple different features (e.g. # of bedrooms, squarefeet, and # of bathrooms) but we can also consider transformations of existing features e.g. the log of the squarefeet or even "interaction" features such as the product of bedrooms and bathrooms.

You will use the logarithm function to create a new feature. so first you should import it from the math library.


```python
from math import log
```

Next create the following 4 new features as column in both TEST and TRAIN data:
* bedrooms_squared = bedrooms\*bedrooms
* bed_bath_rooms = bedrooms\*bathrooms
* log_sqft_living = log(sqft_living)
* lat_plus_long = lat + long 
As an example here's the first one:


```python
train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2)
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)
```


```python
# create the remaining 3 features in both TEST and TRAIN data
train_data['bed_bath_rooms'] = train_data['bedrooms']*train_data['bathrooms'];
test_data['bed_bath_rooms'] = test_data['bedrooms']*test_data['bathrooms'];

train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x));
test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x));

train_data['lat_plus_long'] = train_data['lat'] + train_data['long'];
test_data['lat_plus_long'] = test_data['lat'] + test_data['long'];

```

* Squaring bedrooms will increase the separation between not many bedrooms (e.g. 1) and lots of bedrooms (e.g. 4) since 1^2 = 1 but 4^2 = 16. Consequently this feature will mostly affect houses with many bedrooms.
* bedrooms times bathrooms gives what's called an "interaction" feature. It is large when *both* of them are large.
* Taking the log of squarefeet has the effect of bringing large values closer together and spreading out small values.
* Adding latitude to longitude is totally non-sensical but we will do it anyway (you'll see why)

**Quiz Question: What is the mean (arithmetic average) value of your 4 new features on TEST data? (round to 2 digits)**


```python
print('bedrooms_squared: ' + str(test_data['bedrooms_squared'].mean()));
print('bed_bath_rooms: ' + str(test_data['bed_bath_rooms'].mean()));
print('log_sqft_living: ' + str(test_data['log_sqft_living'].mean()));
print('lat_plus_long: ' + str(test_data['lat_plus_long'].mean()));
```

    bedrooms_squared: 12.446677701584301
    bed_bath_rooms: 7.503901631591394
    log_sqft_living: 7.550274679645938
    lat_plus_long: -74.65333497217307


# Learning Multiple Models

Now we will learn the weights for three (nested) models for predicting house prices. The first model will have the fewest features the second model will add one more feature and the third will add a few more:
* Model 1: squarefeet, # bedrooms, # bathrooms, latitude & longitude
* Model 2: add bedrooms\*bathrooms
* Model 3: Add log squarefeet, bedrooms squared, and the (nonsensical) latitude + longitude


```python
model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2_features = model_1_features + ['bed_bath_rooms']
model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']
```

Now that you have the features, learn the weights for the three different models for predicting target = 'price' using turicreate.linear_regression.create() and look at the value of the weights/coefficients:


```python
# Learn the three models: (don't forget to set validation_set = None)
model_1 = turicreate.linear_regression.create(train_data, target = 'price', features = model_1_features, 
                                                    validation_set = None);
model_2 = turicreate.linear_regression.create(train_data, target = 'price', features = model_2_features, 
                                                    validation_set = None);
model_3 = turicreate.linear_regression.create(train_data, target = 'price', features = model_3_features, 
                                                    validation_set = None);
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17384</pre>



<pre>Number of features          : 5</pre>



<pre>Number of unpacked features : 5</pre>



<pre>Number of coefficients    : 6</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.042022     | 4074878.213132     | 236378.596455                   |</pre>



<pre>| 2         | 3        | 0.082033     | 4074878.213108     | 236378.596455                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17384</pre>



<pre>Number of features          : 6</pre>



<pre>Number of unpacked features : 6</pre>



<pre>Number of coefficients    : 7</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.039762     | 4014170.932952     | 235190.935429                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17384</pre>



<pre>Number of features          : 9</pre>



<pre>Number of unpacked features : 9</pre>



<pre>Number of coefficients    : 10</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.011139     | 3193229.177890     | 228200.043155                   |</pre>



<pre>| 2         | 3        | 0.025022     | 3193229.177873     | 228200.043155                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
# Examine/extract each model's coefficients:
print(model_1.coefficients);
print(model_2.coefficients);
print(model_3.coefficients);
```

    +-------------+-------+---------------------+--------------------+
    |     name    | index |        value        |       stderr       |
    +-------------+-------+---------------------+--------------------+
    | (intercept) |  None |  -56140675.74114427 | 1649985.420135553  |
    | sqft_living |  None |  310.26332577692136 | 3.1888296040737765 |
    |   bedrooms  |  None |  -59577.11606759667 | 2487.2797732245012 |
    |  bathrooms  |  None |  13811.840541653264 | 3593.5421329670735 |
    |     lat     |  None |  629865.7894714845  | 13120.710032363884 |
    |     long    |  None | -214790.28516471002 | 13284.285159576597 |
    +-------------+-------+---------------------+--------------------+
    [6 rows x 4 columns]
    
    +----------------+-------+---------------------+--------------------+
    |      name      | index |        value        |       stderr       |
    +----------------+-------+---------------------+--------------------+
    |  (intercept)   |  None |  -54410676.1071702  | 1650405.1652726454 |
    |  sqft_living   |  None |  304.44929805407946 |  3.20217535637094  |
    |    bedrooms    |  None | -116366.04322451768 | 4805.5496654858225 |
    |   bathrooms    |  None |  -77972.33050970349 | 7565.059910947983  |
    |      lat       |  None |  625433.8349445503  | 13058.353097300462 |
    |      long      |  None | -203958.60289731968 | 13268.128370009661 |
    | bed_bath_rooms |  None |  26961.624907583264 | 1956.3656155588428 |
    +----------------+-------+---------------------+--------------------+
    [7 rows x 4 columns]
    
    +------------------+-------+---------------------+--------------------+
    |       name       | index |        value        |       stderr       |
    +------------------+-------+---------------------+--------------------+
    |   (intercept)    |  None |  -52974974.06892153 | 1615194.942821453  |
    |   sqft_living    |  None |  529.1964205687523  | 7.699134985078978  |
    |     bedrooms     |  None |  28948.527746351134 | 9395.728891110177  |
    |    bathrooms     |  None |  65661.20723969836  | 10795.338070247015 |
    |       lat        |  None |  704762.1484430869  |        nan         |
    |       long       |  None | -137780.02000717327 |        nan         |
    |  bed_bath_rooms  |  None |  -8478.364107167803 | 2858.9539125640354 |
    | bedrooms_squared |  None |  -6072.384661904947 | 1494.9704277794906 |
    | log_sqft_living  |  None |  -563467.7842801767 | 17567.823081204006 |
    |  lat_plus_long   |  None |  -83217.19791002883 |        nan         |
    +------------------+-------+---------------------+--------------------+
    [10 rows x 4 columns]
    


**Quiz Question: What is the sign (positive or negative) for the coefficient/weight for 'bathrooms' in model 1?**

**Quiz Question: What is the sign (positive or negative) for the coefficient/weight for 'bathrooms' in model 2?**

Think about what this means.

# Comparing multiple models

Now that you've learned three models and extracted the model weights we want to evaluate which model is best.

First use your functions from earlier to compute the RSS on TRAINING Data for each of the three models.


```python
# Compute the RSS on TRAINING data for each of the three models and record the values:
print(get_residual_sum_of_squares(model_1, train_data, train_data['price']));
print(get_residual_sum_of_squares(model_2, train_data, train_data['price']));
print(get_residual_sum_of_squares(model_3, train_data, train_data['price']));
```

    971328233545434.4
    961592067859822.1
    905276314551640.9


**Quiz Question: Which model (1, 2 or 3) has lowest RSS on TRAINING Data?** Is this what you expected?

Now compute the RSS on on TEST data for each of the three models.


```python
# Compute the RSS on TESTING data for each of the three models and record the values:
print(get_residual_sum_of_squares(model_1, test_data, test_data['price']));
print(get_residual_sum_of_squares(model_2, test_data, test_data['price']));
print(get_residual_sum_of_squares(model_3, test_data, test_data['price']));
```

    226568089093160.56
    224368799994313.0
    251829318963157.28


**Quiz Question: Which model (1, 2 or 3) has lowest RSS on TESTING Data?** Is this what you expected? Think about the features that were added to each model from the previous.

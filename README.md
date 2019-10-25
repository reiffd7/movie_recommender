# The Best Movie Recommender EVER! ! !
Daniel Reiff, Steven Rouk, Scott Peabody, Sarah Forward

1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Starting Baseline](#starting-baseline)
4. [New Proposed Model](#new-proposed-model)
5. [Precision](#precision)
5. [The App](#the-app)
5. [Proposed Plan of Implimentation](#proposed-plan-of-implimentation)


![EDA](/images/intro.png)


## Overview

In the scenario for this hackathon, we are working for the company, **Movies-Legit**, who has used a production recommenders for many years now. The recommender provides a significant revenue stream so our managers are hesitant to touch it. The issue is that these systems have been around a long time and our head of data science has asked our team to explore new solutions.

## Description of Dataset
 
 100,004 users, 9,066 movies, and 671 users. The rating system is 0-5. 


![EDA](/images/EDA.png)


## Starting Baseline

The solution that has been around for so long is called the Mean of Means. Some users like to rate things highly---others simply do not. Some items are just better or worse. These general trends can be captured through per-user and per-item rating means. The global mean is also incorporated to smooth things out a bit. So if we see a missing value in a given cell, we'll average the global mean with the mean of the column and the mean of the row and use that value to fill it in. Running the Mean of Means algorithms yields a RMSE of 0.9525. What does this mean? On average, we around 1 point off for every rating. 

## New Proposed Model
In our new model, we use a method called Alternating Least Squares (ALS).

<li> We start with a grid of ratings by users and movie. </li>

<li>We use the data of the users to predict the ratings on each movie.</li>

<li>Then, we see how far off we are from the known ratings and correct by using the movie data to estimate new rating predictions.</li>

<li>We repeat back-and-forth between users and movies until our prediction error does not improve with each step.</li>

We tried a number of models. The best performing was ALS. 

Results |
:-----------
![EDA](/images/benchmark.png)

With the ALS model, we see an improvement of 6%. 

![EDA](/images/ALS-v-MOM.png)

![EDA](/images/top100.png)



## Precision
**How do we measure whether or not our model is making a difference?**

We want to avoid cases where we recommended a movie that the user does not like. So we defined a false positive as an instance where we predicted the user would rate the movie at higher than 3.5 stars but the user actually rated the movie at less than 3.5 stars. This is the scenario we would most want to avoid - it could potentially leave the user disatisfied and ready to give up on our service. 

![precision](/images/als_difference.png)

Precision is the number of correct recommendations divided by the number of all recommendations. It penalizes a model for recommendiong a movie that the user does not actually like. With the ALS model, we imporove precision by 6%. 

We estimate a false positive costs us 1 dollar worth of customer churn, then for these 671 users the ALS model will save us 6936 - 4688 = $2248. For 2 million movie ratings, we would save **$44,960**.

## The App

![webapp](/images/web-app-index.png)

![webapp1](/images/web-app-recommendations.png)

http://3.95.7.113:8080/

## Proposed Plan of Implimentation

We propose to roll out this new recommender to 10% of our users for a three month trial period. At the end of that period, we will evaluate the recommender base on:
* User feedback on whether they noticed an improved performance of the recommender.
* Comparing customer churn from the new recommender group and the old recommender group.
If we receive positive survey results and reduced customer churn, we would then roll out the new recommender to the rest of the customer base.

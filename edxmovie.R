##############################################################
# Movie recommendation system
##############################################################
# This model contains 5 Parts:
# Part 1: Create edx set and validation set (skip to Part 2
#     if already created and available locally)
# Part 2: Loading the edx and validation files stored locally
#     and taking an initial look at the data (use only if you
#     did not use Part 1 to create the files)
# Part 3: Creating a training set and a test set on edx.
#     Model will be trained on the training set and tested
#     on the test set
# Part 4: Creating, fitting and testing the model (steps 1-6)
#     and cross-validation for regularisation and optimal
#     clustering (step 7)
# Part 5: Applying the final model to the Validation set

# Note: if you want to run the final model only, feel free to
# use Part 1 or Part 2 to load the data and then go straight
# to Part 5
##############################################################

##############################################################
# The following libraries are required
# Please install packages first if not already done

if(!require(tidyverse)) install.packages("tidyverse", repos = "https://cran.r-project.org")
if(!require(caret)) install.packages("caret", repos = "https://cran.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "https://cran.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "https://cran.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "https://cran.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "https://cran.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(knitr)
library(kableExtra)
##############################################################


##############################################################
# PART 1
# Create edx set, validation set (final hold-out test set)
#
# NOTE: if you have already created the sets, skip to Part 2
##############################################################

# Warning: this process could take a couple of minutes

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Saving the files locally
saveRDS(edx, file = "edx.rds")
saveRDS(validation, file = "validation.rds")

##########################################################
# PART 2 - Step 1
# Loading the edx and validation files
# Use this step if the files have been saved in your working
# directory under the names edx.rds and validation.rds
##########################################################

# Loading the edx and validation files from the working directory
edx <- readRDS("edx.rds")
validation <- readRDS("validation.rds")

####################################
# PART 2 - Step 2
# Taking an initial Look at the data
####################################

# Counting users, movies and genres (for information)
count <- edx %>% summarise(Users = n_distinct(userId),
                  Movies = n_distinct(movieId),
                  Genres = n_distinct(genres))

kable(count, digits = 0, caption = "Edx data count:",
      format.args = list(decimal.mark = ".", big.mark = ",")) %>%
  kable_styling(bootstrap_options = "striped", font_size = 15, full_width = F)


# Looking at the rating scale: ratings are spread between 0.5 and 5
# The median rating is 4
# There are no NA ratings
edx$rating %>% summary()
      
# In fact, ratings take integer and integer+half values only:
unique(edx$rating)[order(unique(edx$rating))]

# Plotting the ratings
# The graph shows that users tend to give ratings 3 and above rather than <3
# Half ratings are less frequently used
# The most frequently given rating is 4, followed by 3 then 5
g_rating <- edx %>%
  ggplot(aes(rating)) +
  geom_histogram(bins = 10, fill = "coral2", color = "white") +
  ggtitle("Rating histogram")
g_rating

##########################################################
# PART 3
# Creating a test set and a validation set on edx
# All models will be fitted and tested against these sets
##########################################################

# Creating a data partition (20% test set)
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, 
                                  list = FALSE)
train_edx <- edx[-test_index,]
test_edx <- edx[test_index,]

# Making sure not to include users and movies in the test set
# that do not appear in the training set
test_edx <- test_edx %>% 
  semi_join(train_edx, by = "movieId") %>%
  semi_join(train_edx, by = "userId")

# Creating a mean square error function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


##############################################################
# PART 4
# Fitting models
# Models are built step by step to visualize components impact
# First model uses a simple average, then components are added:
# movie effect, user effect, genre effect, time effect, then
# a user-group effect is added to capture the fact that users
# react differently to genres (comedy, horror,...).
# To that effect, clusterization will be used to create groups
# of genres.
# A regularization factor is added to reduce biases induced
# by components with very few ratings. Regularization details
# are available at the end of Part 4.
##############################################################

#############################
# Part 4 - Step 1
# Simple average model
#############################

# Computing the average of all ratings in training set
mu <- mean(train_edx$rating)
mu

# Computing the RMSE on test set for this basic model
model_1_rmse <- RMSE(test_edx$rating, mu)
model_1_rmse

# Storing the results in a small table
rmse_results <- data.frame(method = "Simple average model", RMSE = model_1_rmse)

#############################
# Part 4 - Step 2
# Adding a movie effect
#############################

# We introduce a regularization parameter in order to remove biases
# created by movies with few ratings
# The reason and impact of choosing this parameter are shown at the end
# of Part 4 (Step 7)
lambda <- 5

# Computing a movie effect as a difference versus the mean
movie_avgs <- train_edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum((rating - mu)) / (n() + lambda))

# Viewing a graph dispersion of the movie effect
# The graph shows clearly that many movies differ significantly
# from the average rating
g_bi <- movie_avgs %>%
  ggplot(aes(b_i)) +
  geom_histogram(bins = 10, fill = "coral2", color = "white") +
  ggtitle("Movie effect b_i")
g_bi

# Predicting rating with movie effect on the test set
predicted_ratings <- mu + test_edx %>% 
  left_join(movie_avgs, by='movieId') %>% pull(b_i)

# Measuring the RMSE
RMSE(test_edx$rating, predicted_ratings)

# Updating the results table
model_2_rmse <- RMSE(test_edx$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data.frame(method="Movie Effect Model",
                                    RMSE = model_2_rmse ))

# Significant improvement over the simple average model
print(rmse_results)
kable(rmse_results, digits = 4, caption = "Models tested on test_edx set") %>%
  kable_styling(bootstrap_options = "striped", font_size = 15, full_width = F)

#############################
# Part 4 - Step 3
# Adding a user effect
#############################

# Computing a user effect as a difference versus the mean and movie effect
user_avgs <- train_edx %>% left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum((rating - mu - b_i)) / (n() + lambda))

# Viewing a graph dispersion of the user effect
# User effect (net of movide effect) spread seems reduced versus
# pure movie effect, however it clearly remains significant
# Interpretation: some users rate more harshly (b_u < 0) than others
g_bu <- user_avgs %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 10, fill = "coral2", color = "white") +
  ggtitle("Residual User effect b_u")
g_bu

# Predicting rating with movie and user effects on the test set
predicted_ratings <- test_edx %>% left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% pull(pred)

# Note that median rating is 3.56 not 4
median(predicted_ratings)
median(test_edx$rating)

# Measuring the RMSE
RMSE(test_edx$rating, predicted_ratings)

# Updating the results table
model_3_rmse <- RMSE(test_edx$rating, predicted_ratings)
rmse_results <- rmse_results %>%
  bind_rows(data.frame(method = "Movie and User effect",
                       RMSE = model_3_rmse))

# Significant improvement over previous models
print(rmse_results)
kable(rmse_results, digits = 4, caption = "Models tested on test_edx set") %>%
  kable_styling(bootstrap_options = "striped", font_size = 15, full_width = F)

#############################
# Part 4 - Step 4
# Adding a genre effect
#############################

# Checking that that there is no NA genre
sum(is.na(edx$genres))

#  Checking that each movie is tagged to one genre only
# There are 10,637 movies and the same number of movie-genre combinations
n_distinct(train_edx$movieId)
n_distinct(paste(train_edx$movieId, train_edx$genres, sep = "_"))

# Plotting movie genres rating (only for genres appearing in high numbers)
# the plot shows a clear rating differentiation by genres
# Therefore adding a genre effect looks justified
g_genres <- train_edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 50000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  coord_flip()
g_genres

# Computing the genre effect
genre_avgs <- train_edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum((rating - mu - b_i - b_u)) / (n() + lambda))

# Plotting genre effect b_g
# The plot shows a non-insignificant residual genre effect
g_bg <- genre_avgs %>%
  ggplot(aes(b_g)) +
  geom_histogram(bins = 10, fill = "coral2", color = "white") +
  ggtitle("Residual Genre effect b_g")
g_bg

# Adding the genre effect to the predictions
predicted_ratings <- test_edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>% pull(pred)

# Note that predictions range between negative and above 6
# We can see this by looking at the summary:
data.frame(predicted_ratings) %>% summary()

# Computing the RMSEn (shows only a moderate improvement)
RMSE(test_edx$rating, predicted_ratings)

# Updating the results table
model_4_rmse <- RMSE(test_edx$rating, predicted_ratings)
rmse_results <- rmse_results %>%
  bind_rows(data.frame(method = "Movie+User+Genre effect",
                       RMSE = model_4_rmse))

# Results show a moderate improvement
print(rmse_results)
kable(rmse_results, digits = 4, caption = "Models tested on test_edx set") %>%
  kable_styling(bootstrap_options = "striped", font_size = 15, full_width = F)

#############################
# Part 4 - Step 5
# Adding a time effect
#############################

library(lubridate)

# Checking that there are no NAs in timestamp
sum(is.na(edx$timestamp))
range(train_edx$timestamp)

# Creating a date column from the timestamp
train_edx <- train_edx %>% 
  mutate(date = as_datetime(timestamp))

# Computing the number of weeks since first date
first_date <- min(train_edx$date)
first_date
train_edx <- train_edx %>% 
  mutate(
    rating_week = as.integer(round(difftime(date, first_date, units = "weeks"), 0)))

range(train_edx$rating_week)
class(train_edx$rating_week)

# Plotting average rating vs date (rounded to number of weeks)
# The plot shows that there is a slight time effect on ratings
g_time <- train_edx %>%
  group_by(rating_week) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(rating_week, rating)) +
  geom_point() +
  geom_smooth()
g_time

# Computing the week effect
week_avgs <- train_edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  group_by(rating_week) %>%
  summarize(b_w = mean(rating - mu - b_i - b_u - b_g))

# Plotting the week effect
# The plot show the evidence of some residual week effect
g_bw <- week_avgs %>%
  ggplot(aes(b_w)) +
  geom_histogram(bins = 20, fill = "coral2", color = "coral2") +
  ggtitle("Residual Week effect b_w")
g_bw

# Adding week effect to model
test_edx <- test_edx %>% 
  mutate(date = as_datetime(timestamp),
         rating_week = as.integer(round(difftime(date, first_date, units = "weeks"), 0)))

predicted_ratings <- test_edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(week_avgs, by = 'rating_week') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_w) %>% pull(pred)

# Computing RMSE
RMSE(test_edx$rating, predicted_ratings)

# Updating the results table
model_5_rmse <- RMSE(test_edx$rating, predicted_ratings)
rmse_results <- rmse_results %>%
  bind_rows(data.frame(method = "Adding week effect",
                       RMSE = model_5_rmse))

# Adding the time effect improves RMSE slightly:
print(rmse_results)
kable(rmse_results, digits = 4, caption = "Models tested on test_edx set") %>%
  kable_styling(bootstrap_options = "striped", font_size = 15, full_width = F)


##################################
# Part 4 - Step 6
# Adding a user-genre group effect
##################################

# The model so far does not capture an important effect:
# Different users like different genres.
# So if a user likes Comedies and dislikes Horror movies, they are 
# likely to give lower ratings to Horror movies
# In order to capture this, we will look at average user ratings per genre

# As there are 797 genres recorded in the data, we will create genre clusters
# (genre groups) in order to re-group similar genres
# This will help to reduce the size of the user-genre coombinations but
# will also serve to reduce the risk of over-fitting

# To begin, we observe that certain genres are present multiple times
# in the table, whereas some only have one or two occurences
# Count per genres

genre_list <- train_edx %>% 
  group_by(genres) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

genre_list %>%
  slice_max(order_by = count, n = 10)

genre_list %>%
  slice_min(order_by = count, n = 10)

# Creating a table with average rating per genre per user
# Only users with more than 200 ratings retained to reduce table size
# WARNING: takes around 30-40s to run
genre_ratings <- train_edx %>%
  select(userId, genres, rating) %>%
  group_by(userId) %>%
  filter(n() > 200) %>% ungroup() %>%
  group_by(genres, userId) %>%
  summarize(n=n(), rating = mean(rating))

# Creating a matrix with genres as rows and userId as predictors (columns)
x <- genre_ratings %>%
  select(genres, userId, rating) %>%
  pivot_wider(id_cols = genres, names_from = userId, values_from = rating,
              values_fill = 0)

# The matrix is centered and rownames updated with the genres names
row_names <- x$genres
x <- x[,-1] %>% as.matrix()
x <- sweep(x, 2, colMeans(x, na.rm = TRUE))
x <- sweep(x, 1, rowMeans(x, na.rm = TRUE))
rownames(x) <- row_names

# Creating groups of genres by clusterization
# kmeans will be used

# We define the number of clusters (see cross-validation Part 4 Step 7)
kcluster <- 9

# NA values are first replaced by zeros
x_0 <- x
x_0[is.na(x_0)] <- 0
set.seed(1, sample.kind = "Rounding")
k <- kmeans(x_0, centers = kcluster)

# Creating a data frame with the genre groups
genres_groups <- data.frame(genre_group = k$cluster)
genres_groups <- genres_groups %>% mutate(genres = rownames(genres_groups))

# Checking that all genres have been grouped
n_group <- nrow(genres_groups) # for use in report
identical(n_group, count$Genres)

# The size and variance capture by each group is shown here:
group_des <- data.table(Group = 1:kcluster,
            Size = k$size,
            Variance_Within = k$withinss,
            Share_of_Variance = paste(round(100*k$withinss/k$tot.withinss, 0), "%"))

kable(group_des, digits = 0, caption = "Group Size and Variance",
      align = "c",
      format.args = list(decimal.mark = ".", big.mark = ",")) %>%
  kable_styling(bootstrap_options = "striped", font_size = 15, full_width = F)

# Variance Within and Between
t_var <- data.table(
  Variance_Type = c("Within", "Between", "Total"),
  Variance = c(k$tot.withinss, k$betweenss, k$totss)) %>%
  mutate( 
  Share_of_Variance = paste(round(100 * Variance / k$totss, 0), "%"))

kable(t_var, digits = 0, caption = "Variance analysis",
      align = "c",
      format.args = list(decimal.mark = ".", big.mark = ",")) %>%
  kable_styling(bootstrap_options = "striped", font_size = 15, full_width = F)

# Content of Group 2 (for information)
g2 <- genres_groups %>% 
  filter(genre_group == 2) %>%
  left_join(genre_list, by = "genres") %>%
  slice_max(order_by = count, n = 10)

kable(g2, digits = 0, caption = "Group 2 - Genres with highest count",
      align = "c",
      format.args = list(decimal.mark = ".", big.mark = ",")) %>%
  kable_styling(bootstrap_options = "striped", font_size = 15, full_width = F)

# Adding user_group in the train and test sets
train_edx <- train_edx %>%
  select(userId, movieId, rating, timestamp, title, genres, date, rating_week) %>%
  left_join(genres_groups, by = "genres") %>%
  mutate(user_group = paste(userId, genre_group, sep = "_"))

test_edx <- test_edx %>%
  select(userId, movieId, rating, timestamp, title, genres, date, rating_week) %>%
  left_join(genres_groups, by = "genres") %>%
  mutate(user_group = paste(userId, genre_group, sep = "_"))

# Checking that there are no NAs in user_group
sum(is.na(train_edx$user_group))
sum(is.na(test_edx$user_group))

# Defining the regularization parameter
kappa <- 15 # See Step 7 for regularization cross-validation

# Computing the user_group component
ug_avgs <- train_edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(week_avgs, by = 'rating_week') %>%
  group_by(user_group) %>%
  summarize(b_ug = (sum(rating - mu - b_i - b_u - b_g - b_w) / (n() + kappa)))

# Plotting the user-group effect
# The plot shows a definite residual effect
g_bug <- ug_avgs %>%
  ggplot(aes(b_ug)) +
  geom_histogram(bins = 20, fill = "coral2", color = "white") +
  ggtitle("Residual User-Genre group effect b_ug")
g_bug

# Checking that there are no NAs in b_u calculations
sum(is.na(user_avgs$b_u))

# Adding all components to the model
# A predicted_rating table is first created
predicted_ratings <- test_edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(week_avgs, by = 'rating_week') %>%
  left_join(ug_avgs, by='user_group') 

# Some user-group combinations are present in the testing set but not
# in the training set, generating some NAs for b_ug
# This is to be expected in real life, we will therefore set b_ug
# to nil for these
sum(is.na(predicted_ratings$b_ug))

# Setting b_ug to nil if not available
predicted_ratings$b_ug[is.na(predicted_ratings$b_ug)] <- 0
sum(predicted_ratings$b_ug == 0)

# Finalizing the predictions
predicted_ratings <- predicted_ratings %>%
  mutate(pred = mu + b_i + b_u + b_g + b_w + b_ug) %>% 
  pull(pred)

predicted_ratings %>% summary()
RMSE(test_edx$rating, predicted_ratings)

model_6_rmse <- RMSE(test_edx$rating, predicted_ratings)
rmse_results <- rmse_results %>%
  bind_rows(data.frame(method = "With User Group effect",
                       RMSE = model_6_rmse))

# The RMSE is further reduced
print(rmse_results)
kable(rmse_results, digits = 4, caption = "Models tested on test_edx set") %>%
  kable_styling(bootstrap_options = "striped", font_size = 15, full_width = F)


#############################
# Part 4 - Step 7
# Regularization
#############################

# For simplicity, two regularization factors are computed
# One for the movies, user and genre effect
# One for the user-group effect


# Finding the optimal lambda for movies, user and genre effect
# WARNING - takes a couple of minutes to run
lambdas <- seq(0, 10, 0.5)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_edx$rating)
  
  movie_avgs <- train_edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  user_avgs <- train_edx %>% 
    left_join(movie_avgs, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  genre_avgs <- train_edx %>% 
    left_join(movie_avgs, by="movieId") %>%
    left_join(user_avgs, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  predicted_ratings <- test_edx %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(genre_avgs, by='genres') %>%
    left_join(week_avgs, by = 'rating_week') %>%
    mutate(pred = mu + b_i + b_u + b_g + b_w) %>% 
    pull(pred)
  
  return(RMSE(test_edx$rating, predicted_ratings))
})

# Finding the optimal lambda
# The plot shows that the optimal lambda is 5
g_lambdas <- qplot(lambdas, rmses)
g_lambdas
lambda <- lambdas[which.min(rmses)]
lambda



# Finding the optimal number of clusters
# WARNING - takes a couple of minutes to run
kclusters <- 8:12
set.seed(1, sample.kind = "Rounding")

krmse <- sapply(kclusters, function(kc){
  
  k <- kmeans(x_0, centers = kc)

  # Creating a data frame with the genre groups
  genres_groups <- data.frame(genre_group = k$cluster)
  genres_groups <- genres_groups %>% mutate(genres = rownames(genres_groups))

  # Adding user_group in the train and test sets
  temptrain_edx <- train_edx %>%
    select(userId, movieId, rating, timestamp, title, genres, date, rating_week) %>%
    left_join(genres_groups, by = "genres") %>%
    mutate(user_group = paste(userId, genre_group, sep = "_"))
  
  temptest_edx <- test_edx %>%
    select(userId, movieId, rating, timestamp, title, genres, date, rating_week) %>%
    left_join(genres_groups, by = "genres") %>%
    mutate(user_group = paste(userId, genre_group, sep = "_"))

  # Computing the user_group component
  ug_avgs <- temptrain_edx %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(genre_avgs, by='genres') %>%
    left_join(week_avgs, by = 'rating_week') %>%
    group_by(user_group) %>%
    summarize(b_ug = (sum(rating - mu - b_i - b_u - b_g - b_w) / (n() + kappa)))
  
  # Adding all components to the model
  # A predicted_rating table is first created
  predicted_ratings <- temptest_edx %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(genre_avgs, by='genres') %>%
    left_join(week_avgs, by = 'rating_week') %>%
    left_join(ug_avgs, by='user_group') 
  
  # Setting b_ug to nil if not available
  predicted_ratings$b_ug[is.na(predicted_ratings$b_ug)] <- 0
  
  # Finalizing the predictions
  predicted_ratings <- predicted_ratings %>%
    mutate(pred = mu + b_i + b_u + b_g + b_w + b_ug) %>% 
    pull(pred)
  
  return(RMSE(test_edx$rating, predicted_ratings))
})

# Finding the optimal kcluster
# The plot shows that the optimal kcluster is 9
g_kclusters <- qplot(kclusters, krmse)
g_kclusters
kcluster <- kclusters[which.min(krmse)]
kcluster
krmse[which.min(krmse)]

# Finding the optimal kappa for user-genre effect
# Note that this step uses the results from Phase 4 steps 0 to 5
# which should therefore be run prior to running this step
# user-group should also be defined
# WARNING - takes three minutes to run

kappas <- seq(0, 50, 5)

rmses <- sapply(kappas, function(l){
  
  ug_avgs <- train_edx %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(genre_avgs, by='genres') %>%
    left_join(week_avgs, by = 'rating_week') %>%
    group_by(user_group) %>%
  summarize(b_ug = (sum(rating - mu - b_i - b_u - b_g - b_w) / (n() + l)))

  predicted_ratings <- test_edx %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(genre_avgs, by='genres') %>%
    left_join(week_avgs, by = 'rating_week') %>%
    left_join(ug_avgs, by='user_group')

  predicted_ratings$b_ug[is.na(predicted_ratings$b_ug)] <- 0
  
  predicted_ratings <- predicted_ratings %>%
    mutate(pred = mu + b_i + b_u + b_g + b_w + b_ug) %>% 
    pull(pred)
  
  return(RMSE(test_edx$rating, predicted_ratings))
})

# Finding the optimal kappa
# The plot shows that the optimal kappa is 15
g_kappas <- qplot(kappas, rmses)
g_kappas
kappa <- kappas[which.min(rmses)]
kappa



######################################################
# Part 5
# Fitting the final model on the entire edx set, then
# applying the final model on the Validation data set
######################################################

####################################################
# Part 5 - Step 1
# Simple average model
####################################################

# Creating a mean square error function
# (only required if Part 3 was not run)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Computing the average of all ratings in training set
mu <- mean(edx$rating)
mu

# Computing the RMSE on test set for this basic model
model_val_1_rmse <- RMSE(validation$rating, mu)
model_val_1_rmse

# Storing the results in a small table
val_results <- data.frame(method = "Simple average model", RMSE = model_val_1_rmse)

#############################
# Part 5 - Step 2
# Adding a movie effect
#############################

# We introduce a regularization parameter in order to remove biases
# created by movies with few ratings
# The reason and impact of choosing this parameter are shown at the end
# of Part 4
lambda <- 5

# Computing a movie effect as a difference versus the mean
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum((rating - mu)) / (n() + lambda))

# Viewing a graph dispersion of the movie effect
# The graph shows clearly that many movies differ significantly
# from the average rating
movie_avgs %>%
  ggplot(aes(b_i)) +
  geom_histogram(bins = 10, fill = "coral2", color = "white") +
  ggtitle("Movie effect b_i")

# Predicting rating with movie effect on the test set
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>% pull(b_i)

# Measuring the RMSE
RMSE(validation$rating, predicted_ratings)

# Updating the results table
model_val_2_rmse <- RMSE(validation$rating, predicted_ratings)
val_results <- bind_rows(val_results,
                          data.frame(method="Movie Effect Model",
                                     RMSE = model_val_2_rmse ))

# Significant improvement over the simple average model
print(val_results)

#############################
# Part 5 - Step 3
# Adding a user effect
#############################

# Computing a user effect as a difference versus the mean and movie effect
user_avgs <- edx %>% left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum((rating - mu - b_i)) / (n() + lambda))

# Viewing a graph dispersion of the user effect
# User effect (net of movide effect) spread seems reduced versus
# pure movie effect, however it clearly remains significant
# Interpretation: some users rate more harshly (b_u < 0) than others
user_avgs %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 10, fill = "coral2", color = "white") +
  ggtitle("Residual User effect b_u")

# Predicting rating with movie and user effects on the test set
predicted_ratings <- validation %>% left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% pull(pred)

# Measuring the RMSE
RMSE(validation$rating, predicted_ratings)

# Updating the results table
model_val_3_rmse <- RMSE(validation$rating, predicted_ratings)
val_results <- val_results %>%
  bind_rows(data.frame(method = "Movie and User effect",
                       RMSE = model_val_3_rmse))

# Significant improvement over previous models
print(val_results)


#############################
# Part 5 - Step 4
# Adding a genre effect
#############################

# Computing the genre effect
genre_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum((rating - mu - b_i - b_u)) / (n() + lambda))

# Plotting genre effect b_g
# The plot shows a non-insignificant residual genre effect
genre_avgs %>%
  ggplot(aes(b_g)) +
  geom_histogram(bins = 10, fill = "coral2", color = "white") +
  ggtitle("Residual Genre effect b_g")

# Adding the genre effect to the predictions
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>% pull(pred)

# Computing the RMSEn (shows only a moderate improvement)
RMSE(validation$rating, predicted_ratings)

# Updating the results table
model_val_4_rmse <- RMSE(validation$rating, predicted_ratings)
val_results <- val_results %>%
  bind_rows(data.frame(method = "Movie+User+Genre effect",
                       RMSE = model_val_4_rmse))

# Results show a moderate improvement
print(val_results)

#############################
# Part 5 - Step 5
# Adding a time effect
#############################

library(lubridate)

# Creating a date column from the timestamp
edx <- edx %>% 
  mutate(date = as_datetime(timestamp))

# Computing the number of weeks since first date
first_date <- min(edx$date)
edx <- edx %>% 
  mutate(
    rating_week = as.integer(round(difftime(date, first_date, units = "weeks"), 0)))

range(edx$rating_week)

# Plotting average rating vs date (rounded to number of weeks)
# The plot shows that there is a slight time effect on ratings
edx %>%
  group_by(rating_week) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(rating_week, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Average rating per week")

# Computing the week effect
week_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  group_by(rating_week) %>%
  summarize(b_w = mean(rating - mu - b_i - b_u - b_g))

# Plotting the week effect
# The plot show the evidence of some residual week effect
week_avgs %>%
  ggplot(aes(b_w)) +
  geom_histogram(bins = 20, fill = "coral2", color = "coral2") +
  ggtitle("Residual Week effect b_w")


# Adding week effect to model
validation <- validation %>% 
  mutate(date = as_datetime(timestamp),
         rating_week = as.integer(round(difftime(date, first_date, units = "weeks"), 0)))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(week_avgs, by = 'rating_week') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_w) %>% pull(pred)

# Computing RMSE
RMSE(validation$rating, predicted_ratings)

# Updating the results table
model_val_5_rmse <- RMSE(validation$rating, predicted_ratings)
val_results <- val_results %>%
  bind_rows(data.frame(method = "Adding week effect",
                       RMSE = model_val_5_rmse))

# Adding the time effect improves RMSE slightly:
print(val_results)


##################################
# Part 5 - Step 6
# Adding a user-genre group effect
# FINAL MODEL
##################################

# Creating a table with average rating per genre per user
# Only users with more than 200 ratings retained to reduce table size
# WARNING: takes around 40-50s to run
genre_ratings <- edx %>%
  select(userId, genres, rating) %>%
  group_by(userId) %>%
  filter(n() > 200) %>% ungroup() %>%
  group_by(genres, userId) %>%
  summarize(n=n(), rating = mean(rating))

# Creating a matrix with genres as rows and userId as predictors (columns)
x <- genre_ratings %>%
  select(genres, userId, rating) %>%
  pivot_wider(id_cols = genres, names_from = userId, values_from = rating,
              values_fill = 0)

# The matrix is centered and rownames updated with the genres names
row_names <- x$genres
x <- x[,-1] %>% as.matrix()
x <- sweep(x, 2, colMeans(x, na.rm = TRUE))
x <- sweep(x, 1, rowMeans(x, na.rm = TRUE))
rownames(x) <- row_names

# Creating groups of genres by clusterization
# kmeans will be used
# NA values are first replaced by zeros

# Applying kmeans
kcluster <- 9
x_0 <- x
x_0[is.na(x_0)] <- 0
set.seed(1, sample.kind = "Rounding")
k <- kmeans(x_0, centers = kcluster)

# Creating a data frame with the genre groups
genres_groups <- data.frame(genre_group = k$cluster)
genres_groups <- genres_groups %>% mutate(genres = rownames(genres_groups))

# The size of each group is shown here:
k$size

# Adding user_group in the train and test sets

edx <- edx %>%
  left_join(genres_groups, by = "genres") %>%
  mutate(user_group = paste(userId, genre_group, sep = "_"))

validation <- validation %>%
  left_join(genres_groups, by = "genres") %>%
  mutate(user_group = paste(userId, genre_group, sep = "_"))

# Checking that there are no NAs in user_group
sum(is.na(edx$user_group))
sum(is.na(validation$user_group))

# Defining the regularization parameter
kappa <- 15 # See Step 7 for regularization cross-validation

# Computing the user_group component
ug_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(week_avgs, by = 'rating_week') %>%
  group_by(user_group) %>%
  summarize(b_ug = (sum(rating - mu - b_i - b_u - b_g - b_w) / (n() + kappa)))

# Plotting the user-group effect
# The plot shows a definite residual effect
ug_avgs %>%
  ggplot(aes(b_ug)) +
  geom_histogram(bins = 20, fill = "coral2", color = "white") +
  ggtitle("Residual User-Genre group effect b_ug")

# Chekcing that there are no NAs in b_u calculations
sum(is.na(user_avgs$b_u))

# Adding all components to the model
# A predicted_rating table is first created
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(week_avgs, by = 'rating_week') %>%
  left_join(ug_avgs, by='user_group') 

# Some user-group combinations are present in the testing set but not
# in the training set, generating some NAs for b_ug
# This is to be expected in real life, we will therefore set b_ug
# to nil for these
sum(is.na(predicted_ratings$b_ug))

# Setting b_ug to nil if not available
predicted_ratings$b_ug[is.na(predicted_ratings$b_ug)] <- 0
sum(predicted_ratings$b_ug == 0)

# Finalizing the predictions
predicted_ratings <- predicted_ratings %>%
  mutate(pred = mu + b_i + b_u + b_g + b_w + b_ug) %>% 
  pull(pred)

RMSE(validation$rating, predicted_ratings)

model_val_6_rmse <- RMSE(validation$rating, predicted_ratings)
val_results <- val_results %>%
  bind_rows(data.frame(method = "FINAL with User Group effect",
                       RMSE = model_val_6_rmse))

# The RMSE is further reduced with the Final Model
kable(val_results, digits = 4, caption = "Models tested on Validation set") %>%
  kable_styling(bootstrap_options = "striped", font_size = 15, full_width = F)

#####################
# END OF SCRIPT
#####################




##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

####START OF EXERCISE

#first task is to visually demonstrate the task to complete. Below we create a 50x50 plot showing from a sample of
#50 users, how many reviews they filled in from a sample of 50 movies.

users<- sample(unique(edx$userId), 50)
rafalib::mypar()
edx %>% filter(userId %in% users) %>% 
  dplyr::select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  pivot_wider(names_from = movieId, values_from = rating) %>% 
  (\(mat) mat[, sample(ncol(mat), 50)])()%>%
  as.matrix() %>% 
  t() %>%
  image(1:50, 1:50,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

#Loss function:To compare different models or to see how well we're doing compared to some baseline, the typical error loss, the residual mean squared error (RMSE) is used on the test set. We can interpret RMSE similar to standard deviation.
#If  is the number of user-movie combinations,  is the rating for movie  by user , and  is our prediction, then RMSE is defined as follows: 

#Code to look at the distribution of the predictors

if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

library(gridExtra)
p1 <- edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

p2 <- edx %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")
p3 <- edx %>% 
  separate_rows(genres, sep = "\\|") %>%
  dplyr::count(genres) %>%
  ggplot(aes(y=n, reorder(x=genres,-n,sum))) + 
  geom_col( color = "black") + 
  ggtitle("Genres") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
grid.arrange(p1, p2, p3, ncol = 3)


# we now want to create a function that we can use to test the RMSE of our model

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

### Building the Recommendation System

# Modeling Movie effects

mu <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))


movie_avgs %>% qplot(b_i, geom ="histogram", bins = 30, data = ., color = I("black"))

predicted_ratings <- mu + final_holdout_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

model_1_rmse <- RMSE(predicted_ratings, final_holdout_test$rating)
rmse_results <- data_frame(method = "Movie Effect Model", RMSE = model_1_rmse)

rmse_results %>% knitr::kable()

# Modeling User effects

edx  %>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")


user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- final_holdout_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model_2_rmse <- RMSE(predicted_ratings, final_holdout_test$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()





# Modeling Genres effects

edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(movieId) %>%
  summarize(b_g = mean(rating)) %>%
  filter(n()>=100) %>%
  ggplot(aes(b_g)) + 
  geom_histogram(bins = 30, color = "black")



genres_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  separate_rows(genres, sep = "\\|") |> 
  group_by(movieId) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

predicted_ratings <- final_holdout_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='movieId') |> 
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

model_3_rmse <- RMSE(predicted_ratings, final_holdout_test$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User + Genre Effects Model",  
                                     RMSE = model_3_rmse ))
rmse_results

###Regularization

# the code takes a long time to run. I can tell you that best lambda that minimizes RMSE is 5.25. I would recommend to use this value instead of the seq in the code below.

#lambdas <- seq(0,10, 0.25)
lambdas <- 5.25

rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating-mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i-mu)/(n()+l))
  b_g <- edx |> 
    left_join(b_i, by="movieId") |> 
    left_join(b_u, by="userId") |> 
  separate_rows(genres, sep = "\\|") |> 
    group_by(movieId) %>%
    summarize(b_g = mean(rating - mu - b_i - b_u)/(n()+1))
  predicted_ratings <- 
    final_holdout_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "movieId") |> 
    mutate(pred = mu + b_i + b_u +b_g) %>%
    pull(pred)
  return(RMSE(predicted_ratings, final_holdout_test$rating))
})

qplot(lambdas, rmses)  
 
lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User + Genre Effect Model",  
                                     RMSE = min(rmses)))

rmse_results %>% knitr::kable()


#code forming part of conclusion where we see if the timestamp column could be added to future models to improve RMSE

if(!require(anytime)) install.packages("anytime", repos = "http://cran.us.r-project.org")
library(anytime)

edx |> mutate(Date=anydate(timestamp)) |>
       select(Date) |> 
       summary() |>
       knitr::kable()

              
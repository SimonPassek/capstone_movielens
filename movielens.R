# the final algorithm that predicts the ratings on the validation set using the edx set.

# load libraries
if (!require("pacman")) install.packages("pacman"); 
pacman::p_load("tint", "tidyverse", "data.table", "janitor", "caret", "lubridate")



# download and wrangle data -----------------------------------------------


dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
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



# save data ---------------------------------------------------------------

saveRDS(validation, "validation")
saveRDS(edx, "edx")


# create data partition for training and testing algorithm ----------------

# create train and test data set
set.seed(1, sample.kind = "Rounding")

test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)

train_set_incomplete <- edx[-test_index,]
test_set_incomplete <- edx[test_index,]

# to make sure to have the same movieIds and userIds in train- and test_set
test_set <- test_set_incomplete %>% 
  semi_join(train_set_incomplete, by = "movieId") %>% 
  semi_join(train_set_incomplete, by = "userId")

#row_bind the dropped rows to the train_set
removed <- anti_join(test_set_incomplete, test_set)
train_set <- rbind(train_set_incomplete, removed)

# mutate new columns for further analysis

train_set <- train_set %>% mutate(date = lubridate::as_datetime(timestamp))
train_set  <- train_set %>% mutate(week = lubridate::round_date(date, "week"))

test_set <- test_set %>% mutate(date = lubridate::as_datetime(timestamp))
test_set <- test_set %>% mutate(week = lubridate::round_date(date, "week"))

# save train_set and test_set ---------------------------------------------

saveRDS(train_set, "train_set")
saveRDS(test_set, "test_set")



# using bootstrap for picking best lambda ---------------------------------
# here the test_set is not used;
# bootstrapping as sampling with resampling


# define lambdas 
lambdas = seq(4, 6, 0.1)

set.seed(1, sample.kind = "Rounding")

B <- 5 # number of bootstrap iterations

rmse_bootstrapping <- replicate(B, {
  
  #create bootstrap sample without using test_set
  boots_index <- sample(1:nrow(train_set),replace = TRUE, size = 1440018)
  train_set_boots <- train_set[-boots_index,]
  test_sub <- train_set[boots_index,] # 20 % of the test_set as bootstrap validation set
  
  # make sure users and movies of train set are in the test set
  test_set_boots <- test_sub %>% 
    semi_join(train_set_boots, by = "movieId") %>% 
    semi_join(train_set_boots, by = "userId")
  
  lost <- anti_join(test_sub, test_set_boots)
  
  train_set_boots <- rbind(train_set_boots, lost)
  
  
  rmse <- map_df(lambdas, function(x){
    
    mu <- mean(train_set_boots$rating)
    
   # movie effects with regularization 
    movie_effects_reg <- train_set_boots %>% 
      group_by(movieId) %>% 
      summarize(b_i = sum(rating - mu)/(n()+x))
    
   # user effects with regularization  
    user_effects_reg <- train_set_boots %>%  
      left_join(movie_effects_reg, by = "movieId") %>% 
      group_by(userId) %>% 
      summarize(b_u = sum(rating - mu - b_i)/(n()+x))
    
   # genres effects with regularization
    genres_effects_reg <- train_set_boots %>% 
      left_join(movie_effects_reg, by = "movieId") %>% 
      left_join(user_effects_reg, by = "userId") %>% 
      group_by(genres) %>% 
      summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+x))
    
   # week effects with regularization
    week_effects_reg <- train_set_boots %>% 
      left_join(movie_effects_reg, by = "movieId") %>% 
      left_join(user_effects_reg, by = "userId") %>%
      left_join(genres_effects_reg, by = "genres") %>% 
      group_by(week) %>% 
      summarize(b_w = sum(rating - mu - b_i - b_u - b_g)/(n()+x))
    
   # making the prediction on the test set 
    prediction <- test_set_boots %>% 
      left_join(movie_effects_reg, by = "movieId") %>% 
      left_join(user_effects_reg, by = "userId") %>%
      left_join(genres_effects_reg, by = "genres") %>% 
      left_join(week_effects_reg, by = "week") %>%  
      mutate(pred = mu + b_i + b_u + b_g + b_w) %>% 
      pull(pred)
    
    # calculating RMSE
    rmse <- RMSE(prediction[which(!is.na(prediction))], test_set_boots$rating[which(!is.na(prediction))])
    
    tibble(rmse = rmse, lambda = x)
  })
  
  rmse
})


# tibble with results
rmse_bootstrapping_df <- bind_cols(lapply(rmse_bootstrapping, function(x)as_tibble(x)))

# defining colnames of tibble
colnames(rmse_bootstrapping_df) <- sapply(seq_len(B), function(x){
  c(paste("RMSE", x, sep = "_"), paste("lambda", x, sep = "_"))
})

# gather tibble

rmse_bootstrapping_df_long <- rmse_bootstrapping_df %>% pivot_longer(everything(), names_to = c(".value", "bootstrap_iteration"), names_pattern = "(.*)_(.*)" )

# filtering for best lambdas in all bootstrap iterations

rmse_bootstrapping_df_long_best <- rmse_bootstrapping_df_long  %>% group_by(bootstrap_iteration) %>% filter(RMSE == min(RMSE)) %>% arrange(RMSE)

View(rmse_bootstrapping_df_long_best)

# the best/(mean) lambda over all resamplings is 5.1



# using whole test_set for building algorithm and testing on validation set --------



validation <- validation %>% mutate(date = lubridate::as_datetime(timestamp)) %>% 
  mutate(week = lubridate::round_date(date, unit = "week"))

edx <- edx %>% mutate(date = lubridate::as_datetime(timestamp)) %>% 
  mutate(week = lubridate::round_date(date, unit = "week"))


# defining train_set and test_set as edx and validation

train_set <- edx
test_set <- validation

lambdas <- 5.1

rmse_lambdas_all <- map_df(lambdas, function(x){
  
  mu <- mean(train_set$rating)
  
  movie_effects_reg <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+x))
  
  user_effects_reg <- train_set %>% 
    left_join(movie_effects_reg, by = "movieId") %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - mu - b_i)/(n()+x)) 
  
  genre_effects_reg <- train_set %>% 
    left_join(movie_effects_reg, by = "movieId") %>% 
    left_join(user_effects_reg, by = "userId") %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating- mu - b_i - b_u)/(n()+x))
  
  week_effects_reg <- train_set %>% 
    left_join(movie_effects_reg, by = "movieId") %>% 
    left_join(user_effects_reg, by = "userId") %>% 
    left_join(genre_effects_reg, by = "genres") %>% 
    group_by(week) %>% 
    summarize(b_w = sum(rating - mu - b_i - b_u - b_g)/(n()+x))
  
  prediction <- test_set %>% 
    left_join(movie_effects_reg, by = "movieId") %>% 
    left_join(user_effects_reg, by = "userId") %>% 
    left_join(genre_effects_reg, by = "genres") %>% 
    left_join(week_effects_reg, by = "week") %>% 
    mutate(pred = mu+b_i+b_u+b_g+b_w) %>% 
    pull(pred)
  
  
  rmse <- RMSE(test_set$rating, prediction)
  
  tibble(rmse)
})

rmse_lambdas_all_view <- rmse_lambdas_all %>% mutate(lambdas) %>% arrange(rmse)

View(rmse_lambdas_all_view)


#Project 1
###################################
# Create edx set and validation set
###################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1) # if using R 3.6.0: set.seed(1, sample.kind = "Rounding")
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

# Below is the algorithm I developed on the edx set
# For grading, I will run algorithm on validation set to generate ratings

#I will keep the original rating from the validation set in a new variable val_rating for checking at the end.
val_rating <- validation$rating

#I am electing to use matrix factorization, recosystem library to train and validate the model.
#Before training the model using recosystem library, I am trying out the course method on different variables. 

#First I am using the user effect
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Modeling movie effects
mu <- mean(edx$rating)

#Using movie average
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

#I will compute an approximation by computing ^?? and ^bi and estimating ^bu as the average of yu,i ??? ^?? ??? ^bi
user_avgs <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#constructing predictors to see how much the RMSE improves compared to naive method
predicted_ratings <- validation %>%  
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

naive_rmse <- RMSE(validation$rating, mu)
rmse_results <- data_frame(Method = "Just the average", RMSE = naive_rmse)
model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))

#This produces a table with the Method used and the RMSE for that method
rmse_results %>% knitr::kable()



#One more try is to use the timestamp for validation
#convert the timestamp to year only
validation_year <- validation %>% mutate(Year=substring(as.Date(as.POSIXct(validation$timestamp, origin="1970-01-01")),1,4)) %>% select(userId, movieId, rating, Year)

#Using Year Averages
year_avgs <- validation_year %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs,by='userId') %>%
  group_by(Year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u))

#constructing predictors to see how much the RMSE improves compared to previous methods
predicted_ratings <- validation_year %>%  
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs,by='Year') %>%
  mutate(pred = mu + b_i + b_u + b_y) %>%
  .$pred

naive_rmse <- RMSE(validation$rating, mu)
model_4_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie + User Effects + Year Effects Model",  
                                     RMSE = model_4_rmse ))

#This produces a table with the Method used and the RMSE for that method
rmse_results %>% knitr::kable()



#Another attempt is to use the year timestamp without the user effects
year_avgs <- validation_year %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(Year) %>%
  summarize(b_y = mean(rating - mu - b_i))

#constructing predictors to see how much the RMSE improves compared to previous methods
predicted_ratings <- validation_year %>%  
  left_join(movie_avgs, by='movieId') %>%
  left_join(year_avgs,by='Year') %>%
  mutate(pred = mu + b_i + b_y) %>%
  .$pred

naive_rmse <- RMSE(validation$rating, mu)

model_5_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie + Year Effects Model",  
                                     RMSE = model_5_rmse ))

#This produces a table with the Method used and the RMSE for that method
rmse_results %>% knitr::kable()

#Next, instead of year from timestamp. we will use month of the movie release.
#USe the tmiestamp to find the month movies were released and use that as one of the effects
validation_month <- validation %>% mutate(Month=substring(as.Date(as.POSIXct(validation$timestamp, origin="1970-01-01")),6,7)) %>% select(userId, movieId, rating, Month)

#Month Averages
month_avgs <- validation_month %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs,by='userId') %>%
  group_by(Month) %>%
  summarize(b_m = mean(rating - mu - b_i - b_u))

#constructing predictors to see how much the RMSE improves compared to previous methods
predicted_ratings <- validation_month %>%  
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(month_avgs,by='Month') %>%
  mutate(pred = mu + b_i + b_u + b_m) %>%
  .$pred

naive_rmse <- RMSE(validation$rating, mu)

model_6_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie + User Effects + Month Effects Model",  
                                     RMSE = model_6_rmse ))

#This produces a table with the Method used and the RMSE for that method
rmse_results %>% knitr::kable()



#Another attempt is to use the month timestamp without the user effects
month_avgs <- validation_month %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(Month) %>%
  summarize(b_m = mean(rating - mu - b_i))

#constructing predictors to see how much the RMSE improves compared to previous methods
predicted_ratings <- validation_month %>%  
  left_join(movie_avgs, by='movieId') %>%
  left_join(month_avgs,by='Month') %>%
  mutate(pred = mu + b_i + b_m) %>%
  .$pred

naive_rmse <- RMSE(validation$rating, mu)
model_7_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie + Month Effects Model",  
                                     RMSE = model_7_rmse ))

#This produces a table with the Method used and the RMSE for that method
rmse_results %>% knitr::kable()


#Now, we will use genre to see if it has any impact to the RMSE. 
#Separating genres to one line ber genre, where a movie has multiple genres separated by |, I create a new row for each
val_sep <- validation %>% separate_rows(genres, sep ="\\|") %>% group_by(genres)

#Try a prediction based on user effect plus genre
#Genre Averages
genre_avgs <- val_sep %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs,by='userId') %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

#constructing predictors to see how much the RMSE improves compared to previous methods
predicted_ratings <- val_sep %>%  
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs,by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

naive_rmse <- RMSE(validation$rating, mu)
model_8_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie + User Effects + Genre Effects Model",  
                                     RMSE = model_8_rmse ))

#This produces a table with the Method used and the RMSE for that method
rmse_results %>% knitr::kable()



#Repeat above without adding the user effect
#Genre Averages
genre_avgs <- val_sep %>% 
  left_join(movie_avgs, by='movieId') %>%
  summarize(b_g = mean(rating - mu - b_i ))

#constructing predictors to see how much the RMSE improves compared to previous methods
predicted_ratings <- val_sep %>%  
  left_join(movie_avgs, by='movieId') %>%
  left_join(genre_avgs,by='genres') %>%
  mutate(pred = mu + b_i + b_g) %>%
  .$pred

naive_rmse <- RMSE(validation$rating, mu)
model_9_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie + Genre Effects Model",  
                                     RMSE = model_9_rmse ))

#This produces a table with the Method used and the RMSE for that method
rmse_results %>% knitr::kable()





#Finally, we will use recosystem library for matrix factorization method to see if we can get a better RMSE

#First we remove any prediction from previous steps in validation$rating
validation <- validation %>% select(-rating)

#Train the model and predict
#With the recosystem library, we are able to manipulate the matrix in parralle blocks
library(recosystem)

train_data <- data_memory(user_index = edx$userId, item_index = edx$movieId, 
                          rating = edx$rating, index1 = T)
test_data <- data_memory(user_index = validation$userId, item_index = validation$movieId, 
                         rating = validation$rating, index1 = T)
recommender <- Reco()
recommender$train(train_data, opts = c(dim = 30, costp_l2 = 0.1, costq_l2 = 0.1, 
                                       lrate = 0.1, niter = 100, nthread = 6, verbose = F)) 
validation$rating <- recommender$predict(test_data, out_memory())


naive_rmse <- RMSE(val_rating, mu)
model_3_rmse <- RMSE(val_rating, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Recosystem",  
                                     RMSE = model_3_rmse ))

rmse_results %>% knitr::kable()

#The recosystem gave us the best RMSE


#Doing some high level check for our prediction
#This part is just for fun
#original rating
head(val_rating,10)
#predicted rating from the recosystem method
head(validation$rating,10)


# round to one decimal place
pr<- round(predicted_ratings,1)
valr <- round(validation$rating,1)
mean(pr)
mean(valr)
#end of checking


cat("The Final RMSE for this project is: ", model_3_rmse)



#Final step is to create the submission file and cleanup memory used 

# Ratings will go into the CSV submission file below:

write.csv(validation %>% select(userId, movieId, rating),
          "submission.csv", na = "", row.names=FALSE)
rm(dl, ratings, movies, test_index, temp, movielens, removed)


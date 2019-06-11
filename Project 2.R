#####################################
# Create train set and validation set
#####################################

#removing all variable from global environment before running the code (cleanup)
rm(list=ls())

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("RandomForest", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")
#if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
#if(!require(ggpubr)) install.packages("ggpubr", repos = "http://cran.us.r-project.org")
#if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")



# NYC Property Sales dataset:
# https://www.kaggle.com/new-york-city/nyc-property-sales/downloads/nyc-property-sales.zip/1
# Data is downloaded onto my local machine and read into variable data

data <- read.csv("nyc-rolling-sales.csv")

#I am removing columns not used in this analysis
cleandata <- select(data,-c(X,TAX.CLASS.AT.PRESENT, EASE.MENT, 
                            BUILDING.CLASS.AT.PRESENT,TAX.CLASS.AT.TIME.OF.SALE,BUILDING.CLASS.AT.TIME.OF.SALE))

#Data might be skewed, I am removing rows with blank, 0 or dash for sale price, 0 for Year Built or 0 for zipcode
cleandata <- cleandata[!(is.na(cleandata$SALE.PRICE) | cleandata$SALE.PRICE==" -  " | cleandata$SALE.PRICE==0 | cleandata$YEAR.BUILT==0 | cleandata$ZIP.CODE==0),]

#For further analysis, we want to remove all properties without gross square footage
cleandata <- cleandata[-which(cleandata$GROSS.SQUARE.FEET ==" -  "),]


#convert Borough and Sale Price from factor to numeric
cleandata$BOROUGH <- as.numeric(as.character(cleandata$BOROUGH))
cleandata$SALE.PRICE <- as.numeric(as.character(cleandata$SALE.PRICE))

#Remove any duplicated rows
cleandata <- cleandata %>% distinct()

#Visualizing the resulting dataset
boxplot(cleandata)
histogram(cleandata$YEAR.BUILT) # We can see most sales are for houses built in the 1900s and later)


#Plotting the data to see where the price concentration occurs in relation to year house is built 
ggplot(data=cleandata, mapping = aes(x=SALE.PRICE, y= YEAR.BUILT)) + geom_point() +
  scale_x_continuous(breaks= round(seq(min(cleandata$SALE.PRICE), max(cleandata$SALE.PRICE), by=450000000),1)) +
  scale_y_continuous(breaks= round(seq(min(cleandata$YEAR.BUILT), max(cleandata$YEAR.BUILT), by=100),1))


#Replot the data to show the concentrated area only. Remove the very few outliers in price and year built.        
plot(cleandata$SALE.PRICE,cleandata$YEAR.BUILT,xlim=c(1,3e+08), ylim=c(1800,2020))

#Addiotnal cleaning
#remove all prices below $100K or above $300 Million. Also remove year built prior to 1800
cleandata <- cleandata[which(cleandata$SALE.PRICE > 100000 & cleandata$SALE.PRICE <= 300000000),]
cleandata <- cleandata[which(cleandata$YEAR.BUILT >= 1800),]
plot(cleandata$SALE.PRICE,cleandata$YEAR.BUILT, main="Price Vs Year Built",xlab="Sales Price",
     ylab="Year Built", col="green")


cyl <- c("magenta","blue","green","yellow","cyan","red")

#Labels for ggplot
labs(title = waiver(), subtitle = waiver(), caption = waiver(),
     tag = waiver())



#Seeing sales by Residential Units vs Building Category
ggplot(data=cleandata, mapping = aes(x=reorder(BUILDING.CLASS.CATEGORY,-RESIDENTIAL.UNITS) ,y=RESIDENTIAL.UNITS)) + geom_bar(stat = "identity") +
  labs(x="Building Category",y="Residential") + theme(axis.text.x = element_text(angle=90,hjust=1, colour = "blue"))
  


#Seeing sales by Commercial Units vs Building Category
ggplot(data=cleandata, mapping = aes(x=reorder(BUILDING.CLASS.CATEGORY,-COMMERCIAL.UNITS) ,y=COMMERCIAL.UNITS)) + geom_bar(stat = "identity") + 
  labs(x="Building Category",y="Commercial") + theme(axis.text.x = element_text(angle=90,hjust=1))


#Seeing property allocation by Borough
histogram(cleandata$BOROUGH, xlab="Borough", col= cyl)



#Validation set will be 10% of NYC Property Sales cleaned data
RNGkind("Super")
set.seed(1, "Super")
test_index <- createDataPartition(y = cleandata$BOROUGH, times = 1, p = 0.1, list = FALSE)
trainset <- cleandata[-test_index,]
validation <- cleandata[test_index,]

#To avoid memory shortage, I decided to use a subset of the columns. In reality I would add more columns
#and use more powerful machine to do this. 
trainset_subset <- select(trainset, c(BOROUGH, NEIGHBORHOOD,SALE.PRICE,YEAR.BUILT))
validation_subset <- select(validation, c(BOROUGH, NEIGHBORHOOD,SALE.PRICE,YEAR.BUILT))

#Create a model using Lasso Regression
tr.control <- trainControl(method="repeatedcv", number = 10,repeats = 10)
lambdas <- seq(1,0,-.001)
set.seed(1957, "Super")
lasso_model <- train(SALE.PRICE~., data=trainset_subset,method="glmnet",metric="RMSE",
                     maximize=FALSE,trControl=tr.control,
                     tuneGrid=expand.grid(alpha=1,lambda=c(1,0.1,0.05,0.01,seq(0.009,0.001,-0.001), 0.00075,0.0005,0.0001)))

#Prediction using validation subset
lassopreds <- round(predict(lasso_model,newdata = validation_subset), 2)

#Final plot of predicted sales price vs year built
plot(lassopreds,validation$YEAR.BUILT, main="Price Vs Year Built", xlab = "Predicted Sales Price", ylab = "Validation Year Built", col="cyan")


#Calculate the RMSE to see how well our model is doing
RMSE <- function(true_prices, predicted_prices){
  sqrt(mean(log(true_prices) - log(predicted_prices))^2)
}

cat("LASSO RMSE: ", RMSE(lassopreds,validation_subset$SALE.PRICE))

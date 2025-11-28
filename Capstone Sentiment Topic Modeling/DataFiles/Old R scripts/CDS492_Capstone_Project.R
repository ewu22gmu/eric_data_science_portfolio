#---------------------------------------------+
# R SCRIPT FILE FOR CDS492 CAPSTONE PROJECT   |
# AUTHOR: ERIC WU                             |
# DATE: 09.26.2025                            |
# VERSION: 0.1                                |                               
#---------------------------------------------+

###Initialize==================================================================+

#Load in libraries
library(conflicted) #fix dplyr and stats conflicts
library(tidyverse)
#prefer dplyr library for filter() and lag()
conflicts_prefer(dplyr::filter(),dplyr::lag(),.quiet=TRUE)
library(readr) #for reading in .csv files with automatic encoding detection
library(lubridate) #used for date + time objects 
library(TTR) #used for SMA
library(quanteda) #used for NLP text preprocessing
library(udpipe)
library(stringr)
library(vader) #used for sentiment analysis

#Set working directory
file_dir = "/Users/eric/Documents/University/CDS492/Capstone Proposal/DataFiles"
setwd(file_dir)

#Read in data and clean
sp500 <- read.csv('SPX.csv') %>%
  mutate(
    Date = mdy(Date), # mdy() is for Month-Day-Year format
    across(c(Open, High, Low, Close), parse_number)
  )
nasdaq <- read.csv('IXIC.csv') %>%
  mutate(
    Date = mdy(Date), # mdy() is for Month-Day-Year format
    across(c(Open, High, Low, Close), parse_number)
  )
ts <- read.csv('truth_archive.csv') %>% 
  mutate(
    created_at = ymd_hms(created_at)
  )




###Data Cleansing==============================================================+

#Clean sp500 and nasdaq for date formating-------------------------------------+
nasdaq <- nasdaq %>% mutate(Date = mdy(Date))
sp500 <- sp500 %>% mutate(Date = mdy(Date))

#CLEAN ts----------------------------------------------------------------------+
#Filter 'truth social' for posts within the time frame of 01/01/2025 - 09/26/2026
ts$created_at <- ymd_hms(ts$created_at)
#define start and end dates of the time frame
start_date <- ymd("2025-01-01")
end_date <- ymd("2025-09-26")

#filter df
ts1 <- ts %>% filter(created_at >= start_date, created_at <= end_date)
#filter df for blank content
ts1 <- ts1 %>% filter(trimws(content) != '')
#filter df
ts1 <- ts %>% filter(created_at >= start_date, created_at <= end_date)
#filter df for blank content
ts1 <- ts1 %>% filter(trimws(content) != '')




### Feature Generation=========================================================+

#Feature generation for the indexes--------------------------------------------+
#S&P500
sp500 <- sp500 %>% 
  mutate(
    #finance features
    daily_change = Close - Open, #add a change in price per day
    daily_volatility = High - Low, 
    daily_return = (Close - lag(Close)) / lag(Close), #pct change day by day
    sma20 = SMA(Close, n = 20), #20 day simple moving avg
    
    #time based features
    day_of_week = wday(Date, label = TRUE, abbr = FALSE), #add a day of week label
    month = month(Date, label = TRUE, abbr = FALSE) #add a month label
  )

#add a movement feature
sp500 <- sp500 %>% 
  mutate(
    Result = case_when(
      daily_change > 0 ~ "gain",
      daily_change < 0 ~ "loss",
      daily_change == 0 ~ "none",
      TRUE ~ NA_character_ #fallback for NA values
    )
  )

#NASDAQ
nasdaq <- nasdaq %>% 
  mutate(
    #finance features
    daily_change = Close - Open, #add a change in price per day
    daily_volatility = High - Low, 
    daily_return = (Close - lag(Close)) / lag(Close), #pct change day by day
    sma20 = SMA(Close, n = 20), #20 day simple moving avg
    
    #time based features
    day_of_week = wday(Date, label = TRUE, abbr = FALSE), #add a day of week label
    month = month(Date, label = TRUE, abbr = FALSE) #add a month label
  )

#add a movement feature
nasdaq <- nasdaq %>% 
  mutate(
    Result = case_when(
      daily_change > 0 ~ "gain",
      daily_change < 0 ~ "loss",
      daily_change == 0 ~ "none",
      TRUE ~ NA_character_ #fallback for NA values
    )
  )


#Apply VADER lexicon based sentiment analysis----------------------------------+

ts1_vader_sentiment <- vader_df(ts1$content) # apply vader lexicon based sentiment analysis
ts1 <- cbind(ts1, ts1_vader_sentiment) #join sentiment results back to original df
ts1 <- ts1 %>%
  mutate(sentiment_class = case_when(
    compound >= 0.05 ~ "Positive",
    compound > -0.05 & compound < 0.05 ~ "Neutral",
    compound <= -0.05 ~ "Negative",
    TRUE ~ NA_character_ # Handle any unexpected cases
  ))
vader_sentiment_counts <- ts1 %>%
  count(sentiment_class)


#Merge Tables------------------------------------------------------------------+
#join indexes 
indices <- sp500 %>% 
  inner_join(nasdaq, 
             by = 'Date',
             suffix = c('_sp500','_nasdaq')
  )




### EDA Visualizations=========================================================+

#Visualization of DJT posts per day against the closing price of the indices---+
#get posts per day from ts1
ppd <- ts1 %>% 
  mutate(Date = as_date(created_at)) %>%
  group_by(Date) %>%
  summarise(post_count = n()) %>%
  ungroup()

#combine ppd and indicies data
ppd_nasdaq <- nasdaq %>%
  left_join(ppd, by = 'Date') %>%
  mutate(post_count = replace_na(post_count, 0))

#scale 
scale_factor <- max(ppd_nasdaq$Close, na.rm = TRUE) / max(ppd_nasdaq$post_count, na.rm = TRUE)

#plot
p1 <- ggplot(ppd_nasdaq, aes(x = Date)) +
  geom_line(aes(y = Close, color = "Closing Price"), linewidth = 1) +
  geom_col(aes(y = post_count * scale_factor, fill = "Tweet Count"), alpha = 0.5) +
  scale_y_continuous(
    name = "Stock Price (USD)",
    # Add a secondary axis for the tweet count, using the scaling factor
    sec.axis = sec_axis(~ . / scale_factor, name = "Count of Tweets")
  ) +
  scale_color_manual(name = "Data", values = "darkblue") +
  scale_fill_manual(name = "Data", values = "coral") +
  labs(
    title = "NASDAQ Closing Price vs. Tweet Count",
    x = "Date",
    color = "Data",
    fill = "Data"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

plot(p1)

#Plot normalized indices over time---------------------------------------------+
indices_normalized <- indices %>%
  mutate(
    Close_sp500_normalized = (Close_sp500 / first(Close_sp500)) * 100,
    Close_nasdaq_normalized = (Close_nasdaq / first(Close_nasdaq)) * 100
  )

p4 <- ggplot(data = indices_normalized, aes(x = Date)) +
  geom_line(aes(y = Close_sp500_normalized, color = "S&P 500"), linewidth = 1) +
  geom_line(aes(y = Close_nasdaq_normalized, color = "NASDAQ"), linewidth = 1) +
  labs(
    title = "S&P 500 vs. NASDAQ Performance",
    subtitle = "Indexed to start date (Base = 100)",
    y = "Normalized Price",
    x = "Date",
    color = "Index"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
plot(p4)

#Plot seperate boxplots of numerical features of indices-----------------------+

#sp500
sp500_long <- sp500 %>%
  pivot_longer(
    cols = where(is.numeric),
    names_to = "metric",
    values_to = "value"
  )

p8 <- ggplot(sp500_long, aes(x = metric, y = value)) +
  geom_boxplot() +
  labs(
    title = "Distribution of Metrics by Index",
    x = "Metric",
    y = "Value"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

plot(p8)

#nasdaq
nasdaq_long <- nasdaq %>%
  pivot_longer(
    cols = where(is.numeric),
    names_to = "metric",
    values_to = "value"
  )

p9 <- ggplot(nasdaq_long, aes(x = metric, y = value)) +
  geom_boxplot() +
  labs(
    title = "Distribution of Metrics by Index",
    x = "Metric",
    y = "Value"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

plot(p9)


#Plot Sentiment score distribution---------------------------------------------+
p10 <- ggplot(vader_sentiment_counts, aes(x = sentiment_class, y = n, fill = sentiment_class)) +
  geom_bar(stat = "identity") +
  labs(
    title = "Sentiment Distribution of Social Media Posts",
    x = "Sentiment",
    y = "Number of Posts"
  ) +
  theme_minimal()

plot(p10)


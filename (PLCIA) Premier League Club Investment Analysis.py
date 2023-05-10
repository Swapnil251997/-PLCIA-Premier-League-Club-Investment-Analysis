#!/usr/bin/env python
# coding: utf-8

# # Premier League Club Investment Analysis for an Investment company (EDA) 
# 
# **Domain: Sports**
# 
# **Context:**
#     
# A renowned investment firm usually invest in top-tier sports teams that have potential. The dataset in their possession comprises crucial information about all the clubs that have participated in premier league (assume that it has the data for all clubs). It includes data on the number of goals scored, the number of times they have finished in the top two positions and other relevant details.
# 
# **Data:**
# Premier League Final Data.csv- : The data set contains information on all the clubs so far participated in all the premier league tournaments.
# 
# **Data Dictionary:**
# 
# * Club: Name of the football club
# * Matches: Number of matches the club has played in the Premier League
# * Wins: Number of matches won by the club in the Premier League
# * Loss: Number of matches lost by the club in the Premier League
# * Draws: Number of matches drawn by the club in the Premier League
# * Clean Sheets: Number of matches in which the club has prevented the opposing side from scoring
# * Team Launch: Year in which the club was founded
# * Winners: Number of times the club has won the Premier League
# * Runners-up: Number of times the club has finished as runners-up in the Premier League
# * lastplayed_pl: Year in which the team last played in the Premier League
# 
# **Project Objective**
# 
# The management of the firm aims to invest in one of the top-performing club in the English Premier League. To aid in their decision-making process, the analytics department has been tasked with creating a comprehensive report on the performance of various clubs. However, some of the more established clubs have already been owned by the competitors. As a result, the firm wishes to identify the clubs they can approach and potentially invest to ensure a successful and profitable deal.

# **Key learning after this project:**
# 
# - Data cleaning is the process of identifying and correcting or removing errors, inconsistencies, and inaccuracies in a dataset.
# - Observation writing involves examining the data and noting any notable findings, anomalies, or areas of interest.
# - Exploratory Data Analysis (EDA) is the process of examining and visualizing a dataset to understand its main characteristics, such as the distribution of data, the relationships between variables, and any anomalies or patterns that may exist. The goal of EDA is to uncover insights and trends that can help inform further analysis or decision-making. It is often the first step in any data analysis project, as it provides a foundation for more advanced statistical methods and models.
# - Treat Null values basis domain knowledge aka using Domain-specific imputation

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv(r'D:\conda\Premier_League_Final_Data_batch2.csv')


# ### Section A: Explore the Dataset

# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


# Get more information about datatypes and null values in the dataframe

df.info()


# ### Section B: Clean the Dataset

# In[7]:


# Let us first start with Club column

df['Club']= df['Club'].str.replace('\d+', '')


# In[8]:


df.head()


# Now that the numbers have been removed from the front of each club name in the Club column, it has been cleaned and is ready to be used for analysis

# In[9]:


df["Winners"].isnull()


# In[10]:


df['Winners'].value_counts() 

# The code returns the count of unique values in the "Winners" column and the number of times each value occurs.


# Upon inspecting the dataset, it can be observed that there are a total of 25 non-null values. Furthermore, it is noteworthy that out of the 18 football clubs listed, none of them have won the Premier League title, as the "Winners" column displays a count of 0 for each club.
# After looking at the counts, it has been determined that there have been a total of 30 Premier League tournaments held in the past (1992-2022 per year one tournament). Out of the 25 football clubs (Non zero non nulls in winner columns) listed in the dataset, 3 clubs have won the Premier League title once, 1 club has won it thrice, 1 club has won it 5 times, another club has won it 6 times, and 1 club has won it a remarkable 13 times, totaling to 30 victories.

# In[11]:


# Replace null values with 0 in the "Winners" column
df["Winners"].fillna(0, inplace=True)


# It would be appropriate to update the "Winners" column by replacing the null values with 0, as these clubs have not won the Premier League title. This data cleaning step will ensure that the dataset accurately reflects the historical performance of each club in terms of Premier League wins.

# In[12]:


# Check for null values in the "Winners" column after data cleaning
df["Winners"].isnull().any()


# In[13]:


# Next, let is look at Runners-up, As seen earlier even this column has Null value

df['Runners-up'].value_counts()


# Teams have different numbers of runner-up finishes. One team has finished as runner-up 7 times, another 6 times, one team 5 times, another 4 times, another 3 times, one team 2 times and three teams have finished as runner-up once each.
# We also notice some inconsistency in data,this column particularly has null values, 0's and '-' We need to clean

# In[14]:


# No. of runner-ups
1+1+1+6+4+5+3+7+2


# In[15]:


# Since we know the no. of times Premier League was conducted is 30 and we have data for all we will convert the null & '-' to 0 for all other clubs

# replace '-' and null values with zero
df['Runners-up'].fillna(0, inplace=True)
df['Runners-up'].replace('-', 0, inplace=True)


# In[16]:


# Also we have seen it earlier that 'Runners-up' column is "Object" type let us convert it into int type
df['Runners-up'] = pd.to_numeric(df['Runners-up'], errors='coerce')
df['Runners-up'] = df['Runners-up'].astype('Int64')


# In[17]:


# Check the datatype

df.info()


# In[18]:


# We also observed TeamLaunch column has data inconsistency

df['TeamLaunch'].value_counts()


# In[19]:


# We need to convert the 'TeamLaunch' into 'YYYY'

# convert the column to datetime format
df['TeamLaunch'] = pd.to_datetime(df['TeamLaunch'], errors='coerce')

# convert the column to YYYY format
df['TeamLaunch'] = df['TeamLaunch'].dt.strftime('%Y')


# In[20]:


df['TeamLaunch'].value_counts()


# In[21]:


# Let us explore column 'lastplayed_pl'
df.dtypes['lastplayed_pl']


# In[24]:


# Let us extract only the year in lastplayed_pl column
df['lastplayed_pl'] = (pd.to_datetime(df['lastplayed_pl'], format='%b-%y')).dt.year


# In[25]:


df.head()


# ### Section C: Deep dive into Data Analysis

# In[26]:


# Calculate basic data summaries

df.describe()


# The average number of matches played by each team in the tournament is 573.75, while the mean number of goals scored by all teams is 769. However, the median number of goals scored is much lower at 462, indicating that some teams have scored significantly more goals than others. 
# 
# Interestingly, the median number of wins and runners-up positions are both 0, suggesting that most teams have not won or finished as runners-up in the tournament. However, there is one team that has won the tournament a remarkable 13 times and another team that has been the runners-up 7 times. It would be interesting to find out which teams these are.

# In[27]:


# Team that has won Premier League 13 times
df[df['Winners']==13]['Club']


# In[28]:


# Team that has been runner-up 7 times
df[df['Runners-up']==7]['Club']


# We see that Manchester United has won Premier league 13 times and have been runner-up 7 times.

# In[29]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


# Let us visualize each column

# First let us start with Matches Played column
# plot histogram
plt.hist(df['Matches Played'])

# add labels and title
plt.xlabel('No. of Matches Played')
plt.ylabel('Frequency')
plt.title('Histogram of Matches Played')


# We can see from the histogram that a majority of teams have played less than 400 matches. However, there are a few teams that have played an exceptionally high number of matches, exceeding 900. 
# 
# As per the project requirements, it is worth noting that some of the more established clubs have already been owned by the competitors. Therefore, the client is interested in identifying potential clubs that may perform well in the future, even if they have less experience in the Premier League.

# In[31]:


# Identify teams who have played more than 900 matches

df[df['Matches Played']>=900]['Club']


# Upon analysis, we have observed that there are a total of 11 clubs who have significantly more experience in the Premier League as compared to the others. These clubs have played a higher number of matches and have established themselves as experienced players in the league.
# 
# As per the client's requirements, we are interested in identifying potential clubs that may perform well in the future, even if they have less experience in the Premier League. Therefore, we have decided to drop these 11 clubs from our analysis, as their established presence in the league may skew our results and make it difficult to identify less experienced clubs with high potential.
# 
# By removing these clubs, we can focus our analysis on the remaining clubs and potentially identify hidden gems that may have been overlooked due to their lack of experience in the league.

# In[32]:


df[df['Matches Played'] < 900]


# In[33]:


df = df[df['Matches Played'] < 900].reset_index(drop=True)
# view data

df.head()


# ### Now let us look at Win, Loss, Drawn, and clean sheets column
# It is essential to understand that the values in all the columns represent the cumulative scores over all the matches played. 
# 
# To accurately analyze the performance of the teams, we must normalize the data by dividing the no. of wins, loss, drawn, clean sheet, goals by the number of matches played. 
# 
# This normalization will provide us with a fair idea of the winning, losing, draw, and clean sheet percentages of each team along with goals per match.

# In[34]:


# Create new columns for Winning Rate, Loss Rate, Draw Rate, & Clean Sheet Rate
df['Winning Rate'] = (df['Win'] / df['Matches Played'])*100
df['Loss Rate'] = (df['Loss'] / df['Matches Played'])*100
df['Drawn Rate'] = (df['Drawn'] / df['Matches Played'])*100
df['Clean Sheet Rate'] = (df['Clean Sheets'] / df['Matches Played'])*100


# In[35]:


# Create a column for average goals scored per match

df['Avg Goals Per Match']=df['Goals']/df['Matches Played']

df['Avg Goals Per Match']=df['Avg Goals Per Match'].round()


# In[37]:


#view data
df.head(10)


# In[39]:


# Now let us visualize Winning, Loss, Drawn rate, and Clean Sheet

# Set the figure size
plt.figure(figsize=(10, 6))

# Create the boxplot
boxplot = plt.boxplot([df['Winning Rate'], df['Drawn Rate'], df['Loss Rate'], df['Clean Sheet Rate']], 
                      patch_artist=True,
                      labels=['Winning Rate', 'Drawn Rate', 'Loss Rate', 'Clean Sheet Rate'])

# Set the title and axis labels
plt.title('Distribution of Winning Rate, Drawn Rate, Loss Rate and Clean Sheet Rate')
plt.xlabel('Winning, Drawn ,Lost Game & Clean Sheet')
plt.ylabel('Rate')

# Show the plot
plt.show()


# #### Winning Rate
# 
# We observe that there are a few outliers in the Winning Rate boxplot, which are located above the upper whisker. It is safe to conclude that these outlier clubs have shown exceptional winning rates compared to the other clubs. Let us identify them ahead.
# 
# Also let us identify the club that has least "Winning Rate"
# 
# #### Drawn Rate
# 
# We observe an outlier in the drawn rate boxplot, indicating that there is one clubs has a much higher drawn rate compared to others. This may not necessarily be a positive indication, as it suggests that the club may struggle to secure wins in their matches. Going further let us identify which club is this.
# 
# #### Loss Rate
# 
# We can see very clearly that loss rates for these clubs are high compared to winning rate. 
# 
# #### Clean Sheet Rate
# 
# We see that data for Clean Sheet rate is pretty symmetric.

# In[40]:


# Winning Rate further analysis. Identify clubs with high winning rate

# Calculate the interquartile range for the "Winning Rate" column
Q1 = df['Winning Rate'].quantile(0.25)
Q3 = df['Winning Rate'].quantile(0.75)
IQR = Q3 - Q1

# Calculate the upper boundaries for potential outliers <-- Expectional high winning rate compared to other teams
upper_bound = Q3 + 1.5 * IQR

# Identify the clubs with high winning rate 
highwinningrate = df[(df['Winning Rate'] > upper_bound)]
highwinningrate


# Upon analyzing the data, we have found that two teams, Leeds United and Blackburn Rovers, have exceptionally high winning rates of 39% and 38% respectively.

# In[41]:


# Winning Rate further analysis. Identify club with low winning rate

# Calculate the lower boundaries for potential outliers <-- Low winning rate compared to other teams
lower_bound = Q1 - 1.5 * IQR

# Identify the clubs with lowest winning rate 
lowwinningrate = df[(df['Winning Rate'] < lower_bound)]
lowwinningrate


# Club is lowest winning rate of 22% is Hull City

# In[42]:


# Drawn Rate further analysis. Identify club with high drawn rate
# Calculate the interquartile range for the "Drawn Rate" column
Q1 = df['Drawn Rate'].quantile(0.25)
Q3 = df['Drawn Rate'].quantile(0.75)
IQR = Q3 - Q1

# Calculate the upper boundaries for potential outliers <-- Expectional high winning rate compared to other teams
upper_bound = Q3 + 1.5 * IQR

# Identify the clubs with high winning rate 
highwinningrate = df[(df['Drawn Rate'] > upper_bound)]
highwinningrate


# In[43]:


# Let us explore columns 'Winners' and 'Runners-up'

df['Winners'].value_counts()


# In[44]:


df['Runners-up'].value_counts()


# We observe that out of the 29 clubs, only 2 clubs have won the Premier League, and one club has been a runner-up. Let us identify these clubs

# In[45]:


df[(df['Winners']==1) | (df['Runners-up']==1)]


# Blackburn Rovers have won Premier League once and been an Runners-up once and Leicester City has won Premier League once

# In[46]:


# Let us look at "lastplayed_pl" column
df['lastplayed_pl'].value_counts()


# Out of the total 29 teams, eight are currently playing in the Premier League. Since these teams are currently active in the league, it makes sense to prioritize them in our analysis. However, there are also teams that date back as early as 2000. It may be appropriate to assign these teams less weight.

# In[47]:


# Let us check the eight teams that are currently playing in the Premier League

df[df['lastplayed_pl']==2023]['Club']


# Giving more priority to teams that have more recent experience playing in the Premier League is ideal. When making the final decision, we will assign higher weight to teams that have played more recently, and lesser weight to those that have not played recently.

# ## Section D: Final Recommendations Framework

# Let's create a plan to Score each team on the pre defined metric.
# 
# * Give a score of 10 if club have a relatively high experience in the Premier League above average (372)
# * Give a score of 15 if club has winning rate above Q3
# * Give a score of 15 if club has lossing rate below Q1
# * Give a score of 10 if club drawn rate below Q1 and losing rate is below Q1
# * Give a score of 10 if club has clean sheet above Q3 and winning rate is above Q3
# * Give a score of 15 if club has won premier league
# * Give a score of 10 if club has been a runners-up in premier league
# * Give a score of 15 if club has been currently playing in premier league

# In[48]:


# Calculate the upper bound for the "Winning Rate" column
upper_bound_WinningRate = df['Winning Rate'].quantile(0.75)

# Calculate the lower bound for the "Loss Rate" column
lower_bound_LosingRate = df['Loss Rate'].quantile(0.25)

# Calculate the lower bound for the "Drawn Rate" column
lower_bound_DrawnRate = df['Drawn Rate'].quantile(0.25)

# Calculate the upper bound for the "Drawn Rate" column
upper_bound_CleanSheetRate = df['Clean Sheet Rate'].quantile(0.75)


# In[49]:


len(df)
df['scores']=np.zeros(len(df))


# In[50]:


df.head()


# In[51]:


df.loc[df['Matches Played'] >= 372, 'scores'] += 10
df.loc[df['Winning Rate'] >= upper_bound_WinningRate, 'scores'] += 15
df.loc[df['Loss Rate'] <= lower_bound_LosingRate, 'scores'] += 15
df.loc[(df['Drawn Rate'] <= lower_bound_DrawnRate) & (df['Loss Rate'] <= lower_bound_LosingRate), 'scores'] += 10
df.loc[(df['Clean Sheet Rate'] >= upper_bound_CleanSheetRate) & (df['Winning Rate'] >= upper_bound_WinningRate), 'scores'] += 10
df.loc[df['Winners'] == 1, 'scores'] += 15
df.loc[df['Runners-up'] == 1, 'scores'] += 10
df.loc[df['lastplayed_pl'] == 2023, 'scores'] += 15


# In[54]:


# sort the DataFrame by score in descending order
df_sort = df.sort_values(by='scores', ascending=False)

# create a bar chart of team scores
plt.figure(figsize=(25,8))
plt.bar(df_sort['Club'], df_sort['scores'], color='green')

# add labels and title to the chart
plt.ylabel('Scores', fontsize=16)
plt.title('Football Club v/s performance score', fontsize=18)

# add legend to explain the blue bars
plt.legend(['Scores'], fontsize=14)

# rotate the team names on the x-axis for readability
plt.xticks(rotation=90, fontsize=14)
plt.yticks(fontsize=14)

# set the y-axis limit to start from 0 and end at 100
plt.ylim(0, 100)

# display the chart
plt.show()


# **Based on the above chart, Blackburn Rovers has the highest score basis our analysis and next best Leicester City**

# To ensure a thorough evaluation of football club performance we must consider clubs current form.
# 
# Let us check the score of those clubs that have played in the last three years. Specifically, suggest including clubs that have played in 2023, as well as those that last played in 2022 and 2021. 
# 
# This approach allows us to pinpoint those clubs that are currently in good form and have consistently performed well over the past few years.

# In[55]:


# sort the DataFrame by score in descending order
df_sort = df[(df['lastplayed_pl']==2023) | (df['lastplayed_pl']==2022) | (df['lastplayed_pl']==2021)].sort_values(by='scores', ascending=False)

# create a bar chart of team scores
plt.figure(figsize=(25,8))
plt.bar(df_sort['Club'], df_sort['scores'], color='green')

# add labels and title to the chart
plt.ylabel('Scores', fontsize=16)
plt.title('Football Club v/s performance score', fontsize=18)

# add legend to explain the blue bars
plt.legend(['Scores'], fontsize=14)

# rotate the team names on the x-axis for readability
plt.xticks(rotation=90, fontsize=14)
plt.yticks(fontsize=14)

# set the y-axis limit to start from 0 and end at 100
plt.ylim(0, 100)

# display the chart
plt.show()


# We believe that Leicester City's recent form and performance make them a better choice for investment.
# 
# Leicester City, the 2016 Premier League champions, have consistently finished in the top 10 in recent years. They placed 5th in both the 2019-2020 and 2020-2021 seasons and finished 8th in 2021-2022. With sufficient financial backing, Leicester City has the potential to achieve even greater success in the near future. Therefore, it would be reasonable to recommend Leicester City to our clients.

# ### I recommend investing in Leicester City based on my analysis
# # Thank You

# In[ ]:





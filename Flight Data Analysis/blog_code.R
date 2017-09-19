airportsDB <- read.csv('http://s3.amazonaws.com/jsw.dsprojects/AirlinePredictions/Airport_Lookup.csv',
                       header = TRUE, stringsAsFactors = FALSE)
carriersDB <- read.csv('http://s3.amazonaws.com/jsw.dsprojects/AirlinePredictions/Carrier_Lookup.csv',
                       header = TRUE, stringsAsFactors = FALSE)
setwd('D:/UNCC/Fall 2016/ML/Project/Data/2015')
#flightsDB <- read.csv('flightData.csv', header=TRUE,stringsAsFactors=FALSE)
flightsDB <- read.csv('trainv2.csv', header=TRUE,stringsAsFactors=FALSE)
head(flightsDB)
#flightsDB <- subset(flightsDB, select = -c(X, YEAR, X.1))
flightsDB <- subset(flightsDB, select = -c(YEAR))
summary(flightsDB$ARR_DELAY)
dim(flightsDB)

#omitting NA's as they are fewer rows.

flightsDB <- na.omit(flightsDB)

summary(flightsDB)

holidays <- c('2015-01-01', '2015-01-20', '2015-02-17', '2015-05-26',
              '2015-07-04', '2015-09-01', '2015-10-13', '2014-11-11',
              '2014-11-28', '2014-12-25')

holidayDates <- as.Date(holidays)

#function for holiday dates

DaysToHoliday <- function(month, day){ # Input a month and day from the flightsDB
  
  # Get our year.
  year <- 2015
  if (month > 10){
    year <- 2014
  }
  # Paste on a 2013 for November and December dates.
  
  currDate <- as.Date(paste(year,month,day,sep = '-')) # Create a DATE object we can use to calculate the time difference
  
  
  numDays <- as.numeric(min(abs(currDate-holidayDates))) # Now find the minimum difference between the date and our holidays
  return(numDays)                                        # We can vectorize this to automatically find the minimum closest
  # holiday by subtracting all holidays at once
  
}

datesOfYear <- unique(flightsDB[,1:2])
datesOfYear$HDAYS <- mapply(DaysToHoliday, datesOfYear$MONTH, datesOfYear$DAY_OF_MONTH)
head(datesOfYear)

InputDays <- function(month,day){
  finalDays <- datesOfYear$HDAYS[datesOfYear$MONTH == month & datesOfYear$DAY_OF_MONTH == day] # Find which row to get
  return(finalDays)
}

flightsDB$HDAYS <- mapply(InputDays, flightsDB$MONTH, flightsDB$DAY_OF_MONTH)
head(flightsDB)
head(flightsDB)

flightsDB$ARR_HOUR <- trunc(flightsDB$ARR_TIME/100) # Cuts off the minutes, essentially.
flightsDB$DEP_HOUR <- trunc(flightsDB$DEP_TIME/100)

flightsDB$CARRIER_CODE <- as.numeric(as.factor(flightsDB$UNIQUE_CARRIER))

numericDB <- select(flightsDB, -c(DEP_TIME, ARR_TIME))

write.csv(numericDB, 'finalAtlNyc.csv')


# -*- coding: utf-8 -*-
"""
Created on 2022-02-19

@author: Patrick Bosworth and Karen Farley (partners)
PS 627 Group 14
Team: PK Data Analytics Inc. 
Project: Support for a Global Pandemic Vaccination Policy
"""


import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

# needed for calculating natural log of a dataframe column
import numpy as np

# Use this library to match country names which are non-standard
# Library was installed with a pip command in Anaconda Console: pip install hdx-python-country
from hdx.location.country import Country
# Library setup: use non-live data from the repo rather than fetching from API, also override default country name
Country.countriesdata(use_live=False, country_name_overrides={"PSE": "oPt"})



# --- DATAFILE IMPORT ---
# Creates a dataframe out of the six data sources
vaccineData =   pd.read_csv('vaccination-data.csv') # clean enough to use without filtering
covidData   =   pd.read_csv('WHO-COVID-19-global-data.csv')
countryData =   pd.read_csv('WDICountry.csv')
gdpData     =   pd.read_csv('c235b7d1-5b9f-48fe-ac64-babb9ca06872_Data.csv')
hdiData     =   pd.read_csv('2020_statistical_annex_all.csv')
popData     =   pd.read_csv('API_SP.POP.TOTL_DS2_en_csv_v2_3731322.csv',  
                            skiprows = 4)

# --- CLEANING & FILTERING START ---
# 1/6 Filter and clean countryData: 
    # 1) drop unwanted cols by listing cols to keep, drop the rest
columnsToKeep1 = countryData.columns[:5]
columnsToKeep2 = countryData.columns[7:9]
columnsToKeep = [*columnsToKeep1, *columnsToKeep2]
countryData = countryData[columnsToKeep]
    # 2) Clean up Column Names: by passing in a list of new column names.
countryData.columns = ['ISO3', 'ShortName', 'TableName', 
                       'LongName', 'ISO2', 'Region', 'IncomeGroup' ]

# 2/6 Clean vaccineData: 
    # Clean up Column Names: by passing in a list of new column names.
vaccineData.columns = ['Country2', 'ISO3', 'WHORegion2', 'DataSource', 
                       'DateUpdated', 'TotalVax', 'Vax1Plus', 'VaxPer100', 
                       'Vax1PlusPer100', 'VaxFull', 'VaxFullPer100', 
                       'VaxTypesUsed', 'VaxFirstDate', 'VaxTypesNum' ]
# 3/6 Filter and clean gdpData: 
    # 1) drop unwanted cols by listing cols to keep, drop the rest
columnsToKeep1 = gdpData.columns[2:4]
columnsToKeep2 = gdpData.columns[14:]
columnsToKeep = [*columnsToKeep1, *columnsToKeep2]
gdpData = gdpData[columnsToKeep]
    # 2) Clean up Column Names: by passing in a list of new column names.
gdpData.columns = ['Country3', 'ISO3', 'GDP2019', 'GDP2020']
    # 3) Convert data to numbers, put nan where data is missing
gdpData['GDP2019'] = pd.to_numeric(gdpData['GDP2019'], errors = 'coerce')
gdpData['GDP2020'] = pd.to_numeric(gdpData['GDP2020'], errors = 'coerce')
                                    
# 4/6 Filter and clean hdiData: 
    # 1) drop unwanted cols by listing cols to keep, drop the rest
columnsToKeep = hdiData.columns[:3]
hdiData = hdiData[columnsToKeep]
    # 2) Rename columns
hdiData.columns = ['HDIRank', 'HDICountry', 'HDIValue']
    # 3) Clean Rows - select and keep rows where HDIRank is not NULL
    # Force all non-numeric data in the HDIRank to null, rest to numeric
hdiData['HDIRank'] = pd.to_numeric(hdiData['HDIRank'], errors = 'coerce')
    # Our fixed data is in 2 parts, 1st is countries with good data...
newHDIData1 = hdiData[hdiData['HDIRank'].isnull() == False]
    # ... 2nd is countries with bad data...
newHDIData2 = hdiData[hdiData['HDIValue'] == '..']    
    # ... and now concatenate the 2
hdiFrames = [newHDIData1, newHDIData2]
hdiData = pd.concat(hdiFrames)
    # 4) Convert data to numbers, put nan where data is missing
hdiData['HDIValue'] = pd.to_numeric(hdiData['HDIValue'], errors = 'coerce')

    # 5) Critical issue with hdiData is the non-standard country names and lack of ISO country codes
    # Use hdx Country library, which has fuzzy matching for country names to generate ISO codes
#TODO: Delete this attempt to match the non-standard countrynames in hdiData with other data
'''countryMatch = joinedData
columnsToDrop = ['WHORegion2', 'DateUpdated', 'VaxPer100', 'Vax1PlusPer100','VaxFullPer100','VaxFirstDate','Region','IncomeGroup','GDP2019Per100','GDP2020Per100','CasesCumPer100','DeathsCumPer100']
countryMatch = countryMatch.drop(columns = columnsToDrop)
##joining hdiData by Country2 name string match
countryMatch = countryMatch.reset_index()
countryMatch = countryMatch.set_index(['Country2'])
hdiData = hdiData.set_index(['HDICountry'])
countryMatchLeft = countryMatch.join(hdiData, how = 'left')
countryMatchRight = countryMatch.join(hdiData, how = 'right')
##check used to identify data with no match at each join stage above
nullDataLeft=countryMatchLeft[countryMatchLeft['HDIRank'].isnull()== True]
nullDataRight=countryMatchRight[countryMatchRight['Country4'].isnull()== True]'''

    # Usage: perform a fuzzy match and return ("SLE", False). The False indicates a fuzzy rather than exact match.
#testISO3 = Country.get_iso3_country_code_fuzzy("Sierra")
#print(testISO3[0])

    # For a single column, run the country fuzzy matching function to make a column of ISO3 data
    # Code adapted from a StackOverflow post: https://stackoverflow.com/questions/54653528/pandas-call-function-for-each-row-of-a-dataframe 
for index, row in hdiData.iterrows():
   fuzzyCountryCode = Country.get_iso3_country_code_fuzzy(row['HDICountry'])
   hdiData.loc[index, 'HDIISO3'] = fuzzyCountryCode[0]

# 5/6 Filter and clean covidData: data here is by date and country, and we want all the cols
    # 1) Clean up Column Names: by passing in a list of new column names.
covidData.columns = ['DateReported', 'ISO2', 'Country', 'WHORegion', 
                     'CasesNew', 'CasesCum', 'DeathsNew', 'DeathsCum']
    # 2) Pull out the date of first case, and first death, for use later (maybe)
covidFirstCaseDate = covidData[covidData['CasesCum'] > 0]
covidFirstCaseDate = covidFirstCaseDate.sort_values('DateReported').drop_duplicates(subset=['ISO2'])
covidFirstDeathDate = covidData[covidData['DeathsCum'] > 0]
covidFirstDeathDate = covidFirstDeathDate.sort_values('DateReported').drop_duplicates(subset=['ISO2'])

    # 3) Rebuild the filter with just the cumulative columns
covidGroupedByCountry = covidData.groupby(by = ['ISO2'])

    # 4) Aggregate, preserving important columns    
    # Note: 'unique' tends to return a list of the uniques in that column. 
    # To preserve just one (sorted alphabetically), with the original datatype, use 'min' or 'max'.
covidGroupedByCountry = covidGroupedByCountry.agg({'Country': 'min', 
                                       'WHORegion': 'min',                                    
                                       'CasesCum': 'max', 
                                       'DeathsCum': 'max'})    

# 6/6 Filter popData: remove unneeded columns, rename the rest
columnsToKeep1 = popData.columns[:2]
columnsToKeep2 = popData.columns[63:65]
columnsToKeep = [*columnsToKeep1, *columnsToKeep2]
popData = popData[columnsToKeep]
    # 2) Clean up Column Names: by passing in a list of new column names.
popData.columns = ['Country4', 'ISO3', 'Pop2019', 'Pop2020' ]

# --- CLEANING & FILTERING COMPLETE ---



## --- JOINING DATA START ---
## Join data using vaccineData as the primary dataset upon which the others are joined
## Generally, inner join used to exclude regional data and countries without covid/vaccine/HDI data
## Extra column cleanup will be performed after joins are complete

joinedData=vaccineData

## 1/6 joining countryData by ISO3 match
joinedData = joinedData.set_index(['ISO3'])
countryData = countryData.set_index(['ISO3'])
joinedData = joinedData.join(countryData, how = 'inner')

## 2/6 joining gdpData by ISO3 match
gdpData = gdpData.set_index(['ISO3'])
joinedData = joinedData.join(gdpData, how = 'inner')

## 3/6 joining popData by ISO3 match
popData = popData.set_index(['ISO3'])
joinedData = joinedData.join(popData, how = 'inner')

## 4/6 joining hdiData by ISO3 match
hdiData = hdiData.set_index(['HDIISO3'])
joinedData = joinedData.join(hdiData, how = 'inner')

## 5/6 joining covidGroupedByCountry by ISO2 match
joinedData = joinedData.reset_index()
joinedData = joinedData.set_index(['ISO2'])
joinedData = joinedData.join(covidGroupedByCountry, how = 'inner')

## 6/6 Joined DataFrame Cleanup
## A cross-check used to identify any data with no match at each join stage above
nullData=joinedData[joinedData['HDICountry'].isnull()== True]

## Generate some calculated columns to standardize our data on rate-per-100-people
joinedData['GDP2019Per100'] = joinedData['GDP2019'] / joinedData['Pop2019'] *100
joinedData['GDP2020Per100'] = joinedData['GDP2020'] / joinedData['Pop2020'] *100
joinedData['CasesCumPer100'] = joinedData['CasesCum'] / joinedData['Pop2020'] *100
joinedData['DeathsCumPer100'] = joinedData['DeathsCum'] / joinedData['Pop2020'] *100

## Remove unwanted data and duplicate columns
columnsToDrop=['DataSource', 'TotalVax', 'Vax1Plus', 'VaxFull', 'VaxTypesUsed', 
               'VaxTypesNum', 'GDP2019', 'GDP2020', #'Pop2019', 'Pop2020', 
               'CasesCum', 'DeathsCum', 'WHORegion', 'ShortName', 'TableName', 
               'LongName', 'Country3', 'Country4', 'HDIRank', 'HDICountry', 'Country']
joinedData=joinedData.drop(columns=columnsToDrop)

# Because GDP data appears to logistic in character, we create a calculated col
# with the natural log of GDP, then run a standard linear regression later
# numpy is used because it works well on Pandas dataframes
natLogCol = joinedData['GDP2020Per100']
joinedData['NatLogGDP2020Per100'] = np.log(natLogCol)
## --- JOINING DATA COMPLETE ---


# --- EXPORT CLEANED DATA START ---
joinedData.to_csv('PKJoinedData.csv', index = False)
# --- EXPORT CLEANED DATA COMPLETE ---

# --- REGRESSION AND DATA ANALYSIS START ---

#running vaccine and economic indicator regressions
model1=smf.ols(formula='VaxFullPer100 ~ GDP2020Per100', data=joinedData).fit()
print(model1.summary())
#Woohoo!

model2=smf.ols(formula='VaxFullPer100 ~ HDIValue', data=joinedData).fit()
print(model2.summary())

'''model3=smf.ols(formula='VaxPer100 ~ GDP2020Per100', data=joinedData).fit()
print(model3.summary())'''

'''model4=smf.ols(formula='VaxPer100 ~ HDIValue', data=joinedData).fit()
print(model4.summary())'''

model5=smf.ols(formula='Vax1PlusPer100 ~ GDP2020Per100', data=joinedData).fit()
print(model5.summary())

model6=smf.ols(formula='Vax1PlusPer100 ~ HDIValue', data=joinedData).fit()
print(model6.summary())

#running covid numbers and economic indicator regressions
model7=smf.ols(formula='CasesCumPer100 ~ GDP2020Per100', data=joinedData).fit()
print(model7.summary())

model8=smf.ols(formula='CasesCumPer100 ~ HDIValue', data=joinedData).fit()
print(model8.summary())

model9=smf.ols(formula='DeathsCumPer100 ~ GDP2020Per100', data=joinedData).fit()
print(model9.summary())

model10=smf.ols(formula='DeathsCumPer100 ~ HDIValue', data=joinedData).fit()
print(model10.summary())


## LOG-LINEAR REGRESSION
# To do this regression we use GDP data that was converted to the natural log of GDP
# Vaccinations predicted by ln(GDP):
logModel1=smf.ols(formula='VaxFullPer100 ~ NatLogGDP2020Per100', data=joinedData).fit()
print(logModel1.summary())
# --- REGRESSION AND DATA ANALYSIS COMPLETE ---

# --- OUTPUT REGRESSIONS TO TXT FILE ---
with open('PKAnalysisOutput.txt', 'w') as f:
    print(model1.summary(), file=f)
    print(model2.summary(), file=f)
    '''print(model3.summary(), file=f)
    print(model4.summary(), file=f)'''
    print(model5.summary(), file=f)
    print(model6.summary(), file=f)
    print(model7.summary(), file=f)
    print(model8.summary(), file=f)
    print(model9.summary(), file=f)
    print(model10.summary(), file=f)
    print(logModel1.summary(), file=f)

# --- SCATTERPLOT AREA FOR TESTING ---
plt.scatter(x = joinedData['VaxFullPer100'],
            y = joinedData['NatLogGDP2020Per100'],
            marker = 'D',
            #label = 'Data'
            )
plt.xlabel('Vax Rate')
plt.ylabel('ln(GDP)')
plt.title('Linear model for Vaccination given ln(GDP)')
plt.grid()
plt.legend()


vaxCol = joinedData['VaxFullPer100']
# Generate summary statistics, to console
print(vaxCol.describe())

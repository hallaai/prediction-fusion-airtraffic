# Fused prediction model
Time series prediction of real world data which in reality affected by many factors but they are not seen, so some safictication should be used here. Using just simple ARIMA or XGboost is not a good option. 
Model uses daily airtraffic - number of passangers, which it tries to predict with several tecniques blinded together into one model. 
Among many other examples, particulary was implemented: (Prophet + XGBoost) with a Flask web service.

### Base Models Selection
- Prophet/Neural Prophet: Handles holidays, seasonality, and abrupt changes.
- XGBoost/LightGBM: Utilizes engineered features (e.g., lags, rolling statistics).
- SARIMA/ETS: For linear trends, seasonality, and stationarity.
You might also want to add:
- LSTM/GRU Networks: Models complex non-linear patterns and long-term dependencies.

### Models Fusion
Fusion Technique: Stacking. Models are fused into one by just finding average. Ideally it should be tuned for each particular case. 
```python
np.mean([prophet_pred.values, xgb_pred, sarima_pred.values, ets_pred.values])
```
You can choose which combination of models works the best way for you, for example, by choosing the least error or you might be interested in jusrt prediction a particular time window

### Installation
```
pip install flask pandas numpy plotly prophet xgboost holidays statsmodels
```
or
```
python3.12 -m pip install flask pandas numpy plotly prophet xgboost holidays statsmodels
```

## Prediction steps
1. Preprocess data (log-transform, handle missing values).
2. Split into train/validation/test with temporal ordering.
3. Train SARIMA, Prophet, LSTM on train set.
4. Generate validation predictions (OOF) from each model.
5. Train meta-model (e.g., XGBoost) on validation predictions + features.
6. Evaluate stacked model on test data.


## Results
Only some periods are enough to train the model. 
After Training it fills in gaps nicely
![prediction with missing periods](./Screenshot_20250305_103003.png)

## Possible improvements
- Generate out-of-fold (OOF) predictions on validation data to avoid overfitting.
- Use these predictions as input features for a meta-model, such as:
  - Linear Regression (simple, explainable).
  - XGBoost/Neural Network (captures non-linear interactions).

### Key Considerations
- Temporal Cross-Validation: Use forward chaining (e.g., TimeSeriesSplit in sklearn) to preserve time order.
- Feature Engineering: Include lagged values, rolling averages, and calendar features (month, holidays).
- Hybrid Approaches: Combine statistical models (SARIMA) with ML (XGBoost) outputs in a final layer.

### Alternative: Weighted Averaging
- Optimize weights for base models (e.g., inverse of validation RMSE) for a blended prediction.


# Booking Price Prediction Models: Forecasting Ticket Costs for Flights and Events

The prediction of booking prices for time-sensitive services such as airline tickets and concert admissions represents a sophisticated application of machine learning and time series forecasting. These prediction systems aim to advise consumers on the optimal purchase timing while helping vendors maximize revenue through dynamic pricing strategies. As the date of the activity approaches, these models continuously refine their predictions, incorporating new data to progressively reduce forecasting errors. This report examines the methodologies, models, and real-world applications in this domain, with particular attention to Ryanair's formerly public prediction system.

## Advanced Machine Learning Models for Price Prediction

The landscape of booking price prediction is dominated by several sophisticated machine learning approaches, each bringing distinct advantages to the forecasting challenge. These models work with large datasets comprising temporal data, consumer behavior patterns, and market conditions to generate increasingly accurate predictions.

### Time Series and Neural Network Approaches

Long Short-Term Memory (LSTM) networks have emerged as powerful tools for booking price prediction due to their exceptional capability to model temporal dependencies in sequential data. Research indicates that LSTM models excel at capturing the relationship between pricing strategies and seasonal demand patterns, with price fluctuations and discount strategies contributing approximately 30% variance in prediction models[^1]. The architecture of LSTM allows it to remember patterns over extended sequences, making it particularly suitable for identifying how room or ticket pricing interacts with cyclical demand patterns throughout the year.

These neural network approaches can process vast amounts of historical booking data and identify complex patterns that might escape traditional statistical methods. For flight price prediction specifically, systems like AirTrackBot leverage neural networks to analyze past and current fare trends to estimate whether prices will rise or fall in the coming days or weeks[^2]. This approach considers multiple factors simultaneously, including days until departure, seasonal demand fluctuations, and route popularity to forecast airfare changes with minimal deviations.

### Ensemble and Boosting Methods

Random Forest models have demonstrated remarkable efficacy in the prediction of booking trends. These ensemble learning methods identify pricing and discounts as crucial variables, contributing up to 34% to the variance in booking behaviors[^1]. The importance of Random Forest models lies in their ability to handle non-linear relationships between variables while providing interpretable insights through feature importance rankings. These models have revealed significant price sensitivity in budget and mid-range market segments, demonstrating that even small price increases can lead to considerable reductions in bookings.

Complementing Random Forest approaches, XGBoost algorithms have proven particularly adept at capturing the complex interactions between price and other features such as competitor pricing and lead time. Research shows that XGBoost can effectively model how price and discount factors collectively contribute approximately 32% to demand forecasting accuracy[^1]. The algorithm's strength lies in handling non-linear interactions between variables, revealing how customers respond to both a preferred service provider's price and competing alternatives, especially during promotional periods.

### Specialized Forecasting Systems

Facebook's Prophet algorithm offers specialized capabilities for time series forecasting in the booking domain. This model decomposes time series data into trend, seasonality, and holiday effects, making it particularly effective at capturing cyclic patterns in booking behavior[^1]. In contrast to other approaches that may emphasize pricing as the dominant factor, Prophet analyses have revealed that room pricing is not always the highest contributing feature to prediction accuracy, highlighting the importance of a multi-factorial approach to forecasting.

## Key Factors Influencing Prediction Accuracy

The success of booking price prediction models depends heavily on the inclusion of relevant factors that influence pricing dynamics. These factors vary depending on the specific industry but share common principles across domains.

### Temporal and Competitive Dynamics

Lead time—the period between booking and service delivery—represents a critical variable in price prediction models. As the event date approaches, airlines typically adjust their pricing strategies based on remaining inventory and booking velocity. Prediction systems must account for this temporal dimension, recalibrating forecasts as new information becomes available. Similarly, competitor pricing has emerged as a significant factor, contributing between 15-22% of variance in booking predictions according to different models[^1]. The competitive landscape creates a dynamic environment where prices fluctuate in response to rival offerings, particularly in markets with high service provider density.

Local events such as festivals, conferences, and concerts significantly impact demand and pricing patterns, creating temporal anomalies that prediction models must account for[^1]. These events can cause sudden demand spikes that disrupt normal pricing patterns, requiring models to incorporate calendar data and event schedules into their predictions.

### Industry-Specific Considerations

For flight bookings specifically, route popularity, budget airline availability, and seasonal demand patterns represent crucial variables that prediction systems must incorporate[^2]. AirTrackBot and similar systems analyze these factors alongside real-time pricing data to generate buy-or-wait recommendations for consumers considering flight purchases[^2].

In the concert and event ticketing domain, different variables take precedence. A GitHub project focused on predicting StubHub ticket markups collected data from multiple sources including StubHub's API for ticket pricing, web scraped data from SongKick.com for face values and sell-out status, and artist popularity metrics from the EchoNest API[^4]. This comprehensive approach demonstrates how prediction models for event tickets must consider venue-specific attributes, artist popularity, and regional market conditions.

## Yield Management and Dynamic Pricing

Underlying many booking price prediction challenges is the practice of yield management—a sophisticated pricing strategy employed by service providers to maximize revenue through dynamic price adjustments. Airlines like Ryanair employ these techniques to optimize revenue by selling every seat at the highest price the market will bear at any given moment[^5].

Yield management systems constantly adjust prices based on multiple factors including current demand, time remaining before the service date, historical booking patterns, and competitor pricing[^5]. This creates the fluctuating price environment that prediction models attempt to forecast, essentially requiring the models to predict the behavior of other algorithmic systems.

The complexity of yield management systems has created the market opportunity for consumer-oriented prediction tools that attempt to reverse-engineer pricing algorithms. These tools aim to help consumers determine optimal purchase timing by forecasting price movements based on historical patterns and current market conditions[^5].

## Ryanair's Price Prediction System

Ryanair, Europe's largest low-cost carrier, previously offered a publicly accessible price prediction tool that appears to have been associated with or called "Airhint"[^5]. This tool was designed to address the fundamental consumer question when faced with Ryanair's dynamic pricing: "Should I buy now, or wait?" The system likely employed sophisticated algorithms to analyze Ryanair's yield management patterns and provide guidance on expected price movements.

While the current search results don't provide comprehensive details on the technical implementation of Airhint, the system likely incorporated analyses of historical pricing data, seasonality patterns, days-to-departure effects, and route-specific demand variables to generate its predictions[^5]. The tool represented an interesting counterbalance to the airline's own revenue optimization strategy, essentially providing consumers with intelligence to navigate Ryanair's dynamic pricing landscape.

Ryanair's business model centers on offering extremely low base fares while generating additional revenue through ancillary services and dynamic pricing strategies[^3]. Their reservation system, called Airtime, provides a comprehensive booking platform, but the publicly available prediction tool represented a separate service that analyzed the airline's pricing patterns to benefit consumers[^3][^5].

## Implementation Approaches and Training Data

The development of effective booking price prediction models requires substantial datasets and careful feature engineering. A research paper on airline seating purchases leveraged a dataset comprising 69 variables and over 1.1 million historical booking records dating back to 2017[^6]. This scale of data is typical for production-grade prediction systems, allowing models to identify subtle patterns across diverse scenarios.

For concert ticket markup prediction, researchers collected data for 3,126 concerts across 16 metropolitan areas in the United States, incorporating data from multiple sources to build comprehensive prediction models[^4]. This multi-source approach highlights the importance of integrating diverse data streams to capture the full complexity of pricing dynamics.

Effective booking price prediction models typically require:

1. Historical pricing data over extended periods to capture seasonal patterns
2. Competitor pricing information to understand market positioning
3. Demand indicators such as search volume and partial bookings
4. Event calendars and local factors affecting demand
5. Lead time and inventory availability metrics

The integration of these data sources creates the foundation for progressively more accurate predictions as the service date approaches and more information becomes available to the model.

## Conclusion

Booking price prediction represents a sophisticated application of machine learning techniques to temporal data with complex underlying dynamics. The most effective approaches employ multiple algorithmic strategies including LSTM networks, ensemble methods like Random Forest, boosting algorithms such as XGBoost, and specialized time series forecasting tools like Prophet. These models analyze factors including lead time, competitive positioning, seasonal patterns, and local events to generate increasingly accurate price predictions as the service date approaches.

Ryanair's publicly available prediction tool exemplified the commercial application of these techniques, providing consumers with guidance on optimal purchase timing. As machine learning and data collection capabilities continue to advance, we can expect booking price prediction systems to achieve greater accuracy and broader application across industries where dynamic pricing is prevalent. The arms race between revenue optimization systems and consumer-oriented prediction tools will likely intensify, driving innovation in this specialized domain of time series forecasting.[^1][^2][^3][^4][^5][^6]

<ins>Sources:</ins>

[^1]: https://thesai.org/Downloads/Volume15No10/Paper_43-Prediction_of_Booking_Trends_and_Customer_Demand.pdf
[^2]: https://airtrackbot.com/flight-price-predictor
[^3]: https://thinkinsights.net/digital/ryanair-business-model
[^4]: https://github.com/epsilon670/predicting_ticket_markups
[^5]: https://shop.riffraft.com/news/airhint-ryanair-16021/
[^6]: https://www.diva-portal.org/smash/get/diva2:1465383/FULLTEXT01.pdf
[^7]: https://dash.harvard.edu/entities/publication/ccecd7d6-2381-4232-a840-cc6c16f60d37
[^8]: https://www.altexsoft.com/blog/hotel-price-prediction/
[^9]: https://www.going.com/guides/travel-predictions-2025
[^10]: https://vizologi.com/business-strategy-canvas/ryanair-business-model-canvas/
[^11]: https://sites.northwestern.edu/pricepoint/sample-page/
[^12]: https://res.org.uk/mediabriefing/everything-you-wanted-to-know-about-ryanair-pricing-but-never-dared-to-test/
[^13]: https://dspace.mit.edu/handle/1721.1/68100
[^14]: https://www.independent.co.uk/travel/news-and-advice/ryanair-price-tickets-b2640821.html
[^15]: https://www.irjmets.com/uploadedfiles/paper/issue_3_march_2024/51520/final/fin_irjmets1711827181.pdf
[^16]: https://www.airhint.com/about
[^17]: https://www.airhint.com/ryanair-predictor
[^18]: https://www.reddit.com/r/Flights/comments/1dhy6aq/ryanair_flight_price_drop/
[^19]: https://airtrackbot.com/ryanair-flight-price-tracker
[^20]: https://www.moneysavingexpert.com/travel/ryanair-tips/
[^21]: https://github.com/RawatMeghna/Flight-Booking-Price-Prediction
[^22]: https://www.flightapi.io/blog/flight-price-prediction-tools/
[^23]: https://www.ryanair.com/fi/en/trip/manage
[^24]: https://readdork.com/features/ticket-prices-beyonce-kendrick-lamar-2025/
[^25]: https://wowfare.com/blog/evaluating-the-effectiveness-of-predictive-pricing-in-flight-booking-tools/
[^26]: https://www.airhint.com
[^27]: https://www.ryanair.com/fi/en/
[^28]: https://www.bbc.com/news/articles/c2kdxlv8x05o
[^29]: https://ieeexplore.ieee.org/document/10099266
[^30]: https://www.alternativeairlines.com/flight-price-predictor
[^31]: https://help.ryanair.com/hc/en-fi/articles/12893493549329-How-do-I-verify-my-booking
[^32]: https://www.reddit.com/r/blacksabbath/comments/1iiaa0g/ticket_price_estimate_for_the_upcoming_show/
[^33]: https://www.tripadvisor.com/ShowTopic-g186605-i90-k11232398-o10-Does_Ryanair_Flight_Price_Fluctuate-Dublin_County_Dublin.html
[^34]: https://www.reuters.com/business/aerospace-defense/ryanair-trims-traffic-forecast-again-boeing-delays-2025-01-27/
[^35]: https://www.mdpi.com/2076-3417/13/10/6032
[^36]: https://www.yeschat.ai/tag/Concert-Tickets
[^37]: https://illumin.usc.edu/the-algorithm-behind-plane-ticket-prices-and-how-to-get-the-best-deal/
[^38]: https://github.com/Kekyei/airline-booking-prediction
[^39]: https://softjourn.com/insights/predictive-analytics-in-ticketing
[^40]: https://github.com/DrPoojaAbhijith/Project-Predicting-flight-booking-prices.
[^41]: https://sciforce.solutions/case-studies/ticket-sales-prediction-59
[^42]: https://www.kaggle.com/code/uditkishoregagnani/airline-booking-prediction
[^43]: https://www.kaggle.com/code/xhlulu/ticket-price-prediction-with-scikit-learn
[^44]: https://www.altexsoft.com/blog/flight-price-predictor/
[^45]: https://wowfare.com/blog/essential-tools-and-applications-to-forecast-flight-prices-prior-to-making-your-booking/

# Key Factors Influencing Flight Price Predictions: A Comprehensive Analysis

Flight price prediction has evolved into a sophisticated domain combining data science, machine learning, and market analysis. As airlines employ increasingly complex dynamic pricing strategies, understanding the factors that drive these price fluctuations has become essential for both travelers seeking optimal booking times and companies developing prediction tools. Modern prediction systems can process billions of data points in real-time to forecast price movements with remarkable accuracy, continually refining their models as new information becomes available. This report examines the critical variables that influence flight price predictions, the methodologies used to process this information, and how these systems progressively improve their forecasting precision as departure dates approach.

## Temporal Dynamics in Price Forecasting

The relationship between time and flight pricing represents one of the most significant dimensions in prediction models. Understanding these temporal patterns allows forecasting systems to identify optimal booking windows and predict price movements with increasing accuracy.

### Lead Time and Booking Patterns

The period between booking and departure serves as a fundamental variable in flight price prediction models. Research indicates that ticket prices often increase as the departure date approaches, with passengers booking closer to travel dates typically paying premium prices while those booking well in advance may secure lower fares[^103]. This pattern creates a temporal curve that prediction algorithms must model accurately. The timing of purchases significantly impacts final pricing, with early bookings generally offering better value compared to last-minute reservations that tend to carry substantial premiums[^105].

Advanced prediction systems continuously analyze how prices fluctuate over time, identifying cyclical patterns and trends through specialized time series analysis algorithms. These systems track historical pricing trajectories across millions of routes to establish predictive patterns that inform future forecasts[^104]. By monitoring how prices have changed for similar flights in the past, these algorithms can project expected price movements with increasingly refined accuracy as more data becomes available.

### Seasonal Trends and Cyclical Patterns

Seasonality exerts a profound influence on flight pricing dynamics, creating predictable waves of demand that algorithms must incorporate. During peak travel seasons like summer or significant holidays, flight demand typically surges, prompting airlines to increase ticket prices to capitalize on heightened demand[^101]. Conversely, during quieter travel periods, carriers often implement discounted fares to stimulate bookings.

Prediction systems specifically analyze day of the week patterns, time of day variations, and seasonal fluctuations to make pricing decisions[^103]. These temporal patterns create recognizable signatures that machine learning models can identify and project forward. Modern AI systems deployed by flight search platforms process historical pricing data, current market trends, seasonal patterns, and external factors like major events to develop multi-dimensional analysis that identifies patterns human analysts would likely miss[^104].

## Market Forces and Competitive Dynamics

The competitive landscape and market conditions create a complex environment of continuously shifting prices that prediction systems must navigate to provide accurate forecasts.

### Demand Fluctuations and Market Responsiveness

Market demand represents a crucial determinant in flight pricing models, often driving significant price variations across similar routes. Dynamic pricing algorithms continuously monitor demand for specific flights, raising prices when demand is high and lowering them when interest wanes to attract more passengers[^103]. This demand-sensitive approach creates a responsive pricing environment that prediction models must track in real-time.

Modern prediction systems analyze load factors and seat availability across carriers to gauge how full flights are becoming, as airlines typically adjust fares based on remaining inventory[^104]. As seats on a flight fill up, prices tend to increase according to predetermined thresholds programmed into yield management systems. By monitoring booking velocities and comparing them to historical patterns, prediction algorithms can anticipate when these pricing thresholds might trigger increases.

### Competitive Positioning and Rival Pricing

The level of competition between airlines significantly influences pricing strategies and creates predictive challenges for forecasting systems. In intensely competitive markets, airlines may engage in price wars to attract passengers, leading to lower fares as companies vie to outshine each other[^101]. Conversely, routes dominated by a handful of major carriers might maintain higher pricing due to limited competition.

Prediction algorithms specifically monitor competitor pricing strategies on identical routes, as airlines frequently adjust their fares in response to competitors' moves[^103]. This competitive dynamic creates a responsive market where prices can change multiple times daily based on rivals' actions. AI-driven prediction tools provide real-time insights into competitors' pricing strategies, enabling both airlines and prediction systems to adjust their approaches accordingly[^107].

## External Factors and Economic Conditions

Beyond the immediate market dynamics, various external factors significantly impact flight pricing and must be incorporated into prediction models for maximum accuracy.

### Fuel Costs and Operational Expenses

Fuel prices exert considerable influence on airline operational costs, with significant implications for ticket pricing. A surge in fuel costs typically prompts airlines to increase ticket prices to offset additional expenses, while declining fuel prices may result in savings passed to customers through reduced fares[^101]. Prediction systems monitor these cost fluctuations to anticipate potential price adjustments based on changing market conditions.

Advanced prediction models incorporate economic indicators affecting travel demand and airline operating costs[^4]. These include broader economic health measures, currency exchange rates for international routes, and industry-specific metrics that might signal changing cost structures. By analyzing how these factors have historically influenced pricing, algorithms can anticipate potential price movements before they materialize.

### Local Events and Destination-Specific Factors

Flight price prediction systems must account for destination-specific events and factors that create temporal anomalies in demand patterns. Factors unique to particular destinations, such as festivals, major sporting events, or conferences, can trigger localized demand spikes that significantly influence pricing[^101]. These events create temporary market conditions that diverge from typical patterns.

Weather patterns that might disrupt travel represent another crucial variable incorporated into sophisticated prediction models[^104]. Anticipated weather events can affect both supply (through potential flight cancellations) and demand (through changing travel preferences), creating complex interactions that prediction systems must model. By incorporating these external factors, prediction algorithms can account for pricing anomalies that might otherwise appear random.

## Airline Strategies and Pricing Methodologies

Understanding the underlying business strategies and pricing methodologies employed by airlines provides essential context for prediction systems attempting to forecast future price movements.

### Dynamic Pricing and Yield Management

Dynamic pricing, often referred to as yield management, represents a sophisticated strategy airlines employ to optimize revenue by adjusting ticket prices based on numerous variables[^103]. This approach helps carriers maximize revenue by selling every seat at the highest price the market will bear at any given moment. Prediction systems essentially attempt to reverse-engineer these yield management algorithms to forecast their behavior.

Airlines implement yield management through complex algorithms that continuously adjust prices based on remaining inventory, booking velocity, historical patterns, and competitive positioning. These systems segment customers based on willingness to pay, offering different prices to different market segments simultaneously. Prediction models analyze these patterns to identify how airlines typically respond to changing market conditions, enabling more accurate forecasts of future price movements.

### Passenger Segmentation and Cabin Classification

Airlines employ sophisticated passenger segmentation strategies that significantly impact pricing structures and create prediction challenges. Carriers may offer different fares for various passenger segments, including business travelers, leisure travelers, or those with flexible schedules[^103]. These segmentation approaches create multiple pricing tiers that prediction systems must model separately.

Cabin class represents a particularly significant variable in prediction models, with higher cabin classes strongly correlating with increased prices[^102]. A comprehensive study of international flight prices found cabin class to be among the most influential factors determining final pricing. Prediction systems must account for these class-based pricing differentials when generating forecasts, as each class operates under somewhat different pricing dynamics.

## Machine Learning Approaches to Price Prediction

The technical implementation of flight price prediction has evolved significantly through advanced machine learning techniques that can process enormous datasets and identify complex patterns.

### Regression and Ensemble Methods

Regression analysis helps establish relationships between various factors and flight prices, identifying which variables have the most significant impact on pricing outcomes[^104]. These models quantify the relative importance of different factors, enabling more precise predictions. Linear regression and ridge regression models have demonstrated strong performance in flight price prediction tasks, explaining significant portions of price variance[^106].

Random forest models have shown particularly impressive results in flight price prediction contexts. Research indicates that random forest algorithms can successfully identify the most significant factors influencing flight prices, with certain implementations explaining up to 98.9% of price variance[^102]. The decision tree model has also demonstrated superior performance in predicting flight prices when compared to other approaches[^106]. These ensemble methods excel at capturing non-linear relationships between variables while providing interpretable insights through feature importance rankings.

### Neural Networks and Advanced AI

Deep learning neural networks represent an increasingly important approach to flight price prediction, mimicking human brain structure to process complex patterns and generate increasingly accurate predictions as they analyze more data[^104]. These sophisticated models can identify subtle relationships between variables that might escape detection through traditional statistical methods.

Modern flight search platforms employ advanced AI systems that process billions of data points in real-time to predict price fluctuations with unprecedented accuracy[^104]. These systems continuously refine their predictions based on new data, creating models that become progressively more precise over time. By incorporating multiple predictive techniques simultaneously, these platforms can overcome the limitations of any single approach, resulting in more robust and accurate price forecasts.

## Conclusion

Flight price prediction represents a sophisticated challenge involving temporal dynamics, market forces, external factors, airline strategies, and advanced machine learning techniques. The most effective prediction systems incorporate diverse data sources including historical pricing trends, competitive positioning, demand indicators, and broader economic factors to generate increasingly precise forecasts. As artificial intelligence and machine learning capabilities continue advancing, prediction systems have demonstrated remarkable accuracy, with top-performing models explaining nearly 99% of price variance under optimal conditions.

The dynamic nature of airline pricing creates both challenges and opportunities for prediction systems. While the complexity of pricing algorithms and the multitude of influencing factors make perfect prediction impossible, modern approaches have achieved impressive results through comprehensive data integration and sophisticated modeling techniques. These systems provide valuable guidance to consumers navigating the complex landscape of flight pricing while offering airlines deeper insights into market dynamics. As technology continues evolving, we can expect flight price prediction systems to achieve even greater accuracy through more sophisticated analysis of the factors detailed in this report.

<ins>Sources:</ins>

[^101]: https://www.aerotime.aero/articles/factors-influencing-flight-prices
[^102]: https://www.ewadirect.com/proceedings/tns/article/view/11228/pdf
[^103]: https://www.linkedin.com/pulse/dynamic-pricing-airlines-science-behind-airfare-fluctuations-pydqf
[^104]: https://valorflights.com/how-ai-for-flight-search-predicts-flight-prices-with-remarkable-accuracy
[^105]: https://wowfare.com/blog/exploring-the-essential-factors-that-determine-the-prices-of-airline-tickets/
[^106]: https://www.jsr.org/hs/index.php/path/article/download/5303/2492/38165
[^107]: https://www.promptcloud.com/blog/dynamic-pricing-in-airlines-with-ai/
[^108]: https://www.mightytravels.com/2025/02/7-most-accurate-flight-price-prediction-apps-put-to-the-test-2025-analysis/
[^109]: https://wowfare.com/blog/key-elements-shaping-airline-ticket-prices-in-2024-that-every-traveler-should-be-aware-of/
[^110]: https://ijatem.com/wp-content/uploads/2024/05/IJATEM24FEB001.pdf
[^111]: https://kodytechnolab.com/blog/dynamic-pricing-in-airlines/
[^112]: https://wowfare.com/blog/a-comprehensive-overview-of-how-flight-ticket-pricing-algorithms-work-and-what-affects-them/
[^113]: https://airtrackbot.com/flight-price-predictor
[^114]: https://www.altexsoft.com/blog/flight-price-predictor/
[^115]: https://www.altexsoft.com/blog/dynamic-pricing-airlines/
[^116]: https://wowfare.com/blog/evaluating-the-effectiveness-of-predictive-pricing-in-flight-booking-tools/
[^117]: https://www.fetcherr.io/blog/airline-pricing-tactics
[^118]: https://wowfare.com/blog/factors-that-affect-flight-ticket-prices/
[^119]: https://www.fetcherr.io/blog/dynamic-pricing-in-aviation
[^120]: https://kitrum.com/blog/how-ai-powered-dynamic-pricing-keeps-travel-companies-ahead/

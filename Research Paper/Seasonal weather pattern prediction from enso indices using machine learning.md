Discover Environment
Research
Seasonal weather pattern prediction from enso indices using machine
learning
Mohammad Mohsin1 · Tanima Ghosh1 · Fahima Akter1 · Sanupa Sarkar1 · Md. Reaz Akter Mullick1
Received: 22 September 2025 / Accepted: 13 January 2026
© The Author(s) 2026 OPEN
Abstract
Seasonal climate prediction in Bangladesh remains challenging due to the nonlinear nature of weather and climate inter-
actions. This study investigates the correlation between nine El Niño–Southern Oscillation (ENSO) indices and seasonal
temperature and rainfall patterns across Bangladesh, using monthly data from 29 meteorological stations (1977–2022).
Six supervised machine-learning models, such as Random Forest (RF), XGBoost (XGB), Decision Tree (DT), Linear Regres-
sion (LR), K-Nearest Neighbors (KNN), and K-Fold Cross-Validation (KFCV) were evaluated using R2, MAE, and RMSE. XGB
achieved the highest accuracy for temperature prediction ( R2 = 0.8824 for Tmax, 0.9706 for Tmin, and 0.9559 for Tavg), with
RF and KFCV performing comparably. Rainfall prediction accuracy was lower, with RF achieving the highest R2 (0.6273).
Overall, the results confirm that multiple ENSO indices significantly influence Bangladesh’s seasonal climate and that
advanced ML models, particularly XGB and RF, offer strong potential for improved prediction.
Keywords Seasonal weather patterns prediction · ENSO · Supervised machine learning · Temperature · Rainfall
1 Introduction
Weather describes the short-term state of the atmosphere at a given place and time, whereas climate reflects the long-
term (e.g., 30-year) average of those conditions [2]. Small variations in weather patterns can substantially influence
temperature and precipitation enough to affect agriculture, hydropower, urban water management, and livelihoods
[11]. Although short-range numerical models now provide reliable 7–10-day forecasts, their skill sharply declines beyond
2 weeks as initial-condition errors grow rapidly [28]. In contrast, seasonal forecasts based on slowly evolving large-scale
climate drivers can offer communities and sectors in Bangladesh a critical window for preparedness [16]. Bangladesh is
highly sensitive to climate anomalies, with nearly 80% of its annual rainfall occurring during the summer monsoon [1],
often leading to severe flooding (e.g., 1987, 1988, 1998, 2007, 2024). Droughts and extreme cold events similarly disrupt
socioeconomic conditions. Improving seasonal prediction capability is therefore essential for minimizing climate-related
risks and enhancing regional resilience.
A major driver of climate variability is the El Niño-Southern Oscillation (ENSO), a coupled ocean–atmosphere phe-
nomenon originating in the tropical Pacific with a recurrence period of 2–7 years [35]. ENSO’s warm (El Niño) and cool
Supplementary Information The online version contains supplementary material available at https:// doi. org/ 10. 1007/ s44274- 026-
00533-6.
* Mohammad Mohsin, mohammad.mohsin.ce@gmail.com; Tanima Ghosh, tanima.ghosh.ce@gmail.com; Fahima Akter, fahima.akter.
ce@gmail.com; Sanupa Sarkar, sanupasarkar24112002@gmail.com; Md. Reaz Akter Mullick, reazmullick@cuet.ac.bd | 1Department of Civil
Engineering, CUET, Chattogram, Bangladesh.
Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Vol.:(0123456789)

Research
Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
(La Niña) phases modulate global rainfall, temperature, and circulation anomalies by altering sea-surface temperatures
and atmospheric pressure gradients [12, 18]. These atmosphere–ocean anomalies are typically tracked using indices
such as the Southern Oscillation Index (SOI), Oceanic Niño Index (ONI), and Niño 3.4 SST. Several other indices, including
the Indian Ocean Dipole (IOD), Pacific Decadal Oscillation (PDO), Trans-Niño Index (TNI), Pacific North American (PNA)
pattern, BEST, and upper-level zonal winds capture different aspects of regional and extra-tropical variability that often
interact with ENSO. A concise summary of key indices, their regions, and phase characteristics is provided in the Sup-
plementary Information.
Recent studies indicate that the performance of Niño 3.4 or ONI-based models has declined, partly due to changes
in ENSO dynamics and the presence of the Spring Predictability Barrier [4]. South Asian studies show that rainfall is
controlled by the combined influence of several climate indices, supporting the relevance of multi-index approaches
[30, 31]. Although both dynamical and statistical models have been applied to ENSO-related forecasting [8, 23, 41], their
skill varies substantially across lead times and regions. Machine-learning approaches have recently shown promise in
capturing nonlinear interactions between teleconnection indices and regional climate variables [36], and Random For-
est models, in particular, have demonstrated strong performance for rainfall prediction [33]. However, existing studies
remain largely limited to single-index predictors or short-term forecasts.
Despite extensive work on ENSO-related variability, significant research gaps remain in current studies focusing on
Bangladesh. Many studies rely on a single ENSO indicator, commonly Niño 3.4 SST or ONI, which captures only part of the
coupled ocean–atmosphere system. This narrow approach often leads to inconsistent correlations, season-dependent
sensitivities, and limited predictive skill. Furthermore, the combined influence of multiple ENSO-related indices, includ-
ing IOD, PDO, TNI, BEST, PNA, and zonal wind anomalies, has not been systematically examined for Bangladesh, even
though these drivers interact and may jointly modulate rainfall and temperature. Their interdependence raises additional
questions about multicollinearity and the relative importance of each predictor.
To address these gaps, this study develops a data-driven framework that uses multiple ENSO-related indices to exam-
ine their combined influence on seasonal climate variability in Bangladesh. To comprehensively capture the role of
large-scale ocean–atmosphere interactions, nine widely recognized ENSO-related and teleconnection indices are incor-
porated: ONI, NINO 3.4 SST, SOI, IOD, PDO, BEST, PNA, TNI, and 200 mb Zonal Winds. These indices were selected based
on extensive climatological evidence demonstrating their influence on temperature anomalies, monsoon dynamics,
and rainfall variability across South Asia and the broader Indo-Pacific region [3, 21, 32]. Each index represents a distinct
physical mechanism—for instance, NINO 3.4 SST and ONI capture central Pacific sea-surface temperature anomalies, SOI
reflects atmospheric pressure variability, IOD characterizes the Indian Ocean Dipole, while PDO, PNA, and TNI describe
broader teleconnection modes that modulate ENSO impacts on regional climates [9, 13]. Because these indices often
exhibit interdependence, potential multicollinearity was evaluated using pairwise correlations and variance inflation
factors (VIF). To ensure robust predictive performance despite correlated predictors, ensemble machine learning mod-
els—specifically Random Forest and XGBoost—were chosen due to their demonstrated ability to handle multicollinearity
and capture nonlinear interactions within high-dimensional climate data [7, 19]. This methodological design enables a
more accurate estimation of how combined teleconnection signals influence rainfall and temperature variability across
Bangladesh. Specifically, the study (i) quantifies the relative contributions of the nine teleconnection indices, (ii) evaluates
the predictive skill of several ML algorithms (Random Forest, XGBoost, Decision Tree, Linear Regression, K-Nearest Neigh-
bors) using MAE, RMSE, and R 2, and (iii) examines how multicollinearity and feature importance affect model stability
and interpretability. Through this integrated approach, the study provides an improved understanding of ENSO-related
teleconnections and supports more reliable seasonal forecasting for Bangladesh.
2 Methodology
2.1 Dataset and study area
Four datasets of monthly maximum temperature, monthly minimum temperature, monthly average temperature and
monthly average rainfall intensity have been processed using the data of 29 stations covering entire Bangladesh shown
in Fig. 1. The stations are namely, Barisal, Bhola, Bogra, Chandpur, Chittagong, Comilla, Cox’s Bazar, Dhaka, Dinajpur, Farid-
pur, Feni, Hatiya, Ishurdi, Jessore, Khepupara, Khulna, M.Court, Madaripur, Mymensingh, Patuakhali, Rajshahi, Rangamati,
Rangpur, Sandwip, Satkhira, Sitakunda, Srimangal, Sylhet, Teknaf.
Vol:.(1234567890)

Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Research
All the datasets also contain nine ENSO indices. They are ONI, NINO 3.4 SST, SOI, IOD, PDO, BEST, PNA, TNI, and 200mb
Zonal Winds. Besides, each dataset also contains other parameters such as Year, Month, Season, Dominant Cycle, ENSO
Event and its Intensity.
2.2 Data source
The ENSO indices were obtained from the NOAA Climate Prediction Center for the period 1977–2022. Temperature and
rainfall data for the same period were collected from the Bangladesh Agricultural Research Council (BARC). Temperature
is provided in degrees Celsius, and rainfall in millimeters.
2.3 Workflow
The research workflow, shown in Fig. 2, began with framing the problem and understanding the types of data required.
Then, the required data were collected. Subsequently, data pre-processing was conducted to handle duplicate & missing
values and remove outliers. After pre-processing, the exploratory data analysis (EDA) was conducted. In the EDA, Univari-
ate, Bi-variate, and multi-variate analysis was carried out. EDA helps to understand the data, find the correlation among
them and facilitate the model selection. Before feeding the data to train the model, data post-processing is required. Data
post-processing is also known as ‘Feature Engineering’. Feature engineering was required for encoding categorical data
and scaling the numerical data. Then, all the datasets were divided into two parts to train and validate the model. 70%
of the total data were used to train the model and later, 30% of the total data were used to validate the model. Then, the
models were assessed by using three performance metrics, namely Mean absolute error (MAE), root mean squared error
(RMSE), and coefficient of determination (R2) metrics. Finally, a comparison was demonstrated between actual scenario
to predicted scenario, which highlighted the worthiness of the ML model.
2.4 Model selection
The Decision Tree (DT) regressor predicts the target variable by recursively partitioning the feature space into homo-
geneous subsets based on an optimal splitting criterion that minimizes within-node variance. For regression problems,
the split at each node is selected by minimizing the sum of squared deviations of the target variable from its mean. The
predictive structure of a decision tree can be expressed as a piecewise constant function over the input space. Model
complexity is governed by parameters such as the maximum tree depth and the minimum number of samples required
for node splitting and leaf formation, which directly control overfitting and stability. These parameters are particularly
important when modeling climate variables, as meteorological datasets often contain noise and nonstationary [25].
ℕ
2
VarSplit =∑( y i −y ) (1)
i=1
where y is the observed value and y is the mean within a node.
i
Random Forest (RF) extends the decision tree framework by constructing an ensemble of trees trained on boot-
strapped subsets of the data, with random feature selection at each split. The final prediction is obtained by averaging
the outputs of all trees, expressed as:
T
1
ŷ = ∑h(x) (2)
T
t=1
where h(x) represents the prediction of the t-th tree. Key model behavior is influenced by the number of trees, tree
t
depth, and the number of predictors sampled at each split. These parameters collectively reduce variance and enhance
generalization, making Random Forest particularly effective for capturing nonlinear and multivariate ENSO–climate
relationships in seasonal forecasting applications [6].
Linear Regression (LR) models the dependent variable as a linear combination of predictor variables and serves as a
baseline approach for assessing linear dependence between ENSO indices and climate variables. The model is defined as
Vol.:(0123456789)

Research
Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
n
y =𝛽 0 +∑𝛽 i 𝜒 i +𝜀 (3)
i=1
where β are regression coefficients and ε denotes random error. Model performance depends on whether an intercept
i
is included and whether regularization is applied to mitigate multicollinearity among predictors. Although simple, linear
regression provides interpretability and acts as a reference against which more complex nonlinear models can be evalu-
ated in climate modeling studies [37].
XGBoost is a boosting-based ensemble method that sequentially builds decision trees to minimize a regularized
objective function combining prediction loss and model complexity. The optimization function is given by
N k
L=∑l ( y i ,ŷ i) +∑Ω ( f k) (4)
i=1 k=1
where Ω(f ) penalizes overly complex trees. Model performance is strongly influenced by hyperparameters such as
k
learning rate, number of boosting iterations, maximum tree depth, and subsampling ratios. These parameters enable
XGBoost to capture complex nonlinear interactions while controlling overfitting, which is essential for modeling climate
variability driven by multiple interacting teleconnection indices [8, 22].
The K-Nearest Neighbors (KNN) algorithm is a non-parametric machine-learning technique used for both classifica-
tion and regression tasks. Known for its simplicity and ease of implementation, KNN does not rely on assumptions about
the data’s underlying distribution. By identifying past seasons with comparable ENSO conditions to the target forecast
season, this non-parametric analogue-based approach can make forecasts based on observable precedents rather than
presumptive functional forms. For inter annular or seasonal weather prediction, KNN combines data mining to historical
data and provides fairly accurate result.
1
ŷ(x)= k ∑ y i (5)
i⋅𝜖N (x)
k
Model behavior depends primarily on the choice of k and the distance metric used to identify nearest neighbors. In
climate applications, KNN functions as an analogue-based forecasting approach, identifying seasons with similar ENSO
conditions and inferring outcomes based on historical precedents rather than parametric assumptions [26].
K-Fold Cross-Validation evaluates model robustness by dividing the dataset into k subsets The model is trained on
k-1 folds and tested on the remaining fold. This process is repeated kkk times, each time with a different fold as the test
set. The CV score is the average performance metric across all folds, which gives a more reliable estimate of model
performance than using a single train-test split.
Mathematically:
k
1
CVScore=
k
∑Metric
i
(6)
i=1
where k = number of folds, and Metric = the evaluation metric (e.g., RMSE, MAE, R2, accuracy) calculated on the iii-th fold.
i
This method is particularly important in weather forecasting applications, where issues such as class imbalance and high
variability can significantly influence predictive accuracy [14, 17].
Table 1 represents the value of different hyperparameters used in different models.
2.5 Model training and validation
This study utilized a data-driven methodology to examine the influence of 13 critical climate indices, including the Year,
Month, ONI, NINO 3.4 SST, SOI, IOD, PDO, PNA, TNI, BEST, 200 mb Zonal Winds, ENSO Events, and their intensity. A data
partitioning strategy was adopted, dividing the dataset into training (70%) and validation (30%) subsets, in line with
standard practices in machine learning research [10, 38]. This splitting approach aimed to mitigate the risk of overfitting
by ensuring the model’s generalizability through performance assessment on unseen data [34].
The trained model was then subjected to rigorous evaluation on the independent validation set to measure its predic-
tive accuracy and uncover potential biases. Such practices align with recommendations by leading machine learning
Vol:.(1234567890)

Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Research
Table 1 represents the value
Model name Hyperparameters with value
of different hyperparameters
used in different models Decision Tree max_depth = 4, random_state = 42
Random Forest random_state = 42
Linear Regression No hyperparameter was used
XGBoost max_depth = 3, random_state = 42,
learning_rate = 0.1, n_estima-
tors = 100
KNN n_neighbors = 5
K Fold Cross Validation random_state = 42, cv = 5
Fig. 1 Geographical Distribu-
tion of Meteorological 29
Stations in Bangladesh
Vol.:(0123456789)

Research
Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Fig. 2 Workflow Diagram
practitioners to ensure robust model validation and avoid spurious conclusions in predictive studies [40]. o evaluate
the predictive performance of the machine learning models for seasonal temperature and rainfall forecasting, three
widely used statistical metrics—Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of
Determination ( R2), were employed. MAE measures the average magnitude of prediction errors and provides a clear
interpretation of overall model accuracy, which is particularly useful for assessing typical deviations in climate variables
such as temperature and rainfall. RMSE emphasizes larger errors by squaring deviations, making it especially relevant for
climate studies where extreme events (e.g., heavy rainfall or temperature extremes) have significant practical implica-
tions for flood risk management and climate resilience planning. R2 quantifies the proportion of variance in the observed
data explained by the model and is commonly used to assess how well multiple climate predictors collectively capture
the variability of weather parameters. The combined use of MAE, RMSE, and R2 ensures a comprehensive evaluation by
Vol:.(1234567890)

Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Research
balancing overall accuracy, sensitivity to extremes, and explanatory power, which is essential in climate and hydrological
modeling applications [27, 29].
By integrating MAE, RMSE, and R2, the study ensures a comprehensive evaluation of predictive performance for both
temperature and rainfall. MAE evaluates overall accuracy, RMSE emphasizes sensitivity to extremes, and R2 assesses how
well multiple input indices explain the variability of each target variable. This approach is critical in climate modeling
and hydrological applications, where accurate predictions of temperature and rainfall are essential for water resource
management, agriculture, and disaster preparedness [20].
3 Results and analysis
3.1 Data pre‑processing
During the data preprocessing phase, we identified a few instances of null values within the dataset. However, no dupli-
cate entries were detected. Given the limited number of null values, we opted to remove the corresponding rows entirely
to ensure data integrity. This was achieved using the dropna function provided by the Pandas library, which allowed
us to eliminate rows containing null values efficiently. This approach preserved the overall quality and consistency of
the dataset while minimizing the impact of missing data on subsequent analyses. After completing the data cleaning
process, the temperature datasets were reduced to dimensions of 539 rows and 47 columns, while the rainfall dataset
was reduced to dimensions of 527 rows and 47 columns.
3.2 Exploratory data analysis (EDA)
To gain a comprehensive understanding of the dataset, exploratory data analysis (EDA) was conducted. This study
involved univariate, bivariate, and multivariate analyses to examine the data from different perspectives and identify
underlying patterns.
3.2.1 Univariate analysis
For categorical variables, such as ENSO Event, and Season, monthly count plots were generated to analyze their frequency
distributions. For numerical variables, Kernel Density Estimate (KDE) plots were utilized to examine the data distribution.
Figure 3 highlights La Nina events occur more frequently than El Nino and Neutral events, showing variability in the
ENSO Events.
On the other hand, Fig. 4 shows the Neutral events have the highest frequency, followed by Weak El Nino. Besides,
Very Strong El Nino have much lower counts, indicating they are less frequent.
Fig. 3 Monthly Count Plot of
ENSO Event
Vol.:(0123456789)

Research
Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Figure 5 represents the Neutral–Neutral phase condition is dominant with a higher frequency compared to remain-
ing conditions. This suggests a higher occurrence of Neutral–Neutral conditions in the dataset and highlights the
count of occurrences for different seasons. The El Nino-Neutral, Neutral-El Nino, Neutral-La Nina, La Nina-Neutral
has the lowest count.
Figure 6 shows that ONI values are concentrated around zero, indicating the dominance of neutral ENSO conditions
with fewer extreme El Niño or La Niña events.
Fig. 4 Monthly Count Plot of
ENSO Event with Intensity
Fig. 5 Monthly Count Plot of
Phase Conditions
Fig. 6 KDE Plot of ONI
Vol:.(1234567890)

Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Research
Figure 7 illustrates that NINO 3.4 SST anomalies are mostly centered near zero, suggesting balanced warm and cool
phase occurrences over the study period.
Figure 8 indicates a near-symmetric distribution of SOI values, reflecting comparable frequencies of positive and
negative atmospheric pressure anomalies.
Figure 9 shows that IOD values cluster around zero, implying that neutral Indian Ocean Dipole conditions are most
common.
Figure 10 highlights that PDO values are largely concentrated near zero, consistent with its low-frequency, decadal-
scale variability.
Figure 11 demonstrates that BEST index values are centered around zero, indicating limited dominance of extreme
SST anomalies.
Figure 12 shows a symmetric distribution of PNA values, suggesting balanced positive and negative teleconnection
phases.
Figure 13 indicates that TNI values are mostly near zero, reflecting weak-to-moderate trans-Pacific SST variability.
Fig. 7 KDE Plot of NINO 3.4
SST
Fig. 8 KDE Plot of SOI
Fig. 9 KDE Plot of IOD
Vol.:(0123456789)

Research
Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Fig. 10 KDE Plot of PDO
Fig. 11 KDE Plot of BEST
Fig. 12 KDE Plot of PNA
Fig. 13 KDE Plot of TNI
Vol:.(1234567890)

Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Research
Figure 14 shows that zonal wind anomalies are evenly distributed around zero, indicating stable upper-level
circulation conditions.
3.2.2 Bi‑variate analysis
Figures 15, 16, 17 and 18 illustrates the distribution of ENSO events (El Nino, Neutral, and La Nina) across four seasons:
Winter, Pre monsoon, Monsoon, and Post monsoon. In all seasons except Post monsoon, Neutral events exhibit the
Fig. 14 KDE Plot of 200mb
Zonal Winds
Fig. 15 Bivariate Analysis
between Winter Season and
ENSO Event
Fig. 16 Bivariate Analysis
between Pre Monsoon Season
and Enso Event
Vol.:(0123456789)

Research
Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Fig. 17 Bivariate Analysis
between Monsoon Season
and Enso Event
Fig. 18 Bivariate Analysis
between Post Monsoon Sea-
son and Enso Event
highest frequency, while El Nino events are the least common. But, La Nina events have the highest frequency in
the Post monsoon.
Figure 15 illustrates the distribution of ENSO events during the winter season. Neutral conditions dominate winter
months, indicating relatively stable large-scale climate conditions. El Niño occurrences are least frequent, while La
Niña events appear moderately, suggesting limited but notable cold-season ENSO influence.
Figure 16 shows the frequency of ENSO events during the pre-monsoon season. Neutral events remain the most
common, reflecting transitional atmospheric conditions. La Niña events occur more frequently than El Niño, indicat-
ing a tendency toward enhanced variability prior to monsoon onset.
Figure 17 presents the distribution of ENSO events during the monsoon season. Neutral phases are again domi-
nant, while La Niña events occur more often than El Niño events. This pattern highlights the stronger association of
La Niña with monsoon-season variability.
Figure 18 depicts ENSO event frequencies in the post-monsoon season. Unlike other seasons, La Niña events are
most frequent, suggesting a stronger ENSO signal during monsoon withdrawal. Neutral conditions decrease, indicat-
ing increased climate variability in this period.
Figure 19 illustrates ENSO phase transitions during winter. Most observations remain within the same phase, indi-
cating phase persistence. No direct transitions between El Niño and La Niña are observed, reflecting gradual seasonal
evolution of ENSO conditions.
Vol:.(1234567890)

Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Research
Fig. 19 Bivariate Analysis
between Winter Season and
Phase Condition
Fig. 20 Bivariate Analysis
between Pre Monsoon Season
and Phase Condition
Figure 20 shows ENSO phase conditions during the pre-monsoon season. Stable phase conditions dominate, with
limited transitions. Changes, when present, mainly occur between Neutral and ENSO-active states, highlighting gradual
pre-monsoon adjustments.
Figure 21 presents ENSO phase behavior during the monsoon season. The results indicate strong phase stability,
with minimal transitions. This persistence suggests that established ENSO conditions tend to maintain their influence
throughout the monsoon period.
Figure 22 illustrates phase conditions during the post-monsoon season. While phase stability remains dominant,
slightly more transitions are observed compared to other seasons. However, direct El Niño–La Niña shifts are still absent,
confirming gradual ENSO evolution.
3.2.3 Multi‑variate analysis
In this study, multivariate analysis is applied to explore the relationships between ENSO indices and weather param-
eters across 29 stations. To gain more specific insights, the datasets for T , T , T , and Rainfall were divided into
max min avg
eight ENSO events: Weak El Nino, Moderate El Nino, Strong El Nino, Very Strong El Nino, Neutral, Weak La Nina, Mod-
erate La Nina, and Strong La Nina, facilitating a detailed examination of event-specific correlations. This approach
enhances the precision of the analysis, uncovering complicated associations that deepen the understanding of how
ENSO indices influence regional climate variability. The heatmap uses a seven-color scale to represent the strength
and direction of correlations: dark green denotes strong negative correlations (–1 to –0.7), medium green indicates
Vol.:(0123456789)

Research
Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Fig. 21 Bivariate Analysis
between Monsoon Season
and Phase Condition
Fig. 22 Bivariate Analysis
between Post Monsoon Sea-
son and Phase Condition
Fig. 23 Correlation of ENSO
Indices with Rainfall Across 29
Stations in Bangladesh during
Strong La Nina Events
moderate negative correlations (–0.7 to –0.3), light green represents weak negative correlations (–0.3–0), white
denotes neutral relationships (0), light blue corresponds to weak positive correlations (0–0.3), medium blue reflects
moderate positive correlations (0.3–0.7), and deep blue captures strong positive correlations (0.7–1).
Figure 23 shows the correlation between various climate indices and rainfall across different regions during strong La
Nina events. The x-axis lists the meteorological stations across Bangladesh, representing regional variations in rainfall
responses, while the y-axis presents the major climate indices, including ONI, NINO 3.4 SST, SOI, IOD, PDO, BEST, PNA, TNI,
and 200 mb Zonal Winds. Notably, IOD and TNI exhibit strong negative correlations in most regions, indicating reduced
Vol:.(1234567890)

Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Research
Fig. 24 Correlation of ENSO
Indices with T Across 29
min
Stations in Bangladesh during
Strong La Nina Events
Fig. 25 Correlation of ENSO
Indices with T Across 29
avg
Stations in Bangladesh during
Strong El Nino Events
rainfall influence. PNA shows a consistently strong positive correlation, especially in southeastern areas. NINO 3.4 SST
also shows moderate positive correlations. These patterns suggest that during strong La Nina, regional rainfall is more
strongly influenced by IOD, PNA, and TNI.
Figure 24 shows strong correlations between climate indices and minimum temperature during strong La Nina events.
The x-axis displays meteorological stations across Bangladesh, capturing spatial variations in minimum temperature pat-
terns, while the y-axis includes the key climate indices—such as ONI, NINO 3.4 SST, SOI, IOD, PDO, BEST, PNA, TNI, and 200
mb Zonal Winds—that influence T variability. PNA exhibits a consistently strong positive correlation across all regions,
min
while IOD and TNI show strong negative correlations, indicating significant influence on minimum temperature. NINO
3.4 SST also shows moderate to strong positive correlations. Overall, the pattern suggests that during strong La Nina,
PNA, IOD, and TNI play key roles in shaping minimum temperature variations.
Figure 25 shows how average temperature in different parts of Bangladesh relates to various climate factors during
strong El Nino events. The x-axis outlines meteorological stations across Bangladesh, reflecting spatial changes in aver-
age temperature behavior, while the y-axis features the important climate indices—ONI, NINO 3.4 SST, SOI, IOD, PDO,
BEST, PNA, TNI, and 200 mb Zonal Winds—that drive variations in Tavg across regions. The PDO has the strongest and
most consistent positive relationship with temperature across all stations. NINO 3.4 SST, TNI, and upper-level winds also
show clear positive links, especially in eastern regions. In contrast, the PNA index is linked with lower temperatures.
Some indices like ONI, SOI, and BEST have little or no clear effect. The color scale makes it easy to compare the strength
and direction of these relationships.
Figure 26 shows how maximum temperature ( T ) in Bangladesh correlates with climate indices during strong El Nino
max
events. The x-axis represents different meteorological stations throughout Bangladesh, highlighting regional differences
in maximum temperature responses. The y-axis lists major climate indices, including ONI, NINO 3.4 SST, SOI, IOD, PDO,
BEST, PNA, TNI, and 200 mb Zonal Winds, which play a role in shaping T variations. NINO 3.4 SST, PDO, and TNI have
max
positive moderate correlations across all stations.
*** In this section, four representative correlation heatmaps are presented, derived from four climate datasets—maximum
temperature (T ), minimum temperature (T ), average temperature (T ), and rainfall, during four major ENSO events.
max min avg
These heatmaps illustrate the key relationships between ENSO indices and climate variables across Bangladesh. The remain-
ing correlation heatmaps are provided in the annexure for completeness. Altogether, a total of 32 correlation heatmaps were
generated, covering all combinations of the four climate variables and the identified ENSO events, enabling a comprehensive
assessment of ENSO–climate interactions over the study area.***
Vol.:(0123456789)

Research
Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Fig. 26 Correlation of ENSO
Indices with T Across 29
max
Stations in Bangladesh during
Strong El Nino Event
3.3 Feature engineering
To prepare the dataset for machine learning modeling, several preprocessing steps were implemented to ensure that
the input features were both interpretable and suitable for numerical algorithms. These steps were necessary because
climate datasets often contain categorical descriptors, varying measurement scales, and redundant variables, all of which
can negatively affect model training if not addressed.
3.3.1 Categorical feature encoding
The dataset included categorical variables such as SEASON and ENSO Event, which required transformation into numeri-
cal format before being used in machine learning algorithms. One-hot encoding was applied to these variables using
Scikit-learn’s OneHotEncoder. Dropping the first category (drop = ’first’) was essential to prevent multicollinearity, which
occurs when redundant dummy variables provide overlapping information. This step was particularly important because
tree-based models (RF, XGB) can overemphasize duplicated categorical information, and linear models (LR) can become
unstable when perfect collinearity exists. Thus, encoding not only enabled numerical processing but also improved
model interpretability and robustness.
3.4 Feature selection
After encoding, non-informative or operational columns such as Index, Phase Change, and Phase Duration were removed.
These variables did not contribute meaningful predictive information and retained metadata rather than physical cli-
mate interpretation. Their inclusion would introduce noise and increase model complexity without improving accuracy.
Separating the dataset into input features (x) and target variables (y) further ensured that only relevant predictors were
used in model training. This step helped reduce overfitting and enhanced the interpretability of feature-importance
measures used later in the analysis.
3.5 Data scaling
To standardize the numerical features and improve the performance of machine learning models, feature scaling was
applied. Standard scaling was performed using the StandardScaler function from Scikit-learn, which transforms the data
to have a mean of 0 and a standard deviation of 1. The scaler was fitted on the training data, and the same transformation
was applied to the test data to maintain consistency. This step was essential because the dataset contains climate vari-
ables measured in different units and on varying scales—such as temperature (°C), rainfall (mm), and multiple unitless
climate indices—which could otherwise distort the learning process. Scaling prevents predictors with larger numeric
ranges from disproportionately influencing model behavior, improves the convergence and stability of optimization-
based algorithms such as XGBoost and Linear Regression, and ensures fair distance calculations in models like KNN.
Additionally, standardized inputs help maintain numerical stability and prevent biased feature importance estimates.
The scaler was fitted only on the training data to avoid data leakage, and the same transformation was applied to the
test data, ensuring consistent and unbiased model evaluation.
Vol:.(1234567890)

Discover Environment            (2026) 4:29   | https://doi.org/10.1007/s44274-026-00533-6
Research

Table 2  Performance Comparison of Machine Learning Models for Predicting Climate Variable (Training Dataset)
| Perfor- Dataset | Model Name |     |     |     |     |     |
| --------------- | ---------- | --- | --- | --- | --- | --- |
mance
Random For- Decision Tree  K fold Cross validation XGBoosting Linear Regression KNN Regressor
Metrics
|     | est Regressor | Regressor |     |     |     |     |
| --- | ------------- | --------- | --- | --- | --- | --- |
R2 Score T  (°C) 0.9700 0.7969 0.7751 ± 0.0318 0.9353 0.6525 0.7686
max
| T  (°C) | 0.9972 | 0.9788 | 0.9787 ± 0.0054 | 0.9956 | 0.9805 | 0.8998 |
| ------- | ------ | ------ | --------------- | ------ | ------ | ------ |
min
| T  (°C) | 0.9928 | 0.9526 | 0.9456 ± 0.0115 | 0.9846 | 0.8828 | 0.8624 |
| ------- | ------ | ------ | --------------- | ------ | ------ | ------ |
avg
Rainfall (mm) 0.9464 0.6923 0.6216 ± 0.0235 0.8970 0.6302 0.6394
MAE T  (°C) 0.3443 0.9061 0.9423 ± 0.0599 0.5279 1.1833 0.9592
max
| T  (°C) | 0.1866 | 0.5313 | 0.5149 ± 0.0542 | 0.2455 | 0.5020 | 1.0857 |
| ------- | ------ | ------ | --------------- | ------ | ------ | ------ |
min
| T  (°C) | 0.2373 | 0.6118 | 0.6449 ± 0.0537 | 0.3571 | 1.0514 | 1.0583 |
| ------- | ------ | ------ | --------------- | ------ | ------ | ------ |
avg
Rainfall (mm) 31.9164 75.7660 85.6627 ± 5.4976 47.8426 92.3468 89.6615
RMSE T  (°C) 0.4698 1.2256 1.2812 ± 0.1035 0.6860 1.6080 1.3114
max
| T  (°C) | 0.2556 | 0.7002 | 0.6917 ± 0.0785 | 0.3201 | 0.6745 | 1.5645 |
| ------- | ------ | ------ | --------------- | ------ | ------ | ------ |
min
| T  (°C) | 0.3253 | 0.8299 | 0.8764 ± 0.0720 | 0.4751 | 1.3160 | 1.4318 |
| ------- | ------ | ------ | --------------- | ------ | ------ | ------ |
avg
Rainfall (mm) 51.6270 124.1331 135.9836 ± 7.6475 71.6564 137.1181 138.1089
Table 3  Performance Comparison of Machine Learning Models for Predicting Climate Variable (Testing Dataset)
| Perfor- Dataset | Model Name |     |     |     |     |     |
| --------------- | ---------- | --- | --- | --- | --- | --- |
mance
Metrics Random For- Decision Tree  K fold Cross validation XGBoosting Linear Regression KNN Regressor
|     | est Regressor | Regressor |     |     |     |     |
| --- | ------------- | --------- | --- | --- | --- | --- |
R2 Score T  (°C) 0.8512 0.7918 0.7751 ± 0.0318 0.8509 0.6425 0.6095
max
| T  (°C) | 0.9693 | 0.9584 | 0.9787 ± 0.0054 | 0.9702 | 0.9679 | 0.8589 |
| ------- | ------ | ------ | --------------- | ------ | ------ | ------ |
min
| T  (°C) | 0.9521 | 0.9386 | 0.9456 ± 0.0115 | 0.9559 | 0.8472 | 0.7873 |
| ------- | ------ | ------ | --------------- | ------ | ------ | ------ |
avg
Rainfall (mm) 0.6253 0.6146 0.6216 ± 0.0235 0.5897 0.5662 0.4325
MAE T  (°C) 0.7921 0.9195 0.9423 ± 0.0599 0.7889 1.1940 1.2683
max
| T  (°C) | 0.5144 | 0.6243 | 0.5149 ± 0.0542 | 0.5120 | 0.5529 | 1.2357 |
| ------- | ------ | ------ | --------------- | ------ | ------ | ------ |
min
| T  (°C) | 0.5719 | 0.6563 | 0.6449 ± 0.0537 | 0.5479 | 1.0951 | 1.2344 |
| ------- | ------ | ------ | --------------- | ------ | ------ | ------ |
avg
Rainfall (mm) 94.3346 94.9789 85.6627 ± 5.4976 101.2635 107.4055 125.6424
RMSE T  (°C) 1.0180 1.2043 1.2812 ± 0.1035 1.0252 1.5982 1.6749
max
| T  (°C) | 0.7772 | 0.9042 | 0.6917 ± 0.0785 | 0.7678 | 0.7953 | 1.6955 |
| ------- | ------ | ------ | --------------- | ------ | ------ | ------ |
min
| T  (°C) | 0.7644 | 0.8630 | 0.8764 ± 0.0720 | 0.7316 | 1.3677 | 1.6149 |
| ------- | ------ | ------ | --------------- | ------ | ------ | ------ |
avg
Rainfall (mm) 149.2662 152.6571 135.9836 ± 7.6475 154.7295 161.8352 188.6645
3.6   Model performance evaluation
Table 2 presents the performance comparison of six machine learning models (Random Forest, Decision Tree, K-Fold
Cross-Validated model, XGBoost, Linear Regression, and KNN) for predicting different climate variables (Tmax, Tmin,
Tavg, and Rainfall) using the training dataset. Overall, the Random Forest and XGBoost models consistently demonstrate
higher R  2 values and lower error metrics, indicating superior predictive performance. The K-Fold cross-validated model
provides stable and reliable results with low variance. In contrast, Linear Regression and KNN generally show weaker
performance, especially for rainfall prediction, with higher MAE and RMSE values.
Table 3 presents the performance of six machine learning models (Random Forest Regressor, Decision Tree Regres-
sor, K-Fold Cross Validation, XGBoosting, Linear Regression, and KNN Regressor) evaluated across four testing datasets
, and Rainfall). Three metrics are used:  R2 Score, MAE (Mean Absolute Error), and RMSE (Root Mean
( T , T  , T
max min avg
Square Error). The best-performing model is Random Forest in predicting Tmax, with an  R2 score of 0.8512, the MAE
of 0.7921, and RMSE of 1.0180. Similarly, Random Forest also excels in predicting Rainfall, achieving the highest  R2
score (0.6253), the MAE (94.3346), and the RMSE (149.2662). On the other hand, XGBoosting leads for  T  dataset, with
avg
Vol.:(0123456789)

Research
Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
an R 2 score of 0.9559, the MAE (0.5479), and RMSE of 0.7316. XGBoosting shows superior performance in T predic-
min
tion, achieving the best balance across metrics with an R2 score of 0.9702, an MAE of 0.5120, and RMSE of 0.7316.
In the training phase (Table 2), Random Forest and XGBoost consistently achieve the highest R2 scores and lowest
error values, indicating strong learning capability and robustness, while the K-Fold cross-validated model ensures
stable and reliable performance. In contrast, Linear Regression and KNN perform relatively poorly, particularly for
rainfall prediction. The testing results in Table 3 confirm these trends. Random Forest shows superior performance
for Tmax and Rainfall, whereas XGBoost outperforms other models for Tmin and Tavg, achieving the best balance
between accuracy and error. The close agreement between training and testing performances indicates good gener-
alization ability of the ensemble models, highlighting Random Forest and XGBoost as the most reliable and accurate
approaches for climate variable prediction.
In particular, the XGBoosting model performed best for T and T due to the consistency in these datasets. These
min avg
datasets exhibit stable patterns, allowing XGBoost’s gradient boosting approach to effectively capture and predict the
underlying relationships with high precision. Its ability to handle non-linear relationships and outliers and fine-tune
its performance through regularization further contributes to its success. The Random Forest Regressor excelled in
predicting T and Rainfall, despite the dataset’s random variability. Random Forest’s ensemble approach, which
max
combines predictions from multiple decision trees, effectively handles random variations and avoids overfitting. This
robustness makes it ideal for datasets with less predictable patterns.
The Fig. 27 serves as a complementary visualization to Table 1 by translating the regression outputs into a classifi-
cation-based interpretation. Each observation was categorized into three conceptual classes based on its deviation
from the 10-year seasonal average: below-average, where actual or predicted values of rainfall, T , T , and T
avg min max
are lower than the corresponding average; above-average, where values exceed the 10-year seasonal mean; and
neutral, where values lie close to the average and the probabilities of being above or below are nearly equal. The
resulting heatmap employs a binary evaluation scheme in which green indicates correct classification, meaning
the predicted class matches the observed class, while red denotes incorrect classification where the predicted and
actual classes differ.
Critically, the model demonstrates strong alignment with actual trends, particularly for temperature variables (T ,
avg
T , T ), where the predicted direction often matches the observed change. However, prediction accuracy appears
min max
more varied for rainfall, which shows several mismatches or uncertain classifications (“Both”), reflecting the complex
and nonlinear nature of precipitation dynamics. Some combinations (e.g., La Nina–El Nino in Pre-Monsoon) show
high concordance across all variables, while others (e.g., Neutral–Neutral in Post-Monsoon) reflect greater ambiguity.
Fig. 27 Comparison between
Actual and Predicted Scenario
under Different Conditions
Vol:.(1234567890)

Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Research
4 Discussion
The results of this study provide a detailed understanding of how ENSO-related indices modulate Bangladesh’s
seasonal climate, but their interpretation must be situated within the context of earlier ENSO–rainfall studies in the
region. Previous research has consistently shown that El Niño events are associated with suppressed monsoon rainfall
and warmer winter temperatures, while La Niña events typically enhance monsoon rainfall and contribute to cooler
winter conditions in Bangladesh [11, 35]. The correlations found in this study, especially reduced rainfall during El
Niño phases and increased rainfall during La Niña, are highly consistent with these well-established teleconnection
patterns.
Where this study advances earlier work is in demonstrating that several non-traditional indices (e.g., PDO, TNI, PNA)
show correlations equal to or stronger than the canonical Niño 3.4 or ONI. This aligns with emerging evidence that
ENSO’s regional impacts are increasingly shaped by extratropical and Indian Ocean processes rather than solely by
central Pacific SST anomalies [9, 13]. For instance, the consistently strong influence of PDO on Tmax, Tmin, and Tavg
observed in this study supports the understanding that Pacific decadal variability can modulate the strength and
spatial pattern of ENSO teleconnections. Similar behavior has been reported in South Asian monsoon studies, which
show that warm PDO phases amplify El Niño drying, while cool PDO phases can weaken it [3, 21].
The results also reveal that IOD and TNI frequently exhibit strong negative correlations with rainfall and tempera-
ture during La Niña phases. This is physically consistent with their documented roles in modulating moisture trans-
port and convective activity across the Indo-Pacific region [5, 9]. A positive IOD event, for example, may counteract
El Niño–induced drying by enhancing westerly moisture flow into the Bay of Bengal, while negative IOD conditions
often reinforce La Niña–related cooling and precipitation anomalies across South Asia [3].
Furthermore, the strong positive correlation between PNA and temperature variables (Tmax, Tmin, Tavg) observed
in multiple ENSO phases suggests an important mid-latitude influence on Bangladesh’s climate. PNA’s documented
role in generating Rossby wave trains that modulate upper-level zonal winds and wintertime temperature variability
[13] supports this finding. The significant involvement of 200 mb zonal winds in the model also reinforces the idea
that upper-tropospheric circulation anomalies contribute meaningfully to seasonal variability over South Asia.
Another important contribution of this study is the spatial differentiation of teleconnections across Bangladesh.
Stations in the northeast and southeast exhibited stronger correlations with Pacific indices, consistent with prior
findings that monsoon trough positioning and Bay of Bengal moisture influx are more sensitive in these regions
[11]. In contrast, weaker teleconnections in the northwest align with earlier climatological studies showing greater
continental influence and reduced monsoon penetration [1]. These results confirm that ENSO teleconnections are
both seasonally and geographically modulated within Bangladesh.
When compared with prior work, the exceptionally high model skill for temperature (T R2 = 0.9693, T
min avg
R2 = 0.9559, T R2 = 0.8512) is consistent with the broader literature: temperature anomalies tend to be more directly
max
controlled by large-scale radiative and circulation drivers and are therefore easier for multi-index, nonlinear mod-
els to predict than precipitation [24]. When benchmarked against prior work, the rainfall R2 obtained in this study
(0.6253) aligns well with expectations for a spatially extensive, multi-decadal national dataset. Earlier ENSO–rainfall
studies in Bangladesh typically report modest explanatory power when using single indices, often corresponding to
R2 values below 0.30 [11]. In contrast, localized or basin-specific machine-learning studies frequently achieve higher
predictive skill, with Random Forest, Gradient Boosting, or LSTM models reporting R2 values in the range of 0.70–0.80
for monthly or seasonal rainfall [15, 39]. These elevated values, however, are generally derived from smaller spatial
domains, shorter temporal windows, and more homogeneous hydroclimatic regimes, which inherently reduce vari-
ability and increase model fit. Therefore, the rainfall R2 obtained in this study represents a robust and realistic estimate
for a heterogeneous 29-station dataset spanning 45 years. Moreover, the comparatively higher skill observed for
temperature variables (T , T , and T ) is consistent with established climatological understanding: temperature
min avg max
responds more directly to large-scale radiative forcing and circulation anomalies, whereas rainfall is influenced by
additional mesoscale and intraseasonal processes that limit predictive skill even in advanced ML frameworks.
Overall, the findings show that ENSO teleconnections affecting Bangladesh arise from a combination of tropical
Pacific, Indian Ocean, and extratropical circulation modes. Unlike earlier Bangladesh-focused studies that relied
primarily on single indices such as Niño 3.4 or SOI, this multi-index approach captures a more comprehensive set of
physical drivers, including decadal variability and cross-basin interactions. The strong performance of ensemble ML
models, Random Forest and XGBoost, further demonstrates the advantage of nonlinear, multi-predictor frameworks
Vol.:(0123456789)

Research
Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
for diagnosing complex climate relationships [28]. By integrating nine teleconnection indices and applying interpret-
able machine learning, this study offers a more holistic explanation for why ENSO impacts vary across seasons, ENSO
phases, and regions within Bangladesh.
5 Conclusion
This study explored the influence of ENSO-related climate indices on temperature and rainfall patterns across Bangladesh
using 45 years of observational data and a suite of supervised machine learning models. Strong associations between
regional weather parameters and indices such as NINO 3.4 SST, SOI, PDO, and TNI were identified, particularly during
pronounced El Niño and La Niña phases. Among the evaluated models, XGBoost delivered the highest predictive accu-
racy for T ( R2 = 0.9702) and T ( R2 = 0.9559), while Random Forest performed best for T ( R2 = 0.8512) and rainfall
min avg max
( R2 = 0.6253), demonstrating the capability of ensemble methods to capture complex climate relationships.
Seasonal forecasts generated using the best-performing models revealed consistent warming trends in T and T
min avg
across most ENSO transitions, with notable declines in T during monsoon periods. Rainfall predictions were more vari-
max
able, showing increases during La Niña–related transitions and declines during El Niño–associated phases, reflecting the
inherent complexity of precipitation dynamics. These findings underscore the practical value of machine learning–based
forecasting for agricultural planning, water resource management, and disaster preparedness, enabling more informed
decision-making in climate-sensitive sectors.
Future work should prioritize the integration of high-resolution remote sensing datasets, the inclusion of a broader
set of teleconnection indices, and the implementation of advanced deep learning architectures—such as LSTM networks
and hybrid ensemble frameworks, to further enhance predictive performance. The relatively lower R2 values obtained
for rainfall prediction can be attributed to the study’s exclusive reliance on ENSO indices, even though rainfall variability
is also strongly influenced by other large-scale atmospheric phenomena, such as the Madden–Julian Oscillation. We
acknowledge this limitation and recommend that future research incorporate additional atmospheric drivers to capture
rainfall dynamics more comprehensively. Furthermore, extending the modeling framework to simulate extreme weather
events, including heatwaves, droughts, and floods, would strengthen early warning systems and support long-term
climate resilience planning for Bangladesh and other ENSO-sensitive regions.
Author contributions M.M.: Conceptualization, Data sourcing and collection, Data processing, Analysis, Validation, Visualization, Result inter-
pretation, Design, Writing—original draft, Writing—review and editing. T.G.: Conceptualization, Data sourcing and collection, Data processing,
Analysis, Validation, Visualization, Result interpretation, Design, Writing—original draft, Writing—review and editing. F.A.: Conceptualization,
Data sourcing and collection, Design, Writing—original draft, Writing—review and editing. S.S: Conceptualization, Data sourcing and collec-
tion, Design, Writing—original draft, Writing—review and editing. M.R.A.M.: Conceptualization, Supervision, Writing—review and editing.
Funding The authors declare that no funds, grants, or other support were received during the preparation of this manuscript.
Data availability The datasets analyzed during the current study are not publicly available due to confidentiality agreements and institutional
data-sharing restrictions, but are available from the corresponding author on reasonable request.
Declarations
Ethics approval and consent to participate Not applicable.
Consent for publication Not applicable.
Competing interests The authors declare no competing interests.
Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which
permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit to
the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You
do not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party
material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If
material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds
the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://c reati veco
mmons.o rg/l icens es/b y-n c-n d/4.0 /.
Vol:.(1234567890)

Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
Research
References
1. Ahmed R, Kim IK. Patterns of daily rainfall in Bangladesh during the summer monsoon season: case studies at three stations. Phys Geogr.
2003;24(4):295–318. https://d oi.o rg/1 0.2 747/0 272-3 646.2 4.4.2 95.
2. Amin A, Mourshed M. Weather and climate data for energy applications. Renew Sustain Energy Rev. 2024;192:114247. https://d oi.o rg/
10.1 016/J.R SER.2 023.1 14247.
3. Ashok K, Guan Z, Yamagata T. Impact of the Indian Ocean dipole on the relationship between the Indian monsoon rainfall and ENSO.
Geophys Res Lett. 2001;28(23):4499–502. https://d oi.o rg/1 0.1 029/2 001GL 01329 4.
4. Barnston AG, Tippett MK, Ranganathan M, L’Heureux ML. Deterministic skill of ENSO predictions from the North American Multimodel
Ensemble. Clim Dyn. 2019;53(12):7215–34. https://d oi.o rg/1 0.1 007/s 00382-0 17-3 603-3.
5. Behera SK, Yamagata T. Influence of the Indian Ocean Dipole on the Southern Oscillation. J Meteorol Soc Japan Ser II. 2003;81(1):169–77.
https://d oi.o rg/1 0.2 151/J MSJ.8 1.1 69.
6. Biau G, Fr GB. Analysis of a Random Forests Model. In Journal of Machine Learning Research (Vol. 13). 2012.
7. Bilbao-Barrenetxea N, Martínez-España R, Jimeno-Sáez P, Faria SH, Senent-Aparicio J. Multi-model ensemble machine learning
approaches to project climatic scenarios in a river basin in the Pyrenees. Earth Syst Environ. 2024;8(4):1159–77. https://d oi.o rg/1 0.1 007/
S41748-0 24-0 0408-X.
8. Chen S, Chen W, Xie SP, Yu B, Wu R, Wang Z, et al. Strengthened impact of boreal winter North Pacific Oscillation on ENSO development
in warming climate. NPJ Clim Atmos Sci. 2024. https://d oi.o rg/1 0.1 038/s 41612-0 24-0 0615-3.
9. Ciasto LM, Simpkins GR, England MH. Teleconnections between Tropical Pacific SST anomalies and extratropical Southern Hemisphere
climate. J Clim. 2015;28(1):56–65. https://d oi.o rg/1 0.1 175/J CLI-D-1 4-0 0438.1.
10. Douglass MJJ. Book review: hands-on machine learning with Scikit-Learn, Keras, and Tensorflow, 2nd edition by Aurélien Géron. Phys
Eng Sci Med. 2020;43(3):1135–6. https://d oi.o rg/1 0.1 007/s 13246-0 20-0 0913-z.
11. Ehsan MA, Tippett MK, Robertson AW, Singh B, Rahman MA. The ENSO fingerprint on Bangladesh summer monsoon rainfall. Earth Syst
Environ. 2023;7(3):617–27. https://d oi.o rg/1 0.1 007/s 41748-0 23-0 0347-z.
12. Fang W, Sha Y, Sheng VS. Survey on the application of artificial intelligence in ENSO forecasting. Mathematics. 2022;10(20):3793. https://
doi.o rg/1 0.3 390/M ATH10 20379 3.
13. Gershunov A, Barnett TP. Interdecadal modulation of ENSO teleconnections. Bull Am Meteorol Soc. 1998;79(12):2715–26. https://d oi.o rg/
10.1 175/1 520-0 477(1998)0 79.
14. Halder RK, Uddin MN, Uddin MA, Aryal S, Khraisat A. Enhancing K-nearest neighbor algorithm: a comprehensive review and performance
analysis of modifications. J Big Data. 2024. https://d oi.o rg/1 0.1 186/s 40537-0 24-0 0973-y.
15. Islam MS, Shafiuzzaman M, Mahmud G, Nowshin N, Reza P, Hasan J, et al. Explainable deep learning for rainfall prediction: a CNN-
XGBoost hybrid approach in the northern region of Bangladesh. Neural Comput Appl. 2025;37(33):28125–60. https://d oi.o rg/1 0.1 007/
S00521-0 25-1 1646-Z.
16. Islam T, Saha S, Evan AA, Halder N, Dey SC. Monthly weather forecasting through ANN model: a case study in Barisal, Bangladesh. IJARCCE.
2016;5(6):1–6. https://d oi.o rg/1 0.1 7148/i jarcc e.2 016.5 601.
17. Jan Z, Abrar M, Bashir S, Mirza AM. Seasonal to Inter-annual Climate Prediction Using Data Mining KNN Technique. Communications in
Computer and Information Science, 20 CCIS. 2008;40–51. https://d oi.o rg/1 0.1 007/9 78-3-5 40-8 9853-5_7
18. Jaroszewicz S, Mariani MC, Tweneboah OK, Beccar-Varela MP. Multifractal analysis of the Southern Oscillation Index. J Atmos Sol-Terr Phys.
2024;254:106161. https://d oi.o rg/1 0.1 016/J.J ASTP.2 023.1 06161.
19. Jordan G. The rise of machine learning in climate modelling. Weather. 2025;80(6):185–6. https://d oi.o rg/1 0.1 002/W EA.7 717.
20. Khoshvaght H, Permala RR, Razmjou A, Khiadani M. A critical review on selecting performance evaluation metrics for supervised machine
learning models in wastewater quality prediction. J Environ Chem Eng. 2025;13(6):119675. https://d oi.o rg/1 0.1 016/J.J ECE.2 025.1 19675.
21. Kumar KK, Rajagopalan B, Hoerling M, Bates G, Cane M. Unraveling the mystery of Indian monsoon failure during El Niño. Science.
2006;314(5796):115–9. https://d oi.o rg/1 0.1 126/S CIENC E.1 13115 2.
22. Kuthuru A. Semantic data contracts: a new integration paradigm for enterprise AI and database systems. Eur J Comput Sci Inf Technol.
2025;13(31):1–9. https://d oi.o rg/1 0.3 7745/e jcsit.2 013/v ol13n 3119.
23. Latif M, Barnett TP, Cane MA, Flügel M, Graham NE, von Storch H, et al. A review of ENSO prediction studies. Clim Dyn. 1994;9(4–5):167–79.
https://d oi.o rg/1 0.1 007/B F0020 8250/M ETRIC S.
24. Li H, Li S, Ghorbani H. Data-driven novel deep learning applications for the prediction of rainfall using meteorological data. Front Environ
Sci. 2024;12:1445967. https://d oi.o rg/1 0.3 389/F ENVS.2 024.1 44596 7/B IBTEX.
25. Liu Z, Leo. Decision Trees. Artificial Intelligence for Engineers. 2025. https://d oi.o rg/1 0.1 007/9 78-3-0 31-7 5953-6_4
26. Nti IK, Nyarko-Boateng O, Aning J. Performance of machine learning algorithms with different K values in K-fold crossvalidation. Int J
Inform Technol Comput Sci. 2021;13(6):61–71. https://d oi.o rg/1 0.5 815/i jitcs.2 021.0 6.0 5.
27. Pawlicki M, Pawlicka A, Uccello F, Szelest S, D’Antonio S, Kozik R, et al. Evaluating the necessity of the multiple metrics for assessing
explainable AI: a critical examination. Neurocomputing. 2024;602:128282. https://d oi.o rg/1 0.1 016/J.N EUCOM.2 024.1 28282.
28. Pinheiro E, Ouarda TBMJ. An interpretable machine learning model for seasonal precipitation forecasting. Commun Earth Environ.
2025;6(1):1–14. https://d oi.o rg/1 0.1 038/s 43247-0 25-0 2207-2.
29. Rodrigues AP, Fernandes R, Vijaya P. A study on the evaluation of different regressors in Weather Prediction. International Conference on
Artificial Intelligence and Data Engineering, AIDE 2022. 2022. pp. 13–18. https://d oi.o rg/1 0.1 109/A IDE57 180.2 022.1 00608 14.
30. Sahu RT, Kumar K, Joshi SS, Memon KK, Kumar C. Climate Long-Distance Interaction Signature for Spatio-Temporal Variability of Indian
Monsoon Over the Tropic of Cancer: India. J Indian Soc Remote Sens. 2025;2025:1–18. https://d oi.o rg/1 0.1 007/S 12524-0 25-0 2328-3.
31. Sahu RT, Verma MK, Ahmad I. Impact of long-distance interaction indicator (monsoon indices) on spatio-temporal variability of precipita-
tion over the Mahanadi River basin. Water Resour Res. 2023;59(6):e2022WR033805. https://d oi.o rg/1 0.1 029/2 022WR 03380 5.
32. Saji NH, Goswami BN, Vinayachandran PN, Yamagata T. A dipole mode in the tropical Indian Ocean. Nature. 1999;401(6751):360–3. https://
doi.o rg/1 0.1 038/4 3854.
Vol.:(0123456789)

Research
Discover Environment (2026) 4:29 | https://doi.org/10.1007/s44274-026-00533-6
33. Saleh MdA, Rasel HM. Performance evaluation of Machine Learning based regression models for rainfall forecasting. 2024. https://d oi.
org/1 0.2 1203/R S.3.R S-3 85674 1/V 1.
34. Sarker IH. Machine learning: algorithms, real-world applications and research directions. SN Comput Sci. 2021;2(3):1–21. https://d oi.o rg/
10.1 007/S 42979-0 21-0 0592-X/M ETRIC S.
35. Sattar MA, Mia S, Shanta AA, Abdul Ahad Biswas AKM, Ludwig F. Remote impacts from el niño and la niña on climate variables and major
crops production in coastal Bangladesh. Atmosphere. 2021. https://d oi.o rg/1 0.3 390/a tmos1 21114 49.
36. Song WJ, Yu JY, Lian T. The efficacy of tropical and extratropical predictors for long-lead El Niño-Southern Oscillation prediction: a study
using a machine learning algorithm. Int J Climatol. 2023;43(14):6887–99. https://d oi.o rg/1 0.1 002/j oc.8 241.
37. Starbuck C. Linear Regression. In The Fundamentals of People Analytics (pp. 181–206). Springer International Publishing. 2023. https://
doi.o rg/1 0.1 007/9 78-3-0 31-2 8674-2_1 0.
38. Tareq WZT. Deep learning. Decision-Making Models: A Perspective of Fuzzy Logic and Machine Learning. 2024. pp. 317–327. https://d oi.
org/1 0.1 016/B 978-0-4 43-1 6147-6.0 0016-5.
39. Wang X, Kingsland G, Poudel D, Fenech A. Urban flood prediction under heavy precipitation. J Hydrol. 2019;577:123984. https://d oi. org/
10.1 016/J.J HYDRO L.2 019.1 23984.
40. Wu Q, Vos P. Inference and prediction. Handb Stat. 2018;38:111–72. https://d oi.o rg/1 0.1 016/B S.H OST.2 018.0 6.0 04.
41. Yeh SW, Kirtman BP. ENSO amplitude changes due to climate change projections in different coupled models. J Climate. 2007;20(2):203–17.
https://d oi.o rg/1 0.1 175/J CLI40 01.1.
Publisher’s Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.
Vol:.(1234567890)
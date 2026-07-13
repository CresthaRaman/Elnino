TRIBHUVAN UNIVERSITY
INSTITUTE OF ENGINEERING
THAPATHALI CAMPUS
A Project Proposal
On
AI-Enhanced Prediction of El Nino and its Regional Climatic Impacts
Submitted By:
Biraj Adhikari (THA080BCT013)
Raman Shrestha (THA080BCT031)
Rojin Dhami (THA080BCT035)
Sandeep Khadka (THA080BCT037)
Submitted To:
Department of Electronics and Computer Engineering
Thapathali Campus
Kathmandu, Nepal
June, 2026

ACKNOWLEDGEMENT
WeexpressoursinceregratitudetotheDepartmentofElectronicsandComputerEngi-
neering, Thapathali Campus, for providing us the opportunity to undertake this minor
projecton“AI-EnhancedPredictionofElNiñoanditsRegionalClimaticImpacts”
We are deeply thankful to our project supervisor Er. Kobid Karkee for his invaluable
guidance, constructive feedback, and continuous encouragement throughout the pro-
posal preparation and planned project work. His expertise in machine learning and
climate modeling has been instrumental in shaping our approach and in helping us an-
ticipateandaddresstechnicalchallenges.
We also appreciate the support from our faculty members and peers who provided in-
sightfulsuggestionsduringdiscussionsontopicselection,literaturereview,andmethod-
ologydesign. Specialthanksgototheopen-sourcecommunity,especiallytheproviders
of the ERA5 and ORAS5 datasets, for making state-of-the-art climate reanalysis data
easilyaccessibletostudentsandresearchers.
Finally, we are grateful to our families and friends for their patience, motivation, and
consistent support during the extended hours of study, experimentation, and documen-
tationrequiredforthisproposal.
BirajAdhikari(THA080BCT013)
RamanShrestha(THA080BCT031)
RojinDhami(THA080BCT035)
SandeepKhadka(THA080BCT037)
ii

ABSTRACT
The 2026 El Niño event is forecasted with high probability, posing significant climatic
risks globally, particularly for South Asia’s monsoon-dependent agriculture and water
security. Thisprojectproposesthedevelopmentofadeeplearningframeworkthatfore-
casts the El Niño-Southern Oscillation (ENSO) state at lead times of 3–6 months and
translates these predictions into deterministic impact assessments for South Asian pre-
cipitationandtemperatureanomalies. Leveragingpubliclyavailablereanalysisdatasets
(ERA5, ORAS5) and ENSO indices, the model will be trained and validated on recent
historical data (1980-2025).The proposed approach employs a Convolutional Neural
Network – Temporal Convolutional Network (CNN-TCN) hybrid architecture wherein
the CNN component captures spatial patterns (east-west SST gradients, spatial ENSO
patterns across the tropical Pacific) from each monthly snapshot and the TCN compo-
nent captures temporal evolution (progressive warming trends, multi-month dependen-
cies)acrossthe12-monthinputwindow. Theexpectedoutcomeisafunctionalforecast-
ingpipelineandadeterministicoutlookfortheJune–September2026monsoonseason,
contributing actionable insights for disaster preparedness and resource management in
theSouthAsianregion.
Keywords: climate prediction, CNN-TCN, deep learning, El Niño, ENSO, monsoon
forecasting,seasurfacetemperature,SouthAsia.
iii

TABLEOFCONTENTS
ACKNOWLEDGEMENT ii
ABSTRACT iii
LISTOFFIGURES vi
LISTOFTABLES vii
LISTOFABBREVIATIONS viii
1 INTRODUCTION 1
1.1 Background . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
1.2 Motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
1.3 ProblemStatement . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
1.4 Objectives . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
1.5 ScopeandLimitations . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.5.1 Scope . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.5.2 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.6 ProjectApplications . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2 LITERATUREREVIEW 5
2.1 ENSOPrediction: FromDynamicalModelstoDeepLearning . . . . . 5
2.2 DeepLearningApproachesforENSOForecasting . . . . . . . . . . . . 5
2.2.1 CNN-BasedMulti-YearENSOForecasting . . . . . . . . . . . 5
2.2.2 Autoencoder-LSTMHybridArchitecture . . . . . . . . . . . . . 6
2.2.3 Physics-GuidedDeepEchoStateNetworks . . . . . . . . . . . 6
2.2.4 ResoNet: RobustandExplainableENSOForecasting . . . . . . 7
2.3 MachineLearningforRegionalClimateImpactAssessment . . . . . . . 8
2.3.1 ENSO-DrivenWeatherPatternPrediction . . . . . . . . . . . . 8
2.4 ComparisonofExistingApproaches . . . . . . . . . . . . . . . . . . . 8
3 METHODOLOGY 9
3.1 SystemOverview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
3.2 SystemArchitecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
iv

3.3 DataSources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
3.3.1 Oceanic&AtmosphericReanalysis . . . . . . . . . . . . . . . 13
3.3.2 ENSOIndices&ModelEnsembles . . . . . . . . . . . . . . . 13
3.4 Workflow . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
3.4.1 DataIngestion . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
3.4.2 DataPreprocessing . . . . . . . . . . . . . . . . . . . . . . . . 14
3.4.3 FeatureEngineering . . . . . . . . . . . . . . . . . . . . . . . . 17
3.4.4 FeatureSelection . . . . . . . . . . . . . . . . . . . . . . . . . 18
3.4.5 BaselineModel . . . . . . . . . . . . . . . . . . . . . . . . . . 20
3.4.6 DeepLearningModel . . . . . . . . . . . . . . . . . . . . . . . 21
3.4.7 ValidationandEvaluation . . . . . . . . . . . . . . . . . . . . . 24
3.4.8 2026Forecasting . . . . . . . . . . . . . . . . . . . . . . . . . 26
3.5 Tools&Environment . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
4 EXPECTEDOUTCOMES 27
5 PROJECTSCHEDULE 28
6 PROJECTBUDGET 29
7 FEASIBILITYANALYSIS 30
7.1 TechnicalFeasibility . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
7.2 EconomicFeasibility . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
7.3 TimeFeasibility . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
7.4 OperationalFeasibility . . . . . . . . . . . . . . . . . . . . . . . . . . 31
REFERENCES 32
v

LISTOFFIGURES
Figure3-1 SystemArchitectureoftheproject . . . . . . . . . . . . . . . . 10
Figure3-2 DataPreprocessingPipeline . . . . . . . . . . . . . . . . . . . 12
Figure5-1 GanttChartoftheProjectSchedule . . . . . . . . . . . . . . . 28
vi

LISTOFTABLES
Table2-1 ComparisonofDeepLearningApproachesforENSOForecasting 8
Table6-1 EstimatedProjectBudget . . . . . . . . . . . . . . . . . . . . . 29
vii

LISTOFABBREVIATIONS
ACC AnomalyCorrelationCoefficient
AI ArtificialIntelligence
MIFS MutualInformationFeatureSelection
CMIP6 CoupledModelIntercomparisonProjectPhase6
CNN ConvolutionalNeuralNetwork
CORDEX CoordinatedRegionalClimateDownscalingExperiment
DESN DeepEchoStateNetwork
DL DeepLearning
ECMWF EuropeanCentreforMedium-RangeWeatherForecasts
ENSO ElNiño-SouthernOscillation
ESN EchoStateNetwork
GCM GeneralCirculationModel
GODAS GlobalOceanDataAssimilationSystem
GPU GraphicsProcessingUnit
JJAS June-July-August-September
LSTM LongShort-TermMemory
MAE MeanAbsoluteError
MEI MultivariateENSOIndex
ML MachineLearning
NOAA NationalOceanicandAtmosphericAdministration
ORAS5 OceanReanalysisSystem5
RF RandomForest
RMSE RootMeanSquareError
RNN RecurrentNeuralNetwork
ResNet ResidualNetwork
SLP SeaLevelPressure
SST SeaSurfaceTemperature
CNN-TCN ConvolutionalNeuralNetwork–TemporalConvolutionalNetwork
XGBoost ExtremeGradientBoosting
viii

1. INTRODUCTION
1.1 Background
TheElNiño-SouthernOscillation(ENSO)isthedominantmodeofinterannualclimate
variability,influencingweatherpatternsworldwidethroughcoupledocean-atmosphere
interactions in the tropical Pacific [1]. ENSO cycles between warm (El Niño) and cool
(La Niña) phases, driven primarily by anomalies in Sea Surface Temperature (SST)
across the equatorial Pacific Ocean. These events have far-reaching teleconnections
that affect precipitation, temperature, and extreme weather events across multiple con-
tinents.
South Asia, where the Indian Summer Monsoon provides over 70% of annual rain-
fall, is highly vulnerable to ENSO-induced climate anomalies [2]. El Niño events are
associated with weakened monsoon circulation, leading to droughts in the Indian sub-
continent,whileLaNiñaeventsoftenintensifymonsoonrainfall[3]. Climateoutlooks
indicate the possibility of ENSO-related anomalies during 2026, highlighting the im-
portanceofimprovedforecastingsystems.
Traditional dynamical climate models, such as General Circulation Models (GCMs),
provide seasonal ENSO forecasts but face limitations in prediction skill beyond 6–9
monthsduetothe“springpredictabilitybarrier”[4]. Recentadvancesindeeplearning
havedemonstratedsuperiorforecastingskillatlongerleadtimes(12–22months)byex-
tracting complex spatiotemporal patterns from vast reanalysis archives [1, 3]. These
data-driven approaches leverage Convolutional Neural Networks (CNNs) and Long
Short-Term Memory (LSTM) networks to capture nonlinear dynamics that traditional
statisticalmethodsanddynamicalmodelsstruggletorepresent.
1.2 Motivation
Themotivationforthisprojectstemsfromthreecriticalfactors:
1. Imminent Climate Risk: The 2026 El Niño poses significant risks to South
1

Asia’sagriculture,waterresources,anddisastermanagementinfrastructure. Early
andaccuratepredictionenablesproactivemitigationstrategies.
2. Limitations of Existing Models: While dynamical models provide reasonable
short-term ENSO forecasts, their skill degrades significantly beyond one season.
Deep learning models have shown promise in extending this predictability hori-
zonbutareoftennottailoredtoregionalimpactassessment.
3. Need for Actionable Regional Forecasts: Most existing ENSO prediction sys-
tems focus on the Niño 3.4 index alone without providing spatially resolved im-
pact assessments for vulnerable regions. There is a critical gap between global
ENSOstatepredictionandlocalizedclimateimpactforecasting.
1.3 ProblemStatement
Despite significant progress in ENSO forecasting using deep learning, existing ap-
proachesfacekeylimitations:
1. Most models predict only the Niño 3.4 index without translating predictions into
regional climate impacts, limiting their practical utility for disaster preparedness
andagriculturalplanning.
2. No unified deep learning framework currently exists that couples ENSO state
forecasting with deterministic regional impact assessment specifically for South
Asianprecipitationandtemperatureanomalies.
Thisprojectaddressesthesegapsbydevelopingamulti-taskCNN-TCNframeworkthat
forecasts ENSO state at 6 month lead times and simultaneously produces deterministic
precipitationandtemperatureanomalymapsoverSouthAsia.
1.4 Objectives
Theprimaryobjectivesofthisprojectareasfollows:
2

1. To develop a CNN-TCN model to forecast the Niño 3.4 index at a 6-month lead
timeusingERA5andORAS5reanalysisdataspanning1980–2025.
2. Todevelopacoupledimpactassessmentmodulethatproducesdeterministicfore-
casts of June–July–August–September precipitation and temperature anomalies
overSouthAsiabasedonthepredictedNiño3.4index.
1.5 ScopeandLimitations
1.5.1 Scope
• ThesystemwillforecastENSOstatesusingtheNiño3.4indexatleadtimesof6
months.
• RegionalimpactassessmentwillcovertheSouthAsiandomain(5◦N–40◦N,60◦E–
100◦E).
• The model will be trained on publicly available datasets including ERA5 and
ORAS5.
• The final deliverable will include a functional forecasting pipeline implemented
inPython.
1.5.2 Limitations
• The model is trained on historical data and may not capture unprecedented cli-
matedynamics.
• Thesystemdoesnotaccountforotherclimatedrivers(e.g.,IndianOceanDipole,
ArcticOscillation)thatmaymodulateENSOimpacts.
• Real-timeoperationaldeploymentisbeyondthescopeofthisproject.
1.6 ProjectApplications
• Agricultural Planning: The system provides early warning for monsoon de-
ficiency or excess rainfall, enabling farmers and agricultural agencies to guide
3

cropplanningandirrigationmanagement.
• Disaster Preparedness: The framework identifies regions at elevated flood or
drought risk, allowing disaster management authorities to undertake proactive
resourceallocation.
• Water Resource Management: Seasonal reservoir operation planning is sup-
ported through deterministic precipitation forecasts generated by the proposed
system.
• Climate Research: The project contributes to the broader scientific understand-
ingofENSOteleconnectionsandadvancesAI-basedclimatepredictionmethod-
ologies.
4

2. LITERATUREREVIEW
2.1 ENSOPrediction: FromDynamicalModelstoDeepLearning
The prediction of ENSO has been a central challenge in climate science for over four
decades. Early approaches relied on linear statistical models and dynamical ocean-
atmospherecoupledmodels[4]. WhiledynamicalmodelsbasedonGeneralCirculation
Models (GCMs) can capture the fundamental physics of ENSO, their predictive skill
drops sharply beyond 6–9 months due to the well-documented “spring predictability
barrier,”whereforecastinitializationerrorsgrowrapidlyduringborealspring[3].
The advent of machine learning and deep learning has opened new avenues for ENSO
prediction. Unlike dynamical models that solve physical equations numerically, data-
drivenapproacheslearncomplexnonlinearmappingsdirectlyfromobservationaldata,
potentiallycapturingdynamicsthatarepoorlyrepresentedinphysicalmodels.
2.2 DeepLearningApproachesforENSOForecasting
2.2.1 CNN-BasedMulti-YearENSOForecasting
Ham et al. [1] demonstrated a breakthrough in ENSO prediction by applying transfer
learning with CNNs to forecast the Niño 3.4 index at lead times exceeding 18 months.
Their approach addressed the fundamental challenge of limited observational data by
pre-training the CNN on historical simulations from CMIP5 climate models and fine-
tuning on reanalysis data (1871–1973 for training, 1984–2017 for testing). The CNN
architectureprocessesmonthlymapsofSST,oceanheatcontent(upper300m),andsea
level pressure as multi-channel inputs, producing Niño 3.4 index predictions at various
leadtimes.
The model achieved an all-season correlation skill of 0.65 at 17-month lead time, sub-
stantially outperforming both dynamical models and conventional statistical methods.
Importantly,theCNNsuccessfullypredictedtheonsetandamplitudeofElNiñoevents
that traditional models missed. Gradient-based saliency analysis revealed that the net-
5

work learned physically meaningful precursor patterns, including subsurface heat con-
tent anomalies in the western Pacific that serve as recharged energy for subsequent El
Niño development. However, the model showed reduced skill for La Niña events and
lackeduncertaintyquantification.
2.2.2 Autoencoder-LSTMHybridArchitecture
Dijkstraetal. [4]proposedahybriddeeplearningarchitecturecombiningautoencoders
with LSTM networks for ENSO forecasting. Their approach first employs a convolu-
tional autoencoder to compress high-dimensional SST fields into a low-dimensional
latent representation, effectively performing nonlinear dimensionality reduction. The
temporal evolution of these latent features is then modeled using an LSTM network,
whichcapturesthesequentialdependenciesinherentinENSOdynamics.
Theautoencoder-LSTMframeworkwastrainedonSODAreanalysisdataandevaluated
against persistence forecasts and linear inverse models. The model demonstrated com-
petitiveskillatleadtimesof6–12months,withtheautoencodersuccessfullycapturing
thedominantspatialpatternsofENSOvariability(comparabletotraditionalEOFanal-
ysis but with nonlinear generalization). The LSTM component effectively learned the
oscillatory nature of ENSO transitions between warm and cool phases. The authors
notedthatthelatentspacerepresentationprovidedphysicalinterpretability,withdiffer-
entlatentdimensionscorrespondingtorecognizableENSOprecursorpatterns.
2.2.3 Physics-GuidedDeepEchoStateNetworks
Mahesh et al. [3] introduced a physics-guided Deep Echo State Network (DESN) ap-
proachtoenhanceENSOpredictabilitylimits. EchoStateNetworks(ESNs)areavari-
ant of recurrent neural networks where the recurrent layer (reservoir) has fixed random
weights, and only the output layer is trained. This architecture offers computational
efficiency and avoids the vanishing gradient problem that plagues conventional RNNs
duringtraining.
The key innovation in their work is the incorporation of physics-based constraints into
the ESN framework. Specifically, they embed known ENSO dynamics, such as the
6

recharge-dischargeoscillatormechanismanddelayedoscillatortheory,asstructuralpri-
ors in the reservoir connectivity. The physics-guided DESN achieved prediction skill
comparable to deep learning baselines (correlation > 0.5 at 18-month lead) while re-
quiringsignificantlyfewertrainableparametersandprovidingenhancedinterpretability.
ThemodelwastrainedonERA5reanalysisdatawithSST,thermoclinedepth,andzonal
windstressasinputs. Anotableadvantageisthemodel’sabilitytoovercomethespring
predictabilitybarriermoreeffectivelythanpurelydata-drivenapproaches,attributedto
the physics constraints preventing the model from learning spurious correlations that
breakdownduringspringtransitions.
2.2.4 ResoNet: RobustandExplainableENSOForecasting
Yeetal. [2]developedResoNet,ahybriddeeplearningframeworkcombiningResidual
Network(ResNet)architecturewithphysicaloceandynamicsforrobustandexplainable
ENSO forecasting. The model integrates residual learning blocks that capture multi-
scale spatiotemporal features from SST, ocean heat content, and wind stress fields. A
key contribution is the attention mechanism that identifies physically meaningful pre-
cursorregions.
ResoNet achieved state-of-the-art performance with correlation skills exceeding 0.7 at
12-month lead time and maintained useful skill (> 0.5) beyond 18 months. The ex-
plainability module generates spatial attribution maps highlighting the oceanic regions
mostinfluentialforeachprediction, enhancingtrustinthemodel’soutputs. Themodel
demonstrated particular strength in predicting the diversity of El Niño events (Eastern
Pacificvs. CentralPacifictypes),acapabilitylackinginmanyearlierdeeplearningap-
proaches. Theframeworkalsoincorporatedensemblepredictiontoprovideuncertainty
estimates,generatingprobabilisticforecastsratherthansingledeterministicpredictions.
7

2.3 MachineLearningforRegionalClimateImpactAssessment
2.3.1 ENSO-DrivenWeatherPatternPrediction
Duttaetal. [5]investigatedtheapplicationofmachinelearningforpredictingseasonal
weatherpatternsfromENSOindices,bridgingthegapbetweenlarge-scaleENSOstate
prediction and regional climate impacts. Their study employed Random Forest, Gradi-
ent Boosting, and Support Vector Machine classifiers to predict seasonal precipitation
andtemperaturecategories(belownormal,normal,abovenormal)conditionedonmul-
tivariate ENSO indicators including Niño 3.4, Southern Oscillation Index (SOI), and
theMultivariateENSOIndex(MEI).
The results demonstrated that ensemble machine learning methods, particularly XG-
Boost and Random Forest, achieved classification accuracies of 70–82% for seasonal
precipitation categories when ENSO indices from 2–3 months prior were used as pre-
dictors. Feature importance analysis revealed that the combination of Niño 3.4 ampli-
tude and rate of change provided the strongest predictive signal for South Asian mon-
soon anomalies. However, the study was limited to categorical predictions and did not
providespatiallyresolvedforecasts.
2.4 ComparisonofExistingApproaches
Table 2-1 summarizes the key characteristics and performance metrics of the reviewed
ENSOforecastingapproaches.
Table2-1ComparisonofDeepLearningApproachesforENSOForecasting
| Study         | Method                       | MaxLead | Corr. Uncert. |
| ------------- | ---------------------------- | ------- | ------------- |
| Hametal.      | [1] CNN+TransferLearning     | 17mo.   | 0.65 No       |
| Dijkstraetal. | [4] Autoencoder+LSTM         | 12mo.   | 0.60 No       |
| Maheshetal.   | [3] Physics-guidedDESN       | 18mo.   | 0.50 No       |
| Yeetal.       | [2] ResoNet(ResNet+Physics)  | 18mo.   | 0.70 Yes      |
| Duttaetal.    | [5] XGBoost/RFClassification | 3mo.    | 0.82 No       |
8

3. METHODOLOGY
3.1 SystemOverview
Theproposedsystemfollowsaneight-stagepipelinearchitecture:
1. Dataingestion
2. Datapreprocessing
3. Featureengineering
4. Featureselection
5. Baselinemodeldevelopment
6. Deeplearningmodeltraining
7. Modelvalidation
8. 2026ENSOforecasting
Theframeworkingestsmulti-variableclimatereanalysisdataandproducesENSOstate
predictions at 6 month lead times along with regional precipitation and temperature
anomalyforecastsforSouthAsia.
9

3.2 SystemArchitecture
Figure3-1SystemArchitectureoftheproject
10

The system architecture comprises eight sequential stages: (1) Data Ingestion, where
multi-variable climate reanalysis datasets (ERA5, ORAS5, NOAA) are acquired from
public repositories; (2) Data Preprocessing, which includes spatial standardisation to
a uniform 2◦×2◦ grid, temporal alignment to the common period 1980–2025, clima-
tological anomaly computation, and min-max normalisation; (3) Feature Engineer-
ing, where 12-month sliding window input tensors of shape (12 × 30 × 100 × 4) are
constructed from the four spatial fields; (4) Feature Selection, applying Mutual In-
formation Feature Selection (MIFS) for the XGBoost baseline; (5) Baseline Model
Development, comprising the XGBoost baseline; (6) Deep Learning Model Train-
ing,comprisingtheCNN-TCNdeeplearningmodel;(7)ModelValidation,wherethe
modelsareevaluated;and(8)Forecasting,wherethetrainedmodelgenerates6-month
leadNiño3.4indexpredictionsandregionalSouthAsianprecipitationandtemperature
anomalyforecasts.
11

Figure3-2DataPreprocessingPipeline
12

The data preprocessing pipeline (Figure 3-2) transforms raw climate reanalysis fields
into analysis-ready inputs through four sequential operations: spatial standardisation
via area-weighted block averaging to a 2◦×2◦ grid, temporal alignment ensuring all
variablessharethe1980–2025monthlytimeaxis,climatologicalanomalycomputation
bysubtractingthelong-termmonthlymean,andzscorenormalizationtoscaleallfields
todatatohaveameanof0andastandarddeviationof1rangeforstablemodeltraining.
3.3 DataSources
Theprojectutilizesthefollowingpubliclyavailabledatasets:
3.3.1 Oceanic&AtmosphericReanalysis
• ERA5 Reanalysis (ECMWF): Monthly SST, sea level pressure (SLP), 2m tem-
perature(T2M),andwindstressfieldsat1°×1°resolution(1950–2025).
• ORAS5(ECMWF):Oceanheatcontentandsubsurfacetemperatureprofilesfor
thetropicalPacific.
• GODAS(NOAA/CPC):Oceantemperatureandcurrentfields.
3.3.2 ENSOIndices&ModelEnsembles
• Niño 3.4 Index (NOAA): Monthly SST anomaly in the Niño 3.4 region (5°N–
5°S,170°W–120°W),servingastheprimaryENSOindicator(1871–present).
• Multivariate ENSO Index (MEI): Composite ENSO indicator combining mul-
tipleatmosphericandoceanicvariables.
3.4 Workflow
Theprojectfollowsaeight-steppipeline:
13

3.4.1 DataIngestion
Alldatasetsareaggregated,resampledtoaconsistentmonthly2°×2°grid,andaligned
to the common period 1980–2025. Data quality checks and gap-filling procedures are
appliedtoensuretemporalcontinuity.
3.4.2 DataPreprocessing
The raw climate datasets ingested from ERA5, ORAS5, and NOAA undergo the fol-
lowingpreprocessingstepspriortofeatureengineering:
•SpatialStandardization
All datasets are ingested at their native 1◦ latitude × 1◦ longitude spatial resolution.
A 1◦ latitude × 1◦ longitude grid cell covers approximately 111 km (along latitude) ×
111 km (along longitude) at the equator, representing an area of roughly 12,300 km2.
At 27◦N (Nepal/India region), 1◦ longitude shrinks to approximately 99 km, making
each1◦ latitude×1◦ longitudecellapproximately111km×99km≈10,989km2. All
datasets are subsequently coarsened to a 2◦ latitude × 2◦ longitude grid using block
averaging(area-weightedmeanovereach2×2gridcellblock),whereeachcoarsened
cell covers approximately 222 km × 198 km ≈ 43,956 km2 at 27◦N, roughly 30%
of Nepal’s total land area. This spatial coarsening reduces the total data volume by
a factor of four while preserving the large-scale spatial patterns and teleconnection
signalscriticalforENSOdynamics,asthedominantmodesofinterannualSeaSurface
Temperature (SST) variability operate at spatial scales well above the 2◦ threshold.
ThecoarsenedfieldsarethenclippedtothetropicalPacific(30◦S–30◦N,120◦E–80◦W)
for ENSO modeling and to South Asia (5◦N–40◦N, 60◦E–100◦E) for regional impact
assessment.
•TemporalAlignment
All variables are resampled to a consistent monthly temporal resolution and aligned
to the common period 1980–2025, yielding 552 monthly time steps. ERA5 fields has
aggregated monthly means, and missing time steps are identified and flagged for gap-
filling.
14

•MissingValueHandling
Missing values arising from sensor gaps or reanalysis artifacts are handled through a
three-tierstrategyappliedsequentiallyacrossallingestedvariables.
TemporalGapFilling: Isolatedmissingmonths(≤2consecutivetimesteps)arefilled
usinglinearinterpolationbetweenadjacentvalidtimesteps,definedas:
t−t
X(t) = X(t )+ before ×[X(t )−X(t )] (3-1)
before after before
t −t
after before
where X(t) is the interpolated value at missing time step t, X(t ) is the observed
before
value at the last valid time step preceding the gap, X(t ) is the observed value at
after
the first valid time step following the gap, and t and t are the time indices of
before after
theboundingobservedvalues. Thisapproachisphysicallyjustifiedasoceanandatmo-
spheric variables evolve smoothly over monthly timescales, rendering a linear estimate
betweentwoadjacentmonthsanappropriateapproximation.
Spatial Gap Filling: Spatially missing grid cells are filled using the climatological
monthly mean computed over the full 1980–2025 period. For a given variable X at
gridcell(i,j),theclimatologicalmeanforcalendarmonthmisdefinedas:
1 (cid:88)
Nm
¯
X (i,j) = X (i,j) (3-2)
m y,m
N
m
y=1
¯
where X (i,j) is the climatological mean at grid cell (i,j) for calendar month m,
m
X (i,j) is the observed value at grid cell (i,j) for month m in year y, and N is
y,m m
the total number of valid observations available for month m across all years in the
1980–2025 period. This strategy is employed in place of temporal interpolation when
novalidneighboringtimestepsexistattheaffectedgridcell.
Variable Exclusion: Variables for which more than 5% of total data points, computed
across all time steps and grid cells, remain missing after the application of the above
fillingstrategiesareexcludedentirelyfromthefeatureset. Retentionofheavilyincom-
plete variables risks introducing spurious patterns into the training data, potentially
15

degradingmodelperformance.
•AnomalyComputation
Monthly climatological means are computed for each variable over the 1980–2010
baseline period and subtracted from the raw fields to obtain anomaly fields. This re-
movestheseasonalcycleandisolatesinterannualvariabilityrelevanttoENSOdynam-
ics:
X′(t) = X(t)−X ¯ (3-3)
m
where X′(t) is the anomaly at time t, X(t) is the raw value, and X ¯ is the climatolog-
m
icalmeanforcalendarmonthm.
•Normalization
All anomaly fields are standardized using Z-score normalization computed exclusively
onthetrainingset(1980–2018)topreventdataleakageintothevalidationandtestsets:
X′(t)−µ
X ˆ (t) = train (3-4)
σ
train
whereµ andσ arethemeanandstandarddeviationofthetrainingperiodanoma-
train train
lies.
•Niño3.4IndexComputation
The target variable for the El Niño–Southern Oscillation (ENSO) forecasting objec-
tiveistheNiño3.4index,definedasthearea-averagedSeaSurfaceTemperature(SST)
anomalyovertheNiño3.4region(5◦S–5◦N,170◦W–120◦W).Theindexiscomputeddi-
rectlyfromERA5SSTanomalyfields,whicharealreadyingestedaspartoftheprimary
datasetpipeline,therebyeliminatingtheneedforanadditionalobservationaldataset.
For each monthly time step t, the Niño 3.4 index is computed as the area-weighted
spatial mean of SST anomalies over all grid cells (i,j) falling within the Niño 3.4
domain:
(cid:80) w(i,j)·X′ (i,j,t)
i,j∈Ω SST
N (t) = (3-5)
3.4 (cid:80)
w(i,j)
i,j∈Ω
where N (t) is the Niño 3.4 index at time step t, X′ (i,j,t) is the SST anomaly at
3.4 SST
16

grid cell (i,j) and time step t computed following the anomaly procedure described in
Section 3.4.2, Ω denotes the set of all grid cells within the Niño 3.4 region (5◦S–5◦N,
170◦W–120◦W), and w(i,j) = cos(ϕ ) is the area weight assigned to grid cell (i,j),
i
whereϕ denotesthelatitudeofgridcelliinradians.
i
A 3-month running mean is subsequently applied to the raw monthly index values to
suppressintra-seasonalnoiseandretainonlythemulti-monthENSOsignal:
2
1 (cid:88)
ˆ
N (t) = N (t−k) (3-6)
3.4 3.4
3
k=0
ˆ
whereN (t)isthesmoothedNiño3.4indexattimesteptandN (t−k)denotesthe
3.4 3.4
raw index value at lag k months. The resulting smoothed time series serves as the pri-
mary prediction target for the CNN-TCN model in Objective 1 and as the conditioning
variablefortheregionalimpactassessmentmoduleinObjective2.
3.4.3 FeatureEngineering
Inputsequencesareconstructedconsistingof12consecutivemonthlysnapshotsoffour
spatial fields extracted over the tropical Pacific domain (30◦S–30◦N, 120◦E–80◦W),
whichrepresentstheprimaryregionofElNiño–SouthernOscillation(ENSO)variabil-
ity. Thefourspatialfieldsareasfollows:
• Sea Surface Temperature (SST) anomaly, sourced from the European Centre
forMedium-RangeWeatherForecastsReanalysisversion5(ERA5),representing
theprimaryindicatorofENSOstateattheoceansurface.
• Sea Level Pressure (SLP) anomaly, sourced from ERA5, capturing the atmo-
spheric pressure patterns associated with the Walker Circulation and trade wind
variability.
• Wind stress anomaly, sourced from ERA5, representing the strength and di-
rection of trade winds along the equatorial Pacific, which drive the east-west
redistributionofwarmsurfacewatercharacteristicofENSOevents.
17

• Upper 300m Ocean Heat Content (OHC) anomaly, sourced from the Ocean
Reanalysis System 5 (ORAS5), capturing subsurface thermal energy accumula-
tion in the tropical Pacific. Subsurface heat content is among the most skill-
ful predictors of ENSO at lead times of 6 months and beyond, as it reflects the
rechargeanddischargeofoceanicheatpriortothesurfaceexpressionofElNiño
orLaNiñaevents.
Eachinputsequenceisthereforerepresentedasafour-dimensionaltensorofshape:
X ∈ RT×N ϕ ×N λ ×C (3-7)
where T = 12 denotes the number of monthly time steps in each input window, N =
ϕ
30denotesthenumberoflatitudegridcells,N = 100denotesthenumberoflongitude
λ
grid cells, and C = 4 denotes the number of input channels corresponding to the four
spatial fields listed above. These sequences serve as the primary input to both the
ConvolutionalNeuralNetwork–TemporalConvolutionalNetwork(CNN-TCN)model
and the XGBoost baseline model, with the latter receiving the tensor flattened into a
one-dimensionalfeaturevectoroflength12×30×100×4 = 144,000priortoMIFS-
baseddimensionalityreduction.
3.4.4 FeatureSelection
Given the high dimensionality of the flattened input (12 × 30 × 100 × 4 = 144,000
features per sample), a feature selection step is applied prior to training the XGBoost
baselinemodeltoidentifyandretainonlythosespatialgridcellsthatcarrystatistically
significant predictive information for El Niño–Southern Oscillation (ENSO) forecast-
ing. Mutual Information Feature Selection (MIFS) is employed as the feature selec-
tion criterion, operating directly on the continuous Niño 3.4 index values as the target
variable. This approach preserves the full information content of the target without
requiring discretisation into categorical phase labels, and captures non-linear statisti-
cal dependencies between input features and the prediction target that linear methods
cannotdetect.
18

The CNN-TCN model does not require an explicit feature selection step. The convolu-
tional layers of the spatial encoder perform implicit feature selection through learned
spatial filters during end-to-end training, automatically identifying and weighting re-
gionsofthetropicalPacificmostrelevanttoENSOprediction. Thefullunfilteredinput
tensorofshape(12×30×100×4)isthereforefeddirectlyintotheCNN-TCNmodel
withoutanypriordimensionalityreduction.
Mutual Information: For each grid cell (i,j) of each input variable, the mutual in-
formation between the grid cell values and the continuous Niño 3.4 target is computed
as:
(cid:88) (cid:88) p(x,y)
I(X ;Y) = p(x,y)log (3-8)
i,j
p(x)p(y)
x∈Xi,j y∈Y
whereX denotesthevaluesofthefeatureatgridcell(i,j)acrossalltrainingsamples,
i,j
Y denotes the corresponding continuous Niño 3.4 index values, p(x,y) is the joint
probability distribution of X and Y, and p(x) and p(y) are the respective marginal
i,j
probability distributions. A high value of I(X ;Y) indicates that knowing the value
i,j
at grid cell (i,j) substantially reduces uncertainty about the Niño 3.4 index, implying
strongpredictiverelevanceforENSOforecasting.
Redundancy Penalization: MIFS explicitly penalizes redundancy between candidate
and already selected features to ensure the retained feature set is both maximally infor-
mative and minimally redundant. For a candidate feature X and the set of already
i,j
selectedfeaturesS,theredundancy-penalizedrelevancescoreisdefinedas:
1 (cid:88)
S(X ) = I(X ;Y)− I(X ;X ) (3-9)
i,j i,j i,j k,l
|S|
X ∈S
k,l
whereI(X ;Y)isthemutualinformationbetweenthecandidatefeatureandtheNiño
i,j
3.4 target, I(X ;X ) is the mutual information between the candidate feature and
i,j k,l
eachalreadyselectedfeatureX ∈ S,and|S|isthecurrentsizeoftheselectedfeature
k,l
set. Ateachiteration,thecandidatefeaturewiththehighestscoreS(X )isaddedtoS.
i,j
19

Thisiterativeprocedureensuresthateachnewlyselectedfeaturecontributesmaximum
new predictive information about the Niño 3.4 target beyond what the already selected
featurescollectivelyprovide.
PhysicalInterpretation: Thespatialregionsexpectedtoachievethehighestmutualin-
formationscoreswiththeNiño3.4targetareconsistentwithestablishedENSOphysics.
Sea Surface Temperature (SST) anomalies in the central and eastern equatorial Pa-
cific,equatorialwindstressanomaliesalongtheWalkerCirculationpathway,andupper
300mOceanHeatContent(OHC)anomaliesinthewesternPacificthermoclineareex-
pectedtocarrythestrongestpredictivesignal. Theredundancypenalizationstepfurther
ensures that highly correlated neighbouring grid cells, which carry largely overlapping
information, do not dominate the selected feature set, yielding a spatially diverse and
physicallymeaningfulsetofpredictors.
Selection Criterion: Features are selected iteratively in descending order of their
redundancy-penalized score S(X ) until a predetermined number of features K is
i,j
retained. The value of K is determined through cross-validation on the validation set
(2019–2022)byevaluatingXGBoostperformanceacrossarangeoffeaturesetsizesand
selectingthevalueofK thatminimisesvalidationRootMeanSquareError(RMSE).
Application: The MIFS procedure is applied exclusively to the XGBoost baseline
model. The selected K features are extracted from the flattened one-dimensional in-
put vector and fed directly into XGBoost for training and inference. The CNN-TCN
modelreceivesthefullunfilteredinputtensorofshape(12×30×100×4)withoutany
prior feature selection, as the spatial encoder handles all feature extraction implicitly
throughlearnedconvolutionalfilters.
3.4.5 BaselineModel
An eXtreme Gradient Boosting (XGBoost) regression model is employed as the base-
linetoestablishaperformanceflooragainstwhichtheproposedCNN-TCNisevaluated.
XGBoost operates on tabular input and therefore receives the MIFS-selected features
as a flattened one-dimensional feature vector per sample, discarding the spatial and
temporalstructurepreservedintheCNN-TCNinputtensor.
20

The model is trained on the training period (1980–2018) to predict the Niño 3.4 index
at a 6-month lead time. Hyperparameters including the number of estimators, maxi-
mum tree depth, and learning rate are tuned on the validation set (2019–2022) using
grid search. Final performance is reported on the held-out test set (2023–2025) using
the Anomaly Correlation Coefficient (ACC) and Root Mean Square Error (RMSE) as
evaluationmetrics,consistentwiththoseappliedtotheCNN-TCNmodel.
• Established Benchmark: XGBoost has been employed in prior ENSO and cli-
mateimpactstudies.Itthereforeservesasacredibleandmeaningfulpointofcom-
parison.
• Contrast with Spatiotemporal Learning: XGBoost operates on flattened tabu-
lar input and cannot exploit the spatial structure of climate fields or the temporal
evolution of monthly snapshots. This contrast directly demonstrates the value of
theCNN-TCNarchitecture.
• Standard Practice: In deep learning research, it is expected to benchmark pro-
posed models against a strong classical machine learning baseline. XGBoost is
widelyacceptedasthestandardchoiceforthisroleinrecentMLliterature.
• Computational Efficiency: XGBoost trains rapidly on CPU resources, serving
as a quick sanity check that the data pipeline and preprocessing steps are func-
tioningcorrectlybeforecommittingGPUresourcestoCNN-TCNtraining.
3.4.6 DeepLearningModel
The primary forecasting model is a multi-task hybrid Convolutional Neural Network
– Temporal Convolutional Network (CNN-TCN) that simultaneously addresses both
project objectives through a shared spatiotemporal encoder and two task-specific pre-
diction heads. The hybrid architecture is motivated by the dual nature of El Niño–
Southern Oscillation (ENSO) variability, ENSO manifests as both a spatial pattern
(east-west Sea Surface Temperature (SST) gradient across the tropical Pacific) and a
temporal evolution (progressive warming or cooling over multiple months) requiring
dedicatedcomponentstocaptureeachdimensionindependentlybeforemodellingtheir
jointevolution.
21

Stage1: SpatialEncoder(CNN):Atwo-dimensionalConvolutionalNeuralNetwork
(CNN) is applied independently to each monthly snapshot within the 12-month input
window. Foreachtimestept,theCNNreceivesaspatialmapofshape(N ×N ×C)
ϕ λ
whereN = 30denotesthenumberoflatitudegridcells,N = 100denotesthenumber
ϕ λ
oflongitudegridcells,andC = 4denotesthenumberofinputchannels(SSTanomaly,
Sea Level Pressure (SLP) anomaly, wind stress anomaly, and upper 300m Ocean Heat
Content (OHC) anomaly). The CNN applies learned convolutional filters across the
spatialdimensionstoextractphysicallymeaningfulspatialfeatures.
The output of the spatial encoder for each time step t is a compact spatial feature vec-
tor f(t) ∈ Rd, where d denotes the number of learned spatial features. Applying the
CNNindependentlyandidenticallyacrossall12timestepsyieldsasequenceofspatial
featurevectors:
F = [f(1),f(2),...,f(12)] ∈ R12×d (3-10)
whereeachvectorf(t)summarisesthespatialstateofthetropicalPacificatmontht.
Stage 2 : Temporal Encoder (TCN): The sequence of spatial feature vectors F is
passed into a Temporal Convolutional Network (TCN) consisting of stacked dilated
causal convolutional layers that model the temporal evolution of the extracted spatial
patterns across the 12-month input window. Causal convolutions ensure that the repre-
sentationatanytimestepisconditionedonlyonpastandpresentobservations,prevent-
ing information leakage from future time steps. Dilation is applied with exponentially
increasingratesacrosssuccessivelayers:
d = 2l−1, l = 1,2,3,...,L (3-11)
l
whered isthedilationrateatlayerl andListhetotalnumberofdilatedconvolutional
l
layers. Thisexponentialdilationallowsthereceptivefieldofthenetworktogrowexpo-
nentiallywithdepth,enablingthetemporalencodertocapturebothshort-rangemonthly
patterns(1–2months)andlong-rangemulti-monthdependencies(8–12months)within
22

theinputwindow. TheTCNcomponentcapturestemporalevolutionpatternssuchas:
• Progressiveeastwardmigrationofthewarmwaterpooloverconsecutivemonths.
• Gradualweakeningorstrengtheningoftradewindsacrossthe12-monthwindow.
• Subsurface heat content buildup preceding the surface expression of El Niño or
LaNiñaevents.
The output of the temporal encoder is a single compact encoded representation vector
z ∈ Rh summarising all spatiotemporal patterns extracted from the 12-month input
sequence,wherehdenotesthehiddendimensionofthetemporalencoder.
Stage 3 : Task-Specific Prediction Heads: The encoded representation z is passed
simultaneouslyintotwoindependentfullyconnectedpredictionheads:
ENSOPredictorHead(Objective1): Aseriesoffullyconnectedlayersregressesthe
Niño3.4indexata6-monthleadtime,producingasingledeterministicscalaroutput:
yˆ = f (z) ∈ R (3-12)
ENSO ENSO
South Asia Impact Head (Objective 2): A separate series of fully connected layers
produces deterministic forecasts of June–July–August–September (JJAS) precipitation
andtemperatureanomaliesovertheSouthAsiandomain(5◦N–40◦N,60◦E–100◦E):
yˆ = f (z) ∈ R2×G (3-13)
impact impact
where G = 340 denotes the total number of South Asian grid cells, and the factor
of 2 accounts for the two predicted variables (precipitation anomaly and temperature
anomaly).
Multi-Task Training Objective: Both prediction heads are trained simultaneously
throughacombinedweightedlossfunction:
23

|     | L = α·L | +(1−α)·L |        |     | (3-14) |
| --- | ------- | -------- | ------ | --- | ------ |
|     | total   | ENSO     | impact |     |        |
whereL istheMeanSquaredError(MSE)lossbetweenthepredictedandobserved
ENSO
Niño3.4indexvalues,L istheMSElossbetweenthepredictedandobservedSouth
impact
Asian precipitation and temperature anomalies, and α is a weighting coefficient tuned
on the validation set (2019–2022) to balance the contributions of both tasks. Gradi-
ents from both loss terms propagate back through the shared spatiotemporal encoder
simultaneously, enabling the encoder to learn representations that are jointly useful for
ENSOstateforecastingandSouthAsianregionalimpactassessment.
ThecompleteCNN-TCNarchitectureissummarisedasfollows:

yˆ ∈ R
|               | C N N        | T C N |                                        | ENSO   |     |
| ------------- | ------------ | ----- | -------------------------------------- | ------ | --- |
| (12×30×100×4) | − − → (12×d) | − − → | z ∈ Rh →−                              |        |     |
|               |              |       | (cid:124) (cid:123)(cid:122) (cid:125) | R2×340 |     |
(cid:124) (cid:123)(cid:122) (cid:125) (cid:124) (cid:123)(cid:122) (cid:125) yˆ ∈
| inputtensor | spatialfeatures | encodedrepresentation |     | impact |     |
| ----------- | --------------- | --------------------- | --- | ------ | --- |
(3-15)
3.4.7 ValidationandEvaluation
A chronological train–validate–test split is employed to evaluate model performance,
ensuringthatnofutureinformationisaccessibleduringtrainingorvalidation:
| • Training: 1980–2018   |     |     |     |     |     |
| ----------------------- | --- | --- | --- | --- | --- |
| • Validation: 2019–2022 |     |     |     |     |     |
| • Testing: 2023–2025    |     |     |     |     |     |
HyperparametersforboththeConvolutionalNeuralNetwork–TemporalConvolutional
Network(CNN-TCN)andtheXGBoostbaselinearetunedexclusivelyonthevalidation
set. Finalperformanceforbothmodelsisreportedontheheld-outtestset(2023–2025).
24

ENSOForecastEvaluation: TheperformanceoftheNiño3.4indexforecastisevalu-
atedusingthefollowingmetrics:
Anomaly Correlation Coefficient (ACC): The ACC measures the temporal correla-
tionbetweenpredictedandobservedNiño3.4anomaliesoverthetestperiod:
(cid:80)N
yˆ(t)·y(t)
ACC = t=1 (3-16)
(cid:113) (cid:113)
(cid:80)N
yˆ(t)2 ·
(cid:80)N
y(t)2
t=1 t=1
where yˆ(t) is the predicted Niño 3.4 index at time step t, y(t) is the corresponding
observed value, and N is the total number of test time steps. An ACC value of 1.0
indicates perfect prediction skill, while a value of 0.0 indicates no skill. A model is
generallyconsideredskillfulwhenACC> 0.5.
Root Mean Square Error (RMSE): The RMSE quantifies the average magnitude of
predictionerrorsinthesameunitsastheNiño3.4index(◦C):
(cid:118)
(cid:117) N
(cid:117) 1 (cid:88)
RMSE = (cid:116) (yˆ(t)−y(t))2 (3-17)
N
t=1
where a lower RMSE indicates better forecast accuracy. Both ACC and RMSE are
reportedfortheCNN-TCNandXGBoostmodelstoenabledirectcomparison.
Impact Forecast Evaluation: The performance of the South Asia precipitation and
temperature anomaly forecasts is evaluated against ERA5 reanalysis data using the fol-
lowingmetrics:
Mean Absolute Error (MAE): The MAE measures the average absolute deviation
between predicted and observed anomaly values across all South Asian grid cells and
testtimesteps:
25

N G
1 (cid:88)(cid:88)
|     | MAE = |     | |yˆ(g,t)−y(g,t)| |     | (3-18) |
| --- | ----- | --- | ---------------- | --- | ------ |
N ·G
t=1 g=1
where yˆ(g,t) is the predicted anomaly at grid cell g and time step t, y(g,t) is the
corresponding observed anomaly, G = 340 is the total number of South Asian grid
cells,andN isthenumberoftesttimesteps.
Pearson Correlation Coefficient: The Pearson correlation coefficient measures the
linear association between predicted and observed anomaly fields across all grid cells
andtesttimesteps:
|           | (cid:80)N (cid:80)G | (cid:0)   | ¯ (cid:1)          |              |        |
| --------- | ------------------- | --------- | ------------------ | ------------ | ------ |
|           |                     | yˆ(g,t)−y | ˆ                  | (y(g,t)−y¯)  |        |
| r =       | t=1                 | g=1       |                    |              |        |
| (cid:113) |                     |           | (cid:113)          |              | (3-19) |
| (cid:80)N | (cid:80)G (cid:0)   |           | (cid:1)2 (cid:80)N | (cid:80)G    |        |
|           | yˆ(g,t)−y           |           | ¯ ˆ ·              | (y(g,t)−y¯)2 |        |
| t=1       | g=1                 |           |                    | t=1 g=1      |        |
¯
whereyˆandy¯arethemeanpredictedandobservedanomalyvaluesrespectivelyacross
all grid cells and test time steps. A Pearson correlation value closer to 1.0 indicates
strongeragreementbetweenthepredictedandobservedspatialanomalypatterns.
3.4.8 2026Forecasting
Generate ensemble forecasts for the Niño 3.4 index and June-July-August-September
2026precipitationandtemperatureanomaliesoverSouthAsiausingthetrainedmodel
withthemostrecentavailableobservationaldataasinput.
3.5 Tools&Environment
• Language: Python3.8+
• Libraries: NumPy,Xarray,Pandas,Matplotlib,Scikit-learn;PyTorchorTensor-
Flow.
• Hardware: GPU(GoogleColabPro,AWSEC2,orlocal)recommendedfordeep
learningtraining.
26

4. EXPECTEDOUTCOMES
Uponcompletion,thisprojectisexpectedtodeliverthefollowingoutcomes:
1. FunctionalDeepLearningPipeline: AcompletePython-basedforecastingsys-
temcapableofingestingreanalysisdataandproducingENSOstatepredictionsat
6monthleadtimes,madeavailableasanopen-sourceGitHubrepository.
2. ENSO Forecast Performance: The CNN-TCN model aims to achieve competi-
tiveforecastingskillrelativetobaselinemachinelearningapproaches.
3. June-July-August-September2026PrecipitationandTemperatureAnomaly
Forecasts: ForecastsofJune-July-August-September2026precipitationandtem-
peratureanomaliesoverSouthAsiaconditionedonthepredictedElNiñostate.
4. Comparative Analysis: A systematic comparison of the proposed CNN-TCN
approach against baseline method (XGBoost) and existing deep learning archi-
tectures,demonstratingthevalueofthechosenarchitectureandmulti-tasklearn-
ing.
27

5. PROJECTSCHEDULE
The project is scheduled over a period of approximately 14 weeks. The Gantt chart be-
lowillustratestheplannedtimelineandsequencingofmajortasks,includingdeliberate
overlapstomaximizeefficiency.
Figure5-1GanttChartoftheProjectSchedule
28

6. PROJECTBUDGET
Theprojectreliesentirelyonopen-sourcesoftwareandpubliclyavailableclimatedatasets,
keepingtheoverallbudgetminimal. Table6-1summarizestheestimatedcosts.
Table6-1EstimatedProjectBudget
S.N. Item EstimatedCost(NPR)
1 GoogleColabPro(3months) 8,000
2 Cloudstorage/backup(GoogleDrive) 500
3 Reportprintingandbinding 1,500
4 Stationeryandmiscellaneous 500
Total 10,500
All datasets (ERA5, ORAS5, GODAS) are freely accessible through their respective
portals for academic use. All software tools including Python, PyTorch, Xarray, and
Scikit-learnareopen-sourcewithnolicensingcosts.
29

7. FEASIBILITYANALYSIS
7.1 TechnicalFeasibility
Theprojectistechnicallyfeasiblegiventheavailabilityofallrequireddatasetsthrough
publicrepositories(ECMWFCDS,NOAA,ESGF).TheConvolutionalNeuralNetwork
– Temporal Convolutional Network (CNN-TCN) architecture is well-supported in Py-
Torch, and the 45-year dataset (1980–2025) provides sufficient temporal coverage for
robust model training. The six-stage pipeline (data ingestion, preprocessing, feature
engineering, MIFS feature selection, model development, and forecasting) is modular
andimplementablewithintheteam’sexistingPythonandmachinelearningskillset.
7.2 EconomicFeasibility
The project incurs near-zero data and software costs, as all datasets and development
tools are freely available. The primary expenditure is cloud compute (Google Colab
Pro)forCNN-TCNtrainingonERA5andORAS5reanalysisfields. Thetotalestimated
budget of NPR 10,500 is well within the means of a student project group, with no
specializedhardwareprocurementrequired.
7.3 TimeFeasibility
Based on the project schedule spanning June 1 to September 7, 2025 (approximately
14 weeks), the timeline is realistic and well-structured. The work begins with liter-
ature review, data collection, and preprocessing in the first four weeks, followed by
feature engineering, MIFS feature selection, and parallel development of the XGBoost
baseline and CNN-TCN model through mid-August. Model training, hyperparameter
tuning, forecast generation, and evaluation are conducted in the final weeks, with doc-
umentation and report writing running concurrently from mid-August through project
completion. Tasksaresequencedwithdeliberateoverlaps(forinstance,XGBoostbase-
lineandCNN-TCNdevelopmentproceedinparallel)tomaximizeefficiencywithinthe
14-weekwindow.
30

7.4 OperationalFeasibility
The project delivers a self-contained Python-based forecasting pipeline publishable as
an open-source GitHub repository. The modular architecture ensures each stage can
be independently tested and debugged. All four team members bring complementary
skillsinmachinelearning,datapreprocessing,andclimatedatahandling,ensuringbal-
ancedworkloaddistribution. Thefinaloutputs(ENSOstateforecastsanddeterministic
South Asian precipitation and temperature anomaly maps for JJAS 2026) are directly
interpretableandactionableforthetargetacademicaudience.
31

REFERENCES
[1] Y.-G. Ham, J.-H. Kim, and J.-J. Luo, “Deep learning for multi-year ENSO fore-
casts,”Nature,vol.573,no.7775,pp.568–572,2019.
[2] F.Ye,J.Hu,T.Liu,andB.Wu,“ResoNet: RobustandexplainableENSOforecasts
withhybridconvolutionandtransformer,”AdvancesinAtmosphericSciences,vol.
41,pp.1–16,2024.
[3] A. Mahesh, M. Evans, G. Weiss, and J. Brajard, “Enhancing the predictability
limitsofENSOwithphysics-guideddeepechostatenetworks,”2024.
[4] H. A. Dijkstra, P. Petersik, E. Hernández-García, and C. López, “Deep learning
withautoencodersandLSTMforENSOforecasting,”ClimateDynamics,vol.53,
pp.5765–5784,2019.
[5] R. Dutta, R. Maity, and A. Sharma, “Seasonal weather pattern prediction from
ENSO indices using machine learning,” Journal of Hydrometeorology, vol. 25,
no.3,pp.421–438,2024.
[6] NOAA Climate Prediction Center, “ENSO Diagnostic Discussion,” 2026. [On-
line]. Available: https://www.cpc.ncep.noaa.gov/products/analysis_
monitoring/enso_advisory/
[7] D.S.Pai,L.Sridhar,M.Rajeevan,O.P.Sreejith,N.S.Satbut,andB.Mukhopad-
hyay, “Development of a new high spatial resolution (0.25◦ × 0.25◦) long period
(1901–2010) daily gridded rainfall data set over India and its comparison with
existingdatasetsovertheregion,”Mausam,vol.65,no.1,pp.1–18,2014.
[8] S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolu-
tionalandrecurrentnetworksforsequencemodeling,”2018.
[9] J. Yan, L. Mu, L. Wang, R. Ranjan, and A. Y. Zomaya, “Temporal convolutional
networksfortheadvancepredictionofENSO,”ScientificReports,vol.10,p.8055,
2020.
[10] C. Rui, Z. Sun, W. Zhang, A.-A. Liu, and Z. Wei, “Enhancing ENSO predictions
with self-attention ConvLSTM and temporal embeddings,” Frontiers in Marine
Science,vol.11,p.1334210,2024.
32

[11] Y. Chen, Y. Jin, Z. Liu, et al., “Combined dynamical–deep learning ENSO fore-
casts,”NatureCommunications,vol.16,p.3845,2025.
[12] T. Lian et al., “A deep learning-based long-term ENSO forecasting model: 3D-
STransformer,” Journal of Geophysical Research: Machine Learning and Com-
putation,2025.
[13] Q. Chen, Y. Cui, G. Hong, et al., “Toward long-range ENSO prediction with an
explainable deep learning model,” npj Climate and Atmospheric Science, vol. 8,
p.259,2025.
[14] W. Xu et al., “Diffusion models bridge deep learning and physics in ENSO fore-
casting,”2025.
[15] P. D. Nooteboom et al., “Artificial intelligence predicts normal summer monsoon
rainfallforIndiain2023,”ScientificReports,2024.
33
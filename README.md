# MultimodalFloodRiskAnalysis

This project won 1st place in Sustainability at CruzHacks 2026. https://devpost.com/software/flood-risk-analysis

## Inspiration
Flooding is one of the most prevalent and deadly natural disasters and is only becoming increasingly common with record rainfalls brought by climate change. Fooding has cost the United States 180 billion per year and has directly caused thousands of death since 2000 (1) (2). Continued urban development is unsafe, unsustainable, and incredibly costly. Current flood risk analysis is often constrained to flood plains, relies on anecdotal evidence, and can be difficult to access making it unreliable to use and just not good for life threatening situations (3). Our project, Flood Risk Analysis, provides a data driven and accessible approach of analyzing flood risk in areas allowing for smarter, more sustainable, and safer building and living choices.

## What it does
Our project leverages 2 deep learning models to accurately identify flood risk across an area. The first model is a lightweight UNET to classify land usage (buildings, water, vegetation, crops, etc) from satellite imagery achieving over 70% accuracy. The second model is a UNET inspired encoder decoder architecture that considers factors such as historical rainfall, land use classification, and elevation maps to assess flood risk achieving over 90% accuracy.

Our platform allows for 2 separate ways to use the model. First, there is a RESTFul API built on FastAPI so our models can be directly integrated into other people's applications. Secondly, we have a web portal which allows users to start projects and upload satellite imagery and depth maps to get flood risk assessment. There is a separate portal for both consumers and enterprise users, where consumers have access to a chatbot built on the Gemini API to learn more about flood risk and enterprise users have the ability to annotate potential building zones or high risk areas.

## How we built it
Our models and platform was built in 3 separate parts: developing the dataset, developing the models, and developing the user interface.

Datasets for flood risk were not readily available because they were often incredibly low resolution, hard to access, and were not built on objective measures and instead reliant on community input. To create our dataset, we pulled elevation mapping data, high resolution satellite imagery, historical rainfall, and historical flood records. Our dataset was built on the Sacramento-San Joaquin Delta because of its ecological and elevation diversity, allowing us to create a representative and diverse dataset that would prevent overfitting while still remaining manageable. Flood risk was then determined using industry standard techniques by using historical flooding records, locating lower vs higher ground, and surface permeability (4). For land use classification, we used the FLAIR HUB Toy dataset which gave us 19 separate classes.

Our Land Classification model was then developed as a 7.8m parameter UNET and was trained on the FLAIR HUB dataset and achieved over 70% accuracy. Then, our risk analysis model was developed as a multimodal encoder decoder architecture and trained on our custom dataset as well as land classification from our land classification model, achieving over 90% accuracy.

To keep our models as useful as possible, we built it as a RESTFUl API service. We built a FastAPI service that would allow users to easily send GET requests with their satellite imagery and depth maps and get risk analysis as a result. However, most consumers and many enterprise users may be unwilling to develop their own application to access our models so we created a NextJS dashboard with Auth0 and MongoDB to keep the models easily accessible. We also implemented tools such as a chatbot built on the Gemini API to allow users to learn more about flood risk and annotation tools to better understand flood risk.

## Challenges we ran into
The largest challenge we ran into was the dataset creation. As there was no readily available dataset, we had to parse over 5GB of data and create our own custom dataset. High resolution satellite imagery also had many rate limits for the free plan, and it was difficult to acquire that data. Additionally, satellite imagery, flood risk, and elevation all used different coordinate systems and it was very difficult to accurately merge the data.

We also struggled a lot with having our models converge. The land classification has 19 classes, and on a small subset of the dataset, the model struggled to accurately classify pixels before extensive hyperparameter tuning. Additionally, despite the large size of the dataset, the flood risk model struggled to generalize and quickly overfit due to the dataset having few areas with higher flood risk. This was ultimately solved by augmenting data to create a more balanced dataset as well as hyperparameter tuning.  

## Accomplishments that we're proud of
We are incredibly proud of successfully creating and shipping a product that includes 2 custom machine learning models, an API, and a full stack application with authentication and database usage. Having heavily time dependent steps such as model training slowed development drastically and made it difficult to complete on time.

## What we learned
Before this product, we had never used the Gemini API, MongoDB, Auth0, or NextJS. Throughout the creation of this project, we had to learn quickly to integrate all these tools into our project. Additionally, a lot was learned about developing more robust machine learning models by augmenting datasets and tuning. With many datasets having extensive bias towards some classes, it is important that we are able to still utilize them.

## What's next for Flood Risk Analysis
The next steps for Flood Risk Analysis involve further fleshing out the consumer and enterprise portals to have more tools, collaboration, and other useful features. Additionally, Flood Risk Analysis still needs to be deployed on a cloud service like AWS to allow people to access it more easily.

## Sources
1. https://www.jec.senate.gov/public/index.cfm/democrats/2024/6/flooding-costs-the-u-s-between-179-8-and-496-0-billion-each-year
2. https://www.ketv.com/article/get-the-facts-deadliest-floods-in-the-us/65321053
3. https://www.floods.org/news-views/research-and-reports/the-us-is-finally-curbing-floodplain-development-research-shows/
4. https://www.fema.gov/flood-maps


# Technical Description

## Machine Learning Architecture

The system uses two custom and specialized deep learning models working together to assess flood risk.

### Land Classification Model
A lightweight U-Net (7.8M parameters) performs semantic segmentation on satellite imagery. It takes 4-channel aerial images (red, green, blue, and near-infrared bands) and classifies each pixel into one of 15 land cover types: buildings, water, vegetation, bare soil, agricultural land, vineyards, and more. This model was trained on the FLAIR-HUB dataset and provides detailed land use context that influences flood behavior. The model achieves over 70% accuracy.  

### Flood Risk Prediction Model
A U-Net inspired encoder decoder network predicts flood risk scores (0-100) for each pixel. This model accepts classified satellite imagery from the land classification model combined with elevation data and outputs a continuous risk score. The architecture uses skip connections to preserve fine spatial details while the encoder captures broader geographic patterns. This model achieves over 90% accuracy.

## Data Processing Pipeline
The system combines multiple data sources to create comprehensive training labels:

1. Historical Flood Events: A database of 672,423 historical flood events is indexed using a KD tree. Each location's proximity to past flooding is weighted by severity and distance using a Gaussian kernel.

2. Elevation Analysis: Digital elevation maps identify low areas most susceptible to flooding.

3. Land Cover Classification: Different surface types have different water absorption properties. Water bodies are high risk, impervious surfaces moderate risk, and vegetation is low risk.

4. Precipitation Factors: Seasonal rainfall patterns influence flood likelihood

These four components are combined with weighted percentages (Proximity 45%, Elevation 30%, Land Cover 15%, Precipitation 10%) to produce the final 0-100 risk score for each location.

## Backend Architecture

The FastAPI backend provides a RESTful API that orchestrates both ML models and manages analysis results:
- Analysis Endpoint: Accepts satellite images with elevation maps, splits them into overlapping tiles (default 128×128 pixels with 50% overlap), and runs both models on each tile
- Visualization Endpoints: Generate PNG outputs showing the original image, flood risk heatmap, and land classification map side-by-side
- Result Caching: Stores analysis results in memory for projects that may require reuse of the data
- Statistics Aggregation: Computes risk breakdowns by land class and spatial distribution of risk levels

The API is also designed to be easily accessed so our foundation models can be used in applications that do not use our GUI.

## Frontend Application

A NextJS web application provides an interactive interface for flood risk analysis. Users upload satellite imagery with elevation data, submit analysis requests, and visualize results through an intuitive dashboard. The frontend displays the flood risk assessment with a color-coded heatmap and overlays the land classification to help users understand which landscape features contribute to risk. Consumer users have the option to chat with a Gemini powered chatbot that can answer questions about sustainability and flood risk. Enterprise customers can annotate data and understand flood risks in different areas better.

## Data Processing Systems

### Tile-Based Processing
Both training and inference use a tiling strategy to handle large geographic areas efficiently:
- Training tiles are 128×128 pixels with 64-pixel stride for spatial robustness
- Inference uses overlapping tiles to smooth boundaries and improve prediction continuity

### Dataset Creation
A comprehensive pipeline generates training data from raw geographic inputs:
- Historical flood events are converted to spatial risk maps via weighting from proximity and severity
- Elevation data is normalized to 0-100 scale based on regional characteristics
- RGB satellite imagery is paired with elevation and land cover labels
- The final dataset contains nearly 60 million data points with risk values

### Multi-Modal Integration
The system treats RGB and elevation as separate channels in a unified 4-channel representation. This allows the neural network to learn how surface appearance (vegetation, buildings, water) and topography jointly predict flood risk, rather than treating them as independent factors.

# Tested On
MacOS Arm Tahoe 26.1
Created for CruzHacks 2026 in 36 hours

# Datasets Used

The [FLAIR-HUB toy dataset](https://huggingface.co/datasets/IGNF/FLAIR-HUB) was used to train the land classification segmentation model. This dataset contains 250 image-label pairs of 4-channel aerial imagery with pixel-level land cover annotations.

Historical flood event data for the SF Bay Area, combined with USGS elevation maps and NOAA precipitation records, provides the ground truth for training the flood risk prediction model. The dataset creation process synthesizes these sources into a single 0-100 risk score per location, enabling supervised learning on the multimodal prediction task.

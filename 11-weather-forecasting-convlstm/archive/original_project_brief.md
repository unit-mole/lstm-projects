I am continuing my GitHub portfolio improvement work.

Previously, I completed my **ANN / Deep Learning portfolio series** and my **Simple RNN portfolio series**. Now I am working on my **LSTM / Long Short-Term Memory project series**.

I currently work as a **Quality Data Scientist**, and I want to improve my GitHub profile so that it looks professional for future roles in:

* Data Science
* Machine Learning
* AI / Applied AI
* Data Analytics
* Business Intelligence
* Quality Analytics
* Analytics Engineering

I have now created / will be creating a new GitHub repository for my LSTM projects named:

```text
lstm-projects
```

This repository will be similar to my earlier repositories:

```text
ann-deep-learning-projects
simple-rnn-projects
lstm-projects
```

The goal is to build a professional LSTM portfolio repository containing multiple LSTM-based projects with clean code, strong README files, model evaluation, deployment-ready Streamlit apps, and live demo links.

Earlier, I worked on LSTM projects such as:

```text
Airline Passenger Forecasting
Bitcoin Price Prediction
Conversational Chatbot using Seq2Seq with Attention
ECG Anomaly Detection using LSTM Autoencoder with Attention
Fake News Detection
Human Activity Recognition using LSTM with Attention
Industrial Equipment Failure Detection using LSTM Autoencoder
Multivariate Time Series Forecasting using Stacked LSTM
Video Frame Prediction using Convolutional LSTM
Traffic Flow Prediction using Stacked LSTM
```

Now I am moving to my next LSTM project:

```text
Weather Forecasting using ConvLSTM
```

I have attached the project files/code files for this Weather Forecasting using ConvLSTM project.

Please review the attached files carefully and help me convert this project into a complete GitHub-ready and demo-ready portfolio project.

---

# Main LSTM Portfolio Repository Context

I am creating a main GitHub repository based on **LSTM / Long Short-Term Memory projects**.

The main repository should be:

```text
lstm-projects
```

Inside this repository, I plan to keep multiple LSTM-based projects.

The planned projects are:

```text
lstm-projects/
│
├── Airline Passenger Forecasting
├── Bitcoin Price Prediction
├── Conversational Chatbot using Seq2Seq with Attention
├── ECG Anomaly Detection using LSTM Autoencoder
├── Fake News Detection
├── Human Activity Recognition using LSTM with Attention
├── Industrial Equipment Failure Detection using LSTM Autoencoder
├── Multivariate Time Series Forecasting using Stacked LSTM
├── Neural Machine Translation with Attention
├── Stock Market Price Prediction
├── Text Summarization using Seq2Seq with Attention
├── Traffic Flow Prediction using Stacked LSTM
├── Video Frame Prediction using Convolutional LSTM
└── Weather Forecasting using ConvLSTM
```

For now, I want to proceed only with:

```text
Weather Forecasting using ConvLSTM
```

The final project should be structured like this:

```text
lstm-projects/
│
└── Weather_Forecasting_ConvLSTM/
```

---

# Current Task

Please take the attached Weather Forecasting using ConvLSTM files and convert them into a professional portfolio project.

I want this project to be:

1. GitHub-ready
2. Recruiter-friendly
3. Technically strong
4. Easy to understand
5. Easy to run locally
6. Suitable for hosting as a live Streamlit demo
7. Cleanly structured inside the `lstm-projects` repository
8. Deployment-ready using Streamlit or another suitable hosting platform

---

# Important Requirement

Please inspect the attached project files before making recommendations or generating updated code.

Do not give generic advice only.

Use my actual project files as the base version and improve them properly.

First identify what the current project is doing:

```text
Weather forecasting
Rainfall / precipitation nowcasting
Temperature forecasting
Radar image sequence forecasting
Satellite image sequence forecasting
Spatiotemporal weather forecasting
ConvLSTM-based weather map prediction
Any other weather-related forecasting task
```

Then improve the project according to the actual project objective.

If the current code is basic, improve it into a complete end-to-end **Weather Forecasting using ConvLSTM** project.

---

# Expected Folder Structure

Please organize the project in a clean GitHub-ready structure.

Recommended structure:

```text
lstm-projects/
│
└── Weather_Forecasting_ConvLSTM/
    │
    ├── README.md
    ├── app/
    │   └── streamlit_app.py
    ├── data/
    │   ├── sample_weather_sequence.npy
    │   ├── sample_weather_grid.csv
    │   ├── sample_weather_frames/
    │   └── README_data.md
    ├── notebooks/
    │   └── weather_forecasting_convlstm.ipynb
    ├── src/
    │   ├── data_preprocessing.py
    │   ├── weather_preprocessing.py
    │   ├── grid_generation.py
    │   ├── sequence_generation.py
    │   ├── model_training.py
    │   ├── model_evaluation.py
    │   ├── forecasting_pipeline.py
    │   ├── inference_pipeline.py
    │   └── visualization.py
    ├── models/
    │   ├── weather_convlstm_model.keras
    │   └── model_metadata.json
    ├── outputs/
    │   ├── sample_weather_frames.png
    │   ├── weather_trend_summary.png
    │   ├── actual_vs_predicted_weather_map.png
    │   ├── forecast_sequence.gif
    │   ├── prediction_error_heatmap.png
    │   ├── training_curve.png
    │   ├── baseline_comparison.png
    │   └── model_metrics.json
    ├── images/
    │   └── demo_screenshot.png
    ├── requirements.txt
    ├── .gitignore
    └── README_HOSTING.md
```

You can modify this structure if a better structure is needed, but keep it clean and professional.

---

# Project Objective

The project should clearly explain the weather forecasting / spatiotemporal forecasting problem:

Weather Forecasting using ConvLSTM is a spatiotemporal sequence prediction problem where the goal is to predict future weather patterns using previous weather observations over time.

The project should show how a **Convolutional LSTM / ConvLSTM model** can learn both:

```text
spatial patterns across weather grids, maps, radar images, or satellite frames
temporal patterns across previous weather time steps
```

The project should clearly explain:

* how weather data was loaded,
* whether the data is grid-based, image-based, or tabular,
* how weather frames or grids were prepared,
* how timestamps were handled,
* how weather variables were selected,
* how missing values were handled,
* how input-output weather sequences were created,
* how ConvLSTM learns spatial and temporal weather patterns,
* how future weather maps or values were forecasted,
* how prediction quality was evaluated,
* how the weather forecasting workflow can be demonstrated through a hosted Streamlit app.

---

# Weather Forecasting / Applied AI Framing

Please present this project as a practical spatiotemporal forecasting and environmental analytics project.

The project should answer:

```text
Given previous weather observations over time, can the model forecast the next weather state or future weather sequence?
```

The model should output:

```text
Input Weather Sequence
Predicted Future Weather Map / Frame / Value
Forecast Horizon
Actual vs Predicted Comparison
Prediction Error
Weather Interpretation
```

Example output:

```text
Input: Previous 6 weather frames
Forecast Horizon: Next 1 frame
Output: Predicted future weather map

Interpretation:
The ConvLSTM model estimates how the weather pattern may evolve based on previous spatial and temporal patterns.
```

Another example:

```text
Input: Previous 12 hourly precipitation grids
Forecast Horizon: Next 3 hours
Output: Predicted precipitation intensity maps

Interpretation:
The model predicts the movement and intensity change of weather patterns across the grid.
```

The exact framing should be adjusted based on the attached dataset.

---

# Responsible Use Requirement

Because this project involves weather forecasting, please include a responsible-use note.

The app and README should clearly state:

```text
This project is for educational and portfolio demonstration purposes only.
It is not an official weather forecasting system.
The forecasts may be inaccurate and should not be used for safety-critical, emergency, aviation, agriculture, transportation, or operational decisions.
Real weather forecasting requires validated meteorological data, physical models, domain expertise, and official weather services.
```

This note should appear clearly in both:

1. `README.md`
2. Streamlit app interface

---

# Technical Expectations

Please improve the project technically if needed.

Check whether the existing code properly handles:

1. Weather data loading
2. Timestamp/date parsing
3. Weather variable identification
4. Grid/frame/image structure identification
5. Missing value handling
6. Duplicate timestamp handling
7. Weather frame normalization
8. Sequence generation
9. Target frame generation
10. Train/validation/test split using chronological order
11. ConvLSTM model building
12. Model training
13. Next-step forecasting
14. Multi-step forecasting if feasible
15. Model evaluation
16. Actual vs predicted visualization
17. Prediction error heatmap
18. GIF or animation generation
19. Model saving/loading
20. Streamlit demo app

If any part is missing, please add it.

---

# Dataset Requirement

Please inspect the attached files and identify the actual weather dataset structure.

The dataset may be:

```text
A sequence of weather radar images
A sequence of satellite images
A NumPy array of weather grids
A NetCDF-style weather dataset
A CSV containing weather readings
A grid of temperature / rainfall / pressure values
A synthetic weather sequence dataset
Any other weather forecasting dataset
```

Use the actual data available in the attached project files.

Clearly identify:

```text
Data format
Timestamp column if available
Weather variable used as target
Frame / grid shape
Number of time steps
Number of sequences
Color format if image-based
Grid dimensions if grid-based
Forecasting horizon
Training / validation / test split
```

If the dataset is tabular and not spatial, explain whether ConvLSTM is appropriate.

If the dataset does not support ConvLSTM naturally, either:

1. convert it into a grid/frame sequence only if valid,
   or
2. explain the limitation and recommend a stacked LSTM or multivariate LSTM alternative.

But since this project is part of the ConvLSTM portfolio, keep the main project focused on weather grid/frame sequence forecasting if the data supports it.

---

# Weather / Frame Preprocessing Requirement

Please include a proper weather preprocessing pipeline.

Depending on the dataset, the pipeline should handle:

```text
Timestamp sorting
Weather variable selection
Grid creation if needed
Frame resizing if image-based
Frame normalization
Missing value interpolation if appropriate
Outlier review
Sequence creation
Target frame creation
Train/validation/test split
```

Important requirements:

1. Preserve chronological order.
2. Do not randomly shuffle frames within a sequence.
3. Avoid data leakage between train and test sequences.
4. Normalize weather variables consistently.
5. Fit scaling / normalization only on training data where applicable.
6. Store frame size, sequence length, forecast horizon, and target variable in metadata.
7. Use the same preprocessing during training and inference.

Please explain preprocessing decisions clearly in both:

1. Code comments
2. README.md

---

# Sequence Generation Requirement

Because this is a ConvLSTM project, please make sure the weather data is handled as spatiotemporal sequence data.

The project should create fixed-length weather sequences such as:

```text
Input shape = samples × time steps × height × width × channels
```

Example setup:

```text
Input sequence length: 6 previous weather frames
Target: next weather frame
Frame size: 64 × 64
Channels: 1 for single weather variable such as precipitation or temperature
```

Example data shape:

```text
X_train shape = [samples, 6, 64, 64, 1]
y_train shape = [samples, 64, 64, 1]
```

For sequence-to-sequence future forecasting:

```text
X_train shape = [samples, input_frames, height, width, channels]
y_train shape = [samples, output_frames, height, width, channels]
```

The sequence generation code should clearly define:

```text
input_sequence_length
forecast_horizon
frame_height
frame_width
number_of_channels
target_weather_variable
X_train
y_train
X_validation
y_validation
X_test
y_test
```

Please choose the correct setup based on the attached project and explain it clearly.

---

# Model Requirement

Since this project belongs under the `lstm-projects` repository, the model should clearly demonstrate Convolutional LSTM usage.

Use a suitable **ConvLSTM2D** architecture for weather forecasting.

Recommended architecture for next-frame weather forecasting:

```text
Input Weather Sequence
↓
ConvLSTM2D Layer
↓
Batch Normalization if useful
↓
ConvLSTM2D Layer if useful
↓
Conv2D Output Layer
↓
Predicted Future Weather Frame
```

Recommended architecture for multi-frame weather forecasting:

```text
Input Weather Sequence
↓
ConvLSTM2D Layer with return_sequences=True
↓
Batch Normalization / Dropout if useful
↓
ConvLSTM2D Layer
↓
Conv3D or TimeDistributed Conv2D Output
↓
Predicted Future Weather Sequence
```

The code should include:

* input sequence shape,
* ConvLSTM2D layer,
* number of filters,
* kernel size,
* activation function,
* padding strategy,
* return sequences setting,
* batch normalization if useful,
* dropout if useful,
* output convolution layer,
* regression-style weather output,
* loss function such as MSE, MAE, or Huber loss,
* optimizer,
* training/validation loss tracking.

Please make sure the model training code is clean, modular, and understandable.

---

# ConvLSTM Explanation Requirement

Please clearly explain what ConvLSTM does.

The README should explain:

```text
A normal LSTM learns temporal patterns from vector sequences.
A CNN learns spatial patterns from images or grids.
A ConvLSTM combines convolution operations with LSTM-style memory to learn spatial and temporal patterns together.
This makes ConvLSTM useful for weather nowcasting, radar forecasting, satellite sequence forecasting, video prediction, and other spatiotemporal forecasting tasks.
```

Please include this explanation in simple recruiter-friendly language.

---

# Forecasting Requirement

The project should generate practical weather forecast outputs.

The output should include:

```text
Input Weather Sequence
Predicted Future Weather Frame / Map
Actual Future Weather Frame if available
Forecast Horizon
Prediction Error
Weather Interpretation
```

If feasible, support:

```text
Single-step forecasting
Multi-step recursive forecasting
Batch prediction for multiple sequences
GIF / animation generation for predicted weather sequence
```

For multi-step recursive prediction, explain that the model feeds predicted frames back into the input sequence to generate future frames.

---

# Evaluation Requirement

This is a weather forecasting and spatiotemporal prediction problem, so evaluation should include both numerical and visual metrics.

Please include:

```text
MSE
MAE
RMSE
MAPE if appropriate
SSIM if image/frame based and feasible
PSNR if image/frame based and feasible
Training and Validation Loss Curve
Actual vs Predicted Weather Map
Prediction Error Heatmap
Forecast Sequence GIF
Qualitative Visual Analysis
```

If this is a precipitation / rainfall nowcasting project and labels support it, also include weather-specific metrics if feasible:

```text
CSI - Critical Success Index
POD - Probability of Detection
FAR - False Alarm Ratio
Precision / Recall for rainfall threshold events
```

The README should explain:

* MSE/MAE measure pixel-level or grid-level forecast error,
* RMSE penalizes larger forecast errors,
* SSIM measures structural similarity between actual and predicted maps,
* error heatmaps show where the model struggles,
* weather-specific event metrics can evaluate rainfall-event detection,
* forecasts should be evaluated on future unseen time periods.

If SSIM, PSNR, CSI, POD, or FAR are too heavy or not implemented, explain them as recommended future metrics.

---

# Baseline Comparison Requirement

Please include a simple baseline comparison if feasible.

Compare the ConvLSTM model against basic baselines such as:

```text
Persistence Baseline: last observed weather frame as next frame
Moving Average Baseline
Simple CNN baseline if suitable
Basic ConvLSTM with fewer layers
```

Recommended comparison table:

```text
Model / Approach            MAE        RMSE        SSIM        Notes
Persistence Baseline        <value>    <value>     <value>     Uses last observed frame
Moving Average Baseline     <value>    <value>     <value>     Smooths recent frames
ConvLSTM Model              <value>    <value>     <value>     Learns spatial-temporal patterns
```

If full baseline implementation is not possible, explain the recommended baseline approach in the README.

---

# Error Analysis / Limitation Requirement

Please include a simple output analysis and limitation section.

The project should discuss:

```text
forecast smoothing
blurry weather maps
difficulty predicting sudden weather changes
error accumulation in multi-step forecasting
small dataset limitations
resolution trade-offs
computational cost
difficulty modeling complex meteorological systems
limitations of ConvLSTM compared to numerical weather prediction and modern transformer-based forecasting models
```

Do not overstate the model capability.

This should be presented as an educational ConvLSTM weather forecasting project, not as a production-grade weather prediction system.

---

# Explainability / Interpretation Requirement

Since weather model outputs should be understandable, include a simple interpretation section.

Recommended outputs:

* input weather frame grid,
* actual future weather map,
* predicted future weather map,
* absolute error heatmap,
* forecast sequence animation,
* explanation of where the prediction is accurate,
* explanation of where the prediction struggles.

If attention or saliency is not used, do not invent interpretability methods. Focus on visual comparison and error heatmaps.

---

# Streamlit Demo Requirement

I want to host this project as a demo so that someone can click a link and interact with it.

Please create a clean and professional Streamlit app for this project.

The Streamlit app should allow users to:

1. Upload a weather sequence file if feasible
2. Upload a CSV / NumPy file containing weather grid data if feasible
3. Use a preloaded sample weather sequence
4. Select input sequence length if feasible
5. Select a sample sequence
6. Generate next-step weather forecast
7. Generate multi-step forecast if feasible
8. Download predicted weather outputs

The app should show:

* project title,
* responsible-use disclaimer,
* uploaded data preview,
* input weather sequence grid,
* predicted future weather map,
* actual future weather map if available,
* actual vs predicted comparison,
* prediction error heatmap,
* model metrics,
* forecast animation / GIF if available,
* explanation of ConvLSTM,
* limitations section,
* GitHub link placeholder.

The app should be suitable for hosting on:

```text
Streamlit Community Cloud
```

or:

```text
Hugging Face Spaces
```

Please recommend the best hosting option for this project and explain why.

The deployed app should load pre-trained model artifacts and should not require retraining during app startup.

---

# Streamlit Design Requirement

Please make the Streamlit app polished and portfolio-friendly.

The app should include:

```text
Project title
Short project explanation
Weather forecasting disclaimer
Sample sequence option
Weather sequence upload option if feasible
Input weather frame preview
Forecast button
Predicted weather map result
Actual vs predicted comparison
Error heatmap
Metrics section
Download forecast output button
Model details section
Limitations section
```

The app should be simple, clean, and easy for recruiters or technical reviewers to test.

If weather file upload is too heavy for hosted deployment, make the preloaded sample weather sequence the default demo flow and explain that upload is optional.

---

# GitHub README Requirement

Create a complete professional `README.md` for this project.

The README should include:

1. Project title
2. Responsible-use note
3. Weather forecasting / spatiotemporal forecasting problem
4. Project objective
5. Dataset description
6. Tools and technologies used
7. Project workflow
8. Weather data preprocessing
9. Sequence generation logic
10. ConvLSTM model architecture
11. Weather forecasting approach
12. Multi-step forecasting approach if implemented
13. Baseline comparison if feasible
14. Evaluation metrics
15. Key results
16. Actual vs predicted weather map examples
17. Error heatmap examples
18. Streamlit demo link placeholder
19. Screenshots section
20. How to run locally
21. How to deploy
22. Folder structure
23. Limitations
24. Future improvements
25. Skills demonstrated

The README should be written in a recruiter-friendly and technical style.

It should make the project look strong on GitHub and clearly show that this is a ConvLSTM weather forecasting project.

---

# Hosting Instructions

Please provide clear hosting instructions.

I want to know how to host this project so that I can share the live demo link on:

* GitHub
* Resume
* LinkedIn
* Portfolio website

Please provide instructions for at least one recommended hosting option.

Preferred options:

```text
Streamlit Community Cloud
Hugging Face Spaces
```

For the selected hosting option, explain:

1. What files are required
2. What should be inside `requirements.txt`
3. How to upload/push to GitHub
4. How to connect GitHub repo to hosting platform
5. How to test the app
6. What final link I can share

---

# Local Run Instructions

Please provide step-by-step local run instructions.

Example:

```text
1. Download the project folder.
2. Open terminal inside Weather_Forecasting_ConvLSTM.
3. Create virtual environment.
4. Install dependencies using pip install -r requirements.txt.
5. Run model training if needed.
6. Run Streamlit app using streamlit run app/streamlit_app.py.
7. Open the local URL in browser.
```

---

# Code File Requirements

Please provide complete updated code files, not partial snippets.

Generate all necessary files such as:

```text
README.md
requirements.txt
.gitignore
app/streamlit_app.py
src/data_preprocessing.py
src/weather_preprocessing.py
src/grid_generation.py
src/sequence_generation.py
src/model_training.py
src/model_evaluation.py
src/forecasting_pipeline.py
src/inference_pipeline.py
src/visualization.py
notebooks/weather_forecasting_convlstm.ipynb
README_HOSTING.md
```

If some files are not needed, explain why.

If model files need to be generated after running training, explain where they will be saved.

---

# Model / Artifact File Requirement

The project should save all required model artifacts for inference and demo use:

```text
Trained ConvLSTM weather forecasting model
Model metadata
Input sequence length
Forecast horizon
Frame height
Frame width
Number of channels
Target weather variable
Normalization method
Training configuration
Evaluation metrics
Sample sequence information
```

The Streamlit app should load these artifacts directly and should not require retraining during app startup.

---

# Data Safety / GitHub Requirement

If the weather dataset is large, private, restricted, copyrighted, sensitive, or not allowed for redistribution, please create a safe approach.

For example:

* keep full weather dataset out of GitHub if needed,
* include only a small safe sample weather sequence,
* use public or synthetic weather sample data where possible,
* add `data/README_data.md` explaining the dataset source and usage,
* add `.gitignore` to prevent large/generated files from being uploaded,
* do not include restricted meteorological datasets unless redistribution is allowed,
* keep large `.npy`, `.nc`, `.h5`, `.tif`, or image-sequence files out of GitHub unless small and safe.

Do not assume that large weather datasets should be pushed to GitHub.

---

# Portfolio Positioning

Please also explain how I should present this project in my GitHub portfolio.

Tell me:

1. How to describe this project in one line.
2. How to describe it in my GitHub pinned repository section.
3. What skills this project demonstrates.
4. What screenshots I should include.
5. What output files I should save.
6. How this project supports my transition from Quality Data Scientist to Data Science / ML / AI roles.

Emphasize that this project demonstrates:

* ConvLSTM modeling,
* weather forecasting,
* spatiotemporal forecasting,
* sequence generation,
* grid / image-based forecasting,
* model evaluation using numerical and visual metrics,
* Streamlit deployment,
* professional portfolio-ready ML project structure.

Also connect the project naturally to my current background as a Quality Data Scientist because this project demonstrates forecasting, monitoring, pattern detection, spatiotemporal analytics, operational decision support, and applied AI deployment.

---

# Main Repository README Update Requirement

Since this is another project inside the `lstm-projects` repository, please also suggest how the main repository README should be updated.

The main repository README should include:

1. Repository title: `lstm-projects`
2. Short professional description
3. My career positioning as a Quality Data Scientist building an LSTM portfolio
4. List of planned LSTM projects
5. Completed projects section showing:

   * Airline Passenger Forecasting
   * Bitcoin Price Prediction
   * Conversational Chatbot using Seq2Seq with Attention
   * ECG Anomaly Detection using LSTM Autoencoder with Attention
   * Fake News Detection
   * Human Activity Recognition using LSTM with Attention
   * Industrial Equipment Failure Detection using LSTM Autoencoder
   * Multivariate Time Series Forecasting using Stacked LSTM
   * Video Frame Prediction using Convolutional LSTM
   * Traffic Flow Prediction using Stacked LSTM
   * Weather Forecasting using ConvLSTM
6. Live demo links placeholder for each project
7. Tech stack
8. How the repository is organized
9. Skills demonstrated
10. Future roadmap

The README should be suitable for GitHub recruiters and technical reviewers.

---

# Final Output Required

Please provide:

1. Complete project overview.
2. Recommended `lstm-projects` repository setup/update.
3. Recommended GitHub repository description and topics.
4. Main repository README update.
5. Recommended project folder structure.
6. Updated / cleaned code files.
7. Streamlit demo app code.
8. `requirements.txt`.
9. `.gitignore`.
10. Professional project-level `README.md`.
11. Hosting guide.
12. Local run guide.
13. Suggestions for screenshots and visuals.
14. Recruiter-friendly project description.
15. Explanation of changes made to the original code.
16. All coding files in a downloadable format so I can run the project locally on my system.

Please generate the complete GitHub-ready and Streamlit-hosting-ready version of the **Weather Forecasting using ConvLSTM** project now.

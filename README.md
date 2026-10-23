# RNN-Based Sequential Data Prediction System

A modern, responsive, frontend-only machine learning demonstration web application illustrating how **Recurrent Neural Networks (RNN)** process sequential and time-series data. Built to satisfy 100% of the requirements specified in the project development prompt without relying on any backend, database, server, or external API.

---

## 🌟 Key Features & Sections

The application implements all 10 main sections required by the project specification:

### 1. Home / Landing Page
- Professional hero section with modern AI/ML visual aesthetics.
- Project title, objectives cards, and clear navigation to all sections.
- Animated signal bars and recurrent information flow status indicators.
- Quick call-to-action buttons for interactive training and prediction.

### 2. About RNN
- Professional theoretical explanation of sequential and time-series data.
- Detailed coverage of hidden states and recurrent memory ($h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$).
- Explanation of information flow and gradient dynamics (BPTT).
- **Interactive Unrolled RNN Visualizer**: Step-by-step inspector (Previous, Next, Auto Play) demonstrating state transitions from $t=1$ to $t=5$.

### 3. Multi-Domain Dataset Section
- **4 Selectable Sequential Datasets**:
  - *Daily Energy Demand (MW)*: Hourly diurnal power consumption with work peaks.
  - *Atmospheric Temperature (°C)*: Seasonal cycle with ambient weather swings.
  - *Financial Market Index ($)*: Non-stationary upward trend with volatility clusters.
  - *Synthetic Sinusoidal Wave*: Multi-frequency harmonic wave for mathematical analysis.
- Live dataset regenerator and noise sampler.
- Real-time statistical metrics: Samples count, Mean, Median, Min, Max, Standard Deviation ($\sigma$), and Variance ($\sigma^2$).
- Paginated/scrollable sequence table with sliding window pairs $[X \to y]$.
- High-DPI interactive dataset trend canvas with moving average overlay.

### 4. Interactive Data Preprocessing Pipeline
- Visual demonstration of the 4 core preprocessing stages:
  1. Data Cleaning & missing timestamp interpolation.
  2. Min-Max Normalization to $[0, 1]$ with side-by-side raw vs. normalized matrix display.
  3. Sequence Generation with adjustable sliding window slider ($k \in [3, 7]$).
  4. Chronological Train/Test Partitioning with interactive split slider (60% to 90%).

### 5. RNN Model Architecture & Parameter Inspector
- Visual layer stack: Input Layer $(Batch, k, 1) \to$ Recurrent Layer $(32 \text{ Units}) \to$ Dense Output $(Batch, 1)$.
- Mathematical formula callouts for hidden updates and dense projection.
- Interactive parameter inspector:
  - Toggle RNN hidden units (16, 32, 64).
  - Hyperbolic tangent ($\tanh$) activation.
  - Adam / RMSprop optimizer with momentum.
  - Mean Squared Error (MSE) loss function.
  - Mini-batch processing and Xavier/Glorot uniform weight initialization.
- Backpropagation Through Time (BPTT) overview.

### 6. Interactive Training Simulation & Live Loss Tracking
- Complete client-side gradient training simulation.
- Control buttons: **Start Training**, **Pause**, and **Reset Model**.
- Adjustable target epochs (20 to 100) and learning rate ($\alpha \in [0.001, 0.01]$).
- Live real-time dual-line canvas chart: **Training Loss** and **Validation Loss** vs. Epoch.
- Animated progress bar and live status indicators.
- Live evaluation metrics scorecard:
  - **MAE** (Mean Absolute Error)
  - **MSE** (Mean Squared Error)
  - **RMSE** (Root Mean Squared Error)
  - **Accuracy / $R^2$ Fit** (%)
- Terminal-style training log console streaming epoch-by-epoch loss reduction.

### 7. Interactive Sequence Prediction Panel
- Custom numerical sequence input textarea.
- Quick-load preset scenarios:
  - 📈 *Upward Trend*
  - 🌊 *Oscillating Wave*
  - 📉 *Sudden Dip*
  - 📂 *Latest Dataset Window*
- **Multi-Step Forecasting**: Selectable horizon (+1 Step, +3 Steps, +5 Steps ahead autoregressively).
- Visual forecast canvas chart showing input sequence connected to predicted future points.
- Real-time hidden state vector activations visualizer ($|h_t|$ magnitudes across hidden neurons).
- Directional trend badge (e.g. $+3.8\%$ Upward Trend) and reset button.

### 8. Visualization Dashboard
Four responsive, high-DPI canvas charts with theme-aware contrast:
1. **Actual vs. Predicted Comparison**: Test sequence actual values overlaid against RNN forecasts.
2. **Residual Error Scatter & Spread**: Absolute error stems with zero-error baseline.
3. **Residual Error Histogram**: Error distribution bins indicating model bias and variance.
4. **Hidden State Activation Profile**: Memory activation distribution across hidden units.

### 9. Project Workflow Pipeline
Interactive 7-stage end-to-end pipeline explorer:
$$\text{Dataset} \to \text{Preprocessing} \to \text{Sequence Generation} \to \text{RNN Model} \to \text{Training} \to \text{Prediction} \to \text{Evaluation}$$
Clicking any stage dynamically displays its input artifacts, mathematical transformation, and produced outputs.

### 10. Conclusion & Real-World Industry Applications
- Summary of recurrent temporal modeling principles.
- 5 Industry application cards with architecture mappings:
  - *Time-Series Forecasting* (Many-to-One / Many-to-Many)
  - *Text Prediction & NLP* (Character-Level RNN)
  - *Sentiment Analysis* (Many-to-One Classification)
  - *Speech Processing* (Sequence-to-Sequence)
  - *Sequence Anomaly Detection* (RNN Autoencoder)
- Comparative architecture matrix: Vanilla RNN vs. LSTM vs. GRU vs. Transformer.

---

## 💻 Tech Stack & Design

- **Structure**: Semantic HTML5 (accessible landmarks, labels, and ARIA attributes).
- **Styling**: Vanilla CSS3 (custom CSS variables, responsive grid, glassmorphism, Dark and Light mode theme toggle with `localStorage` persistence).
- **Logic**: Pure Modern Vanilla JavaScript (ES6+ classes, canvas 2D context with device pixel ratio scaling, no external frameworks or CDNs).
- **Architecture**: 100% Client-Side / Browser-Only. Zero backend, database, or network requirements.

---

## 🚀 How to Run Locally

1. Open a terminal in the project directory:
   ```bash
   cd "d:/Deep Learning/DL Projects/RNN"
   ```
2. Start a local HTTP server:
   ```bash
   python -m http.server 8000
   ```
3. Open your browser and navigate to:
   ```text
   http://localhost:8000
   ```
   *(Alternatively, double-click `index.html` to open it directly in any modern web browser).*

---

## 📁 Project Structure

```text
RNN/
├── index.html                   # Complete semantic application layout (all 10 sections)
├── styles.css                   # Custom styles, responsive grid, dark/light themes
├── app.js                       # Dataset generator, RNN model, training loop, charts
├── README.md                    # Project documentation & run guide
└── documentation/
    └── RNN_Project_Prompt.docx  # Original assignment requirements prompt
```

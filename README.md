# 🚌 Bus Distance Calculator

A practical tool for predicting how far your bus can travel based on current conditions. Upload your historical data, and get distance predictions along with efficiency insights for better fleet management.

##  Key Features

###  Machine Learning Predictions
Train models on your historical bus data to predict travel distances. The system automatically selects the best performing algorithm and provides accuracy metrics.

###  Web Dashboard  
Simple interface for both individual predictions and bulk fleet analysis. Upload CSV files or enter data manually to get instant results.

###  Efficiency Analysis
- Distance predictions based on fuel, load, speed, and route type
- Fuel efficiency scoring and recommendations
- Route optimization insights
- Load impact analysis

### Data Format

Your CSV file should include these columns:

| Column | Description | Example |
|--------|-------------|---------|
| Fuel_Level_Percentage | Current fuel level (0-100%) | 75.5 |
| Vehicle_Load_kg | Total load in kilograms | 1200 |
| Speed_kmph | Average speed in km/h | 65 |
| Temperature_C | Outside temperature in Celsius | 22.5 |
| Route_Type | Highway, Urban, or Rural | "Highway" |
| distance | Actual distance traveled (for training) | 145.2 |

##  How to Get Started

1. **Start the dashboard**
   ```bash
   streamlit run bus.py
   ```

2. **Train your model**
   - Upload your historical bus data CSV
   - Click "Let's Train Your AI!" 
   - Wait for training to complete

3. **Make predictions**
   - Enter individual bus details for quick checks
   - Upload CSV files for fleet-wide analysis

4. **Review results**
   - View efficiency metrics and recommendations
   - Download detailed reports
   - Use insights for route and fuel planning

##  What You'll Get

### Efficiency Metrics
- **Fuel Efficiency**: Distance per fuel percentage (km per % fuel)
- **Load Impact**: How vehicle weight affects performance
- **Temperature Effects**: Weather impact on fuel consumption
- **Route Analysis**: Performance differences between highway, urban, and rural routes
- **Speed Optimization**: Recommended speed ranges for best efficiency

### Visual Reports
The dashboard provides interactive charts and downloadable reports with performance insights and recommendations for fleet optimization.

##  Example Use Cases

**Route Planning**: Check if a bus can complete a specific route before departure based on current fuel and load conditions.

**Fleet Optimization**: Compare efficiency across multiple vehicles to identify top performers and buses that need attention.

**Cost Management**: Understand fuel consumption patterns to optimize scheduling and reduce operational costs.

##  Acknowledgments

- **Scikit-learn**: For machine learning algorithms
- **Streamlit**: For the amazing web framework
- **Plotly**: For interactive visualizations
- **Pandas & NumPy**: For data manipulation

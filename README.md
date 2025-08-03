# Pharma Sales Analytics

## Overview
A Streamlit-powered web application for analyzing pharmaceutical sales performance across teams, products, and territories. The dashboard provides:

- Interactive visualizations of sales trends
- Performance benchmarking
- Target achievement analysis
- Predictive analytics capabilities
- Custom report generation

## Features

### 📊 Dashboard
- Key performance indicators (KPIs) at a glance
- Sales trends by therapeutic group
- Monthly sales performance tracking
- Top product and team performance analysis

### 🔍 Data Explorer
- Interactive filtering by multiple dimensions
- Custom chart creation (bar, line, scatter, etc.)
- Real-time data statistics
- Export capabilities

### 📈 Performance Analysis
- Team/Product/Territory performance breakdowns
- Target vs actual comparisons
- Achievement rate calculations
- Top/bottom performer identification

### 🤖 Predictive Analytics
- Sales forecasting models
- Performance prediction engine
- AI-powered recommendations
- What-if scenario analysis

### 📑 Report Generator
- Customizable report templates
- Automated insights generation
- Multi-format export (PDF, Excel, PPT)

## Data Requirements
The application expects CSV data with the following structure:

| Column | Description | Example |
|--------|-------------|---------|
| `YearMonth` | Date in month-year format | `May-25` or `2025-05` |
| `RSM` | Regional Sales Manager | `B1` |
| `FM` | Field Manager | `B10` |
| `MPO` | Medical Representative | `EMP001` |
| `Team` | Sales team name | `Nugenta` |
| `Pcode` | Product code | `P1234` |
| `Pname` | Product name | `MARVELON TAB` |
| `Brand` | Brand name | `Marvelon` |
| `Thgroup` | Therapeutic group | `Contraceptives` |
| `TargetQnty` | Target quantity | `200` |
| `SoldQnty` | Actual quantity sold | `227` |
| `TargetValue` | Target sales value | `18000` |
| `SoldTP` | Actual sales value | `20606` |

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/AbidHasanRafi/pharma-sales-analytics.git
   cd pharma-sales-analytics
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare your sales data in CSV format (or see data in the repository)
2. Run the application:
   ```bash
   streamlit run app.py
   ```
3. Upload your CSV file when prompted
4. Explore the different analysis modules using the sidebar navigation

## Configuration
The app can be customized by modifying the following in `app.py`:

- **Color scheme**: Edit the CSS in the `st.markdown` section
- **Default filters**: Adjust in the Data Explorer section
- **Model parameters**: Modify in the Predictive Analytics section

## Dependencies
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Matplotlib
- Seaborn

## Support
For issues or feature requests, please [open an issue](https://github.com/abidhasanrafi/pharma-sales-analytics/issues) on GitHub.

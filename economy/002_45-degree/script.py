import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
from scipy import stats

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

# =============================================================================
# STEP 1: LOAD AND PREPARE GDP DATA
# =============================================================================


def load_gdp_data(file_path):
    """
    Load Real GDP data from FRED CSV file
    File should have columns: DATE, GDPC1
    """
    df = pd.read_csv(file_path, parse_dates=["DATE"])
    df.columns = ["Date", "GDP"]
    df = df.dropna()
    df = df.sort_values("Date").reset_index(drop=True)

    # Convert GDP to trillions for better readability
    df["GDP"] = df["GDP"] / 1000  # From billions to trillions

    return df


# Load your data
df = load_gdp_data("GDPC1.csv")

print("=" * 70)
print("45-DEGREE MODEL: US GDP ANALYSIS")
print("=" * 70)
print(f"\nData loaded: {len(df)} quarterly observations")
print(
    f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
)
print(f"GDP range: ${df['GDP'].min():.2f}T to ${df['GDP'].max():.2f}T")
print(f"\nFirst few observations:")
print(df.head())
print(f"\nLast few observations:")
print(df.tail())

# =============================================================================
# STEP 2: CREATE LAGGED VARIABLES FOR KEYNESIAN MODEL
# =============================================================================


def create_keynesian_data(df):
    """
    Create lagged GDP variable for simple Keynesian model:
    Y_t = α + β*Y_{t-1} + ε_t
    """
    data = df.copy()
    data["GDP_t"] = data["GDP"]
    data["GDP_t_1"] = data["GDP"].shift(1)  # Lagged GDP (one quarter back)

    # Remove first row (has NaN for lagged value)
    data = data.dropna().reset_index(drop=True)

    return data


keynesian_data = create_keynesian_data(df)
print(f"\nKeynesian model data prepared: {len(keynesian_data)} observations")

# =============================================================================
# STEP 3: ESTIMATE THE SIMPLE KEYNESIAN MODEL
# =============================================================================


def estimate_keynesian_model(data):
    """
    Estimate the simple Keynesian consumption function model:
    Y_t = α + β*Y_{t-1} + ε_t

    Where:
    - α = autonomous expenditure (intercept)
    - β = marginal propensity to consume (MPC)
    - The multiplier k = 1/(1-β)
    """
    X = data[["GDP_t_1"]].values
    y = data["GDP_t"].values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    alpha = model.intercept_
    beta = model.coef_[0]

    # Calculate fitted values and metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100

    # Calculate residuals
    residuals = y - y_pred

    results = {
        "alpha": alpha,
        "beta": beta,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "model": model,
        "y_pred": y_pred,
        "residuals": residuals,
    }

    return results


# Estimate the model
results = estimate_keynesian_model(keynesian_data)

print("\n" + "=" * 70)
print("SIMPLE KEYNESIAN MODEL ESTIMATION")
print("=" * 70)
print(f"\nEstimated Equation:")
print(f"  Y_t = {results['alpha']:.4f} + {results['beta']:.4f} × Y_{{t-1}}")
print(f"\nParameter Interpretation:")
print(f"  α (Autonomous Expenditure) = ${results['alpha']:.4f} trillion")
print(f"  β (Marginal Propensity to Consume) = {results['beta']:.4f}")
print(f"\nModel Performance:")
print(f"  R² = {results['r2']:.4f} ({results['r2'] * 100:.2f}% of variance explained)")
print(f"  RMSE = ${results['rmse']:.4f} trillion")
print(f"  MAE = ${results['mae']:.4f} trillion")
print(f"  MAPE = {results['mape']:.2f}%")

# =============================================================================
# STEP 4: COMPUTE DYNAMIC MULTIPLIER
# =============================================================================


def calculate_dynamic_multiplier(beta, horizons=20):
    """
    Calculate the dynamic multiplier for different time horizons.

    The impact of a $1 shock to autonomous expenditure at time t:
    - At t: $1
    - At t+1: $β
    - At t+2: $β²
    - At t+h: $β^h

    Cumulative multiplier (long-run): k = 1/(1-β)
    """
    multipliers = []
    cumulative = []
    cum_sum = 0

    for h in range(horizons + 1):
        impact = beta**h
        multipliers.append(impact)
        cum_sum += impact
        cumulative.append(cum_sum)

    # Long-run multiplier (infinite horizon)
    if abs(beta) < 1:
        long_run_multiplier = 1 / (1 - beta)
    else:
        long_run_multiplier = np.inf

    multiplier_df = pd.DataFrame(
        {
            "Horizon": range(horizons + 1),
            "Impact_Multiplier": multipliers,
            "Cumulative_Multiplier": cumulative,
        }
    )

    return multiplier_df, long_run_multiplier


# Calculate multipliers
multiplier_df, long_run_mult = calculate_dynamic_multiplier(results["beta"])

print("\n" + "=" * 70)
print("DYNAMIC MULTIPLIER ANALYSIS")
print("=" * 70)
print(f"\nMarginal Propensity to Consume (β) = {results['beta']:.4f}")
print(f"Marginal Propensity to Save (1-β) = {1 - results['beta']:.4f}")
print(f"\n✓ SHORT-RUN MULTIPLIER (k₀): {1 / (1 - results['beta']):.4f}")
print(f"  → A $1 increase in autonomous expenditure leads to")
print(f"     ${1 / (1 - results['beta']):.4f} trillion increase in GDP (long-run)")
print(
    f"\n✓ STABILITY: {'CONVERGENT' if abs(results['beta']) < 1 else 'DIVERGENT'} (|β| = {abs(results['beta']):.4f})"
)

print(f"\nDynamic Multiplier Path (first 10 quarters):")
print(multiplier_df.head(10).to_string(index=False))

# =============================================================================
# STEP 5: GDP FORECASTING (Y_t) AND FORECAST ERRORS
# =============================================================================


def forecast_gdp_future(last_gdp, alpha, beta, n_periods=8):
    """
    Forecast future GDP for n periods using:
    Y_{t+k} = α + β × Y_{t+k-1}
    """
    forecasts = []
    current_gdp = last_gdp

    for i in range(1, n_periods + 1):
        next_gdp = alpha + beta * current_gdp
        forecasts.append(
            {
                "Period": f"t+{i}",
                "Quarter": i,
                "Forecast_GDP": next_gdp,
                "Previous_GDP": current_gdp,
            }
        )
        current_gdp = next_gdp

    return pd.DataFrame(forecasts)


def calculate_insample_forecast_errors(data, results):
    """
    Calculate in-sample forecast errors for model validation
    """
    y_actual = data["GDP_t"].values
    y_pred = results["y_pred"]

    # Calculate various error metrics
    errors = y_actual - y_pred
    abs_errors = np.abs(errors)
    pct_errors = (errors / y_actual) * 100

    error_stats = {
        "Mean_Error": np.mean(errors),
        "Mean_Absolute_Error": np.mean(abs_errors),
        "RMSE": np.sqrt(np.mean(errors**2)),
        "Mean_Percentage_Error": np.mean(pct_errors),
        "MAPE": np.mean(np.abs(pct_errors)),
        "Max_Error": np.max(abs_errors),
        "Min_Error": np.min(abs_errors),
        "Std_Error": np.std(errors),
    }

    return error_stats, errors


# Generate future forecasts
last_gdp = keynesian_data["GDP_t"].iloc[-1]
last_date = df["Date"].iloc[-1]

forecast_df = forecast_gdp_future(
    last_gdp,
    results["alpha"],
    results["beta"],
    n_periods=8,  # 2 years ahead
)

# Calculate forecast errors
error_stats, errors = calculate_insample_forecast_errors(keynesian_data, results)

print("\n" + "=" * 70)
print("GDP FORECASTING RESULTS")
print("=" * 70)
print(f"\nLast observed GDP (at {last_date.strftime('%Y-%m-%d')}): ${last_gdp:.2f}T")
print(
    f"\nForecast Equation: Y_{{t+k}} = {results['alpha']:.4f} + {results['beta']:.4f} × Y_{{t+k-1}}"
)
print(f"\n8-Quarter Ahead Forecasts (2 years):")
print(forecast_df.to_string(index=False))

# Calculate long-run equilibrium GDP
equilibrium_gdp = results["alpha"] / (1 - results["beta"])
print(f"\nLong-run Equilibrium GDP: ${equilibrium_gdp:.2f}T")
print(f"Current GDP vs Equilibrium: ${last_gdp - equilibrium_gdp:+.2f}T")

print("\n" + "=" * 70)
print("FORECAST ERROR ANALYSIS (In-Sample)")
print("=" * 70)
print(f"\nError Statistics:")
print(f"  Mean Error: ${error_stats['Mean_Error']:.4f}T")
print(f"  Mean Absolute Error (MAE): ${error_stats['Mean_Absolute_Error']:.4f}T")
print(f"  Root Mean Squared Error (RMSE): ${error_stats['RMSE']:.4f}T")
print(f"  Mean Percentage Error: {error_stats['Mean_Percentage_Error']:.2f}%")
print(f"  Mean Absolute Percentage Error (MAPE): {error_stats['MAPE']:.2f}%")
print(f"  Standard Deviation of Errors: ${error_stats['Std_Error']:.4f}T")
print(f"  Max Absolute Error: ${error_stats['Max_Error']:.4f}T")
print(f"  Min Absolute Error: ${error_stats['Min_Error']:.4f}T")

# =============================================================================
# STEP 6: OUT-OF-SAMPLE VALIDATION
# =============================================================================


def perform_out_of_sample_validation(data, test_size=0.15):
    """
    Perform out-of-sample validation using train-test split
    """
    split_point = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]

    # Re-estimate on training data
    X_train = train_data[["GDP_t_1"]].values
    y_train = train_data["GDP_t"].values

    temp_model = LinearRegression()
    temp_model.fit(X_train, y_train)

    # Predict on test data
    X_test = test_data[["GDP_t_1"]].values
    y_test = test_data["GDP_t"].values
    y_test_pred = temp_model.predict(X_test)

    # Calculate test metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    test_results = {
        "train_size": len(train_data),
        "test_size": len(test_data),
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_mape": test_mape,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
    }

    return test_results


# Perform validation
test_results = perform_out_of_sample_validation(keynesian_data)

print("\n" + "=" * 70)
print("OUT-OF-SAMPLE VALIDATION (Last 15% of data)")
print("=" * 70)
print(f"\nTraining observations: {test_results['train_size']}")
print(f"Testing observations: {test_results['test_size']}")
print(f"\nTest Set Performance:")
print(f"  R² = {test_results['test_r2']:.4f}")
print(f"  RMSE = ${test_results['test_rmse']:.4f}T")
print(f"  MAE = ${test_results['test_mae']:.4f}T")
print(f"  MAPE = {test_results['test_mape']:.2f}%")

# =============================================================================
# STEP 7: COMPREHENSIVE VISUALIZATIONS
# =============================================================================


def create_keynesian_visualizations(
    df, keynesian_data, results, forecast_df, multiplier_df, test_results
):
    """
    Create comprehensive visualizations for the 45-degree model
    """
    fig = plt.figure(figsize=(18, 12))

    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Historical GDP Time Series (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df["Date"], df["GDP"], linewidth=2, color="darkblue", label="US Real GDP")
    ax1.set_title("US Real GDP (Quarterly, 1990-2025)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Year", fontsize=11)
    ax1.set_ylabel("GDP (Trillions of 2017$)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: GDP Growth Rate (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    gdp_growth = df["GDP"].pct_change() * 100
    ax2.plot(df["Date"], gdp_growth, linewidth=1.5, color="green")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax2.set_title("GDP Growth Rate", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_ylabel("Growth (%)", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: 45-Degree Diagram (middle left)
    ax3 = fig.add_subplot(gs[1, 0])

    # Create the aggregate expenditure line
    gdp_range = np.linspace(
        keynesian_data["GDP_t_1"].min() * 0.95,
        keynesian_data["GDP_t_1"].max() * 1.05,
        100,
    )
    ae_line = results["alpha"] + results["beta"] * gdp_range

    # Plot 45-degree line (equilibrium)
    ax3.plot(
        gdp_range, gdp_range, "k--", linewidth=2.5, label="45° Line (Y=AE)", alpha=0.7
    )

    # Plot aggregate expenditure function
    ax3.plot(
        gdp_range,
        ae_line,
        "r-",
        linewidth=2.5,
        label=f"AE = {results['alpha']:.2f} + {results['beta']:.3f}Y",
    )

    # Plot historical data points
    ax3.scatter(
        keynesian_data["GDP_t_1"],
        keynesian_data["GDP_t"],
        alpha=0.3,
        s=15,
        color="blue",
        label="Historical Data",
    )

    # Mark equilibrium point
    eq_gdp = results["alpha"] / (1 - results["beta"])
    ax3.plot(
        eq_gdp,
        eq_gdp,
        "go",
        markersize=12,
        label=f"Equilibrium: ${eq_gdp:.2f}T",
        zorder=5,
    )

    ax3.set_title("45-Degree Keynesian Cross Diagram", fontsize=12, fontweight="bold")
    ax3.set_xlabel("$Y_{t-1}$ (Lagged GDP, Trillions $)", fontsize=10)
    ax3.set_ylabel("$Y_t$ (Current GDP, Trillions $)", fontsize=10)
    ax3.legend(fontsize=8, loc="best")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Actual vs Fitted Values (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(
        keynesian_data["GDP_t"], results["y_pred"], alpha=0.5, s=25, color="blue"
    )
    ax4.plot(
        [keynesian_data["GDP_t"].min(), keynesian_data["GDP_t"].max()],
        [keynesian_data["GDP_t"].min(), keynesian_data["GDP_t"].max()],
        "r--",
        linewidth=2,
        label="Perfect Fit",
    )
    ax4.set_title(
        f"Actual vs Fitted GDP (R²={results['r2']:.4f})", fontsize=12, fontweight="bold"
    )
    ax4.set_xlabel("Actual GDP (Trillions $)", fontsize=10)
    ax4.set_ylabel("Fitted GDP (Trillions $)", fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Residuals Over Time (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(
        range(len(results["residuals"])),
        results["residuals"],
        alpha=0.5,
        s=20,
        color="red",
    )
    ax5.axhline(y=0, color="black", linestyle="--", linewidth=2)
    ax5.set_title("Forecast Residuals", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Observation", fontsize=10)
    ax5.set_ylabel("Residual (Trillions $)", fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Dynamic Multiplier (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(
        multiplier_df["Horizon"],
        multiplier_df["Cumulative_Multiplier"],
        "b-o",
        linewidth=2,
        markersize=5,
        label="Cumulative",
    )
    ax6.axhline(
        y=1 / (1 - results["beta"]),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Long-run: {1 / (1 - results['beta']):.2f}",
    )
    ax6.set_title("Dynamic Multiplier Path", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Quarters Ahead", fontsize=10)
    ax6.set_ylabel("Cumulative Multiplier", fontsize=10)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 20)

    # Plot 7: GDP Forecast (bottom center)
    ax7 = fig.add_subplot(gs[2, 1])

    # Historical GDP (last 40 quarters)
    hist_dates = df["Date"].iloc[-40:]
    hist_gdp = df["GDP"].iloc[-40:]
    ax7.plot(hist_dates, hist_gdp, "b-", linewidth=2, label="Historical GDP")

    # Create forecast dates (quarterly)
    last_date = df["Date"].iloc[-1]
    forecast_dates = pd.date_range(last_date, periods=9, freq="QS")[1:]

    # Plot forecasts
    forecast_gdp = forecast_df["Forecast_GDP"].values
    forecast_line = np.concatenate([[last_gdp], forecast_gdp])
    forecast_dates_full = np.concatenate([[last_date], forecast_dates])
    ax7.plot(
        forecast_dates_full,
        forecast_line,
        "r--o",
        linewidth=2,
        markersize=5,
        label="Forecast",
    )

    # Plot equilibrium line
    ax7.axhline(
        y=eq_gdp,
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"Equilibrium: ${eq_gdp:.2f}T",
    )

    ax7.set_title("GDP Forecast (8 Quarters Ahead)", fontsize=12, fontweight="bold")
    ax7.set_xlabel("Date", fontsize=10)
    ax7.set_ylabel("GDP (Trillions $)", fontsize=10)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)

    # Plot 8: Forecast Error Distribution (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.hist(
        results["residuals"], bins=30, color="purple", alpha=0.7, edgecolor="black"
    )
    ax8.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax8.set_title("Forecast Error Distribution", fontsize=12, fontweight="bold")
    ax8.set_xlabel("Forecast Error (Trillions $)", fontsize=10)
    ax8.set_ylabel("Frequency", fontsize=10)
    ax8.grid(True, alpha=0.3, axis="y")

    plt.savefig("keynesian_45degree_analysis.png", dpi=300, bbox_inches="tight")
    print("\n✓ Visualizations saved as 'keynesian_45degree_analysis.png'")
    plt.show()


# Create all visualizations
create_keynesian_visualizations(
    df, keynesian_data, results, forecast_df, multiplier_df, test_results
)

# =============================================================================
# STEP 8: GENERATE COMPREHENSIVE REPORT
# =============================================================================


def generate_keynesian_report(
    df, keynesian_data, results, forecast_df, multiplier_df, error_stats, test_results
):
    """
    Generate a comprehensive summary report for the 45-degree model analysis
    """

    equilibrium_gdp = results["alpha"] / (1 - results["beta"])
    multiplier = 1 / (1 - results["beta"])

    report = f"""
{"=" * 80}
45-DEGREE KEYNESIAN MODEL ANALYSIS - US GDP
{"=" * 80}

1. DATA SUMMARY
   - Economic Variable: Real Gross Domestic Product (GDP)
   - Source: FRED (Federal Reserve Economic Data)
   - Total Observations: {len(df)} quarters
   - Date Range: {df["Date"].min().strftime("%Y-%m-%d")} to {df["Date"].max().strftime("%Y-%m-%d")}
   - Time Span: {(df["Date"].max() - df["Date"].min()).days / 365.25:.1f} years

   GDP Statistics (in Trillions of 2017 Dollars):
   - Mean: ${df["GDP"].mean():.2f}T
   - Std Dev: ${df["GDP"].std():.2f}T
   - Min/Max: ${df["GDP"].min():.2f}T / ${df["GDP"].max():.2f}T
   - Latest GDP: ${df["GDP"].iloc[-1]:.2f}T

2. SIMPLE KEYNESIAN MODEL SPECIFICATION

   Theoretical Foundation:
   The simple Keynesian model with consumption function:

   Y_t = C_t + I + G + NX
   C_t = α + β × Y_{{t-1}}

   This yields the difference equation:
   Y_t = α + β × Y_{{t-1}} + ε_t

3. ESTIMATED PARAMETERS

   Estimated Equation:
   Y_t = {results["alpha"]:.4f} + {results["beta"]:.4f} × Y_{{t-1}}

   Parameter Interpretation:
   - α (Autonomous Expenditure) = ${results["alpha"]:.4f} trillion
     → Represents I + G + NX (investment, govt spending, net exports)

   - β (Marginal Propensity to Consume) = {results["beta"]:.4f}
     → Out of each additional dollar of income, {results["beta"] * 100:.2f}¢ is consumed
     → Marginal Propensity to Save (MPS) = {1 - results["beta"]:.4f}

4. MODEL PERFORMANCE & FORECASTING POWER

   In-Sample Fit:
   - R² = {results["r2"]:.4f} ({results["r2"] * 100:.2f}% of variance explained)
   - RMSE = ${results["rmse"]:.4f} trillion
   - MAE = ${results["mae"]:.4f} trillion
   - MAPE = {results["mape"]:.2f}%
   - Assessment: {"Excellent" if results["r2"] > 0.95 else "Good" if results["r2"] > 0.85 else "Moderate"} forecasting power

   Forecast Error Analysis:
   - Mean Error: ${error_stats["Mean_Error"]:.4f}T (bias check)
   - Mean Absolute Error: ${error_stats["Mean_Absolute_Error"]:.4f}T
   - Standard Deviation: ${error_stats["Std_Error"]:.4f}T
   - Max Absolute Error: ${error_stats["Max_Error"]:.4f}T
   - MAPE: {error_stats["MAPE"]:.2f}%

   Out-of-Sample Validation (Last {test_results["test_size"]} quarters):
   - Test R² = {test_results["test_r2"]:.4f}
   - Test RMSE = ${test_results["test_rmse"]:.4f}T
   - Test MAPE = {test_results["test_mape"]:.2f}%
   - Conclusion: {"Stable and reliable" if test_results["test_r2"] > 0.85 else "Moderate stability"}

5. DYNAMIC MULTIPLIER ANALYSIS ⭐

   Short-Run Multiplier (k₀):
   k = 1/(1-β) = 1/(1-{results["beta"]:.4f}) = {multiplier:.4f}

   Economic Interpretation:
   → A $1 increase in autonomous expenditure (I, G, or NX) leads to
     a ${multiplier:.4f} increase in GDP in the long run

   → If government spending increases by $100 billion, GDP will eventually
     increase by ${multiplier * 0.1:.2f} trillion

   Multiplier Path (First 10 Quarters):
"""

    for idx, row in multiplier_df.head(10).iterrows():
        report += f"   Quarter {row['Horizon']}: Impact = {row['Impact_Multiplier']:.4f}, Cumulative = {row['Cumulative_Multiplier']:.4f}\n"

    report += f"""
   Convergence: {"✓ Converges to long-run equilibrium" if abs(results["beta"]) < 1 else "✗ Divergent dynamics"}
   Speed: {"Fast" if results["beta"] < 0.9 else "Moderate" if results["beta"] < 0.95 else "Slow"} (β = {results["beta"]:.4f})

6. GDP FORECASTING (Y_t) - 8 QUARTERS AHEAD

   Last Observed GDP: ${keynesian_data["GDP_t"].iloc[-1]:.2f}T (as of {df["Date"].iloc[-1].strftime("%Y-%m-%d")})

   Forecast Equation: Y_{{t+k}} = {results["alpha"]:.4f} + {results["beta"]:.4f} × Y_{{t+k-1}}

   Quarterly Forecasts:
"""

    for idx, row in forecast_df.iterrows():
        report += f"   {row['Period']:>4s} (Q{row['Quarter']}): ${row['Forecast_GDP']:>7.2f}T\n"

    report += f"""
   Long-Run Equilibrium GDP:
   Y* = α/(1-β) = ${equilibrium_gdp:.2f}T

   Current vs Equilibrium: ${keynesian_data["GDP_t"].iloc[-1] - equilibrium_gdp:+.2f}T
   → GDP is currently {"above" if keynesian_data["GDP_t"].iloc[-1] > equilibrium_gdp else "below"} equilibrium

7. STABILITY & CONVERGENCE ANALYSIS

   Stability Condition: |β| < 1
   - |β| = {abs(results["beta"]):.4f}
   - Status: {"✓ STABLE" if abs(results["beta"]) < 1 else "✗ UNSTABLE"}
   - Interpretation: GDP {"will converge to equilibrium" if abs(results["beta"]) < 1 else "will diverge"}

   Half-Life (time to close 50% of gap to equilibrium):
   - h = ln(0.5)/ln(β) ≈ {-np.log(0.5) / np.log(results["beta"]):.1f} quarters
   - This means it takes about {-np.log(0.5) / np.log(results["beta"]):.1f} quarters to close half
     the distance to equilibrium

8. ECONOMIC IMPLICATIONS

   ✓ The model captures {results["r2"] * 100:.1f}% of GDP dynamics
   ✓ Marginal propensity to consume is {results["beta"]:.3f} (realistic for US economy)
   ✓ The multiplier effect amplifies fiscal policy by {multiplier:.2f}x
   {"✓" if results["r2"] > 0.9 else "○"} {"Strong" if results["r2"] > 0.9 else "Moderate"} predictive power for policy analysis
   ✓ Forecast errors are relatively small ({error_stats["MAPE"]:.2f}% MAPE)

9. POLICY RECOMMENDATIONS

   Based on the multiplier analysis:
   - Fiscal stimulus has a multiplier of {multiplier:.2f}
   - Each $1 trillion in government spending increases GDP by ${multiplier:.2f}T
   - Tax cuts (affecting disposable income) have similar multiplier effects
   - The economy {"is stable and" if abs(results["beta"]) < 1 else "shows instability and"} responds predictably to shocks

{"=" * 80}
END OF REPORT
{"=" * 80}
"""

    # Save report with UTF-8 encoding
    with open("keynesian_45degree_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print("\n✓ Full report saved as 'keynesian_45degree_report.txt'")


# Generate the comprehensive report
generate_keynesian_report(
    df, keynesian_data, results, forecast_df, multiplier_df, error_stats, test_results
)

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE! ✓")
print("=" * 80)
print("\nGenerated Files:")
print("  1. keynesian_45degree_analysis.png - Comprehensive visualizations")
print("  2. keynesian_45degree_report.txt - Full analysis report")
print("\nKey Results:")
print(f"  • Dynamic Multiplier: {1 / (1 - results['beta']):.4f}")
print(
    f"  • Model R²: {results['r2']:.4f} ({results['r2'] * 100:.1f}% variance explained)"
)
print(f"  • Forecast MAPE: {error_stats['MAPE']:.2f}%")
print(f"  • Long-run Equilibrium GDP: ${results['alpha'] / (1 - results['beta']):.2f}T")
print("\n✓ Ready for your mathematical economics project submission!")
print("=" * 80)

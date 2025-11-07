import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# =============================================================================
# STEP 1: LOAD AND PREPARE DATA
# =============================================================================


def load_coffee_data(file_path):
    """
    Load coffee price data from CSV file
    Assuming file has columns: Date, Price
    """
    df = pd.read_csv(file_path, parse_dates=["DATE"])
    df.columns = ["Date", "Price"]
    df = df.dropna()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# Load your data
# For FRED data: download from https://fred.stlouisfed.org/series/PCOFFOTMUSDM
df = load_coffee_data("coffee - U.S. Cents per Pound.csv")

print(f"Data loaded: {len(df)} observations")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"\nFirst few observations:")
print(df.head())

# =============================================================================
# STEP 2: CREATE LAGGED VARIABLES
# =============================================================================


def create_cobweb_data(df):
    """
    Create lagged price variable for cobweb model
    P_t and P_{t-1}
    """
    data = df.copy()
    data["Price_t"] = data["Price"]
    data["Price_t_1"] = data["Price"].shift(1)  # Lagged price

    # Remove first row (has NaN for lagged value)
    data = data.dropna().reset_index(drop=True)

    return data


cobweb_data = create_cobweb_data(df)
print(f"\nCobweb data prepared: {len(cobweb_data)} observations")

# =============================================================================
# STEP 3: ESTIMATE THE DIFFERENCE EQUATION
# =============================================================================


def estimate_cobweb_parameters(data):
    """
    Estimate parameters for the cobweb difference equation:
    P_t = (c + a)/b - (d/b)*P_{t-1}

    Using linear regression: P_t = beta_0 + beta_1 * P_{t-1}
    Where: beta_0 = (c + a)/b  and  beta_1 = -d/b
    """
    X = data[["Price_t_1"]].values
    y = data["Price_t"].values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    beta_0 = model.intercept_
    beta_1 = model.coef_[0]

    # Calculate R-squared
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    results = {
        "beta_0": beta_0,
        "beta_1": beta_1,
        "r2": r2,
        "rmse": rmse,
        "model": model,
        "y_pred": y_pred,
    }

    return results


# Estimate parameters
results = estimate_cobweb_parameters(cobweb_data)

print("\n" + "=" * 60)
print("COBWEB MODEL ESTIMATION RESULTS")
print("=" * 60)
print(
    f"\nDifference Equation: P_t = {results['beta_0']:.4f} + ({results['beta_1']:.4f})*P_{{t-1}}"
)
print(f"\nModel Performance:")
print(f"  R-squared: {results['r2']:.4f}")
print(f"  RMSE: {results['rmse']:.4f}")

# Interpret parameters
print(f"\nParameter Interpretation:")
print(f"  β₀ = (c + a)/b = {results['beta_0']:.4f}")
print(f"  β₁ = -d/b = {results['beta_1']:.4f}")

# =============================================================================
# STEP 4: STABILITY ANALYSIS
# =============================================================================


def analyze_stability(beta_1):
    """
    Analyze the stability of the cobweb model
    Based on |β₁| = |d/b|
    """
    abs_beta = abs(beta_1)

    print("\n" + "=" * 60)
    print("STABILITY ANALYSIS")
    print("=" * 60)
    print(f"\n|β₁| = |d/b| = {abs_beta:.4f}")

    if abs_beta < 1:
        print("\n✓ CONVERGENT COBWEB (|β₁| < 1)")
        print("  → Prices will converge to equilibrium over time")
        print("  → Oscillations will dampen")
    elif abs_beta == 1:
        print("\n○ PERPETUAL OSCILLATION (|β₁| = 1)")
        print("  → Prices will oscillate with constant amplitude")
        print("  → No convergence or divergence")
    else:
        print("\n✗ DIVERGENT COBWEB (|β₁| > 1)")
        print("  → Prices will diverge from equilibrium")
        print("  → Oscillations will amplify over time")

    return abs_beta


stability_coef = analyze_stability(results["beta_1"])

# =============================================================================
# STEP 5: FORECASTING POWER INVESTIGATION
# =============================================================================


def investigate_forecasting_power(data, results):
    """
    Investigate the forecasting power using various metrics
    """
    print("\n" + "=" * 60)
    print("FORECASTING POWER INVESTIGATION")
    print("=" * 60)

    # 1. In-sample fit
    y_actual = data["Price_t"].values
    y_pred = results["y_pred"]
    residuals = y_actual - y_pred

    print(f"\n1. In-Sample Performance:")
    print(
        f"   R² = {results['r2']:.4f} ({results['r2'] * 100:.2f}% of variance explained)"
    )
    print(f"   RMSE = {results['rmse']:.4f}")
    print(f"   Mean Absolute Error = {np.mean(np.abs(residuals)):.4f}")

    # 2. Residual analysis
    print(f"\n2. Residual Statistics:")
    print(f"   Mean: {np.mean(residuals):.4f} (should be close to 0)")
    print(f"   Std Dev: {np.std(residuals):.4f}")
    print(f"   Min: {np.min(residuals):.4f}")
    print(f"   Max: {np.max(residuals):.4f}")

    # 3. Out-of-sample validation (using last 10% of data)
    split_point = int(len(data) * 0.9)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]

    # Re-estimate on training data
    X_train = train_data[["Price_t_1"]].values
    y_train = train_data["Price_t"].values

    temp_model = LinearRegression()
    temp_model.fit(X_train, y_train)

    # Predict on test data
    X_test = test_data[["Price_t_1"]].values
    y_test = test_data["Price_t"].values
    y_test_pred = temp_model.predict(X_test)

    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"\n3. Out-of-Sample Validation (Last 10% of data):")
    print(f"   Test R² = {test_r2:.4f}")
    print(f"   Test RMSE = {test_rmse:.4f}")
    print(f"   Training periods: {len(train_data)}")
    print(f"   Testing periods: {len(test_data)}")

    return residuals, test_r2, test_rmse


residuals, test_r2, test_rmse = investigate_forecasting_power(cobweb_data, results)

# =============================================================================
# STEP 6: FORECAST 3 FUTURE PERIODS
# =============================================================================


def forecast_future_periods(last_price, beta_0, beta_1, n_periods=3):
    """
    Forecast future prices using the estimated difference equation
    P_{t+k} = β₀ + β₁ * P_{t+k-1}
    """
    forecasts = []
    current_price = last_price

    for i in range(1, n_periods + 1):
        next_price = beta_0 + beta_1 * current_price
        forecasts.append(
            {
                "Period": f"t+{i}",
                "Forecast": next_price,
                "Previous_Price": current_price,
            }
        )
        current_price = next_price

    return pd.DataFrame(forecasts)


# Get the last observed price
last_price = cobweb_data["Price_t"].iloc[-1]
last_date = df["Date"].iloc[-1]

# Generate forecasts
forecasts_df = forecast_future_periods(
    last_price, results["beta_0"], results["beta_1"], n_periods=3
)

print("\n" + "=" * 60)
print("3-PERIOD AHEAD FORECASTS")
print("=" * 60)
print(f"\nLast observed price (at {last_date}): ${last_price:.2f}")
print(
    f"\nForecast equation: P_{{t+k}} = {results['beta_0']:.4f} + ({results['beta_1']:.4f})*P_{{t+k-1}}"
)
print("\nForecasts:")
print(forecasts_df.to_string(index=False))

# Calculate long-run equilibrium
equilibrium_price = results["beta_0"] / (1 - results["beta_1"])
print(f"\nLong-run equilibrium price: ${equilibrium_price:.2f}")

# =============================================================================
# STEP 7: VISUALIZATIONS
# =============================================================================


def create_visualizations(df, cobweb_data, results, forecasts_df):
    """
    Create comprehensive visualizations for the report
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Historical Prices
    ax1 = axes[0, 0]
    ax1.plot(
        df["Date"], df["Price"], linewidth=2, color="darkblue", label="Coffee Prices"
    )
    ax1.set_title("Historical Coffee Prices", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Price (U.S. Cents / lb)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Actual vs Fitted Values
    ax2 = axes[0, 1]
    ax2.scatter(cobweb_data["Price_t"], results["y_pred"], alpha=0.5, s=30)
    ax2.plot(
        [cobweb_data["Price_t"].min(), cobweb_data["Price_t"].max()],
        [cobweb_data["Price_t"].min(), cobweb_data["Price_t"].max()],
        "r--",
        linewidth=2,
        label="Perfect Fit",
    )
    ax2.set_title(
        f"Actual vs Fitted Values (R² = {results['r2']:.4f})",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("Actual Price $P_t$", fontsize=12)
    ax2.set_ylabel("Fitted Price $\\hat{P}_t$", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals Over Time
    ax3 = axes[1, 0]
    residuals = cobweb_data["Price_t"] - results["y_pred"]
    ax3.scatter(range(len(residuals)), residuals, alpha=0.5, s=30, color="red")
    ax3.axhline(y=0, color="black", linestyle="--", linewidth=2)
    ax3.set_title("Residuals Over Time", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Observation", fontsize=12)
    ax3.set_ylabel("Residual", fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cobweb Diagram with Forecasts
    ax4 = axes[1, 1]

    # Plot the difference equation line
    p_range = np.linspace(df["Price"].min() * 0.8, df["Price"].max() * 1.2, 100)
    p_next = results["beta_0"] + results["beta_1"] * p_range
    ax4.plot(p_range, p_next, "b-", linewidth=2, label="$P_t = β_0 + β_1 P_{t-1}$")

    # 45-degree line (equilibrium)
    ax4.plot(p_range, p_range, "k--", linewidth=2, label="45° line")

    # Plot historical data
    ax4.scatter(
        cobweb_data["Price_t_1"],
        cobweb_data["Price_t"],
        alpha=0.3,
        s=20,
        color="gray",
        label="Historical",
    )

    # Plot forecasts with arrows
    prices = [last_price] + forecasts_df["Forecast"].tolist()
    for i in range(len(forecasts_df)):
        ax4.arrow(
            prices[i],
            prices[i],
            0,
            prices[i + 1] - prices[i],
            head_width=2,
            head_length=1,
            fc="red",
            ec="red",
            linewidth=2,
        )
        ax4.arrow(
            prices[i],
            prices[i + 1],
            prices[i + 1] - prices[i],
            0,
            head_width=2,
            head_length=1,
            fc="green",
            ec="green",
            linewidth=2,
        )
        ax4.scatter(
            prices[i],
            prices[i + 1],
            s=100,
            c="red",
            marker="o",
            zorder=5,
            edgecolors="black",
            linewidths=2,
        )

    ax4.set_title(
        "Cobweb Diagram with 3-Period Forecast", fontsize=14, fontweight="bold"
    )
    ax4.set_xlabel("$P_{t-1}$", fontsize=12)
    ax4.set_ylabel("$P_t$", fontsize=12)
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("cobweb_analysis.png", dpi=300, bbox_inches="tight")
    print("\n✓ Visualizations saved as 'cobweb_analysis.png'")
    plt.show()


create_visualizations(df, cobweb_data, results, forecasts_df)

# =============================================================================
# STEP 8: GENERATE SUMMARY REPORT
# =============================================================================


def generate_summary_report(df, results, forecasts_df, stability_coef):
    """
    Generate a comprehensive summary report
    """
    report = f"""
{"=" * 70}
COBWEB MODEL ANALYSIS - COFFEE MARKET
{"=" * 70}

1. DATA SUMMARY
   - Product: Coffee (Arabica)
   - Total Observations: {len(df)}
   - Date Range: {df["Date"].min()} to {df["Date"].max()}
   - Mean Price: U.S. Cents {df["Price"].mean():.2f}/lb
   - Std Dev: U.S. Cents {df["Price"].std():.2f}/lb
   - Min/Max: U.S. Cents {df["Price"].min():.2f} / ${df["Price"].max():.2f}

2. ESTIMATED DIFFERENCE EQUATION
   P_t = {results["beta_0"]:.4f} + ({results["beta_1"]:.4f}) × P_{{t-1}}

   Interpretation:
   - β₀ = (c + a)/b = {results["beta_0"]:.4f} (constant term)
   - β₁ = -d/b = {results["beta_1"]:.4f} (autoregressive coefficient)

3. MODEL PERFORMANCE
   - R² = {results["r2"]:.4f} ({results["r2"] * 100:.1f}% variance explained)
   - RMSE = {results["rmse"]:.4f}
   - This indicates {"good" if results["r2"] > 0.5 else "moderate"} forecasting power

4. STABILITY ANALYSIS
   - |β₁| = {stability_coef:.4f}
   - Model is: {"CONVERGENT" if stability_coef < 1 else "DIVERGENT" if stability_coef > 1 else "NEUTRAL"}
   - Interpretation: Prices will {"converge to equilibrium" if stability_coef < 1 else "diverge from equilibrium" if stability_coef > 1 else "oscillate perpetually"}

5. LONG-RUN EQUILIBRIUM
   - P* = β₀/(1-β₁) = U.S. Cents {results["beta_0"] / (1 - results["beta_1"]):.2f}/lb
   - Current price: U.S. Cents {df["Price"].iloc[-1]:.2f}/lb
   - Distance from equilibrium: U.S. Cents {abs(df["Price"].iloc[-1] - results["beta_0"] / (1 - results["beta_1"])):.2f}

6. THREE-PERIOD FORECASTS
"""

    for idx, row in forecasts_df.iterrows():
        report += f"   {row['Period']}: U.S. Cents {row['Forecast']:.2f}/lb\n"

    report += f"""
7. FORECASTING POWER ASSESSMENT
   ✓ The model explains {results["r2"] * 100:.1f}% of price variation
   {"✓" if results["r2"] > 0.5 else "⚠"} {"Good" if results["r2"] > 0.5 else "Moderate"} predictive accuracy
   {"✓" if stability_coef < 1 else "⚠"} {"Stable" if stability_coef < 1 else "Unstable"} dynamics suggest {"reliable" if stability_coef < 1 else "cautious"} forecasts

8. ECONOMIC INTERPRETATION
   - Supply elasticity (d): {"Positive" if results["beta_1"] < 0 else "Negative"} (as expected)
   - Demand elasticity (b): Determines price sensitivity
   - The cobweb pattern {"is evident" if 0.3 < stability_coef < 0.9 else "may be weak"}
   - Agricultural production lag is captured by P_{{t-1}} term

{"=" * 70}
"""

    # Save report
    with open("cobweb_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print("\n✓ Full report saved as 'cobweb_analysis_report.txt'")


generate_summary_report(df, results, forecasts_df, stability_coef)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print("\nGenerated files:")
print("  1. cobweb_analysis.png - Comprehensive visualizations")
print("  2. cobweb_analysis_report.txt - Full analysis report")
print("\n✓ Ready for your mathematical economy project submission!")

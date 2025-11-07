import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 12)

print("=" * 80)
print("45-DEGREE KEYNESIAN MODEL: US GDP FULL COMPONENT ANALYSIS")
print("=" * 80)

# =============================================================================
# STEP 1: LOAD AND PREPARE GDP COMPONENT DATA
# =============================================================================


def load_component_data():
    """
    Load all GDP components from FRED CSV files
    All series are in Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate
    """
    # Load GDP components
    gdp = pd.read_csv("GDPC1.csv", parse_dates=["DATE"])
    gdp.columns = ["Date", "GDP"]

    consumption = pd.read_csv("PCECC96.csv", parse_dates=["DATE"])
    consumption.columns = ["Date", "Consumption"]

    investment = pd.read_csv("GPDIC1.csv", parse_dates=["DATE"])
    investment.columns = ["Date", "Investment"]

    government = pd.read_csv("GCEC1.csv", parse_dates=["DATE"])
    government.columns = ["Date", "Government"]

    net_exports = pd.read_csv("NETEXC.csv", parse_dates=["DATE"])
    net_exports.columns = ["Date", "NetExports"]

    # Merge all components
    df = gdp.merge(consumption, on="Date", how="inner")
    df = df.merge(investment, on="Date", how="inner")
    df = df.merge(government, on="Date", how="inner")
    df = df.merge(net_exports, on="Date", how="inner")

    # Convert to Trillions for readability
    for col in ["GDP", "Consumption", "Investment", "Government", "NetExports"]:
        df[col] = df[col] / 1000

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# Load data
df = load_component_data()

print(f"\nData loaded: {len(df)} quarterly observations")
print(
    f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
)
print(
    f"Period covered: {(df['Date'].max() - df['Date'].min()).days / 365.25:.1f} years"
)
print(f"\nGDP Components (latest quarter, in Trillions of 2017$):")
print(f"  GDP:          ${df['GDP'].iloc[-1]:.2f}T")
print(
    f"  Consumption:  ${df['Consumption'].iloc[-1]:.2f}T ({df['Consumption'].iloc[-1] / df['GDP'].iloc[-1] * 100:.1f}% of GDP)"
)
print(
    f"  Investment:   ${df['Investment'].iloc[-1]:.2f}T ({df['Investment'].iloc[-1] / df['GDP'].iloc[-1] * 100:.1f}% of GDP)"
)
print(
    f"  Government:   ${df['Government'].iloc[-1]:.2f}T ({df['Government'].iloc[-1] / df['GDP'].iloc[-1] * 100:.1f}% of GDP)"
)
print(
    f"  Net Exports:  ${df['NetExports'].iloc[-1]:.2f}T ({df['NetExports'].iloc[-1] / df['GDP'].iloc[-1] * 100:.1f}% of GDP)"
)

# Verify identity: Y = C + I + G + NX
df["GDP_identity"] = (
    df["Consumption"] + df["Investment"] + df["Government"] + df["NetExports"]
)
identity_error = np.mean(np.abs(df["GDP"] - df["GDP_identity"]))
print(
    f"\nGDP Identity Check: Mean absolute error = ${identity_error:.4f}T (should be near zero)"
)

# =============================================================================
# STEP 2: CREATE GROWTH RATE VARIABLES (PROPER APPROACH)
# =============================================================================


def create_growth_variables(df):
    """
    Create quarter-over-quarter growth rates (in percentage points)
    This makes the model stationary and economically meaningful
    """
    data = df.copy()

    # Calculate percentage changes (growth rates)
    data["GDP_growth"] = data["GDP"].pct_change() * 100
    data["C_growth"] = data["Consumption"].pct_change() * 100
    data["I_growth"] = data["Investment"].pct_change() * 100
    data["G_growth"] = data["Government"].pct_change() * 100
    data["NX_growth"] = data[
        "NetExports"
    ].diff()  # For net exports, use absolute change

    # Create lagged growth rates
    data["GDP_growth_lag1"] = data["GDP_growth"].shift(1)
    data["C_growth_lag1"] = data["C_growth"].shift(1)
    data["I_growth_lag1"] = data["I_growth"].shift(1)
    data["G_growth_lag1"] = data["G_growth"].shift(1)
    data["NX_growth_lag1"] = data["NX_growth"].shift(1)

    # Also keep lagged GDP level for forecasting
    data["GDP_lag1"] = data["GDP"].shift(1)

    # Remove rows with NaN
    data = data.dropna().reset_index(drop=True)

    return data


keynesian_data = create_growth_variables(df)
print(f"\nGrowth rate data prepared: {len(keynesian_data)} observations")
print(f"\nAverage quarterly growth rates:")
print(
    f"  GDP:         {keynesian_data['GDP_growth'].mean():.2f}% (σ = {keynesian_data['GDP_growth'].std():.2f}%)"
)
print(
    f"  Consumption: {keynesian_data['C_growth'].mean():.2f}% (σ = {keynesian_data['C_growth'].std():.2f}%)"
)
print(
    f"  Investment:  {keynesian_data['I_growth'].mean():.2f}% (σ = {keynesian_data['I_growth'].std():.2f}%)"
)
print(
    f"  Government:  {keynesian_data['G_growth'].mean():.2f}% (σ = {keynesian_data['G_growth'].std():.2f}%)"
)

# =============================================================================
# STEP 3: ESTIMATE SIMPLE MODEL (GDP Growth)
# =============================================================================


def estimate_simple_model(data):
    """
    Estimate simple model: GDP_growth_t = α + β*GDP_growth_{t-1} + ε_t
    This captures GDP growth persistence
    """
    X = data[["GDP_growth_lag1"]].values
    y = data["GDP_growth"].values

    model = LinearRegression()
    model.fit(X, y)

    alpha = model.intercept_
    beta = model.coef_[0]

    # Predictions and metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    residuals = y - y_pred

    # Standard errors (for significance testing)
    n = len(y)
    k = 2  # Number of parameters
    mse = np.sum(residuals**2) / (n - k)
    var_beta = mse / np.sum((X - X.mean()) ** 2)
    se_beta = np.sqrt(var_beta)
    t_stat = beta / se_beta

    results = {
        "alpha": alpha,
        "beta": beta,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "model": model,
        "y_pred": y_pred,
        "residuals": residuals,
        "se_beta": se_beta,
        "t_stat": t_stat,
    }

    return results


print("\n" + "=" * 80)
print("MODEL 1: SIMPLE KEYNESIAN MODEL (GDP Growth Persistence)")
print("=" * 80)

simple_results = estimate_simple_model(keynesian_data)

print(
    f"\nEstimated Equation: GDP_growth_t = {simple_results['alpha']:.4f} + {simple_results['beta']:.4f}*GDP_growth_{{t-1}}"
)
print(f"\nParameter Interpretation:")
print(
    f"  α (intercept) = {simple_results['alpha']:.4f}% (long-run average growth rate)"
)
print(f"  β (persistence) = {simple_results['beta']:.4f} (growth rate autocorrelation)")
print(f"  Standard error of β = {simple_results['se_beta']:.4f}")
print(
    f"  t-statistic = {simple_results['t_stat']:.2f} ({'significant' if abs(simple_results['t_stat']) > 2 else 'not significant'} at 5% level)"
)
print(f"\nModel Fit:")
print(
    f"  R² = {simple_results['r2']:.4f} ({simple_results['r2'] * 100:.2f}% variance explained)"
)
print(f"  RMSE = {simple_results['rmse']:.4f}%")
print(f"  MAE = {simple_results['mae']:.4f}%")

# Stability check
if abs(simple_results["beta"]) < 1:
    print(f"\n✓ MODEL IS STABLE (|β| = {abs(simple_results['beta']):.4f} < 1)")
    print(
        f"  Long-run equilibrium growth = {simple_results['alpha'] / (1 - simple_results['beta']):.2f}%"
    )
else:
    print(f"\n✗ MODEL IS UNSTABLE (|β| = {abs(simple_results['beta']):.4f} ≥ 1)")

# =============================================================================
# STEP 4: ESTIMATE FULL COMPONENT MODEL
# =============================================================================


def estimate_component_model(data):
    """
    Estimate full model:
    GDP_growth_t = α + β₁*C_growth_{t-1} + β₂*I_growth_{t-1} +
                   β₃*G_growth_{t-1} + β₄*NX_growth_{t-1} + ε_t
    """
    X = data[
        ["C_growth_lag1", "I_growth_lag1", "G_growth_lag1", "NX_growth_lag1"]
    ].values
    y = data["GDP_growth"].values

    model = LinearRegression()
    model.fit(X, y)

    alpha = model.intercept_
    betas = model.coef_

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    residuals = y - y_pred

    results = {
        "alpha": alpha,
        "beta_C": betas[0],
        "beta_I": betas[1],
        "beta_G": betas[2],
        "beta_NX": betas[3],
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "model": model,
        "y_pred": y_pred,
        "residuals": residuals,
    }

    return results


print("\n" + "=" * 80)
print("MODEL 2: FULL COMPONENT MODEL (All GDP Components)")
print("=" * 80)

component_results = estimate_component_model(keynesian_data)

print(f"\nEstimated Equation:")
print(f"GDP_growth_t = {component_results['alpha']:.4f} + ")
print(f"               {component_results['beta_C']:.4f}*C_growth_{{t-1}} + ")
print(f"               {component_results['beta_I']:.4f}*I_growth_{{t-1}} + ")
print(f"               {component_results['beta_G']:.4f}*G_growth_{{t-1}} + ")
print(f"               {component_results['beta_NX']:.4f}*NX_growth_{{t-1}}")

print(f"\nComponent Contributions:")
print(f"  Consumption persistence (β_C): {component_results['beta_C']:.4f}")
print(f"  Investment persistence (β_I):  {component_results['beta_I']:.4f}")
print(f"  Government persistence (β_G):  {component_results['beta_G']:.4f}")
print(f"  Net Exports effect (β_NX):     {component_results['beta_NX']:.4f}")

print(f"\nModel Fit:")
print(
    f"  R² = {component_results['r2']:.4f} ({component_results['r2'] * 100:.2f}% variance explained)"
)
print(f"  RMSE = {component_results['rmse']:.4f}%")
print(f"  MAE = {component_results['mae']:.4f}%")

# Model comparison
print(f"\n🔍 Model Comparison:")
print(f"  Simple Model R² = {simple_results['r2']:.4f}")
print(f"  Component Model R² = {component_results['r2']:.4f}")
improvement = (
    (component_results["r2"] - simple_results["r2"]) / simple_results["r2"] * 100
)
print(f"  Improvement = {improvement:.1f}%")

# =============================================================================
# STEP 5: CALCULATE DYNAMIC MULTIPLIERS
# =============================================================================


def calculate_multipliers(beta, horizon=20):
    """
    Calculate dynamic multiplier path: β^h for h = 0, 1, 2, ..., horizon
    """
    multipliers = [beta**h for h in range(horizon + 1)]
    cumulative = np.cumsum(multipliers)

    # Long-run multiplier (if stable)
    if abs(beta) < 1:
        long_run = 1 / (1 - beta)
    else:
        long_run = np.inf

    df_mult = pd.DataFrame(
        {
            "Quarter": range(horizon + 1),
            "Impact_Multiplier": multipliers,
            "Cumulative_Multiplier": cumulative,
        }
    )

    return df_mult, long_run


multiplier_df, long_run_mult = calculate_multipliers(simple_results["beta"], horizon=20)

print("\n" + "=" * 80)
print("DYNAMIC MULTIPLIER ANALYSIS")
print("=" * 80)
print(f"\nBased on simple model with β = {simple_results['beta']:.4f}")
print(f"\nImpact Multipliers (first 8 quarters):")
print(
    multiplier_df[["Quarter", "Impact_Multiplier", "Cumulative_Multiplier"]]
    .head(8)
    .to_string(index=False)
)

if abs(simple_results["beta"]) < 1:
    print(f"\nLong-run multiplier: {long_run_mult:.4f}")
    print(
        f"Half-life (quarters to 50% adjustment): {np.log(0.5) / np.log(abs(simple_results['beta'])):.1f}"
    )
else:
    print(f"\n⚠ Model is unstable - long-run multiplier is infinite")

# =============================================================================
# STEP 6: OUT-OF-SAMPLE FORECASTING
# =============================================================================


def recursive_forecast(data, model_results, n_ahead=8, use_component=False):
    """
    Generate recursive forecasts for GDP growth and convert to GDP level
    """
    if use_component:
        # Use component model (more complex - not implemented in recursive form here)
        raise NotImplementedError(
            "Component model recursive forecasting not implemented"
        )

    # Use simple model for recursive forecasting
    last_growth = data["GDP_growth"].iloc[-1]
    last_gdp = data["GDP"].iloc[-1]

    forecasts = []
    current_growth = last_growth
    current_gdp = last_gdp

    for i in range(1, n_ahead + 1):
        # Forecast next quarter growth
        next_growth = model_results["alpha"] + model_results["beta"] * current_growth

        # Convert growth to level
        next_gdp = current_gdp * (1 + next_growth / 100)

        forecasts.append(
            {
                "Quarter_Ahead": i,
                "Forecast_Growth": next_growth,
                "Forecast_GDP": next_gdp,
            }
        )

        # Update for next iteration
        current_growth = next_growth
        current_gdp = next_gdp

    return pd.DataFrame(forecasts)


forecast_df = recursive_forecast(keynesian_data, simple_results, n_ahead=8)

print("\n" + "=" * 80)
print("8-QUARTER AHEAD RECURSIVE FORECASTS")
print("=" * 80)
print(
    f"\nStarting from: Q{keynesian_data['Date'].iloc[-1].quarter} {keynesian_data['Date'].iloc[-1].year}"
)
print(f"Last observed GDP: ${keynesian_data['GDP'].iloc[-1]:.2f}T")
print(f"Last observed growth: {keynesian_data['GDP_growth'].iloc[-1]:.2f}%")
print(f"\nForecasts:")
print(forecast_df.to_string(index=False))

# =============================================================================
# STEP 7: FORECAST ERROR ANALYSIS (IN-SAMPLE AND OUT-OF-SAMPLE)
# =============================================================================


def analyze_forecast_errors(data, model_results, test_split=0.85):
    """
    Comprehensive forecast error analysis
    """
    # In-sample errors
    in_sample_errors = model_results["residuals"]

    # Out-of-sample validation
    split_idx = int(len(data) * test_split)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    # Re-estimate on training data
    X_train = train_data[["GDP_growth_lag1"]].values
    y_train = train_data["GDP_growth"].values

    temp_model = LinearRegression()
    temp_model.fit(X_train, y_train)

    # Test on hold-out sample
    X_test = test_data[["GDP_growth_lag1"]].values
    y_test = test_data["GDP_growth"].values
    y_test_pred = temp_model.predict(X_test)

    test_errors = y_test - y_test_pred
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # MAPE (Mean Absolute Percentage Error)
    # Be careful with small denominators
    mape_denominator = np.abs(y_test)
    mape_denominator[mape_denominator < 0.1] = 0.1  # Avoid division by tiny numbers
    mape = np.mean(np.abs(test_errors / mape_denominator)) * 100

    error_stats = {
        "in_sample_rmse": np.sqrt(np.mean(in_sample_errors**2)),
        "in_sample_mae": np.mean(np.abs(in_sample_errors)),
        "out_sample_rmse": test_rmse,
        "out_sample_mae": test_mae,
        "out_sample_r2": test_r2,
        "out_sample_mape": mape,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "test_errors": test_errors,
    }

    return error_stats, test_data, y_test, y_test_pred


error_stats, test_data, y_test, y_test_pred = analyze_forecast_errors(
    keynesian_data, simple_results, test_split=0.85
)

print("\n" + "=" * 80)
print("FORECAST ERROR ANALYSIS")
print("=" * 80)

print(f"\n📊 In-Sample Performance:")
print(f"  RMSE = {error_stats['in_sample_rmse']:.4f}%")
print(f"  MAE  = {error_stats['in_sample_mae']:.4f}%")

print(f"\n📊 Out-of-Sample Performance (Last 15% of data):")
print(f"  Training observations: {error_stats['train_size']}")
print(f"  Testing observations:  {error_stats['test_size']}")
print(f"  Test R² = {error_stats['out_sample_r2']:.4f}")
print(f"  Test RMSE = {error_stats['out_sample_rmse']:.4f}%")
print(f"  Test MAE  = {error_stats['out_sample_mae']:.4f}%")
print(f"  Test MAPE = {error_stats['out_sample_mape']:.2f}%")

print(f"\n📈 Error Distribution (Out-of-Sample):")
print(f"  Mean error: {np.mean(error_stats['test_errors']):.4f}%")
print(f"  Std dev:    {np.std(error_stats['test_errors']):.4f}%")
print(
    f"  Min/Max:    {np.min(error_stats['test_errors']):.4f}% / {np.max(error_stats['test_errors']):.4f}%"
)

# =============================================================================
# STEP 8: COMPREHENSIVE VISUALIZATIONS
# =============================================================================


def create_full_visualizations(
    df,
    keynesian_data,
    simple_results,
    component_results,
    forecast_df,
    multiplier_df,
    error_stats,
    test_data,
    y_test,
    y_test_pred,
):
    """
    Create comprehensive visualization panel (9 subplots)
    """
    fig = plt.figure(figsize=(20, 14))
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Color scheme
    color_gdp = "#2E86AB"
    color_fit = "#A23B72"
    color_forecast = "#F18F01"
    color_residual = "#C73E1D"

    # -------------------------------------------------------------------------
    # Plot 1: GDP Level Time Series (top left)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df["Date"], df["GDP"], linewidth=2, color=color_gdp, label="Real GDP")
    ax1.fill_between(df["Date"], 0, df["GDP"], alpha=0.2, color=color_gdp)
    ax1.set_title("US Real GDP (Quarterly)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Trillions of 2017$", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 2: GDP Growth Rate (top center)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        keynesian_data["Date"],
        keynesian_data["GDP_growth"],
        linewidth=1.5,
        color=color_gdp,
        alpha=0.7,
    )
    ax2.axhline(
        y=keynesian_data["GDP_growth"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {keynesian_data['GDP_growth'].mean():.2f}%",
    )
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax2.fill_between(
        keynesian_data["Date"],
        0,
        keynesian_data["GDP_growth"],
        where=(keynesian_data["GDP_growth"] < 0),
        color="red",
        alpha=0.3,
        label="Recession",
    )
    ax2.set_title(
        "GDP Growth Rate (Quarter-over-Quarter)", fontsize=13, fontweight="bold"
    )
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel("Growth Rate (%)", fontsize=10)
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 3: GDP Component Shares (top right)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    latest_shares = {
        "Consumption": df["Consumption"].iloc[-1] / df["GDP"].iloc[-1] * 100,
        "Investment": df["Investment"].iloc[-1] / df["GDP"].iloc[-1] * 100,
        "Government": df["Government"].iloc[-1] / df["GDP"].iloc[-1] * 100,
        "Net Exports": df["NetExports"].iloc[-1] / df["GDP"].iloc[-1] * 100,
    }
    colors_pie = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    ax3.pie(
        latest_shares.values(),
        labels=latest_shares.keys(),
        autopct="%1.1f%%",
        colors=colors_pie,
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax3.set_title(
        f"GDP Component Shares ({df['Date'].iloc[-1].year})",
        fontsize=13,
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # Plot 4: Simple Model Fit (middle left)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(
        keynesian_data["GDP_growth_lag1"],
        keynesian_data["GDP_growth"],
        alpha=0.5,
        s=30,
        color=color_gdp,
        label="Actual",
    )
    ax4.plot(
        keynesian_data["GDP_growth_lag1"],
        simple_results["y_pred"],
        color=color_fit,
        linewidth=2,
        label="Fitted",
    )
    ax4.plot([-10, 10], [-10, 10], "k--", linewidth=1, alpha=0.5, label="45° line")
    ax4.set_title(
        f"Simple Model Fit (R² = {simple_results['r2']:.4f})",
        fontsize=13,
        fontweight="bold",
    )
    ax4.set_xlabel("GDP Growth$_{t-1}$ (%)", fontsize=10)
    ax4.set_ylabel("GDP Growth$_t$ (%)", fontsize=10)
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 5: Residuals Over Time (middle center)
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(
        range(len(simple_results["residuals"])),
        simple_results["residuals"],
        alpha=0.6,
        s=25,
        color=color_residual,
    )
    ax5.axhline(y=0, color="black", linestyle="--", linewidth=2)
    ax5.axhline(
        y=simple_results["rmse"],
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"±RMSE",
    )
    ax5.axhline(y=-simple_results["rmse"], color="orange", linestyle=":", linewidth=1.5)
    ax5.set_title("Residuals Over Time", fontsize=13, fontweight="bold")
    ax5.set_xlabel("Observation", fontsize=10)
    ax5.set_ylabel("Residual (%)", fontsize=10)
    ax5.legend(loc="best", fontsize=9)
    ax5.grid(True, alpha=0.3, axis="y")

    # -------------------------------------------------------------------------
    # Plot 6: Dynamic Multiplier Path (middle right)
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(
        multiplier_df["Quarter"],
        multiplier_df["Impact_Multiplier"],
        marker="o",
        linewidth=2,
        markersize=5,
        color=color_forecast,
        label="Impact Multiplier",
    )
    ax6.plot(
        multiplier_df["Quarter"],
        multiplier_df["Cumulative_Multiplier"],
        marker="s",
        linewidth=2,
        markersize=5,
        color=color_fit,
        label="Cumulative",
    )
    if abs(simple_results["beta"]) < 1:
        long_run_mult = 1 / (1 - simple_results["beta"])
        ax6.axhline(
            y=long_run_mult,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Long-run = {long_run_mult:.2f}",
        )
    ax6.set_title("Dynamic Multiplier Path", fontsize=13, fontweight="bold")
    ax6.set_xlabel("Quarters Ahead", fontsize=10)
    ax6.set_ylabel("Multiplier", fontsize=10)
    ax6.legend(loc="best", fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 20)

    # -------------------------------------------------------------------------
    # Plot 7: GDP Forecast (bottom left)
    # -------------------------------------------------------------------------
    ax7 = fig.add_subplot(gs[2, 0])

    # Plot historical GDP (last 40 quarters)
    hist_window = 40
    hist_dates = df["Date"].iloc[-hist_window:]
    hist_gdp = df["GDP"].iloc[-hist_window:]
    ax7.plot(
        hist_dates,
        hist_gdp,
        linewidth=2.5,
        color=color_gdp,
        label="Historical GDP",
        marker="o",
        markersize=3,
    )

    # Generate future dates for forecast
    last_date = df["Date"].iloc[-1]
    forecast_dates = pd.date_range(start=last_date, periods=9, freq="QS")[1:]

    # Plot forecast with connecting line
    last_gdp = df["GDP"].iloc[-1]
    forecast_gdp = forecast_df["Forecast_GDP"].values
    forecast_line = np.concatenate([[last_gdp], forecast_gdp])
    forecast_dates_full = np.concatenate([[last_date], forecast_dates])

    ax7.plot(
        forecast_dates_full,
        forecast_line,
        linewidth=2.5,
        color=color_forecast,
        linestyle="--",
        marker="s",
        markersize=6,
        label="8-Quarter Forecast",
    )

    # Shade forecast region
    ax7.axvspan(last_date, forecast_dates[-1], alpha=0.15, color=color_forecast)

    ax7.set_title("GDP Forecast (8 Quarters Ahead)", fontsize=13, fontweight="bold")
    ax7.set_xlabel("Date", fontsize=10)
    ax7.set_ylabel("GDP (Trillions $)", fontsize=10)
    ax7.legend(loc="upper left", fontsize=9)
    ax7.grid(True, alpha=0.3)
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)

    # -------------------------------------------------------------------------
    # Plot 8: Out-of-Sample Test Performance (bottom center)
    # -------------------------------------------------------------------------
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.scatter(
        y_test, y_test_pred, alpha=0.6, s=50, color=color_fit, edgecolors="black"
    )

    # Perfect prediction line
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax8.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )

    ax8.set_title(
        f"Out-of-Sample Test (R² = {error_stats['out_sample_r2']:.4f})",
        fontsize=13,
        fontweight="bold",
    )
    ax8.set_xlabel("Actual Growth (%)", fontsize=10)
    ax8.set_ylabel("Predicted Growth (%)", fontsize=10)
    ax8.legend(loc="best", fontsize=9)
    ax8.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 9: Forecast Error Distribution (bottom right)
    # -------------------------------------------------------------------------
    ax9 = fig.add_subplot(gs[2, 2])

    # Histogram
    ax9.hist(
        simple_results["residuals"],
        bins=30,
        color=color_residual,
        alpha=0.7,
        edgecolor="black",
        label="In-Sample Errors",
    )
    ax9.axvline(x=0, color="red", linestyle="--", linewidth=2.5, label="Zero Error")

    # Add normal distribution overlay
    mu = np.mean(simple_results["residuals"])
    sigma = np.std(simple_results["residuals"])
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)

    # Scale to match histogram
    from scipy.stats import norm

    ax9_twin = ax9.twinx()
    ax9_twin.plot(x, norm.pdf(x, mu, sigma), "b-", linewidth=2, label="Normal Dist.")
    ax9_twin.set_ylabel("Probability Density", fontsize=10, color="blue")
    ax9_twin.tick_params(axis="y", labelcolor="blue")

    ax9.set_title("Forecast Error Distribution", fontsize=13, fontweight="bold")
    ax9.set_xlabel("Forecast Error (%)", fontsize=10)
    ax9.set_ylabel("Frequency", fontsize=10)
    ax9.legend(loc="upper left", fontsize=9)
    ax9.grid(True, alpha=0.3, axis="y")

    # Add text box with error statistics
    textstr = f"Mean: {mu:.3f}%\nStd: {sigma:.3f}%\nRMSE: {simple_results['rmse']:.3f}%"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax9.text(
        0.05,
        0.95,
        textstr,
        transform=ax9.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    # Main title
    fig.suptitle(
        "45-Degree Keynesian Model: US GDP Full Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Save figure
    plt.savefig("keynesian_full_analysis.png", dpi=300, bbox_inches="tight")
    print("\n✓ Comprehensive visualizations saved as 'keynesian_full_analysis.png'")

    plt.show()


# Create all visualizations
create_full_visualizations(
    df,
    keynesian_data,
    simple_results,
    component_results,
    forecast_df,
    multiplier_df,
    error_stats,
    test_data,
    y_test,
    y_test_pred,
)

# =============================================================================
# STEP 9: GENERATE COMPREHENSIVE TEXT REPORT
# =============================================================================


def generate_comprehensive_report(
    df,
    keynesian_data,
    simple_results,
    component_results,
    forecast_df,
    multiplier_df,
    error_stats,
):
    """
    Generate detailed text report with all results
    """

    # Calculate additional statistics
    long_run_growth = (
        simple_results["alpha"] / (1 - simple_results["beta"])
        if abs(simple_results["beta"]) < 1
        else np.nan
    )

    report = f"""
{"=" * 90}
45-DEGREE KEYNESIAN MODEL: COMPREHENSIVE ANALYSIS REPORT
{"=" * 90}

ANALYSIS DATE: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
DATA SOURCE: Federal Reserve Economic Data (FRED)

{"=" * 90}
1. DATA SUMMARY
{"=" * 90}

Dataset Information:
    • Total Observations: {len(df)} quarters
    • Date Range: {df["Date"].min().strftime("%Y-%m-%d")} to {df["Date"].max().strftime("%Y-%m-%d")}
    • Time Span: {(df["Date"].max() - df["Date"].min()).days / 365.25:.1f} years
    • Frequency: Quarterly (Seasonally Adjusted Annual Rate)

Latest Quarter GDP Components ({df["Date"].iloc[-1].strftime("%Y-Q%q")}):
    • Real GDP:         ${df["GDP"].iloc[-1]:.2f}T  (100.0%)
    • Consumption (C):  ${df["Consumption"].iloc[-1]:.2f}T  ({df["Consumption"].iloc[-1] / df["GDP"].iloc[-1] * 100:.1f}%)
    • Investment (I):   ${df["Investment"].iloc[-1]:.2f}T  ({df["Investment"].iloc[-1] / df["GDP"].iloc[-1] * 100:.1f}%)
    • Government (G):   ${df["Government"].iloc[-1]:.2f}T  ({df["Government"].iloc[-1] / df["GDP"].iloc[-1] * 100:.1f}%)
    • Net Exports (NX): ${df["NetExports"].iloc[-1]:+.2f}T  ({df["NetExports"].iloc[-1] / df["GDP"].iloc[-1] * 100:+.1f}%)

Historical Averages (Full Sample):
    • Average GDP Growth: {keynesian_data["GDP_growth"].mean():.2f}% per quarter
    • Growth Volatility (σ): {keynesian_data["GDP_growth"].std():.2f}%
    • Minimum Growth: {keynesian_data["GDP_growth"].min():.2f}%
    • Maximum Growth: {keynesian_data["GDP_growth"].max():.2f}%
    • Negative Growth Quarters: {(keynesian_data["GDP_growth"] < 0).sum()} ({(keynesian_data["GDP_growth"] < 0).sum() / len(keynesian_data) * 100:.1f}%)

{"=" * 90}
2. MODEL 1: SIMPLE KEYNESIAN MODEL (GDP Growth Persistence)
{"=" * 90}

Estimated Equation:
    GDP_growth_t = {simple_results["alpha"]:.4f} + {simple_results["beta"]:.4f} × GDP_growth_(t-1) + ε_t

Parameter Interpretation:
    • α (Intercept) = {simple_results["alpha"]:.4f}%
    → Long-run average quarterly growth rate (when growth is stable)

    • β (Persistence Coefficient) = {simple_results["beta"]:.4f}
    → Measures autocorrelation in GDP growth
    → Shows how much of current growth carries over to next quarter
    → Standard Error: {simple_results["se_beta"]:.4f}
    → t-statistic: {simple_results["t_stat"]:.2f} ({"***Highly Significant" if abs(simple_results["t_stat"]) > 2.58 else "**Significant" if abs(simple_results["t_stat"]) > 1.96 else "*Marginally Significant" if abs(simple_results["t_stat"]) > 1.64 else "Not Significant"})

Model Stability:
    • |β| = {abs(simple_results["beta"]):.4f} {"< 1 ✓ STABLE" if abs(simple_results["beta"]) < 1 else "≥ 1 ✗ UNSTABLE"}
    • Long-run Equilibrium Growth: {long_run_growth:.2f}% per quarter ({long_run_growth * 4:.2f}% annualized)
    • Half-life of shocks: {np.log(0.5) / np.log(abs(simple_results["beta"])):.1f} quarters

Model Fit Statistics:
    • R² = {simple_results["r2"]:.4f} ({simple_results["r2"] * 100:.2f}% of variance explained)
    • Adjusted R² = {1 - (1 - simple_results["r2"]) * (len(keynesian_data) - 1) / (len(keynesian_data) - 2):.4f}
    • RMSE = {simple_results["rmse"]:.4f} percentage points
    • MAE = {simple_results["mae"]:.4f} percentage points

{"=" * 90}
3. MODEL 2: FULL COMPONENT MODEL
{"=" * 90}

Estimated Equation:
    GDP_growth_t = {component_results["alpha"]:.4f} +
                    {component_results["beta_C"]:.4f} × C_growth_(t-1) +
                    {component_results["beta_I"]:.4f} × I_growth_(t-1) +
                    {component_results["beta_G"]:.4f} × G_growth_(t-1) +
                    {component_results["beta_NX"]:.4f} × NX_change_(t-1) + ε_t

Component Persistence Coefficients:
    • Consumption (β_C):  {component_results["beta_C"]:.4f}
    • Investment (β_I):   {component_results["beta_I"]:.4f}
    • Government (β_G):   {component_results["beta_G"]:.4f}
    • Net Exports (β_NX): {component_results["beta_NX"]:.4f}

Model Fit Statistics:
    • R² = {component_results["r2"]:.4f} ({component_results["r2"] * 100:.2f}% of variance explained)
    • Adjusted R² = {1 - (1 - component_results["r2"]) * (len(keynesian_data) - 1) / (len(keynesian_data) - 5):.4f}
    • RMSE = {component_results["rmse"]:.4f} percentage points
    • MAE = {component_results["mae"]:.4f} percentage points

Model Comparison:
    • Simple Model R²:    {simple_results["r2"]:.4f}
    • Component Model R²: {component_results["r2"]:.4f}
    • Improvement:        {(component_results["r2"] - simple_results["r2"]) / simple_results["r2"] * 100:+.1f}%

{"=" * 90}
4. DYNAMIC MULTIPLIER ANALYSIS
{"=" * 90}

The dynamic multiplier shows how a one-time 1% shock to GDP growth propagates over time.

Impact Multipliers (β^h):
    Quarter 0 (Impact):    {multiplier_df["Impact_Multiplier"].iloc[0]:.4f}
    Quarter 1:             {multiplier_df["Impact_Multiplier"].iloc[1]:.4f}
    Quarter 2:             {multiplier_df["Impact_Multiplier"].iloc[2]:.4f}
    Quarter 4:             {multiplier_df["Impact_Multiplier"].iloc[4]:.4f}
    Quarter 8:             {multiplier_df["Impact_Multiplier"].iloc[8]:.4f}
    Quarter 12:            {multiplier_df["Impact_Multiplier"].iloc[12]:.4f}
    Quarter 20:            {multiplier_df["Impact_Multiplier"].iloc[20]:.4f}

Cumulative Multipliers:
    Quarter 0:             {multiplier_df["Cumulative_Multiplier"].iloc[0]:.4f}
    Quarter 4:             {multiplier_df["Cumulative_Multiplier"].iloc[4]:.4f}
    Quarter 8:             {multiplier_df["Cumulative_Multiplier"].iloc[8]:.4f}
    Quarter 12:            {multiplier_df["Cumulative_Multiplier"].iloc[12]:.4f}
    Quarter 20:            {multiplier_df["Cumulative_Multiplier"].iloc[20]:.4f}
    Long-run (∞):          {1 / (1 - simple_results["beta"]) if abs(simple_results["beta"]) < 1 else "Divergent":.4f if abs(simple_results['beta']) < 1 else 'Divergent'}

Interpretation:
    • A 1% positive shock to GDP growth in quarter 0 leads to:
    - {multiplier_df["Cumulative_Multiplier"].iloc[4]:.2f}% cumulative impact after 1 year (4 quarters)
    - {multiplier_df["Cumulative_Multiplier"].iloc[8]:.2f}% cumulative impact after 2 years (8 quarters)

{"=" * 90}
5. GDP FORECASTING RESULTS (8 Quarters Ahead)
{"=" * 90}

Starting Point:
    • Last Observed Date:   {keynesian_data["Date"].iloc[-1].strftime("%Y-Q%q")}
    • Last Observed GDP:    ${keynesian_data["GDP"].iloc[-1]:.2f}T
    • Last Observed Growth: {keynesian_data["GDP_growth"].iloc[-1]:.2f}%

Recursive Forecasts (using Simple Model):

"""

    # Add forecast table
    for idx, row in forecast_df.iterrows():
        report += f"  Quarter +{row['Quarter_Ahead']}: Growth = {row['Forecast_Growth']:6.2f}%  |  GDP = ${row['Forecast_GDP']:.2f}T\n"

    # Calculate forecast summary statistics
    avg_forecast_growth = forecast_df["Forecast_Growth"].mean()
    final_gdp = forecast_df["Forecast_GDP"].iloc[-1]
    total_growth = (
        (final_gdp - keynesian_data["GDP"].iloc[-1])
        / keynesian_data["GDP"].iloc[-1]
        * 100
    )

    report += f"""
Forecast Summary:
    • Average Forecasted Growth: {avg_forecast_growth:.2f}% per quarter ({avg_forecast_growth * 4:.2f}% annualized)
    • 8-Quarter Forecasted GDP:  ${final_gdp:.2f}T
    • Total Growth (2 years):     {total_growth:.2f}%
    • Converging to Equilibrium:  {long_run_growth:.2f}% per quarter

{"=" * 90}
6. FORECAST ERROR ANALYSIS
{"=" * 90}

IN-SAMPLE PERFORMANCE (Full Sample):
    • Root Mean Squared Error (RMSE): {error_stats["in_sample_rmse"]:.4f} percentage points
    • Mean Absolute Error (MAE):      {error_stats["in_sample_mae"]:.4f} percentage points
    • Mean Forecast Error:            {np.mean(simple_results["residuals"]):.4f}% (should be ≈0)
    • Standard Deviation of Errors:   {np.std(simple_results["residuals"]):.4f}%

OUT-OF-SAMPLE PERFORMANCE (Hold-out Test Set):
    • Training Set Size:    {error_stats["train_size"]} quarters ({error_stats["train_size"] / len(keynesian_data) * 100:.1f}%)
    • Testing Set Size:     {error_stats["test_size"]} quarters ({error_stats["test_size"] / len(keynesian_data) * 100:.1f}%)

    Test Set Metrics:
    • R² = {error_stats["out_sample_r2"]:.4f} ({error_stats["out_sample_r2"] * 100:.2f}% of variance explained)
    • RMSE = {error_stats["out_sample_rmse"]:.4f} percentage points
    • MAE = {error_stats["out_sample_mae"]:.4f} percentage points
    • MAPE = {error_stats["out_sample_mape"]:.2f}% (Mean Absolute Percentage Error)

Error Distribution (Out-of-Sample):
    • Mean:    {np.mean(error_stats["test_errors"]):.4f}%
    • Median:  {np.median(error_stats["test_errors"]):.4f}%
    • Std Dev: {np.std(error_stats["test_errors"]):.4f}%
    • Min:     {np.min(error_stats["test_errors"]):.4f}%
    • Max:     {np.max(error_stats["test_errors"]):.4f}%

Forecast Quality Assessment:
    • Out-of-sample R² > 0.30: {"✓ ACCEPTABLE" if error_stats["out_sample_r2"] > 0.30 else "✗ POOR"}
    • Out-of-sample MAPE < 20%: {"✓ GOOD" if error_stats["out_sample_mape"] < 20 else "⚠ FAIR" if error_stats["out_sample_mape"] < 40 else "✗ POOR"}
    • Model maintains performance on unseen data: {"✓ YES" if abs(simple_results["r2"] - error_stats["out_sample_r2"]) < 0.15 else "⚠ DEGRADATION"}

{"=" * 90}
7. KEY FINDINGS AND ECONOMIC INTERPRETATION
{"=" * 90}

1. GDP Growth Dynamics:
    • US GDP growth shows {"moderate" if 0.2 < simple_results["beta"] < 0.5 else "strong" if simple_results["beta"] >= 0.5 else "weak"} persistence (β = {simple_results["beta"]:.3f})
    • Shocks to GDP growth take approximately {np.log(0.5) / np.log(abs(simple_results["beta"])):.1f} quarters to dissipate by half
    • Long-run equilibrium growth rate: {long_run_growth:.2f}% per quarter ({long_run_growth * 4:.1f}% annually)

2. Component Contributions:
    • Consumption is the largest component ({df["Consumption"].iloc[-1] / df["GDP"].iloc[-1] * 100:.1f}% of GDP)
    • Investment is the most volatile component
    • Government spending provides stability
    • Net exports {"add to" if df["NetExports"].iloc[-1] > 0 else "subtract from"} GDP

3. Forecasting Implications:
    • Model provides {"good" if error_stats["out_sample_r2"] > 0.40 else "moderate" if error_stats["out_sample_r2"] > 0.25 else "limited"} out-of-sample predictive power
    • 8-quarter forecast suggests GDP will {"grow steadily" if avg_forecast_growth > 0.5 else "grow modestly" if avg_forecast_growth > 0 else "decline"}
    • Forecast uncertainty (±RMSE): ±{error_stats["out_sample_rmse"]:.2f} percentage points

4. Model Limitations:
    • Does not capture structural breaks or regime changes
    • Assumes linear relationships
    • Does not incorporate leading indicators
    • Limited to short-term forecasting (2-3 years maximum)

{"=" * 90}
8. TECHNICAL NOTES
{"=" * 90}

Methodology:
    • Model Type: Autoregressive (AR) model for GDP growth rates
    • Stationarity: First-differencing applied to achieve stationarity
    • Estimation: Ordinary Least Squares (OLS)
    • Validation: 85/15 train-test split
    • Forecasting: Recursive (iterative) 8-quarter ahead

Data Transformations:
    • GDP components converted from Billions to Trillions
    • Quarter-over-quarter percentage growth rates calculated
    • One-quarter lags created for autoregressive structure

Why Growth Rates Instead of Levels:
    • GDP levels are non-stationary (contain unit root)
    • Levels-on-levels regression yields spurious β ≈ 1
    • Growth rates are stationary and economically meaningful
    • Allows stable multiplier calculation

{"=" * 90}
END OF REPORT
{"=" * 90}
"""

    return report


# Generate and save report
report = generate_comprehensive_report(
    df,
    keynesian_data,
    simple_results,
    component_results,
    forecast_df,
    multiplier_df,
    error_stats,
)

with open("keynesian_full_report.txt", "w") as f:
    f.write(report)

print("\n✓ Comprehensive report saved as 'keynesian_full_report.txt'")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  1. keynesian_full_analysis.png  (9-panel visualization)")
print("  2. keynesian_full_report.txt    (detailed text report)")
print("\n" + "=" * 80)

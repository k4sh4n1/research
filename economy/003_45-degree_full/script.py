"""
45-Degree Model for GDP Analysis
Mathematical Economics Project
Analyzes US GDP data using Keynesian Cross framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Set style for clean visualizations
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class GDPModel45Degree:
    """
    Implements 45-degree line model for GDP analysis
    Includes simple and full component models
    """

    def __init__(self):
        self.data = None
        self.simple_model = None
        self.full_model = None
        self.multipliers = {}

    def load_data(self, start_year=1990):
        """Load and prepare data from CSV files"""
        print("Loading data...")

        # Load all components
        gdp = pd.read_csv("GDPC1.csv", parse_dates=["DATE"])
        consumption = pd.read_csv("PCECC96.csv", parse_dates=["DATE"])
        investment = pd.read_csv("GPDIC1.csv", parse_dates=["DATE"])
        government = pd.read_csv("GCEC1.csv", parse_dates=["DATE"])
        net_exports = pd.read_csv("NETEXC.csv", parse_dates=["DATE"])

        # Merge datasets
        df = gdp.merge(consumption, on="DATE", how="left")
        df = df.merge(investment, on="DATE", how="left")
        df = df.merge(government, on="DATE", how="left")
        df = df.merge(net_exports, on="DATE", how="left")

        # Filter for desired time period
        df = df[df["DATE"] >= f"{start_year}-01-01"].copy()

        # Handle missing net exports (interpolate)
        df["NETEXC"] = df["NETEXC"].interpolate(method="linear")

        # Create lagged variables for dynamic analysis
        df["GDP_LAG"] = df["GDPC1"].shift(1)
        df["C_LAG"] = df["PCECC96"].shift(1)

        # Calculate growth rates
        df["GDP_GROWTH"] = df["GDPC1"].pct_change() * 100
        df["C_GROWTH"] = df["PCECC96"].pct_change() * 100

        # Drop NaN rows
        self.data = df.dropna().reset_index(drop=True)

        print(
            f"Data loaded: {len(self.data)} quarters from {self.data['DATE'].min().year} to {self.data['DATE'].max().year}"
        )

    def build_simple_model(self):
        """Build simple consumption-based model: Y = a + bC"""
        print("\n" + "=" * 50)
        print("SIMPLE MODEL: Y = α + βC")
        print("=" * 50)

        X = self.data[["PCECC96"]].values
        y = self.data["GDPC1"].values

        model = LinearRegression()
        model.fit(X, y)

        self.simple_model = {
            "model": model,
            "alpha": model.intercept_,
            "beta": model.coef_[0],
            "predictions": model.predict(X),
        }

        # Calculate marginal propensity to consume (MPC)
        mpc = model.coef_[0]
        simple_multiplier = 1 / (1 - mpc)
        self.multipliers["simple"] = simple_multiplier

        # Model statistics
        r2 = model.score(X, y)
        rmse = np.sqrt(mean_squared_error(y, self.simple_model["predictions"]))

        print(f"α (Autonomous spending): ${self.simple_model['alpha']:.2f} billion")
        print(f"β (MPC): {self.simple_model['beta']:.4f}")
        print(f"Simple Multiplier: {simple_multiplier:.3f}")
        print(f"R²: {r2:.4f}")
        print(f"RMSE: ${rmse:.2f} billion")

    def build_full_model(self):
        """Build full component model: Y = C + I + G + NX"""
        print("\n" + "=" * 50)
        print("FULL MODEL: Y = C + I + G + NX")
        print("=" * 50)

        components = ["PCECC96", "GPDIC1", "GCEC1", "NETEXC"]
        X = self.data[components].values
        y = self.data["GDPC1"].values

        model = LinearRegression()
        model.fit(X, y)

        self.full_model = {
            "model": model,
            "intercept": model.intercept_,
            "coefficients": dict(zip(components, model.coef_)),
            "predictions": model.predict(X),
        }

        # Model statistics
        r2 = model.score(X, y)
        rmse = np.sqrt(mean_squared_error(y, self.full_model["predictions"]))

        print(f"Intercept: ${self.full_model['intercept']:.2f} billion")
        print("\nCoefficients:")
        for comp, coef in self.full_model["coefficients"].items():
            print(f"  {comp:12s}: {coef:.4f}")
        print(f"\nR²: {r2:.4f}")
        print(f"RMSE: ${rmse:.2f} billion")

    def calculate_dynamic_multiplier(self):
        """Calculate dynamic multiplier using VAR approach"""
        print("\n" + "=" * 50)
        print("DYNAMIC MULTIPLIER ANALYSIS")
        print("=" * 50)

        # Dynamic model: ΔY_t = α + β₁ΔC_t + β₂ΔY_{t-1}
        df = self.data.copy()
        df["DELTA_GDP"] = df["GDPC1"].diff()
        df["DELTA_C"] = df["PCECC96"].diff()
        df["DELTA_GDP_LAG"] = df["DELTA_GDP"].shift(1)

        # Remove NaN
        df = df.dropna()

        X = df[["DELTA_C", "DELTA_GDP_LAG"]].values
        y = df["DELTA_GDP"].values

        model = LinearRegression()
        model.fit(X, y)

        # Impact multiplier (immediate effect)
        impact_mult = model.coef_[0]

        # Long-run multiplier
        persistence = model.coef_[1]
        lr_mult = impact_mult / (1 - persistence)

        self.multipliers["impact"] = impact_mult
        self.multipliers["long_run"] = lr_mult
        self.multipliers["persistence"] = persistence

        print(f"Impact Multiplier: {impact_mult:.3f}")
        print(f"Persistence Parameter: {persistence:.3f}")
        print(f"Long-Run Multiplier: {lr_mult:.3f}")

        # Calculate cumulative multipliers over time
        periods = 12  # 3 years quarterly
        cumulative = []
        current = impact_mult
        cumulative.append(current)

        for t in range(1, periods):
            effect = impact_mult * (persistence**t)
            current += effect
            cumulative.append(current)

        self.multipliers["cumulative"] = cumulative

    def forecast_gdp(self, periods=8):
        """Forecast GDP using both models"""
        print("\n" + "=" * 50)
        print(f"FORECASTING ({periods} quarters ahead)")
        print("=" * 50)

        # Split data for validation
        split_point = len(self.data) - periods
        train = self.data[:split_point]
        test = self.data[split_point:]

        forecasts = {}

        # Simple model forecast
        X_test_simple = test[["PCECC96"]].values
        y_test = test["GDPC1"].values
        simple_pred = self.simple_model["model"].predict(X_test_simple)

        # Full model forecast
        X_test_full = test[["PCECC96", "GPDIC1", "GCEC1", "NETEXC"]].values
        full_pred = self.full_model["model"].predict(X_test_full)

        # Calculate errors
        simple_mape = mean_absolute_percentage_error(y_test, simple_pred) * 100
        full_mape = mean_absolute_percentage_error(y_test, full_pred) * 100

        simple_rmse = np.sqrt(mean_squared_error(y_test, simple_pred))
        full_rmse = np.sqrt(mean_squared_error(y_test, full_pred))

        print("\nForecast Accuracy:")
        print(f"Simple Model - MAPE: {simple_mape:.2f}%, RMSE: ${simple_rmse:.2f}B")
        print(f"Full Model   - MAPE: {full_mape:.2f}%, RMSE: ${full_rmse:.2f}B")

        self.forecast_results = {
            "test_dates": test["DATE"].values,
            "actual": y_test,
            "simple": simple_pred,
            "full": full_pred,
        }

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(16, 12))

        # 1. 45-degree diagram
        ax1 = plt.subplot(3, 3, 1)
        actual = self.data["GDPC1"].values
        predicted = self.simple_model["predictions"]

        ax1.scatter(actual, predicted, alpha=0.5, s=20)
        ax1.plot(
            [actual.min(), actual.max()],
            [actual.min(), actual.max()],
            "r--",
            lw=2,
            label="45° line",
        )
        ax1.set_xlabel("Actual GDP ($B)")
        ax1.set_ylabel("Predicted GDP ($B)")
        ax1.set_title("45-Degree Model Fit (Simple)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Time series - actual vs predicted
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(self.data["DATE"], self.data["GDPC1"], label="Actual", lw=2)
        ax2.plot(
            self.data["DATE"],
            self.simple_model["predictions"],
            label="Simple Model",
            lw=1,
            alpha=0.7,
        )
        ax2.plot(
            self.data["DATE"],
            self.full_model["predictions"],
            label="Full Model",
            lw=1,
            alpha=0.7,
        )
        ax2.set_xlabel("Date")
        ax2.set_ylabel("GDP ($B)")
        ax2.set_title("GDP: Actual vs Predicted")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Residuals analysis
        ax3 = plt.subplot(3, 3, 3)
        residuals = self.data["GDPC1"] - self.full_model["predictions"]
        ax3.plot(self.data["DATE"], residuals, lw=1)
        ax3.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        ax3.fill_between(self.data["DATE"], residuals, 0, alpha=0.3)
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Residuals ($B)")
        ax3.set_title("Model Residuals (Full Model)")
        ax3.grid(True, alpha=0.3)

        # 4. Dynamic multiplier path
        ax4 = plt.subplot(3, 3, 4)
        periods = range(len(self.multipliers["cumulative"]))
        ax4.plot(periods, self.multipliers["cumulative"], marker="o", lw=2)
        ax4.axhline(
            y=self.multipliers["long_run"],
            color="r",
            linestyle="--",
            alpha=0.5,
            label="Long-run",
        )
        ax4.set_xlabel("Quarters")
        ax4.set_ylabel("Cumulative Multiplier")
        ax4.set_title("Dynamic Multiplier Path")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Component contributions
        ax5 = plt.subplot(3, 3, 5)
        components = ["PCECC96", "GPDIC1", "GCEC1", "NETEXC"]
        recent_data = self.data.iloc[-1]
        values = [recent_data[c] for c in components]
        colors = sns.color_palette("husl", len(components))
        ax5.pie(np.abs(values), labels=components, autopct="%1.1f%%", colors=colors)
        ax5.set_title("GDP Components (Latest Quarter)")

        # 6. Forecast comparison
        ax6 = plt.subplot(3, 3, 6)
        dates = self.forecast_results["test_dates"]
        ax6.plot(dates, self.forecast_results["actual"], "ko-", label="Actual", lw=2)
        ax6.plot(
            dates, self.forecast_results["simple"], "s-", label="Simple", alpha=0.7
        )
        ax6.plot(dates, self.forecast_results["full"], "^-", label="Full", alpha=0.7)
        ax6.set_xlabel("Date")
        ax6.set_ylabel("GDP ($B)")
        ax6.set_title("Forecast Performance")
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis="x", rotation=45)

        # 7. GDP Growth Distribution
        ax7 = plt.subplot(3, 3, 7)
        ax7.hist(
            self.data["GDP_GROWTH"].dropna(), bins=30, alpha=0.7, edgecolor="black"
        )
        ax7.axvline(
            self.data["GDP_GROWTH"].mean(),
            color="r",
            linestyle="--",
            label=f"Mean: {self.data['GDP_GROWTH'].mean():.2f}%",
        )
        ax7.set_xlabel("GDP Growth Rate (%)")
        ax7.set_ylabel("Frequency")
        ax7.set_title("GDP Growth Distribution")
        ax7.legend()

        # 8. Consumption vs GDP Growth Correlation
        ax8 = plt.subplot(3, 3, 8)
        sns.regplot(x="C_GROWTH", y="GDP_GROWTH", data=self.data, ax=ax8)
        r, p = stats.pearsonr(self.data["C_GROWTH"], self.data["GDP_GROWTH"])
        ax8.set_title(f"Correlation (C vs Y Growth): r={r:.2f}, p={p:.3f}")
        ax8.set_xlabel("Consumption Growth (%)")
        ax8.set_ylabel("GDP Growth (%)")

        # 9. Summary text box
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis("off")
        summary_text = (
            f"45° GDP Model Summary\n\n"
            f"Period: {self.data['DATE'].min().year}–{self.data['DATE'].max().year}\n\n"
            f"Simple Model:\n"
            f"  α = {self.simple_model['alpha']:.2f}\n"
            f"  β (MPC) = {self.simple_model['beta']:.4f}\n"
            f"  Multiplier = {self.multipliers['simple']:.3f}\n\n"
            f"Dynamic Multipliers:\n"
            f"  Impact = {self.multipliers['impact']:.3f}\n"
            f"  Persistence = {self.multipliers['persistence']:.3f}\n"
            f"  Long-run = {self.multipliers['long_run']:.3f}\n\n"
            f"Forecast Accuracy:\n"
            f"  Simple MAPE = {mean_absolute_percentage_error(self.forecast_results['actual'], self.forecast_results['simple']) * 100:.2f}%\n"
            f"  Full MAPE = {mean_absolute_percentage_error(self.forecast_results['actual'], self.forecast_results['full']) * 100:.2f}%"
        )
        ax9.text(0.05, 0.95, summary_text, va="top", fontsize=10, family="monospace")

        plt.tight_layout()
        plt.show()


# -------------------------------
# EXECUTION SECTION
# -------------------------------
if __name__ == "__main__":
    model = GDPModel45Degree()
    model.load_data(start_year=1990)
    model.build_simple_model()
    model.build_full_model()
    model.calculate_dynamic_multiplier()
    model.forecast_gdp(periods=8)
    model.visualize_results()

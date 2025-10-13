# Cross domain investigation: applying structural engineering methodologies to financial market prediction and risk assessment

A research proposal by kashani@alum.sharif.edu

PhD student at University of Tabriz

Civil engineering department

Structural engineering

## Highlights

* Structural systems have similarities with financial systems
* Structural engineering methodologies are mostly unexplored in financial markets
* There are innovation opportunities due to the cross-domain gap
* Novel models can be presented for financial systems based on structural ones

## Keywords

Structural system, financial market, econophysics, algorithmic 
trading, cross-domain modeling, quantitative finance

## Abstract

Financial markets exhibit stochastic behaviors similar to structural engineering systems e.g. oscillation, shock response, and collapse. Nonetheless the sophisticated mathematical toolkit developed for structural systems remains largely unapplied to financial analysis. This research proposes a systematic investigation for applying structural engineering methodologies — probably those derived from structural health monitoring, structural control, and finite element modeling - into quantitative finance. With algorithmic trading now comprising `92%` of market volume and foreign-exchange being a `$7.5` trillion-a-day market in 2022 and a `$9.6` trillion-a-day market in 2025, even marginal improvements in prediction accuracy could provide significant economic impact. Preliminary results by proof-of-concept implementations on financial timeseries are revealing patterns and insights not visible by traditional financial tools. This multidisciplinary approach could provide opportunities for market understanding, risk management, and algorithmic trading strategy development.

## Research philosophy and motivation

Structural systems and financial markets share foundational similarities:

* Stochastic behavior under external forces
   * Financial forces of buyers against sellers or bulls against bears
   * Stochastic nature of seismic loads and financial shocks
* Extreme events
   * Structural collapse
   * Market crashes
* Characteristic frequency and damping behavior
* Need for real-time monitoring and anomaly detection
   * Structural health monitoring or SHM
   * Market surveillance
* Forensic similarities
   * Structural forensic analysis involves investigation and inspection of structures
   * Technical analysis involves study of past market data for forecasting price direction
* Support and resistance terminology for both structures and price action
* Similar patterns of stress buildup and sudden release
* Structural energy and market momentum
* Similarities between structural vibrations and market fluctuations
* Equilibrium concept
   * Structural forces
   * Supply and demand across macro and micro economy
* Financial friction and mechanical one
* Nonlinear dynamics behavior
* Shock propagation through the financial network and structural network
* Thresholds
   * Financial tipping points
   * Structural buckling
* ...

For example, when a trader follows the price action methodology, it feels like they are telling a story of bulls and bears. The story of how price momentum is shaped by buyers and sellers. This story-telling remarkably resembles how external forces are exerted upon a physical body to determine its momentum and behavior.

Despite the similarities, the rich structural toolkit remains mostly unexplored in quantitative finance. This gap presents opportunities for practical innovation for:

* Theoretical contributions to econophysics
* Novel predictive tools based on physical systems theory
* Improved understanding of financial markets through physical systems

## Related work

The book "The Physics of Wall Street" by James Owen Weatherall mentions a geophysicist who used a model designed for earthquakes to predict a massive stock market crash [@weatherall2013]. Also, it discusses about a physicist-run hedge fund that earned `2478.6%` over the course of the 1990s. Financial industry has been using physical models to effectively figure out the financial behavior. There are some academic studies carried out, but systematic investigation of structural methodologies remains mostly absent.

## Literature gap

The literature review reveals several critical gaps. While physics-inspired approaches to financial markets have gained traction, the specific application of structural engineering methodologies remains underexplored.

The rich mathematical frameworks available for structural hazard analysis, structural health monitoring, structural control, and finite element method, have seen minimal systematic application. Although econophysics has applied concepts from mechanics and thermodynamics to financial markets [@mantegna2000] [@bouchaud2003].

Some recent studies have investigated the impact of global news events on major equity and bond markets through the lens of seismology, but without leveraging the full toolkit of structural engineering [@pagnottoni2021].

There is no comprehensive framework mapping structural parameters, including mass, stiffness, damping, to financial variables in a systematic way. The few attempts that exist focus on specific phenomena rather than developing a general theoretical framework [@filimonov2013].

Traditional financial analysis relies on Fourier transforms and wavelet analysis [@ramsey2002]. However, the specific advantages of structural modal analysis for identifying market regimes and predicting transitions remain unexplored. Modal analysis is a prominent tool of structural dynamics used to identify natural frequencies and mode shapes. It can be systematically applied to decompose financial time series.

The finite element method or FEM has revolutionized structural engineering. FEM application to financial markets exists but is limited. Some FEM applications exist in option pricing and financial derivatives [@pironneau2011] [@topper2001]. Discretizing the market into elements and analyzing stress distribution by FEM represents a significant potential opportunity.

The literature review reveals a huge gap in cross-domain expertise. Financial experts generally lack the knowledge of physical systems, and structural engineers rarely dive into finance. Exceptions like Didier Sornette [@sornette_wiki] have indicated the potential, but systematic investigation remains absent.

Structural health monitoring or SHM, which is able to detect structural damage in real-time, is parallel to market surveillance and crash prediction. However, no evidence of systematic SHM adaptation to finance is found.

Market crashes are equivalent to structural failures, however the sophisticated mathematical tools for analyzing shock response in structures, including response spectra, time-history analysis, and stochastic dynamic analysis, have not been systematically adapted for financial shock events. Some studies have been carried out but they see the problem from a network model perspective not the dynamic response characteristics central to structural analysis [@gai2010].

In structural engineering the physical experiments would validate theoretical models. Frameworks for validating structural models against market data can extend to both back-testing and real-time testing. The performance metrics could be specifically developed for physics-based trading strategies. The literature lacks frameworks for validating physics-based financial models against market data. This gap presents an opportunity for academic and practical innovation. Considering the increasing dominance of algorithmic trading, there is a need for novel validated forecasting in financial markets.

## Background

The author's educational and industrial background is:

* Civil engineering, BSc
* Earthquake engineering, MSc
* Structural engineering, MSc
* 10+ years of professional software development by C++, Go, Python, JS, C#, and more
* Experience with algorithm development on MetaTrader5 platform by MQL5 language

This multidisciplinary background provides the opportunity to:

* Investigate mathematical similarities between physical and financial systems
* Implement efficient algorithms
* Innovate theoretical toolkits for practical applications

## Significance

This research could have significant impact due to:

* Significant academic gap between engineering fields and quantitative finance
   * Lack of enough multidisciplinary researchers in both fields
* Significant trading volume of financial markets
   * In 2022, currency market alone had a `$7.5` trillion _daily_ volume [@bis2022]
   * In 2025, foreign exchange has become a `$9.6` trillion-a-day market [@johnson2025]
   * `92%` of trading is done by algorithms and computers in 2019 [@groette2024]

## General research questions

Two main general questions are to be answered:

* Can structural engineering methodologies help with _technical_ analysis of financial markets?
* Can structural engineering methodologies help with _fundamental_ analysis of financial markets?

A bit more specific:

* Can structural engineering methodologies identify specific characteristics of financial markets?
* Can structural engineering methodologies help with financial timeseries forecasting?
* Can structural engineering methodologies predict market crashes?
* Can structural engineering methodologies detect market regime changes?
* How do structural engineering methodologies apply to financial markets?
* ...

## Specific research hypotheses

Beyond the general questions, this research might test specific hypotheses:

* Hypothesis statement 1:
   * Traditional structural engineering methods can predict the price action with a win rate of above `50%`.
* Hypothesis statement 2:
   * Stochastic structural engineering methods can predict the price action with a win rate of above `53%`.
* Hypothesis statement 3:
   * ML-assisted structural engineering methods can predict the price action with a win rate of above `55%`.
* Hypothesis statement 4:
   * Finite element modeling of financial market network will identify risk `24-48` hours in advance.
* Hypothesis statement 5:
   * Earthquake power laws can approximate the price action after news announcements with a precision above `60%`.
* Hypothesis statement 6:
   * Structural engineering methods can provide reliable fundamental analysis `1` month in advance.
* Hypothesis statements...

## Research methodology

The following plan can be outlined.

* Stage 1: literature review
* Stage 2: agile investigation of traditional structural engineering methods
   * Agile approach comes from software development practices
   * Candidates of traditional methods:
      - Mass-spring systems
      - Modal analysis
      - Natural frequency detection
      - Spectral analysis
      - Fourier transform, FFT, and HHT
      - Structural health monitoring methods
      - Damping ratio
      - Finite element analysis, FEA
      - ...
* Stage 3: agile investigation of stochastic and ML-assisted structural engineering methods
   * Agile approach will be followed, coming from software development practices
* Stage 4: Selection and optimizations of methodologies according to investigation results
   * C++ for high performance computing
   * Python for investigation
   * Visualization on MetaTrader platform
* Stage 5: validation and testing
   * Back-testing with multiple decades of historic market data
   * Live testing on demo trading accounts

Potential mapping table between physical and financial systems:

| Physics Term | Symbol     | Financial Interpretation and Possible Equivalent |
|--------------|------------|--------------------------------------------------|
| Mass         | m          | market capitalization or liquidity               |
| Stiffness    | k          | mean reversion strength                          |
| Damping      | c          | volatility decay rate                            |
| Displacement | x          | price deviation from equilibrium                 |
| Velocity     | $\dot{x}$  | returns or price change rate                     |
| Acceleration | $\ddot{x}$ | return acceleration or volatility                |
| Force        | F          | order flow imbalance or buy-sell pressure        |
| ...          | ...        | ...                                              |

## Expected contributions

Scientific contributions:

* Novel systematic application of structural engineering methodologies into finance
* New class of theoretical models for market behavior
* Mathematical framework bridging physical and financial systems

Practical contributions:

* Improved market prediction algorithms
* Novel trading signal generation methods
* Innovative risk detection tools
* Novel financial frameworks for technical and fundamental analysis

Possibly and potentially, the following targets might be kept in mind.

* Expected journal publications, like The Journal of Computational Finance
* Potential industry partnerships
* Opensource software deliverables

## Timeline

A rough timeline might be approximately estimated as follows.

* Year 1:
   * literature review
   * Agile sprints for investigating traditional methods
      * Implementation, visualization, and validation
* Year 2 and 3
   * Agile sprints for investigating stochastic methods
      * Implementation, visualization, and validation
   * Agile sprints for ML-assisted methods
      * Implementation, visualization, and validation
* Year 4: optimization and dissertation writing

The concepts of _agile_ and _sprint_ are coming from software development industry best practices.

## Preliminary results

As a proof-of-concept, an MQL5 code on MetaTrader5 platform is developed. Hilbert-Huang Transform (HHT) approach is employed to carry out the frequency spectrum analysis. HHT is particularly useful for non-linear and non-stationary signals. HHT is commonly applied to seismic signals, here HHT is applied to financial timeseries.

Some specifics of the proof-of-concept test:

* HHT and FFT methods are implemented to analyze the price data
* Currency: EUR/USD pair
* Timeframe: M1 or 1-minutes bars
* Real-time testing is done by a demo account on MetaTrader platform
* Data points analyzed: 200, it is an adjustable input parameter, therefore it's trivial to change

Dominant periods and cycles are identified and visualized on the chart. Sample results on Oct. 9, 2025 around 6:37 PM local time:

* Main periods:
   * 3.0 bars or 3 minutes
   * 9.2 bars or 9.2 minutes
   * 14.3 bars or 14.3 minutes
   * 20.0 bars or 20.0 minutes
   * 23.0 bars or 23.0 minutes
* Frequencies (cycles/bar):
   * 0.3378
   * 0.1088
   * 0.0700
   * 0.0500
   * 0.0435

The preliminary results indicate that systematic investigation could contribute multidisciplinary innovations inspired by structural dynamics into financial market analysis. Novel contributions to both theory and practice are highly probable.

## References

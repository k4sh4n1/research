//+------------------------------------------------------------------+
//|                                              FrequencySpectrum.mq5 |
//|                                   Currency Frequency Analyzer      |
//|                                                                    |
//+------------------------------------------------------------------+
#property copyright "Currency Frequency Analyzer"
#property link      ""
#property version   "1.00"
#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   2
#property indicator_minimum 0
#property indicator_maximum 1.1

//--- plot Spectrum
#property indicator_label1  "Spectrum"
#property indicator_type1   DRAW_HISTOGRAM
#property indicator_color1  clrDodgerBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

//--- plot Dominant
#property indicator_label2  "Dominant Cycles"
#property indicator_type2   DRAW_ARROW
#property indicator_color2  clrRed
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

//--- input parameters
input int      InpBarsToAnalyze = 256;        // Number of bars to analyze (power of 2 recommended)
input double   InpMinPeriod = 5.0;            // Minimum period of interest (in bars)
input double   InpMaxPeriod = 100.0;          // Maximum period of interest (in bars)
input bool     InpShowDominantCycles = true;  // Show dominant cycles
input int      InpDominantCycles = 5;         // Number of dominant cycles to identify
input bool     InpDetrend = true;             // Detrend data before FFT
input color    InpTextColor = clrWhite;       // Text color for labels
input int      InpSpectrumBars = 200;         // Number of bars to display spectrum

//--- indicator buffers
double         SpectrumBuffer[];
double         DominantBuffer[];
double         PeriodBuffer[];

//--- global variables
double         g_priceData[];
double         g_realPart[];
double         g_imagPart[];
double         g_spectrum[];
int            g_dataLength;
int            g_fftLength;
string         g_prefix;
bool           g_calculated = false;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                          |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Calculate FFT length (next power of 2)
   g_dataLength = InpBarsToAnalyze;
   g_fftLength = 1;
   while(g_fftLength < g_dataLength)
      g_fftLength *= 2;

//--- Set indicator properties
   string short_name = StringFormat("Frequency Spectrum (%d bars, %.1f-%.1f period range)",
                                    g_dataLength, InpMinPeriod, InpMaxPeriod);
   IndicatorSetString(INDICATOR_SHORTNAME, short_name);
   IndicatorSetInteger(INDICATOR_LEVELS, 2);
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 0, 0.5);
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 1, 0.75);
   IndicatorSetString(INDICATOR_LEVELTEXT, 0, "50%");
   IndicatorSetString(INDICATOR_LEVELTEXT, 1, "75%");
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 0, clrGray);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 1, clrGray);
   IndicatorSetInteger(INDICATOR_LEVELSTYLE, 0, STYLE_DOT);
   IndicatorSetInteger(INDICATOR_LEVELSTYLE, 1, STYLE_DOT);
   IndicatorSetInteger(INDICATOR_DIGITS, 4);

//--- Map indicator buffers
   SetIndexBuffer(0, SpectrumBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, DominantBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, PeriodBuffer, INDICATOR_CALCULATIONS);

//--- Set arrays as series
   ArraySetAsSeries(SpectrumBuffer, true);
   ArraySetAsSeries(DominantBuffer, true);
   ArraySetAsSeries(PeriodBuffer, true);

//--- Set empty value
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);

//--- Set arrow code for dominant cycles
   PlotIndexSetInteger(1, PLOT_ARROW, 159);

//--- Initialize arrays
   ArrayResize(g_priceData, g_fftLength);
   ArrayResize(g_realPart, g_fftLength);
   ArrayResize(g_imagPart, g_fftLength);
   ArrayResize(g_spectrum, g_fftLength/2);

//--- Create unique prefix for text objects
   g_prefix = "FreqSpec_" + IntegerToString(ChartID()) + "_";

//--- Clear any existing text objects
   ObjectsDeleteAll(ChartWindowFind(), g_prefix);

//--- Create period scale labels
   CreatePeriodScale();

   Print("Frequency Spectrum Indicator initialized. FFT Length: ", g_fftLength);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                        |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Clean up text objects
   ObjectsDeleteAll(ChartWindowFind(), g_prefix);
  }

//+------------------------------------------------------------------+
//| Custom indicator iteration function                               |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
//--- Check if we have enough data
   if(rates_total < g_dataLength)
     {
      Print("Not enough data. Need ", g_dataLength, " bars, have ", rates_total);
      return(0);
     }

//--- Set arrays as series
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(time, true);

//--- Calculate only on new bar
   if(prev_calculated > 0 && prev_calculated == rates_total && g_calculated)
      return(rates_total);

//--- Initialize all buffers
   int bars_to_show = MathMin(InpSpectrumBars, rates_total);
   for(int i = 0; i < bars_to_show; i++)
     {
      SpectrumBuffer[i] = EMPTY_VALUE;
      DominantBuffer[i] = EMPTY_VALUE;
      PeriodBuffer[i] = EMPTY_VALUE;
     }

//--- Prepare price data
   ArrayInitialize(g_priceData, 0.0);
   ArrayInitialize(g_realPart, 0.0);
   ArrayInitialize(g_imagPart, 0.0);

//--- Copy close prices
   for(int i = 0; i < g_dataLength; i++)
     {
      g_priceData[i] = close[i];
     }

//--- Detrend if requested
   if(InpDetrend)
     {
      DetrendData(g_priceData, g_dataLength);
     }

//--- Apply window function to reduce spectral leakage
   ApplyHammingWindow(g_priceData, g_dataLength);

//--- Prepare data for FFT
   for(int i = 0; i < g_fftLength; i++)
     {
      if(i < g_dataLength)
         g_realPart[i] = g_priceData[i];
      else
         g_realPart[i] = 0.0;
      g_imagPart[i] = 0.0;
     }

//--- Perform FFT
   FastFourierTransform(g_realPart, g_imagPart, g_fftLength, false);

//--- Calculate power spectrum
   CalculatePowerSpectrum();

//--- Map spectrum to indicator buffer
   MapSpectrumToBuffer(bars_to_show);

//--- Find and display dominant cycles
   if(InpShowDominantCycles)
     {
      FindDominantCycles();
     }

//--- Update period scale if window size changed
   UpdatePeriodScale();

   g_calculated = true;

   return(rates_total);
  }

//+------------------------------------------------------------------+
//| Fast Fourier Transform implementation                            |
//+------------------------------------------------------------------+
void FastFourierTransform(double &real[], double &imag[], int n, bool inverse)
  {
//--- Bit reversal
   int j = 0;
   for(int i = 0; i < n; i++)
     {
      if(i < j)
        {
         double temp_real = real[i];
         double temp_imag = imag[i];
         real[i] = real[j];
         imag[i] = imag[j];
         real[j] = temp_real;
         imag[j] = temp_imag;
        }

      int m = n / 2;
      while(m >= 1 && j >= m)
        {
         j -= m;
         m /= 2;
        }
      j += m;
     }

//--- FFT computation
   int mmax = 1;
   while(mmax < n)
     {
      int istep = 2 * mmax;
      double theta = (inverse ? 2.0 : -2.0) * M_PI / istep;
      double wpr = MathCos(theta);
      double wpi = MathSin(theta);
      double wr = 1.0;
      double wi = 0.0;

      for(int m = 0; m < mmax; m++)
        {
         for(int i = m; i < n; i += istep)
           {
            int j = i + mmax;
            double tr = wr * real[j] - wi * imag[j];
            double ti = wr * imag[j] + wi * real[j];
            real[j] = real[i] - tr;
            imag[j] = imag[i] - ti;
            real[i] += tr;
            imag[i] += ti;
           }
         double temp = wr;
         wr = wr * wpr - wi * wpi;
         wi = temp * wpi + wi * wpr;
        }
      mmax = istep;
     }

//--- Normalize if inverse
   if(inverse)
     {
      for(int i = 0; i < n; i++)
        {
         real[i] /= n;
         imag[i] /= n;
        }
     }
  }

//+------------------------------------------------------------------+
//| Detrend data using linear regression                             |
//+------------------------------------------------------------------+
void DetrendData(double &data[], int length)
  {
   double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;

//--- Calculate sums for linear regression
   for(int i = 0; i < length; i++)
     {
      sum_x += i;
      sum_y += data[i];
      sum_xy += i * data[i];
      sum_x2 += i * i;
     }

//--- Calculate slope and intercept
   double slope = (length * sum_xy - sum_x * sum_y) / (length * sum_x2 - sum_x * sum_x);
   double intercept = (sum_y - slope * sum_x) / length;

//--- Remove trend
   for(int i = 0; i < length; i++)
     {
      data[i] -= (intercept + slope * i);
     }
  }

//+------------------------------------------------------------------+
//| Apply Hamming window to reduce spectral leakage                  |
//+------------------------------------------------------------------+
void ApplyHammingWindow(double &data[], int length)
  {
   for(int i = 0; i < length; i++)
     {
      double window = 0.54 - 0.46 * MathCos(2.0 * M_PI * i / (length - 1));
      data[i] *= window;
     }
  }

//+------------------------------------------------------------------+
//| Calculate power spectrum from FFT result                          |
//+------------------------------------------------------------------+
void CalculatePowerSpectrum()
  {
   ArrayInitialize(g_spectrum, 0.0);

//--- Calculate magnitude for each frequency
   for(int i = 0; i < g_fftLength/2; i++)
     {
      g_spectrum[i] = MathSqrt(g_realPart[i] * g_realPart[i] + g_imagPart[i] * g_imagPart[i]);
     }

//--- Normalize spectrum (skip DC component at index 0)
   double max_value = 0;
   for(int i = 1; i < g_fftLength/2; i++)
     {
      if(g_spectrum[i] > max_value)
         max_value = g_spectrum[i];
     }

   if(max_value > 0)
     {
      for(int i = 1; i < g_fftLength/2; i++)
        {
         g_spectrum[i] /= max_value;
        }
     }
  }

//+------------------------------------------------------------------+
//| Map spectrum to indicator buffer based on period range            |
//+------------------------------------------------------------------+
void MapSpectrumToBuffer(int bars_to_show)
  {
//--- Map each bar to a period within the specified range
   double period_step = (InpMaxPeriod - InpMinPeriod) / (double)(bars_to_show - 1);

   for(int bar = 0; bar < bars_to_show; bar++)
     {
      //--- Calculate the period for this bar position
      double target_period = InpMinPeriod + bar * period_step;

      //--- Find the corresponding frequency bin
      if(target_period > 2.0) // Avoid division by very small periods
        {
         double frequency = (double)g_fftLength / target_period;
         int freq_bin = (int)MathRound(frequency);

         //--- Ensure we're within valid range
         if(freq_bin >= 1 && freq_bin < g_fftLength/2)
           {
            SpectrumBuffer[bars_to_show - 1 - bar] = g_spectrum[freq_bin];
            PeriodBuffer[bars_to_show - 1 - bar] = target_period;
           }
         else
           {
            SpectrumBuffer[bars_to_show - 1 - bar] = 0.0;
            PeriodBuffer[bars_to_show - 1 - bar] = target_period;
           }
        }
     }

//--- Smooth the spectrum display
   for(int i = 1; i < bars_to_show - 1; i++)
     {
      if(SpectrumBuffer[i] != EMPTY_VALUE && SpectrumBuffer[i-1] != EMPTY_VALUE && SpectrumBuffer[i+1] != EMPTY_VALUE)
        {
         SpectrumBuffer[i] = (SpectrumBuffer[i-1] + 2*SpectrumBuffer[i] + SpectrumBuffer[i+1]) / 4.0;
        }
     }
  }

//+------------------------------------------------------------------+
//| Find dominant cycles in the spectrum                             |
//+------------------------------------------------------------------+
void FindDominantCycles()
  {
   int window = ChartWindowFind();
   if(window < 0)
      return;

//--- Get window dimensions
   int window_height = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS, window);
   int window_width = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);

//--- Clear previous labels
   for(int i = 0; i < 10; i++)
     {
      string obj_name = g_prefix + "dominant_" + IntegerToString(i);
      ObjectDelete(0, obj_name);
     }

//--- Clear dominant buffer first
   int bars_to_show = MathMin(InpSpectrumBars, iBars(Symbol(), Period()));
   for(int i = 0; i < bars_to_show; i++)
     {
      DominantBuffer[i] = EMPTY_VALUE;
     }

//--- Create array of spectrum values with their periods
   struct SpectrumPeak
     {
      double         value;
      double         period;
      int            bar_index;
     };

   SpectrumPeak peaks[];
   int peak_count = 0;

//--- Find peaks in spectrum by checking the buffer values
   for(int i = 1; i < bars_to_show - 1; i++)
     {
      if(SpectrumBuffer[i] != EMPTY_VALUE && PeriodBuffer[i] != EMPTY_VALUE &&
         SpectrumBuffer[i] > SpectrumBuffer[i-1] &&
         SpectrumBuffer[i] > SpectrumBuffer[i+1] &&
         SpectrumBuffer[i] > 0.1) // Threshold to avoid noise
        {
         ArrayResize(peaks, peak_count + 1);
         peaks[peak_count].value = SpectrumBuffer[i];
         peaks[peak_count].period = PeriodBuffer[i];
         peaks[peak_count].bar_index = i;
         peak_count++;
        }
     }

//--- Sort peaks by value (descending)
   for(int i = 0; i < peak_count - 1; i++)
     {
      for(int j = i + 1; j < peak_count; j++)
        {
         if(peaks[j].value > peaks[i].value)
           {
            SpectrumPeak temp = peaks[i];
            peaks[i] = peaks[j];
            peaks[j] = temp;
           }
        }
     }

//--- Display top dominant cycles
   int labels_to_show = MathMin(InpDominantCycles, peak_count);

   for(int i = 0; i < labels_to_show; i++)
     {
      //--- Set the dominant buffer to show red dots
      DominantBuffer[peaks[i].bar_index] = peaks[i].value;

      //--- Create label
      string obj_name = g_prefix + "dominant_" + IntegerToString(i);
      string label_text = "Period " + DoubleToString(peaks[i].period, 1) +
                          ": " + DoubleToString(peaks[i].value * 100, 0) + "%";

      //--- Create label inside the window
      ObjectCreate(0, obj_name, OBJ_LABEL, window, 0, 0);
      ObjectSetString(0, obj_name, OBJPROP_TEXT, label_text);
      ObjectSetString(0, obj_name, OBJPROP_FONT, "Arial");
      ObjectSetInteger(0, obj_name, OBJPROP_FONTSIZE, 9);
      ObjectSetInteger(0, obj_name, OBJPROP_COLOR, InpTextColor);

      //--- Position from left side with proper margin
      ObjectSetInteger(0, obj_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, obj_name, OBJPROP_XDISTANCE, window_width - 150);
      ObjectSetInteger(0, obj_name, OBJPROP_YDISTANCE, 5 + (i * 15));
      ObjectSetInteger(0, obj_name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
     }
  }

//+------------------------------------------------------------------+
//| Create period scale labels on the horizontal axis                |
//+------------------------------------------------------------------+
void CreatePeriodScale()
  {
   int window = ChartWindowFind();
   if(window < 0)
      return;

//--- Create labels for key period values
   int num_labels = 10; // Number of labels to display

   for(int i = 0; i <= num_labels; i++)
     {
      string obj_name = g_prefix + "period_" + IntegerToString(i);

      //--- Calculate period for this position
      double period = InpMinPeriod + (InpMaxPeriod - InpMinPeriod) * i / num_labels;

      //--- Create the label
      ObjectCreate(0, obj_name, OBJ_LABEL, window, 0, 0);
      ObjectSetString(0, obj_name, OBJPROP_TEXT, StringFormat("%.0f", period));
      ObjectSetInteger(0, obj_name, OBJPROP_COLOR, clrLightGray);
      ObjectSetInteger(0, obj_name, OBJPROP_FONTSIZE, 8);
      ObjectSetInteger(0, obj_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);

      //--- Position will be updated in UpdatePeriodScale()
     }
  }

//+------------------------------------------------------------------+
//| Update period scale positions based on window size               |
//+------------------------------------------------------------------+
void UpdatePeriodScale()
  {
   int window = ChartWindowFind();
   if(window < 0)
      return;

//--- Get window dimensions
   int window_height = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS, window);

//--- Calculate positions for labels
   int num_labels = 10;
   int bars_to_show = MathMin(InpSpectrumBars, iBars(Symbol(), Period()));

   for(int i = 0; i <= num_labels; i++)
     {
      string obj_name = g_prefix + "period_" + IntegerToString(i);

      if(ObjectFind(0, obj_name) >= 0)
        {
         //--- Calculate horizontal position as a percentage of window width
         int x_position = (int)((double)i / num_labels * (ChartGetInteger(0, CHART_WIDTH_IN_PIXELS) - 100)) + 50;

         //--- Update position - place at bottom of window
         ObjectSetInteger(0, obj_name, OBJPROP_XDISTANCE, x_position);
         ObjectSetInteger(0, obj_name, OBJPROP_YDISTANCE, window_height - 15);
         ObjectSetInteger(0, obj_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
         ObjectSetInteger(0, obj_name, OBJPROP_ANCHOR, ANCHOR_CENTER);
        }
     }
  }
//+------------------------------------------------------------------+

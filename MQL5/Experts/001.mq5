//+------------------------------------------------------------------+
//|                                     HHT_CEEMDAN_Visual_EA.mq5   |
//|                                    Copyright 2024, YourCompany  |
//|                                         https://www.company.net |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024"
#property link      "https://www.yourcompany.net"
#property version   "1.00"
#property strict

//--- Input parameters
input string   InpServerURL = "http://localhost:8080/algorithm/001"; // Server URL
input int      InpDataPoints = 100;          // Number of data points to send
input int      InpUpdateSeconds = 5;         // Update interval (seconds)
input ENUM_APPLIED_PRICE InpPriceType = PRICE_CLOSE; // Price type
input int      InpVisibleBars = 50;          // Number of bars to visualize
input bool     InpShowDashboard = true;      // Show info dashboard
input bool     InpEnableAlerts = true;       // Enable alerts
input double   InpSignalThreshold = 0.7;     // Signal strength threshold (0-1)

//--- Visual settings
input color    InpTrendColor = clrDodgerBlue;      // Trend line color
input color    InpUpperBandColor = clrLimeGreen;   // Upper band color
input color    InpLowerBandColor = clrOrangeRed;   // Lower band color
input color    InpBuySignalColor = clrGreen;       // Buy signal arrow color
input color    InpSellSignalColor = clrRed;        // Sell signal arrow color
input int      InpLineWidth = 2;                   // Main line width

//--- Global variables for data storage
double g_trendBuffer[];
double g_upperBandBuffer[];
double g_lowerBandBuffer[];
double g_oscillatorBuffer[];
double g_signalLineBuffer[];
double g_imfHighBuffer[];
double g_imfMidBuffer[];
double g_imfLowBuffer[];

//--- Signal points
int g_buyPoints[];
int g_sellPoints[];

//--- Current values
double g_currentPrice = 0;
double g_currentTrend = 0;
double g_currentOscillator = 0;
double g_currentSignalLine = 0;
string g_trendDirection = "";
double g_trendStrength = 0;
string g_volatility = "";
string g_currentSignal = "";
double g_signalStrength = 0;

//--- Levels
double g_support = 0;
double g_resistance = 0;
double g_mean = 0;
double g_upperBand = 0;
double g_lowerBand = 0;

//--- Statistics
double g_stdDev = 0;
double g_kurtosis = 0;
double g_skewness = 0;

//--- Control variables
datetime g_lastUpdateTime = 0;
string g_prefix = "HHT_EA_";
bool g_initialized = false;
string g_lastAlert = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Check if URL is allowed
   if(!CheckURLPermission(InpServerURL))
     {
      Print("URL not allowed. Please add ", InpServerURL, " to allowed URLs in Terminal settings");
      return(INIT_FAILED);
     }

//--- Initialize arrays
   ArrayResize(g_trendBuffer, InpDataPoints);
   ArrayResize(g_upperBandBuffer, InpDataPoints);
   ArrayResize(g_lowerBandBuffer, InpDataPoints);
   ArrayResize(g_oscillatorBuffer, InpDataPoints);
   ArrayResize(g_signalLineBuffer, InpDataPoints);
   ArrayResize(g_imfHighBuffer, InpDataPoints);
   ArrayResize(g_imfMidBuffer, InpDataPoints);
   ArrayResize(g_imfLowBuffer, InpDataPoints);

   ArraySetAsSeries(g_trendBuffer, true);
   ArraySetAsSeries(g_upperBandBuffer, true);
   ArraySetAsSeries(g_lowerBandBuffer, true);
   ArraySetAsSeries(g_oscillatorBuffer, true);
   ArraySetAsSeries(g_signalLineBuffer, true);
   ArraySetAsSeries(g_imfHighBuffer, true);
   ArraySetAsSeries(g_imfMidBuffer, true);
   ArraySetAsSeries(g_imfLowBuffer, true);

//--- Initialize with zeros
   ArrayInitialize(g_trendBuffer, 0);
   ArrayInitialize(g_upperBandBuffer, 0);
   ArrayInitialize(g_lowerBandBuffer, 0);
   ArrayInitialize(g_oscillatorBuffer, 0);
   ArrayInitialize(g_signalLineBuffer, 0);
   ArrayInitialize(g_imfHighBuffer, 0);
   ArrayInitialize(g_imfMidBuffer, 0);
   ArrayInitialize(g_imfLowBuffer, 0);

//--- Clear any existing objects
   ClearAllObjects();

//--- Create dashboard
   if(InpShowDashboard)
      CreateDashboard();

//--- Perform initial update
   UpdateData();

   g_initialized = true;

   Print("HHT_CEEMDAN Visual EA initialized successfully");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Clear all objects
   ClearAllObjects();

//--- Clean up
   Comment("");

   Print("HHT_CEEMDAN Visual EA deinitialized. Reason: ", reason);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Check if it's time to update
   datetime currentTime = TimeCurrent();
   if(currentTime - g_lastUpdateTime >= InpUpdateSeconds)
     {
      UpdateData();
      g_lastUpdateTime = currentTime;
     }

//--- Redraw visualization on every tick
   if(g_initialized)
     {
      RedrawVisualization();
      CheckAlerts();
     }
  }

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
   UpdateData();
  }

//+------------------------------------------------------------------+
//| Check URL permission                                             |
//+------------------------------------------------------------------+
bool CheckURLPermission(string url)
  {
//--- Extract domain from URL
   string domain = url;
   int pos = StringFind(domain, "://");
   if(pos > 0)
      domain = StringSubstr(domain, pos + 3);

   pos = StringFind(domain, "/");
   if(pos > 0)
      domain = StringSubstr(domain, 0, pos);

   pos = StringFind(domain, ":");
   if(pos > 0)
      domain = StringSubstr(domain, 0, pos);

   Print("Checking permission for domain: ", domain);
   return true; // In real implementation, check against allowed URLs
  }

//+------------------------------------------------------------------+
//| Update data from server                                          |
//+------------------------------------------------------------------+
void UpdateData()
  {
//--- Prepare price data
   double prices[];
   ArrayResize(prices, InpDataPoints);

   int copied = CopyClose(_Symbol, PERIOD_CURRENT, 0, InpDataPoints, prices);
   if(copied != InpDataPoints)
     {
      Print("Failed to copy price data. Copied: ", copied);
      return;
     }

//--- Reverse array to match time series
   ArrayReverse(prices);

//--- Convert to JSON string
   string jsonData = "[";
   for(int i = 0; i < InpDataPoints; i++)
     {
      if(i > 0)
         jsonData += ",";
      jsonData += DoubleToString(prices[i], _Digits);
     }
   jsonData += "]";

//--- Send request to server
   string response = SendHTTPRequest(jsonData);

//--- Parse response
   if(response != "")
     {
      ParseServerResponse(response);
      RedrawVisualization();
     }
  }

//+------------------------------------------------------------------+
//| Send HTTP request to server                                      |
//+------------------------------------------------------------------+
string SendHTTPRequest(string jsonData)
  {
//--- Prepare headers
   string headers = "Content-Type: application/json\r\n";

//--- Prepare request
   char post[], result[];
   StringToCharArray(jsonData, post);
   ArrayResize(post, StringLen(jsonData));

   string resultHeaders;

//--- Send request
   ResetLastError();
   int res = WebRequest("POST", InpServerURL, headers, 5000, post, result, resultHeaders);

   if(res == -1)
     {
      int error = GetLastError();
      Print("WebRequest failed. Error: ", error);
      if(error == 4014)
         Print("URL not allowed. Please add ", InpServerURL, " to allowed URLs in Terminal settings");
      return "";
     }

//--- Convert response to string
   string response = CharArrayToString(result);
   return response;
  }

//+------------------------------------------------------------------+
//| Parse server response                                            |
//+------------------------------------------------------------------+
void ParseServerResponse(string response)
  {
//--- This is a simplified parser. In production, use proper JSON parsing

//--- Extract current values
   g_currentPrice = ExtractDouble(response, "\"price\":");
   g_currentTrend = ExtractDouble(response, "\"trend_value\":");
   g_currentOscillator = ExtractDouble(response, "\"oscillator_value\":");
   g_currentSignalLine = ExtractDouble(response, "\"signal_line_value\":");
   g_trendDirection = ExtractString(response, "\"trend_direction\":");
   g_trendStrength = ExtractDouble(response, "\"trend_strength\":");
   g_volatility = ExtractString(response, "\"volatility\":");
   g_currentSignal = ExtractString(response, "\"signal\":");
   g_signalStrength = ExtractDouble(response, "\"signal_strength\":");

//--- Extract levels
   g_support = ExtractDouble(response, "\"support\":");
   g_resistance = ExtractDouble(response, "\"resistance\":");
   g_mean = ExtractDouble(response, "\"mean\":");
   g_upperBand = ExtractDouble(response, "\"upper_band\":");
   g_lowerBand = ExtractDouble(response, "\"lower_band\":");

//--- Extract statistics
   g_stdDev = ExtractDouble(response, "\"std\":");
   g_kurtosis = ExtractDouble(response, "\"kurtosis\":");
   g_skewness = ExtractDouble(response, "\"skewness\":");

//--- Extract buffers
   ExtractArray(response, "\"trend_line\":", g_trendBuffer);
   ExtractArray(response, "\"upper_band\":", g_upperBandBuffer);
   ExtractArray(response, "\"lower_band\":", g_lowerBandBuffer);
   ExtractArray(response, "\"oscillator\":", g_oscillatorBuffer);
   ExtractArray(response, "\"signal_line\":", g_signalLineBuffer);
   ExtractArray(response, "\"imf_high_freq\":", g_imfHighBuffer);
   ExtractArray(response, "\"imf_mid_freq\":", g_imfMidBuffer);
   ExtractArray(response, "\"imf_low_freq\":", g_imfLowBuffer);

//--- Extract signal points
   ExtractIntArray(response, "\"buy_points\":", g_buyPoints);
   ExtractIntArray(response, "\"sell_points\":", g_sellPoints);

   Print("Data updated successfully. Trend: ", g_trendDirection,
         " Signal: ", g_currentSignal, " Strength: ", g_signalStrength);
  }

//+------------------------------------------------------------------+
//| Extract double value from JSON                                   |
//+------------------------------------------------------------------+
double ExtractDouble(string json, string key)
  {
   int pos = StringFind(json, key);
   if(pos < 0)
      return 0;

   pos += StringLen(key);
   int end = StringFind(json, ",", pos);
   if(end < 0)
      end = StringFind(json, "}", pos);

   string value = StringSubstr(json, pos, end - pos);
   StringTrimLeft(value);
   StringTrimRight(value);

   return StringToDouble(value);
  }

//+------------------------------------------------------------------+
//| Extract string value from JSON                                   |
//+------------------------------------------------------------------+
string ExtractString(string json, string key)
  {
   int pos = StringFind(json, key);
   if(pos < 0)
      return "";

   pos += StringLen(key);
   pos = StringFind(json, "\"", pos) + 1;
   int end = StringFind(json, "\"", pos);

   return StringSubstr(json, pos, end - pos);
  }

//+------------------------------------------------------------------+
//| Extract array from JSON                                          |
//+------------------------------------------------------------------+
void ExtractArray(string json, string key, double &array[])
  {
   int pos = StringFind(json, key);
   if(pos < 0)
      return;

   pos += StringLen(key);
   pos = StringFind(json, "[", pos) + 1;
   int end = StringFind(json, "]", pos);

   string arrayStr = StringSubstr(json, pos, end - pos);

//--- Parse array values
   string values[];
   int count = StringSplit(arrayStr, ',', values);

   ArrayResize(array, count);
   for(int i = 0; i < count; i++)
     {
      StringTrimLeft(values[i]);
      StringTrimRight(values[i]);
      array[i] = StringToDouble(values[i]);
     }
  }

//+------------------------------------------------------------------+
//| Extract integer array from JSON                                  |
//+------------------------------------------------------------------+
void ExtractIntArray(string json, string key, int &array[])
  {
   int pos = StringFind(json, key);
   if(pos < 0)
      return;

   pos += StringLen(key);
   pos = StringFind(json, "[", pos) + 1;
   int end = StringFind(json, "]", pos);

   string arrayStr = StringSubstr(json, pos, end - pos);

//--- Parse array values
   string values[];
   int count = StringSplit(arrayStr, ',', values);

   ArrayResize(array, count);
   for(int i = 0; i < count; i++)
     {
      StringTrimLeft(values[i]);
      StringTrimRight(values[i]);
      array[i] = (int)StringToInteger(values[i]);
     }
  }

//+------------------------------------------------------------------+
//| Clear all objects                                                |
//+------------------------------------------------------------------+
void ClearAllObjects()
  {
   ObjectsDeleteAll(0, g_prefix);
   ChartRedraw();
  }

//+------------------------------------------------------------------+
//| Redraw all visualization elements                                |
//+------------------------------------------------------------------+
void RedrawVisualization()
  {
//--- Clear previous objects
   ClearAllObjects();

//--- Draw main chart indicators
   DrawMainChartIndicators();

//--- Draw oscillator
   DrawOscillator();

//--- Draw signal arrows
   DrawSignalArrows();

//--- Update dashboard
   if(InpShowDashboard)
      UpdateDashboard();

//--- Redraw chart
   ChartRedraw();
  }

//+------------------------------------------------------------------+
//| Draw main chart indicators                                       |
//+------------------------------------------------------------------+
void DrawMainChartIndicators()
  {
   datetime time[];
   ArraySetAsSeries(time, true);
   int copied = CopyTime(_Symbol, PERIOD_CURRENT, 0, InpVisibleBars, time);
   if(copied <= 0)
      return;

//--- Trend line
   for(int i = 0; i < copied - 1; i++)
     {
      string obj = g_prefix + "Trend_" + IntegerToString(i);
      ObjectCreate(0, obj, OBJ_TREND, 0,
                   time[i+1], g_trendBuffer[i+1],
                   time[i],   g_trendBuffer[i]);
      ObjectSetInteger(0, obj, OBJPROP_COLOR, InpTrendColor);
      ObjectSetInteger(0, obj, OBJPROP_WIDTH, InpLineWidth);
      ObjectSetInteger(0, obj, OBJPROP_RAY_RIGHT, false);
     }

//--- Upper band
   for(int i = 0; i < copied - 1; i++)
     {
      string obj = g_prefix + "Upper_" + IntegerToString(i);
      ObjectCreate(0, obj, OBJ_TREND, 0,
                   time[i+1], g_upperBandBuffer[i+1],
                   time[i],   g_upperBandBuffer[i]);
      ObjectSetInteger(0, obj, OBJPROP_COLOR, InpUpperBandColor);
      ObjectSetInteger(0, obj, OBJPROP_STYLE, STYLE_DOT);
     }

//--- Lower band
   for(int i = 0; i < copied - 1; i++)
     {
      string obj = g_prefix + "Lower_" + IntegerToString(i);
      ObjectCreate(0, obj, OBJ_TREND, 0,
                   time[i+1], g_lowerBandBuffer[i+1],
                   time[i],   g_lowerBandBuffer[i]);
      ObjectSetInteger(0, obj, OBJPROP_COLOR, InpLowerBandColor);
      ObjectSetInteger(0, obj, OBJPROP_STYLE, STYLE_DOT);
     }
  }

//+------------------------------------------------------------------+
//| Draw oscillator bars as histogram                                |
//+------------------------------------------------------------------+
void DrawOscillator()
  {
   datetime time[];
   ArraySetAsSeries(time, true);
   int copied = CopyTime(_Symbol, PERIOD_CURRENT, 0, InpVisibleBars, time);
   if(copied <= 0)
      return;

   for(int i = 0; i < copied; i++)
     {
      string obj = g_prefix + "Osc_" + IntegerToString(i);
      double val = g_oscillatorBuffer[i];
      ObjectCreate(0, obj, val >= 0 ? OBJ_ARROW_UP : OBJ_ARROW_DOWN, 0, time[i], g_signalLineBuffer[i]);
      ObjectSetInteger(0, obj, OBJPROP_COLOR, val >= 0 ? clrLimeGreen : clrOrangeRed);
      ObjectSetInteger(0, obj, OBJPROP_WIDTH, 1);
     }
  }

//+------------------------------------------------------------------+
//| Draw buy/sell arrows                                             |
//+------------------------------------------------------------------+
void DrawSignalArrows()
  {
   datetime time[];
   double high[], low[];
   ArraySetAsSeries(time, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   int copied = CopyHigh(_Symbol, PERIOD_CURRENT, 0, InpVisibleBars, high);
   CopyLow(_Symbol, PERIOD_CURRENT, 0, InpVisibleBars, low);
   CopyTime(_Symbol, PERIOD_CURRENT, 0, InpVisibleBars, time);

   for(int i = 0; i < ArraySize(g_buyPoints); i++)
     {
      int idx = g_buyPoints[i];
      if(idx < copied)
        {
         string obj = g_prefix + "Buy_" + IntegerToString(i);
         ObjectCreate(0, obj, OBJ_ARROW_UP, 0, time[idx], low[idx] - 10*_Point);
         ObjectSetInteger(0, obj, OBJPROP_COLOR, InpBuySignalColor);
        }
     }
   for(int i = 0; i < ArraySize(g_sellPoints); i++)
     {
      int idx = g_sellPoints[i];
      if(idx < copied)
        {
         string obj = g_prefix + "Sell_" + IntegerToString(i);
         ObjectCreate(0, obj, OBJ_ARROW_DOWN, 0, time[idx], high[idx] + 10*_Point);
         ObjectSetInteger(0, obj, OBJPROP_COLOR, InpSellSignalColor);
        }
     }
  }

//+------------------------------------------------------------------+
//| Create dashboard                                                 |
//+------------------------------------------------------------------+
void CreateDashboard()
  {
   string label = g_prefix + "Dashboard";
   ObjectCreate(0, label, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, label, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, label, OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(0, label, OBJPROP_YDISTANCE, 20);
   ObjectSetInteger(0, label, OBJPROP_FONTSIZE, 10);
   ObjectSetString(0, label, OBJPROP_FONT, "Arial");
  }

//+------------------------------------------------------------------+
//| Update dashboard contents                                        |
//+------------------------------------------------------------------+
void UpdateDashboard()
  {
   string label = g_prefix + "Dashboard";
   string text =
      "Price: " + DoubleToString(g_currentPrice, _Digits) + "\n" +
      "Trend: " + g_trendDirection + " (" + DoubleToString(g_trendStrength, 2) + ")\n" +
      "Signal: " + g_currentSignal + " (" + DoubleToString(g_signalStrength, 2) + ")\n" +
      "Osc: " + DoubleToString(g_currentOscillator, 2) + "\n" +
      "Volatility: " + g_volatility + "\n" +
      "Support: " + DoubleToString(g_support, _Digits) + "\n" +
      "Resistance: " + DoubleToString(g_resistance, _Digits);
   ObjectSetString(0, label, OBJPROP_TEXT, text);
  }

//+------------------------------------------------------------------+
//| Check alert conditions                                           |
//+------------------------------------------------------------------+
void CheckAlerts()
  {
   if(!InpEnableAlerts)
      return;
   string alertMsg = "";
   if(g_signalStrength >= InpSignalThreshold)
      alertMsg = "Signal: " + g_currentSignal + " Strength: " + DoubleToString(g_signalStrength, 2);
   if(alertMsg != "" && alertMsg != g_lastAlert)
     {
      Alert(alertMsg);
      g_lastAlert = alertMsg;
     }
  }
//+------------------------------------------------------------------+

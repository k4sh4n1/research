# Note

Apparently, MetaQuotes saves the robot algorithm `*.mq5` files here:

```bash
C:\Users\username\AppData\Roaming\MetaQuotes\Terminal\A_LONG_HASH\MQL5\Experts\*.mq5
```

# MetaTrader

To have access to MQL5 code on the MetaTrader platform:

```bash
cd C:\Users\m3\AppData\Roaming\MetaQuotes\Terminal\A_LONG_HASH\MQL5

mklink /J "research-MQL5" "C:\Users\m3\repos\research-proposal\MQL5"
```

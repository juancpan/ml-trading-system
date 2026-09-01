Changes to algos/backtest_code/yfinance_downloader_v5.py
1. Added import json at the top (line 26)
2. Added --json CLI flag to the argparse section (lines 2077-2089) with help text documenting the expected format and behavior
3. Added JSON loading logic (lines 2097-2120) that:
   - Validates the file exists (sys.exit(1) if not)
   - Validates the JSON is parseable (sys.exit(1) on JSONDecodeError)
   - Validates the parsed data is a non-empty dict
   - Builds current_tickers_map as {key: key} from the JSON keys, overriding the hardcoded map
4. Updated usage comments to document the new flag
Usage
# Use hardcoded tickers (default, unchanged behavior)
python yfinance_downloader_v5.py
# Load ticker universe from a JSON file (keys are tickers, values ignored)
python yfinance_downloader_v5.py --json algos/backtest_code/data/hrp_weights.json
# Combine with timezone flag
python yfinance_downloader_v5.py --json data/hrp_weights.json --tz UTC
Note: The Rolls_Royce key in hrp_weights.json is not a valid yfinance ticker (the actual ticker is RR.L). With the "keys as-is" strategy, yfinance will log a warning for it and skip it. If you need that mapping, you'd update the JSON key from Rolls_Royce to RR.L.

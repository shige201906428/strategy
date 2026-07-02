import datetime
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

def get_sp500_tickers():
    """S&P 500の全500銘柄のティッカーリストを確実に入手する"""
    print("S&P 500の最新銘柄リストを取得中...")
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"-> S&P 500の {len(tickers)} 銘柄を正常に取得しました。")
        return tickers
    except Exception as e:
        print(f"一次ソースのエラーのため、代替ルートで取得を試みます... ({e})")
        try:
            url_fallback = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url_fallback)
            df = tables[0]
            tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
            return tickers
        except Exception:
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

def load_tickers_from_file(file_path, is_japan=False):
    """テキストファイルからティッカーを読み込む（日本株用）"""
    if not os.path.exists(file_path):
        return []
    tickers = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            ticker = line.strip()
            if not ticker or ticker.startswith("#"):
                continue
            if is_japan and not ticker.endswith(".T"):
                ticker = f"{ticker}.T"
            tickers.append(ticker)
    return tickers

def calculate_supertrend(df, period=10, multiplier=4.5):
    """スーパートレンド(Supertrend)を計算するロジック"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # ATRの計算
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    hl2 = (high + low) / 2
    final_upperband = hl2 + multiplier * atr
    final_lowerband = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if close.iloc[i-1] < final_upperband.iloc[i-1]:
            final_upperband.iloc[i] = min(final_upperband.iloc[i], final_upperband.iloc[i-1])
        if close.iloc[i-1] > final_lowerband.iloc[i-1]:
            final_lowerband.iloc[i] = max(final_lowerband.iloc[i], final_lowerband.iloc[i-1])
            
        if direction[i-1] == 1 or direction[i-1] == 0:
            if close.iloc[i] > final_upperband.iloc[i]:
                direction[i] = -1
            else:
                direction[i] = 1
        else:
            if close.iloc[i] < final_lowerband.iloc[i]:
                direction[i] = 1
            else:
                direction[i] = -1
                
        supertrend[i] = final_lowerband.iloc[i] if direction[i] == -1 else final_upperband.iloc[i]
        
    df['Supertrend'] = supertrend
    df['ST_Direction'] = direction
    return df

def check_trend_strategy(ticker_list, sma_len=100, st_len=10, st_mult=4.5):
    """王道転換ストラテジーの条件に合致するか判定する"""
    results = []
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365 * 2)

    print(f"\nYahoo Financeから全株価データを一括ダウンロード中...")
    try:
        all_data = yf.download(ticker_list, start=start_date, end=end_date, group_by='ticker', progress=False)
    except Exception as e:
        print(f"データのダウンロード中にエラーが発生しました: {e}")
        return pd.DataFrame()

    for ticker_symbol in tqdm(ticker_list, desc="スクリーニング中"):
        try:
            if ticker_symbol not in all_data.columns.levels[0]:
                continue
            df = all_data[ticker_symbol].dropna()

            if len(df) < sma_len + 10:
                continue

            df['SMA_Long'] = df['Close'].rolling(window=sma_len).mean()
            df = calculate_supertrend(df, period=st_len, multiplier=st_mult)

            c_close = df['Close'].iloc[-1]
            c_sma = df['SMA_Long'].iloc[-1]
            c_dir = df['ST_Direction'].iloc[-1]
            c_st = df['Supertrend'].iloc[-1]
            
            p_close = df['Close'].iloc[-2]
            p_dir = df['ST_Direction'].iloc[-2]
            p_st = df['Supertrend'].iloc[-2]

            is_bull_market = c_close > c_sma

            if is_bull_market:
                signal_type = ""
                
                if p_dir == 1 and c_dir == -1:
                    signal_type = "初動 (ST転換)"
                elif c_dir == -1 and p_close <= p_st and c_close > c_st:
                    signal_type = "押し目 (ST上抜け)"

                if signal_type:
                    results.append({
                        "Ticker": ticker_symbol,
                        "Signal_Type": signal_type,
                        "Current_Price": round(c_close, 2),
                        "SMA_Value": round(c_sma, 2),
                        "ST_Value": round(c_st, 2)
                    })
        except Exception:
            continue

    res_df = pd.DataFrame(results)
    return res_df

def generate_html_report(df, output_path, title_suffix=""):
    """index.htmlとして概要・チャート両方のリンク付きレポートを出力する"""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    table_rows = ""
    for idx, row in df.iterrows():
        ticker = row['Ticker']
        sig_type = row['Signal_Type']
        
        if "初動" in sig_type:
            badge_color = "badge-danger"
        else:
            badge_color = "badge-warning"
            
        if ".T" in ticker:
            code = ticker.split('.')[0]
            symbols_url = f"https://jp.tradingview.com/symbols/TSE-{code}/"
            chart_url = f"https://jp.tradingview.com/chart/?symbol=TSE:{code}"
            currency_prefix = "¥"
        else:
            symbols_url = f"https://jp.tradingview.com/symbols/{ticker}/"
            chart_url = f"https://jp.tradingview.com/chart/?symbol={ticker}"
            currency_prefix = "$"

        table_rows += f"""
        <tr>
            <td>
                <span class="ticker-text">{ticker}</span>
                <div class="tv-links mt-1">
                    <a href="{symbols_url}" target="_blank" class="badge badge-info mr-1">概要 📄</a>
                    <a href="{chart_url}" target="_blank" class="badge badge-primary">チャート 📈</a>
                </div>
            </td>
            <td class="text-center vertical-middle">
                <span class="badge {badge_color}" style="font-size: 0.9rem; padding: 6px 12px;">{sig_type}</span>
            </td>
            <td class="vertical-middle">{currency_prefix}{row['Current_Price']:,}</td>
            <td class="vertical-middle">{currency_prefix}{row['SMA_Value']:,}</td>
            <td class="vertical-middle">{currency_prefix}{row['ST_Value']:,}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚡ 王道転換ストラテジー検出 {title_suffix}</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.10.21/css/dataTables.bootstrap4.min.css">
    <style>
        body {{ background-color: #f8f9fa; color: #333; font-family: 'Helvetica Neue', Arial, sans-serif; }}
        .container {{ margin-top: 30px; margin-bottom: 50px; }}
        .header-section {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .card {{ border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-radius: 8px; }}
        .table thead th {{ background-color: #2c3e50; color: white; border: none; vertical-align: middle; }}
        .vertical-middle {{ vertical-align: middle !important; }}
        .ticker-text {{ font-size: 1.1rem; font-weight: bold; color: #2c3e50; }}
        .badge {{ padding: 5px 8px; font-size: 0.75rem; }}
        footer {{ text-align: center; margin-top: 30px; color: #777; font-size: 0.9rem; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header-section">
            <h1 class="display-5">🎯 王道転換シグナル スクリーニングランキング {title_suffix}</h1>
            <p class="lead mb-0">100日移動平均線の上で、スーパートレンドが転換（初動）または押し目を形成した銘柄を抽出しています</p>
            <hr class="my-2" style="border-color: rgba(255,255,255,0.2);">
            <small>最終更新日時: <strong>{now_str}</strong> | ヒット数: {len(df)} 銘柄</small>
        </div>

        <div class="card p-4">
            <div class="table-responsive">
                <table id="screenerTable" class="table table-hover table-striped table-bordered" style="width:100%">
                    <thead>
                        <tr>
                            <th>ティッカー (Ticker)</th>
                            <th class="text-center">シグナル種類 (Flag)</th>
                            <th>直近終値 (Current Price)</th>
                            <th>100日移動平均 (100 SMA)</th>
                            <th>STライン (Supertrend)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>
        <footer>
            <p>Generated by Trend Reversal Screener Tool</p>
        </footer>
    </div>

    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.10.21/js/dataTables.bootstrap4.min.js"></script>
    <script>
        $(document).ready(function() {{
            $('#screenerTable').DataTable({{
                "order": [[ 1, "asc" ]],
                "pageLength": 50,
                "language": {{
                    "search": "絞り込み検索:",
                    "lengthMenu": "表示 _MENU_ 件",
                    "info": "全 _TOTAL_ 銘柄中 _START_ から _END_ まで表示",
                    "paginate": {{ "next": "次", "previous": "前" }}
                }}
            }});
        }});
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    choice = os.environ.get("SCREENER_CHOICE", "3")
    tickers = []
    title_suffix = ""
    os.makedirs("data", exist_ok=True)

    if choice in ["1", "3"]:
        sp500_tickers = get_sp500_tickers()
        tickers.extend(sp500_tickers)
        title_suffix += "[S&P 500] "

    if choice in ["2", "3"]:
        topix_file = os.path.join("data", "tickers_topix.txt")
        if not os.path.exists(topix_file):
            with open(topix_file, "w") as f:
                f.write("# 日本株コード\n1306\n7203\n6758\n9984\n")
        topix_tickers = load_tickers_from_file(topix_file, is_japan=True)
        tickers.extend(topix_tickers)
        title_suffix += "[TOPIX] "

    if not tickers:
        print("対象ティッカーがありません。")
        exit()

    result_df = check_trend_strategy(tickers, sma_len=100, st_len=10, st_mult=4.5)
    html_name = "index.html"
    
    if result_df is None or result_df.empty:
        result_df = pd.DataFrame(columns=["Ticker", "Signal_Type", "Current_Price", "SMA_Value", "ST_Value"])
        
    generate_html_report(result_df, html_name, title_suffix)
    print(f"\n[成功] スクリーニング結果を '{html_name}' に上書き保存しました。")

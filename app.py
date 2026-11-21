import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="اسکنر پژوهشی بورس ایران", page_icon="📊", layout="wide")
st.markdown("""
<style>
html, body, [class*="css"] { direction: rtl; text-align: right; }
[data-testid="stMetricValue"] { direction: ltr; }
</style>
""", unsafe_allow_html=True)

st.title("📊 اسکنر پژوهشی بورس ایران")
st.caption("نسخه آزمایشی — رتبه‌بندی تکنیکالی آموزشی، نه توصیه خریدوفروش یا تضمین سود")

REQUIRED = ["date", "symbol", "open", "high", "low", "close", "volume"]
SAMPLE = Path("data/sample_prices.csv")

def calc_rsi(s, period=14):
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def analyze(g):
    g = g.sort_values("date").copy()
    for col in ["open", "high", "low", "close", "volume"]:
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g = g.dropna(subset=["high", "low", "close", "volume"])
    if len(g) < 30:
        return None, g
    c = g.close
    g["ema20"] = c.ewm(span=20, adjust=False).mean()
    g["ema50"] = c.ewm(span=50, adjust=False).mean()
    g["rsi14"] = calc_rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    g["macd"] = macd
    g["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    g["vol_avg20"] = g.volume.rolling(20).mean()
    last = g.iloc[-1]
    prev_high = g.high.tail(21).iloc[:-1].max()
    support = g.low.tail(20).min()
    resistance = g.high.tail(20).max()
    trend = "صعودی" if last.close > last.ema20 > last.ema50 else (
        "نزولی" if last.close < last.ema20 < last.ema50 else "خنثی")
    vol_ratio = last.volume / last.vol_avg20 if pd.notna(last.vol_avg20) and last.vol_avg20 else np.nan
    breakout = bool(last.close > prev_high)
    score = 0
    score += 30 if trend == "صعودی" else 0
    score += 20 if pd.notna(last.rsi14) and 50 <= last.rsi14 <= 70 else 0
    score += 20 if last.macd > last.macd_signal else 0
    score += 15 if pd.notna(vol_ratio) and vol_ratio >= 1.5 else 0
    score += 15 if breakout else 0
    result = {
        "نماد": str(g.symbol.iloc[-1]),
        "آخرین تاریخ": str(pd.to_datetime(last.date).date()),
        "روند": trend,
        "قیمت پایانی": round(float(last.close), 2),
        "RSI": round(float(last.rsi14), 1) if pd.notna(last.rsi14) else None,
        "حجم نسبت به میانگین": round(float(vol_ratio), 2) if pd.notna(vol_ratio) else None,
        "حمایت محاسباتی": round(float(support), 2),
        "مقاومت محاسباتی": round(float(resistance), 2),
        "عبور از سقف قبلی": "بله" if breakout else "خیر",
        "امتیاز پژوهشی": score,
    }
    return result, g

with st.sidebar:
    st.header("داده‌های ورودی")
    uploaded = st.file_uploader("بارگذاری CSV تاریخچه قیمت", type=["csv"])
    st.markdown("**ستون‌های موردنیاز**")
    st.code("date,symbol,open,high,low,close,volume")
    st.caption("برای هر نماد حداقل ۳۰ روز داده روزانه لازم است.")
    st.divider()
    st.caption("اتصال زنده به TSETMC یا ایزی‌تریدر در این نسخه فعال نیست.")

if uploaded:
    df = pd.read_csv(uploaded)
elif SAMPLE.exists():
    df = pd.read_csv(SAMPLE)
    st.info("این صفحه با داده ساختگی نمونه نمایش داده می‌شود؛ داده‌ها قیمت واقعی بازار نیستند.")
else:
    st.warning("برای شروع، فایل CSV داده‌های قیمت را بارگذاری کنید.")
    st.stop()

missing = [x for x in REQUIRED if x not in df.columns]
if missing:
    st.error("ستون‌های لازم موجود نیستند: " + "، ".join(missing))
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "symbol"])
rows, series = [], {}
for sym, group in df.groupby("symbol"):
    item, hist = analyze(group)
    if item:
        rows.append(item)
        series[str(sym)] = hist

if not rows:
    st.error("داده کافی برای تحلیل وجود ندارد. برای هر نماد حداقل ۳۰ ردیف لازم است.")
    st.stop()

ranked = pd.DataFrame(rows).sort_values("امتیاز پژوهشی", ascending=False)
a, b, c = st.columns(3)
a.metric("نمادهای تحلیل‌شده", len(ranked))
b.metric("روندهای صعودی", int((ranked["روند"] == "صعودی").sum()))
c.metric("عبور از سقف قبلی", int((ranked["عبور از سقف قبلی"] == "بله").sum()))

st.subheader("۱۰ نمودار با بالاترین امتیاز پژوهشی")
st.caption("این رتبه‌بندی صرفاً خروجی قواعد تکنیکالی ساده است و به معنی ارزش خرید یا پیش‌بینی سود نیست.")
top = ranked.head(10)
st.dataframe(top, use_container_width=True, hide_index=True)
st.download_button("دریافت گزارش ۱۰ نماد (CSV)", top.to_csv(index=False).encode("utf-8-sig"),
                   "top10_research_scan.csv", "text/csv")

st.subheader("بررسی نمودار")
symbol = st.selectbox("انتخاب نماد", sorted(series.keys()))
h = series[symbol]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=h.date, open=h.open, high=h.high, low=h.low, close=h.close, name="قیمت"))
fig.add_trace(go.Scatter(x=h.date, y=h.ema20, name="EMA 20"))
fig.add_trace(go.Scatter(x=h.date, y=h.ema50, name="EMA 50"))
fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white",
                  legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

selected = ranked[ranked["نماد"] == symbol].iloc[0]
st.subheader("خلاصه وضعیت تکنیکالی")
st.write(f"**روند:** {selected['روند']} | **RSI:** {selected['RSI']} | **امتیاز پژوهشی:** {selected['امتیاز پژوهشی']} از ۱۰۰")
st.write(f"**حمایت محاسباتی:** {selected['حمایت محاسباتی']} | **مقاومت محاسباتی:** {selected['مقاومت محاسباتی']}")
st.warning("این نسخه شکست سقف ساده را بررسی می‌کند؛ تشخیص پولبک، تعدیل قیمت، فیلتر نقدشوندگی، توقف نماد و بک‌تست معتبر هنوز اضافه نشده‌اند. اعداد حمایت و مقاومت، کمینه و بیشینه ۲۰ جلسه اخیرند و محدوده قطعی محسوب نمی‌شوند.")
st.caption("از این ابزار برای آموزش و ارزیابی داده استفاده کنید. خروجی آن سیگنال شخصی خریدوفروش نیست.")

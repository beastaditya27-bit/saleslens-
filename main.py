"""
SalesLens Backend — FastAPI
----------------------------------------------------
Run locally:
    pip install fastapi uvicorn pandas openpyxl python-multipart
    python -m uvicorn main:app --reload

API Endpoints:
    GET  /          → serves index.html
    GET  /health    → health check
    POST /upload    → upload CSV or Excel, get full analysis
"""

# ── Imports ─────────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import io


# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="SalesLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production: replace with your domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Serve Frontend ───────────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("index.html")


# ── Constants ────────────────────────────────────────────────────────────────
REQUIRED_COLS = {"date", "product", "quantity", "revenue", "cost"}

COLUMN_ALIASES = {
    "date":     ["date", "order_date", "sale_date", "transaction_date", "month", "day"],
    "product":  ["product", "product_name", "item", "item_name", "sku", "description"],
    "quantity": ["quantity", "qty", "units", "units_sold", "amount"],
    "revenue":  ["revenue", "sales", "total", "total_sales", "income", "price", "total_revenue"],
    "cost":     ["cost", "cogs", "cost_of_goods", "expenses", "cost_price", "total_cost"],
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map user's column names → standard names."""
    col_map = {}
    lowered = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                col_map[lowered[alias]] = standard
                break
    df = df.rename(columns=col_map)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Could not find required columns: {missing}. Your file has: {list(df.columns)}"
        )
    return df


def parse_upload(file: UploadFile) -> pd.DataFrame:
    """Read uploaded CSV or Excel into a DataFrame."""
    contents = file.file.read()
    if file.filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    elif file.filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        raise HTTPException(status_code=400, detail="Only .csv, .xlsx, .xls files are supported.")
    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the DataFrame with profit, margin, and month columns."""
    df = df.copy()
    df["date"]     = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["revenue"]  = pd.to_numeric(df["revenue"],  errors="coerce").fillna(0)
    df["cost"]     = pd.to_numeric(df["cost"],      errors="coerce").fillna(0)
    df["profit"]   = df["revenue"] - df["cost"]
    df["margin"]   = (df["profit"] / df["revenue"].replace(0, float("nan"))) * 100
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


# ── Analysis Functions ───────────────────────────────────────────────────────
def revenue_trend(df: pd.DataFrame) -> dict:
    """Monthly revenue + profit trend with month-over-month growth."""
    trend = (
        df.groupby("month")
        .agg(revenue=("revenue", "sum"), profit=("profit", "sum"), orders=("quantity", "sum"))
        .reset_index()
        .sort_values("month")
    )
    rev = trend["revenue"].tolist()
    growth = [None] + [
        round((rev[i] - rev[i-1]) / rev[i-1] * 100, 1) if rev[i-1] else 0
        for i in range(1, len(rev))
    ]
    return {
        "labels":     trend["month"].tolist(),
        "revenue":    [round(v, 2) for v in rev],
        "profit":     [round(v, 2) for v in trend["profit"].tolist()],
        "orders":     [int(v) for v in trend["orders"].tolist()],
        "growth_pct": growth,
    }


def product_performance(df: pd.DataFrame) -> dict:
    """Best & worst selling products by revenue and margin."""
    by_product = (
        df.groupby("product")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            units=("quantity", "sum"),
            avg_margin=("margin", "mean"),
        )
        .reset_index()
    )
    by_product["avg_margin"] = by_product["avg_margin"].round(1)
    by_product = by_product.sort_values("revenue", ascending=False)

    return {
        "top_sellers":    by_product.head(5).to_dict(orient="records"),
        "worst_sellers":  by_product.tail(5).sort_values("revenue").to_dict(orient="records"),
        "total_products": int(len(by_product)),
    }


def profit_margin_analysis(df: pd.DataFrame) -> dict:
    """Overall margin stats and per-product margin breakdown."""
    total_rev    = float(df["revenue"].sum())
    total_profit = float(df["profit"].sum())
    overall_margin = round(total_profit / total_rev * 100, 2) if total_rev else 0

    by_product = df.groupby("product").agg(
        revenue=("revenue", "sum"), profit=("profit", "sum")
    ).reset_index()
    by_product["margin"] = (
        by_product["profit"] / by_product["revenue"].replace(0, float("nan")) * 100
    ).round(1)

    high   = by_product[by_product["margin"] >= 40]
    medium = by_product[(by_product["margin"] >= 15) & (by_product["margin"] < 40)]
    low    = by_product[by_product["margin"] < 15]

    return {
        "overall_margin_pct": overall_margin,
        "total_revenue":      round(total_rev, 2),
        "total_profit":       round(total_profit, 2),
        "total_cost":         round(float(df["cost"].sum()), 2),
        "margin_buckets": {
            "high_margin_40plus":  int(len(high)),
            "medium_margin_15_40": int(len(medium)),
            "low_margin_under_15": int(len(low)),
        },
        "low_margin_products": (
            low.sort_values("margin")[["product", "margin", "revenue"]]
            .head(5)
            .to_dict(orient="records")
        ),
    }


def ai_recommendations(df: pd.DataFrame, product_data: dict, margin_data: dict, trend_data: dict) -> list:
    """Rule-based AI recommendations."""
    tips = []

    # Low margin warning
    low = margin_data["low_margin_products"]
    if low:
        worst, m = low[0]["product"], low[0]["margin"]
        tips.append({
            "type": "warning", "icon": "⚠️",
            "title": "Low Margin Product Detected",
            "detail": f'"{worst}" has only a {m}% profit margin. Consider raising its price, reducing supplier cost, or discontinuing it.',
            "impact": "high",
        })

    # Best seller opportunity
    if product_data["top_sellers"]:
        best = product_data["top_sellers"][0]
        tips.append({
            "type": "opportunity", "icon": "🚀",
            "title": "Double Down on Your Best Seller",
            "detail": f'"{best["product"]}" is your top earner (${best["revenue"]:,.0f} revenue, {best["avg_margin"]}% margin). Increase ad spend, create bundles, or upsell accessories.',
            "impact": "high",
        })

    # Revenue trend
    revenues = trend_data["revenue"]
    if len(revenues) >= 2:
        if revenues[-1] < revenues[-2]:
            drop = round((revenues[-2] - revenues[-1]) / revenues[-2] * 100, 1)
            tips.append({
                "type": "warning", "icon": "📉",
                "title": "Revenue Declining This Month",
                "detail": f"Revenue dropped {drop}% compared to last month. Consider a flash sale, email campaign, or reviewing pricing on slow-moving items.",
                "impact": "high",
            })
        else:
            growth = trend_data["growth_pct"][-1]
            tips.append({
                "type": "positive", "icon": "📈",
                "title": "Revenue is Growing!",
                "detail": f"You're up {growth}% this month. Maintain momentum by restocking top sellers and running a loyalty reward for repeat customers.",
                "impact": "medium",
            })

    # Worst seller action
    if product_data["worst_sellers"]:
        worst_p = product_data["worst_sellers"][0]
        tips.append({
            "type": "action", "icon": "🗑️",
            "title": "Consider Dropping a Dead-Weight Product",
            "detail": f'"{worst_p["product"]}" generated only ${worst_p["revenue"]:,.0f} in revenue. Liquidate stock with a clearance sale and reinvest that capital.',
            "impact": "medium",
        })

    # Overall margin health
    overall = margin_data["overall_margin_pct"]
    if overall < 20:
        tips.append({
            "type": "warning", "icon": "💸",
            "title": "Overall Margin is Thin",
            "detail": f"Your overall margin is {overall}% — below the healthy 20–30% benchmark. Audit your top 3 cost drivers and negotiate with suppliers.",
            "impact": "high",
        })
    elif overall >= 35:
        tips.append({
            "type": "positive", "icon": "💰",
            "title": "Healthy Profit Margins",
            "detail": f"Your {overall}% overall margin is strong. Use excess profit to invest in marketing or expand your product line.",
            "impact": "low",
        })

    return tips


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "SalesLens API"}


@app.post("/upload")
async def upload_and_analyse(file: UploadFile = File(...)):
    """
    Main endpoint. Accepts a CSV or Excel file, returns full analysis as JSON.

    Response shape:
    {
        "summary":         { total_revenue, total_profit, total_orders, date_range, ... },
        "revenue_trend":   { labels[], revenue[], profit[], orders[], growth_pct[] },
        "products":        { top_sellers[], worst_sellers[], total_products },
        "margins":         { overall_margin_pct, total_revenue, total_profit, ... },
        "recommendations": [ { type, icon, title, detail, impact } ]
    }
    """
    df      = parse_upload(file)
    df      = normalize_columns(df)
    df      = clean(df)

    trend   = revenue_trend(df)
    prods   = product_performance(df)
    margins = profit_margin_analysis(df)
    recs    = ai_recommendations(df, prods, margins, trend)

    summary = {
        "total_revenue":      margins["total_revenue"],
        "total_profit":       margins["total_profit"],
        "total_orders":       int(df["quantity"].sum()),
        "total_products":     prods["total_products"],
        "overall_margin_pct": margins["overall_margin_pct"],
        "date_range": {
            "from": str(df["date"].min().date()),
            "to":   str(df["date"].max().date()),
        },
    }

    return JSONResponse({
        "summary":         summary,
        "revenue_trend":   trend,
        "products":        prods,
        "margins":         margins,
        "recommendations": recs,
    })
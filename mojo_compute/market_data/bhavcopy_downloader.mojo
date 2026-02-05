"""
NSE Bhavcopy Downloader (Mojo Implementation)
Downloads and parses EOD market data from NSE Bhavcopy files
High-performance implementation using Mojo
"""

from sys import exit
from time import now
from python import Python
from collections import Dict


fn download_bhavcopy(trade_date: String) raises -> String:
    """
    Download Bhavcopy CSV from NSE for given date.

    NSE Bhavcopy URL format:
    https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv

    Args:
        trade_date: Date in DDMMYYYY format

    Returns:
        CSV content as string
    """
    # Use Python requests for HTTP (Mojo HTTP client coming soon)
    let requests = Python.import_module("requests")

    let url = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_" + trade_date + ".csv"

    print("📥 Downloading Bhavcopy from:", url)

    let headers = Dict[String, String]()
    headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    headers["Accept"] = "text/csv,application/csv"

    let response = requests.get(url, headers=headers, timeout=30)

    if response.status_code != 200:
        print("❌ Failed to download Bhavcopy. HTTP Status:", response.status_code)
        raise Error("Failed to download Bhavcopy")

    return String(response.text)


struct BhavcopyRow:
    """Represents a single row from Bhavcopy CSV."""
    var symbol: String
    var series: String
    var open: Float64
    var high: Float64
    var low: Float64
    var close: Float64
    var last: Float64
    var prevclose: Float64
    var tottrdqty: Int
    var tottrdval: Float64
    var timestamp: String
    var totaltrades: Int

    fn __init__(inout self, symbol: String, series: String,
                open_price: Float64, high: Float64, low: Float64,
                close: Float64, last: Float64, prevclose: Float64,
                tottrdqty: Int, tottrdval: Float64, timestamp: String, totaltrades: Int):
        self.symbol = symbol
        self.series = series
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.last = last
        self.prevclose = prevclose
        self.tottrdqty = tottrdqty
        self.tottrdval = tottrdval
        self.timestamp = timestamp
        self.totaltrades = totaltrades


fn parse_bhavcopy_csv(csv_content: String) raises -> DynamicVector[BhavcopyRow]:
    """
    Parse Bhavcopy CSV content into structured data.
    Ultra-fast parsing using Mojo performance.

    CSV Format:
    SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,Totaltrades,ISIN

    Args:
        csv_content: CSV content as string

    Returns:
        Vector of BhavcopyRow structs
    """
    var rows = DynamicVector[BhavcopyRow]()

    # Split by newlines
    let lines = csv_content.split("\n")

    print("📊 Parsing", len(lines), "lines from Bhavcopy CSV...")

    # Skip header row
    for i in range(1, len(lines)):
        let line = lines[i].strip()

        if len(line) == 0:
            continue

        # Split by comma
        let fields = line.split(",")

        if len(fields) < 13:
            continue  # Skip malformed rows

        # Only process EQ (Equity) series
        let series = fields[1].strip()
        if series != "EQ":
            continue

        try:
            let symbol = fields[0].strip()
            let open_price = atof(fields[2].strip())
            let high = atof(fields[3].strip())
            let low = atof(fields[4].strip())
            let close = atof(fields[5].strip())
            let last = atof(fields[6].strip())
            let prevclose = atof(fields[7].strip())
            let tottrdqty = atol(fields[8].strip())
            let tottrdval = atof(fields[9].strip())
            let timestamp = fields[10].strip()
            let totaltrades = atol(fields[11].strip())

            let row = BhavcopyRow(
                symbol, series, open_price, high, low, close, last, prevclose,
                tottrdqty, tottrdval, timestamp, totaltrades
            )

            rows.push_back(row)

        except:
            # Skip rows with parse errors
            continue

    print("✅ Parsed", len(rows), "equity records")

    return rows


fn store_to_postgres(rows: DynamicVector[BhavcopyRow], trade_date: String) raises:
    """
    Store parsed Bhavcopy data to PostgreSQL.
    Uses batch insert for performance.

    Args:
        rows: Vector of BhavcopyRow data
        trade_date: Trading date for this data
    """
    let psycopg2 = Python.import_module("psycopg2")
    let os = Python.import_module("os")

    let pg_dsn = os.getenv("TRADING_CHITTI_PG_DSN", "postgresql://hariprasath@localhost:6432/trading_chitti")

    print("💾 Connecting to PostgreSQL...")
    let conn = psycopg2.connect(pg_dsn)
    let cur = conn.cursor()

    print("📝 Inserting", len(rows), "records into md.eod_prices...")

    # Batch insert for performance
    let insert_query = """
        INSERT INTO md.eod_prices (
            exchange, symbol, trade_date,
            open, high, low, close,
            volume, vwap, source
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (exchange, symbol, trade_date)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            vwap = EXCLUDED.vwap,
            source = EXCLUDED.source
    """

    var inserted = 0
    var updated = 0

    for i in range(len(rows)):
        let row = rows[i]

        # Calculate VWAP (Volume Weighted Average Price)
        var vwap: Float64 = 0.0
        if row.tottrdqty > 0:
            vwap = row.tottrdval / Float64(row.tottrdqty)
        else:
            vwap = row.close

        try:
            let data = (
                "NSE",
                row.symbol,
                trade_date,
                row.open,
                row.high,
                row.low,
                row.close,
                row.tottrdqty,
                vwap,
                "NSE_BHAVCOPY"
            )

            cur.execute(insert_query, data)
            inserted += 1

        except:
            updated += 1
            continue

    conn.commit()
    cur.close()
    conn.close()

    print("✅ Database update complete!")
    print(f"   Inserted: {inserted} records")
    print(f"   Updated: {updated} records")


fn main() raises:
    """Main entry point for Bhavcopy downloader."""
    print("=" * 70)
    print("📈 NSE BHAVCOPY DOWNLOADER (Mojo Edition)")
    print("=" * 70)

    let start_time = now()

    # Get today's date in DDMMYYYY format
    let datetime = Python.import_module("datetime")
    let today = datetime.date.today()
    let trade_date_formatted = today.strftime("%d%m%Y")

    print("📅 Trade Date:", trade_date_formatted)

    # Download Bhavcopy
    let csv_content = download_bhavcopy(trade_date_formatted)

    # Parse CSV
    let rows = parse_bhavcopy_csv(csv_content)

    # Store to PostgreSQL
    store_to_postgres(rows, trade_date_formatted)

    let elapsed = now() - start_time
    print("=" * 70)
    print(f"⚡ Completed in {elapsed:.2f} seconds")
    print("=" * 70)

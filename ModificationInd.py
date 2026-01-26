from bmll2 import get_market_data_range
from pyspark.sql import functions as F

MICS = [
    "XPAR","CEUX","AQEU","XETR","BOTC","XAMS","XLIS","XMIL","XBRU",
    "XSWX","XOSL","XHEL","XSTO","XWAR","XCSE","XWBO","XATH","XPRA",
    "XBUD","XDUB","XTAL","TQEX","XEQT","XMAD",
]
START_DATE = "2024-06-27"
END_DATE   = "2024-06-31"
OUTPUT_DIR = None

def pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def save_df(df, name):
    if OUTPUT_DIR:
        df.coalesce(1).write.mode("overwrite").option("header", True).csv(
            OUTPUT_DIR.rstrip("/") + f"/{name}"
        )

trades_raw = get_market_data_range(MICS, START_DATE, END_DATE, "trades")
cols0 = trades_raw.columns

mic_col     = pick(cols0, ["MIC","mic"])
exec_col    = pick(cols0, ["ExecutionVenue","execution_venue"])
ts_col      = pick(cols0, ["TradeTimestamp","trade_timestamp","TradeTime","TradeDateTime","LocalTradeTimestamp"])
mod_col     = pick(cols0, ["ModificationIndicator","modification_indicator"])
tradeid_col = pick(cols0, ["TradeId","trade_id","tradeId"])

price_col   = pick(cols0, ["Price","price"])
size_col    = pick(cols0, ["Size","size"])
seq_src_col = pick(cols0, ["BMLLSequenceSource","bmll_sequence_source"])
seq_no_col  = pick(cols0, ["BMLLSequenceNo","bmll_sequence_no"])

need = [mic_col, ts_col, mod_col, tradeid_col]
if any(v is None for v in need):
    raise ValueError(f"Missing required columns. Columns: {cols0}")

trades = (
    trades_raw
      .withColumnRenamed(mic_col, "group_mic")
      .withColumnRenamed(ts_col, "trade_timestamp")
      .withColumnRenamed(mod_col, "modification_indicator")
      .withColumnRenamed(tradeid_col, "trade_id")
)

if exec_col:
    trades = trades.withColumnRenamed(exec_col, "exec_venue")
else:
    trades = trades.withColumn("exec_venue", F.lit(None).cast("string"))

if price_col and price_col != "price":
    trades = trades.withColumnRenamed(price_col, "price")
elif price_col is None:
    trades = trades.withColumn("price", F.lit(None).cast("double"))

if size_col and size_col != "size":
    trades = trades.withColumnRenamed(size_col, "size")
elif size_col is None:
    trades = trades.withColumn("size", F.lit(None).cast("double"))

if seq_src_col and seq_src_col != "bmll_seq_src":
    trades = trades.withColumnRenamed(seq_src_col, "bmll_seq_src")
elif seq_src_col is None:
    trades = trades.withColumn("bmll_seq_src", F.lit(None).cast("string"))

if seq_no_col and seq_no_col != "bmll_seq_no":
    trades = trades.withColumnRenamed(seq_no_col, "bmll_seq_no")
elif seq_no_col is None:
    trades = trades.withColumn("bmll_seq_no", F.lit(None).cast("long"))

trades = (
    trades
      .withColumn("trade_timestamp", F.col("trade_timestamp").cast("timestamp"))
      .withColumn("trade_date", F.to_date("trade_timestamp"))
      .withColumn(
          "join_scope",
          F.when(F.col("exec_venue").isNotNull(), F.col("exec_venue"))
           .otherwise(F.col("group_mic"))
      )
)

originals = (
    trades.filter(F.col("modification_indicator") == "-")
          .select(
              "group_mic","join_scope","trade_id",
              F.col("trade_timestamp").alias("orig_ts"),
              F.col("trade_date").alias("orig_date"),
          )
)

mods = (
    trades.filter(F.col("modification_indicator").isin("A","C"))
          .select(
              "group_mic","join_scope","trade_id",
              F.col("trade_timestamp").alias("event_ts"),
              F.col("trade_date").alias("event_date"),
              "modification_indicator",
          )
          .withColumn(
              "event_type",
              F.when(F.col("modification_indicator")=="A","amend").otherwise("cancel")
          )
          .select("group_mic","join_scope","trade_id","event_ts","event_date","event_type")
)

orig_min = (
    originals.groupBy("join_scope","trade_id")
             .agg(
                 F.min("orig_ts").alias("orig_ts"),
                 F.min("orig_date").alias("orig_date"),
                 F.first("group_mic", ignorenulls=True).alias("group_mic_from_orig"),
             )
)

mods_with_orig = (
    mods.join(orig_min, on=["join_scope","trade_id"], how="inner")
        .withColumn("group_mic", F.coalesce(F.col("group_mic"), F.col("group_mic_from_orig")))
        .drop("group_mic_from_orig")
        .filter(F.col("event_ts") >= F.col("orig_ts"))
        .withColumn("delay_seconds", F.col("event_ts").cast("long") - F.col("orig_ts").cast("long"))
        .withColumn("delay_minutes", F.col("delay_seconds")/60.0)
        .withColumn("delay_days",    F.col("delay_seconds")/86400.0)
)

day_counts_all = (
    trades.groupBy("group_mic","trade_date")
          .agg(F.count(F.lit(1)).alias("total_msgs"))
)

day_counts_events = (
    trades.filter(F.col("modification_indicator").isin("A","C"))
          .withColumn("is_amend",  (F.col("modification_indicator")=="A").cast("int"))
          .withColumn("is_cancel", (F.col("modification_indicator")=="C").cast("int"))
          .groupBy("group_mic","trade_date")
          .agg(
              F.sum("is_amend").alias("amend_cnt"),
              F.sum("is_cancel").alias("cancel_cnt"),
          )
)

day_originals = (
    trades.filter(F.col("modification_indicator") == "-")
          .groupBy("group_mic","trade_date")
          .agg(F.count(F.lit(1)).alias("orig_cnt"))
)

day_counts = (
    day_counts_all
      .join(day_counts_events, on=["group_mic","trade_date"], how="left")
      .join(day_originals,     on=["group_mic","trade_date"], how="left")
      .fillna({"amend_cnt":0,"cancel_cnt":0,"orig_cnt":0})
)

stats_by_mic = (
    day_counts.groupBy("group_mic")
              .agg(
                  F.countDistinct("trade_date").alias("coverage_days"),
                  F.avg("amend_cnt").alias("avg_amend_per_day"),
                  F.max("amend_cnt").alias("max_amend_per_day"),
                  F.avg("cancel_cnt").alias("avg_cancel_per_day"),
                  F.max("cancel_cnt").alias("max_cancel_per_day"),
                  F.sum("amend_cnt").alias("total_amend"),
                  F.sum("cancel_cnt").alias("total_cancel"),
                  F.sum("orig_cnt").alias("total_originals"),
                  F.sum("total_msgs").alias("total_msgs"),
              )
              .withColumn(
                  "events_per_1k_originals",
                  (F.col("total_amend")+F.col("total_cancel"))*1000.0/F.col("total_originals")
              )
              .withColumn(
                  "events_share_of_all_msgs_pct",
                  (F.col("total_amend")+F.col("total_cancel"))*100.0/F.col("total_msgs")
              )
              .orderBy("group_mic")
)

global_day = (
    day_counts.groupBy("trade_date")
              .agg(
                  F.sum("amend_cnt").alias("amend_cnt"),
                  F.sum("cancel_cnt").alias("cancel_cnt"),
                  F.sum("total_msgs").alias("total_msgs"),
                  F.sum("orig_cnt").alias("orig_cnt"),
              )
)

global_stats = (
    global_day.agg(
        F.countDistinct("trade_date").alias("coverage_days"),
        F.avg("amend_cnt").alias("avg_amend_per_day"),
        F.max("amend_cnt").alias("max_amend_per_day"),
        F.avg("cancel_cnt").alias("avg_cancel_per_day"),
        F.max("cancel_cnt").alias("max_cancel_per_day"),
        F.sum("amend_cnt").alias("total_amend"),
        F.sum("cancel_cnt").alias("total_cancel"),
        F.sum("orig_cnt").alias("total_originals"),
        F.sum("total_msgs").alias("total_msgs"),
    )
)

delay_stats_by_mic = (
    mods_with_orig.groupBy("group_mic","event_type")
                  .agg(
                      F.count(F.lit(1)).alias("events"),
                      F.avg("delay_minutes").alias("mean_delay_min"),
                      F.expr("percentile_approx(delay_minutes,0.5)").alias("median_delay_min"),
                      F.expr("percentile_approx(delay_minutes,0.9)").alias("p90_delay_min"),
                      F.max("delay_minutes").alias("max_delay_min"),
                  )
)

dups_tradeid = (
    trades.groupBy("group_mic","trade_id")
          .agg(F.count(F.lit(1)).alias("rows"))
          .filter(F.col("rows") > 1)
)

dup_summary_tradeid = (
    dups_tradeid.groupBy("group_mic")
                .agg(
                    F.count(F.lit(1)).alias("dup_keys_tradeid"),
                    F.sum("rows").alias("dup_rows_tradeid"),
                )
)

dups_payload = (
    trades.groupBy("group_mic","trade_id","trade_timestamp","price","size")
          .agg(F.count(F.lit(1)).alias("rows"))
          .filter(F.col("rows") > 1)
)

dup_summary_payload = (
    dups_payload.groupBy("group_mic")
                .agg(
                    F.count(F.lit(1)).alias("dup_keys_payload"),
                    F.sum("rows").alias("dup_rows_payload"),
                )
)

dups_seq = (
    trades.filter(F.col("bmll_seq_src").isNotNull() & F.col("bmll_seq_no").isNotNull())
          .groupBy("group_mic","bmll_seq_src","bmll_seq_no")
          .agg(F.count(F.lit(1)).alias("rows"))
          .filter(F.col("rows") > 1)
)

dup_summary_seq = (
    dups_seq.groupBy("group_mic")
            .agg(
                F.count(F.lit(1)).alias("dup_keys_seqno"),
                F.sum("rows").alias("dup_rows_seqno"),
            )
)

save_df(stats_by_mic,        "stats_by_mic")
save_df(global_stats,        "global_stats")
save_df(delay_stats_by_mic,  "delay_stats_by_mic")
save_df(dup_summary_tradeid, "dups_tradeid_summary")
save_df(dup_summary_payload, "dups_payload_summary")
save_df(dup_summary_seq,     "dups_seqno_summary")

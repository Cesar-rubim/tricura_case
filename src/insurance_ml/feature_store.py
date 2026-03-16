from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import pick_col, safe_join_unique


@dataclass
class FeatureStoreConfig:
    data_dir: Path
    date_start: str = "2023-10-01"
    date_end: str = "2024-12-30"
    output_file: str = "weekly_model_dataset.parquet"


def _build_resident_weekly(data_dir: Path) -> pd.DataFrame:
    residents = pd.read_parquet(data_dir / "residents.parquet").copy()
    for c in ["admission_date", "discharge_date", "deceased_date", "date_of_birth"]:
        if c in residents.columns:
            residents[c] = pd.to_datetime(residents[c], errors="coerce")

    if "date_of_birth" not in residents.columns:
        residents["date_of_birth"] = pd.NaT

    base = residents.dropna(subset=["admission_date"]).copy()

    cutoff = pd.concat(
        [
            base["admission_date"],
            base["discharge_date"] if "discharge_date" in base.columns else pd.Series(dtype="datetime64[ns]"),
            base["deceased_date"] if "deceased_date" in base.columns else pd.Series(dtype="datetime64[ns]"),
        ],
        ignore_index=True,
    ).max()

    end_candidates = [c for c in ["discharge_date", "deceased_date"] if c in base.columns]
    base["end_date"] = base[end_candidates].min(axis=1) if end_candidates else pd.NaT
    base["end_date"] = base["end_date"].fillna(cutoff)
    base = base[base["end_date"] >= base["admission_date"]].copy()

    rows: list[tuple] = []
    cols = ["resident_id", "facility_id", "admission_date", "end_date", "date_of_birth"]
    for r in base[cols].itertuples(index=False):
        for p in pd.period_range(r.admission_date, r.end_date, freq="W-SUN"):
            week_start = p.start_time
            week_end = p.end_time.floor("D")
            age_years = np.nan
            if pd.notna(r.date_of_birth):
                age_years = (week_start - r.date_of_birth).days / 365.25
            rows.append((r.resident_id, r.facility_id, week_start, week_end, age_years))

    return pd.DataFrame(rows, columns=["resident_id", "facility_id", "week_start", "week_end", "resident_age"])


def _build_vitals_weekly(data_dir: Path) -> pd.DataFrame:
    vitals = pd.read_parquet(data_dir / "vitals.parquet").copy()
    time_col = pick_col(list(vitals.columns), ["measured_at", "created_at"], required=True)

    vitals[time_col] = pd.to_datetime(vitals[time_col], errors="coerce")
    vitals = vitals.dropna(subset=["resident_id", time_col, "vital_type"])
    vitals["value"] = pd.to_numeric(vitals["value"], errors="coerce")

    if "dystolic_value" not in vitals.columns:
        vitals["dystolic_value"] = np.nan
    vitals["dystolic_value"] = pd.to_numeric(vitals["dystolic_value"], errors="coerce")

    vitals_pivot = (
        vitals.pivot_table(index=["resident_id", time_col], columns="vital_type", values="value", aggfunc="mean").reset_index()
    )

    dystolic_at_time = vitals.groupby(["resident_id", time_col], as_index=False)["dystolic_value"].mean()

    vitals_pivot = vitals_pivot.merge(dystolic_at_time, on=["resident_id", time_col], how="left")
    vitals_pivot = vitals_pivot.rename(columns={time_col: "event_at"})
    vitals_pivot["week_start"] = vitals_pivot["event_at"].dt.to_period("W").dt.start_time

    metric_cols = [c for c in vitals_pivot.columns if c not in ["resident_id", "event_at", "week_start"]]
    agg_map = {c: ["mean", "std"] for c in metric_cols if c != "dystolic_value"}
    agg_map["dystolic_value"] = ["mean"]

    out = vitals_pivot.groupby(["resident_id", "week_start"]).agg(agg_map)
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    return out.reset_index().sort_values(["resident_id", "week_start"]).reset_index(drop=True)


def _build_claims_weekly(data_dir: Path) -> pd.DataFrame:
    incidents = pd.read_parquet(data_dir / "incidents.parquet").copy()
    injuries = pd.read_parquet(data_dir / "injuries.parquet").copy()
    adm = pd.read_parquet(data_dir / "hospital_admissions.parquet").copy()
    trf = pd.read_parquet(data_dir / "hospital_transfers.parquet").copy()

    inc_time_col = pick_col(list(incidents.columns), ["occurred_at", "reported_at", "created_at"], required=True)
    incidents["incident_at"] = pd.to_datetime(incidents[inc_time_col], errors="coerce")
    incidents = incidents.dropna(subset=["incident_id", "resident_id", "incident_at"]).copy()

    if "incident_type" not in incidents.columns:
        incidents["incident_type"] = "Unknown"
    incidents["incident_type"] = incidents["incident_type"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    inj_id_col = pick_col(list(injuries.columns), ["injury_id"])
    if inj_id_col is None:
        injuries = injuries.reset_index(drop=False).rename(columns={"index": "injury_id"})
        inj_id_col = "injury_id"

    if "incident_id" in injuries.columns:
        inj_agg = (
            injuries[["incident_id", inj_id_col]].dropna(subset=["incident_id"]).groupby("incident_id", as_index=False).agg(injury_count=(inj_id_col, "nunique"))
        )
    else:
        inj_agg = pd.DataFrame(columns=["incident_id", "injury_count"])

    def hospital_link_count(events_df: pd.DataFrame, id_candidates: list[str], eff_candidates: list[str], inef_candidates: list[str]) -> pd.DataFrame:
        e = events_df.copy()
        id_col = pick_col(list(e.columns), id_candidates)
        if id_col is None:
            e = e.reset_index(drop=False).rename(columns={"index": "event_id"})
            id_col = "event_id"

        if "resident_id" not in e.columns:
            return pd.DataFrame(columns=["incident_id", "count", "first_event_at"])

        eff_col = pick_col(list(e.columns), eff_candidates)
        if eff_col is None:
            return pd.DataFrame(columns=["incident_id", "count", "first_event_at"])
        inef_col = pick_col(list(e.columns), inef_candidates)

        e["event_id"] = e[id_col]
        e["event_effective_at"] = pd.to_datetime(e[eff_col], errors="coerce")
        e["event_ineffective_at"] = pd.to_datetime(e[inef_col], errors="coerce") if inef_col else pd.NaT
        if "facility_id" not in e.columns:
            e["facility_id"] = pd.NA

        e = e[["event_id", "resident_id", "facility_id", "event_effective_at", "event_ineffective_at"]].dropna(
            subset=["resident_id", "event_effective_at"]
        )

        m = incidents[["incident_id", "resident_id", "facility_id", "incident_at"]].merge(e, on="resident_id", how="left", suffixes=("_inc", "_evt"))
        fac_ok = m["facility_id_inc"].isna() | m["facility_id_evt"].isna() | (m["facility_id_inc"] == m["facility_id_evt"])
        interval_contains = (m["event_effective_at"] <= m["incident_at"]) & (
            m["event_ineffective_at"].isna() | (m["event_ineffective_at"] >= m["incident_at"])
        )
        post_7d = (m["event_effective_at"] >= m["incident_at"]) & (m["event_effective_at"] <= m["incident_at"] + pd.Timedelta(days=7))

        m = m[fac_ok & (interval_contains | post_7d)]
        if m.empty:
            return pd.DataFrame(columns=["incident_id", "count", "first_event_at"])
        return m.groupby("incident_id", as_index=False).agg(count=("event_id", "nunique"), first_event_at=("event_effective_at", "min"))

    adm_agg = hospital_link_count(
        adm,
        ["admission_id", "hospital_admission_id"],
        ["effective_date", "admitted_at", "start_at", "created_at"],
        ["ineffective_date", "discharged_at", "end_at"],
    ).rename(columns={"count": "admission_count", "first_event_at": "first_admission_at"})

    trf_agg = hospital_link_count(
        trf,
        ["transfer_id", "hospital_transfer_id"],
        ["effective_date", "transferred_at", "start_at", "created_at"],
        ["ineffective_date", "end_at"],
    ).rename(columns={"count": "transfer_count", "first_event_at": "first_transfer_at"})

    claims_incident = (
        incidents[["incident_id", "resident_id", "facility_id", "incident_at", "incident_type"]]
        .merge(inj_agg, on="incident_id", how="left")
        .merge(adm_agg, on="incident_id", how="left")
        .merge(trf_agg, on="incident_id", how="left")
    )

    for c in ["injury_count", "admission_count", "transfer_count"]:
        claims_incident[c] = claims_incident[c].fillna(0).astype("int64")

    claims_incident["had_injury"] = (claims_incident["injury_count"] > 0).astype("int8")
    claims_incident["had_admission"] = (claims_incident["admission_count"] > 0).astype("int8")
    claims_incident["had_transfer"] = (claims_incident["transfer_count"] > 0).astype("int8")
    claims_incident["is_claim"] = 1

    claims_incident["week_start"] = claims_incident["incident_at"].dt.to_period("W").dt.start_time
    claims_incident["week_end"] = claims_incident["incident_at"].dt.to_period("W").dt.end_time.dt.floor("D")

    return (
        claims_incident.groupby(["resident_id", "week_start", "week_end"], as_index=False)
        .agg(
            claim_count=("incident_id", "nunique"),
            incident_type_list=("incident_type", safe_join_unique),
            incident_type_primary=("incident_type", lambda s: s.value_counts().index[0] if len(s) else "Unknown"),
            injury_count_sum=("injury_count", "sum"),
            admission_count_sum=("admission_count", "sum"),
            transfer_count_sum=("transfer_count", "sum"),
            had_injury_week=("had_injury", "max"),
            had_admission_week=("had_admission", "max"),
            had_transfer_week=("had_transfer", "max"),
            is_claim_week=("is_claim", "max"),
        )
        .sort_values(["resident_id", "week_start"])
        .reset_index(drop=True)
    )


def build_weekly_model_dataset(config: FeatureStoreConfig) -> Path:
    date_start = pd.Timestamp(config.date_start)
    date_end = pd.Timestamp(config.date_end)

    resident_weekly = _build_resident_weekly(config.data_dir)
    vitals_weekly = _build_vitals_weekly(config.data_dir)
    claims_weekly = _build_claims_weekly(config.data_dir)

    weekly_dataset = (
        resident_weekly.merge(vitals_weekly, on=["resident_id", "week_start"], how="left")
        .merge(claims_weekly, on=["resident_id", "week_start", "week_end"], how="left")
        .sort_values(["resident_id", "week_start"])
        .reset_index(drop=True)
    )

    weekly_dataset["claim_count"] = weekly_dataset["claim_count"].fillna(0).astype("int64")
    weekly_dataset["is_claim_week"] = weekly_dataset["is_claim_week"].fillna(0).astype("int8")
    weekly_dataset["incident_type_primary"] = weekly_dataset["incident_type_primary"].fillna("NoClaim")
    weekly_dataset["incident_type_list"] = weekly_dataset["incident_type_list"].fillna("NoClaim")

    weekly_dataset = weekly_dataset[(weekly_dataset["week_start"] >= date_start) & (weekly_dataset["week_start"] <= date_end)].copy()
    weekly_dataset = weekly_dataset.sort_values(["resident_id", "week_start"]).reset_index(drop=True)

    weekly_dataset["target_claim_next_week"] = (
        weekly_dataset.groupby("resident_id")["claim_count"].shift(-1).fillna(0) > 0
    ).astype("int8")
    weekly_dataset["target_claim_type_next_week"] = weekly_dataset.groupby("resident_id")["incident_type_primary"].shift(-1).fillna("NoClaim")

    has_future = weekly_dataset.groupby("resident_id")["week_start"].transform("max") > weekly_dataset["week_start"]
    weekly_dataset_final = weekly_dataset[has_future].copy()

    incident_current_week_cols = [
        "claim_count",
        "incident_type_list",
        "incident_type_primary",
        "injury_count_sum",
        "admission_count_sum",
        "transfer_count_sum",
        "had_injury_week",
        "had_admission_week",
        "had_transfer_week",
        "is_claim_week",
    ]
    drop_cols = [c for c in incident_current_week_cols if c in weekly_dataset_final.columns]
    weekly_dataset_final = weekly_dataset_final.drop(columns=drop_cols)

    out_path = config.data_dir / config.output_file
    weekly_dataset_final.to_parquet(out_path, index=False)
    return out_path

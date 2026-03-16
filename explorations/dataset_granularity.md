# Dataset Granularity

| Dataset | Granularity (one row = ...) | Main keys / notes |
|---|---|---|
| `residents.parquet` | one resident enrollment profile in a facility | `resident_id`, `facility_id`; includes DOB, admission/discharge/deceased dates |
| `incidents.parquet` | one incident event record | `incident_id`; tied to `resident_id`, `facility_id`, `occurred_at` |
| `injuries.parquet` | one injury linked to an incident | `injury_id`, `incident_id`; many injuries can map to one incident |
| `factors.parquet` | one contributing factor linked to an incident | `factor_id`, `incident_id`; many factors per incident |
| `vitals.parquet` | one vital measurement observation | `vital_id`; resident/facility + `vital_type`, `measured_at` |
| `lab_reports.parquet` | one lab report/result record | `lab_report_id`; resident/facility + reported/collected timestamps |
| `medications.parquet` | one medication administration/scheduled event | `medication_id`; resident/facility + scheduled/administered times |
| `physician_orders.parquet` | one physician order record | `order_id`; resident/facility + ordered/start/end dates |
| `adl_responses.parquet` | one ADL response item for an activity at an assessment time | `adl_response_id`; resident/facility + `activity`, `assessment_date` |
| `gg_responses.parquet` | one GG functional response item | `gg_response_id`; resident/facility + task + response |
| `care_plans.parquet` | one care plan lifecycle record | `care_plan_id`; initiated/closed/review timestamps |
| `needs.parquet` | one need item (usually under a care plan) | `need_id`, `care_plan_id`; initiated/resolved lifecycle |
| `diagnoses.parquet` | one diagnosis episode/record | `diagnosis_id`; resident/facility + onset/resolution |
| `hospital_admissions.parquet` | one hospital admission interval/event | `admission_id`; resident/facility + effective/ineffective dates |
| `hospital_transfers.parquet` | one transfer interval/event | `transfer_id`; resident/facility + effective/ineffective dates |
| `therapy_tracks.parquet` | one therapy track episode (discipline-specific) | `therapy_id`; resident/facility + start/end |
| `document_tags.parquet` | one extracted document tag hit | `document_tag_id`; resident/facility + doc type/tag/confidence |

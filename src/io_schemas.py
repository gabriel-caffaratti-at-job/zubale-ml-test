# src/io_schemas.py
from typing import List, Literal
from pydantic import BaseModel, Field

Plan = Literal["Basic","Standard","Pro"]
Contract = Literal["Monthly","Annual"]
YesNo = Literal["Yes","No"]

class ChurnRow(BaseModel):
    plan_type: Plan
    contract_type: Contract
    autopay: YesNo
    is_promo_user: YesNo
    add_on_count: float
    tenure_months: float
    monthly_usage_gb: float
    avg_latency_ms: float
    support_tickets_30d: float
    discount_pct: float
    payment_failures_90d: float
    downtime_hours_30d: float

import math
from datetime import datetime
from typing import List, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import BaseModel, Field

# Check for Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ==============================================================================
# 1. PAGE CONFIG & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Bradford Box Office Initial Forecaster",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 14px 18px;
        border-radius: 10px;
        color: #f8fafc;
    }
    div[data-testid="stMetricLabel"] > div { color: #94a3b8 !important; font-weight: 600; }
    div[data-testid="stMetricValue"] > div { color: #f8fafc !important; }
    .ai-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. BRADFORD VENUE PRESETS & GENRE ARCHETYPE PARAMETERS
# ==============================================================================
BRADFORD_VENUES = {
    "Alhambra Theatre": {
        "default_per_perf": 1450,
        "default_run": 8,
        "default_gp": 340000.0,
        "default_weeks": 38,
        "desc": "No. 1 large-scale touring house (Musicals, Panto, Drama, Dance)",
    },
    "St George's Hall": {
        "default_per_perf": 1335,
        "default_run": 1,
        "default_gp": 42000.0,
        "default_weeks": 28,
        "desc": "Premier concert hall (Comedy, Tribute, Live Music, Spoken Word, Classical)",
    },
    "King's Hall, Ilkley": {
        "default_per_perf": 490,
        "default_run": 1,
        "default_gp": 13500.0,
        "default_weeks": 22,
        "desc": "Victorian touring & community hall (Mid-scale Comedy, Music, Talks)",
    },
    "The Studio": {
        "default_per_perf": 200,
        "default_run": 2,
        "default_gp": 7200.0,
        "default_weeks": 16,
        "desc": "Intimate studio space (New Writing, Fringe, Stand-Up, Community)",
    },
    "Custom Configuration": {
        "default_per_perf": 1200,
        "default_run": 8,
        "default_gp": 250000.0,
        "default_weeks": 32,
        "desc": "Manual capacity and run settings",
    },
}

GENRE_ARCHETYPES = {
    "No. 1 Touring Musical (Blockbuster)": {
        "base_gp_pct": 0.84,
        "ref_run": 8,
        "dilution_exp": 0.18,
        "launch_ticket_share": 0.31,
        "early_yield_skew": 1.17,
        "pre_open_share": 0.92,
        "group_cliff_week": 8,
        "curve_shape": [0.36, 0.44, 0.51, 0.61, 0.74, 0.84, 0.92, 1.00],
    },
    "Jukebox / Feel-Good Touring Musical": {
        "base_gp_pct": 0.74,
        "ref_run": 8,
        "dilution_exp": 0.22,
        "launch_ticket_share": 0.22,
        "early_yield_skew": 1.15,
        "pre_open_share": 0.89,
        "group_cliff_week": 6,
        "curve_shape": [0.25, 0.32, 0.39, 0.50, 0.65, 0.78, 0.89, 1.00],
    },
    "Annual Pantomime": {
        "base_gp_pct": 0.88,
        "ref_run": 48,
        "dilution_exp": 0.14,
        "launch_ticket_share": 0.24,
        "early_yield_skew": 1.12,
        "pre_open_share": 0.85,
        "group_cliff_week": 10,
        "curve_shape": [0.27, 0.33, 0.35, 0.38, 0.55, 0.73, 0.85, 1.00],
    },
    "Touring Drama / Star-Led Play": {
        "base_gp_pct": 0.62,
        "ref_run": 8,
        "dilution_exp": 0.27,
        "launch_ticket_share": 0.12,
        "early_yield_skew": 1.13,
        "pre_open_share": 0.81,
        "group_cliff_week": 6,
        "curve_shape": [0.14, 0.19, 0.25, 0.35, 0.51, 0.66, 0.81, 1.00],
    },
    "Set-Text / Classic Drama (Schools Heavy)": {
        "base_gp_pct": 0.68,
        "ref_run": 8,
        "dilution_exp": 0.24,
        "launch_ticket_share": 0.14,
        "early_yield_skew": 1.06,
        "pre_open_share": 0.86,
        "group_cliff_week": 8,
        "curve_shape": [0.15, 0.22, 0.31, 0.48, 0.68, 0.79, 0.86, 1.00],
    },
    "One-Nighter: Arena/TV Stand-Up Comedy": {
        "base_gp_pct": 0.90,
        "ref_run": 1,
        "dilution_exp": 0.20,
        "launch_ticket_share": 0.48,
        "early_yield_skew": 1.08,
        "pre_open_share": 0.98,
        "group_cliff_week": 0,
        "curve_shape": [0.52, 0.60, 0.67, 0.75, 0.84, 0.92, 0.98, 1.00],
    },
    "One-Nighter: Tribute / Nostalgia / Concert": {
        "base_gp_pct": 0.73,
        "ref_run": 1,
        "dilution_exp": 0.32,
        "launch_ticket_share": 0.19,
        "early_yield_skew": 1.10,
        "pre_open_share": 0.96,
        "group_cliff_week": 0,
        "curve_shape": [0.21, 0.27, 0.34, 0.46, 0.63, 0.79, 0.96, 1.00],
    },
    "Family / Children's Branded Adaptations": {
        "base_gp_pct": 0.70,
        "ref_run": 4,
        "dilution_exp": 0.25,
        "launch_ticket_share": 0.16,
        "early_yield_skew": 1.09,
        "pre_open_share": 0.84,
        "group_cliff_week": 4,
        "curve_shape": [0.17, 0.23, 0.30, 0.40, 0.56, 0.71, 0.84, 1.00],
    },
    "Opera / Ballet / Classical": {
        "base_gp_pct": 0.58,
        "ref_run": 3,
        "dilution_exp": 0.28,
        "launch_ticket_share": 0.21,
        "early_yield_skew": 1.18,
        "pre_open_share": 0.88,
        "group_cliff_week": 6,
        "curve_shape": [0.25, 0.32, 0.40, 0.52, 0.66, 0.78, 0.88, 1.00],
    },
}

BRAND_TIER_MAP = {
    1: ("Tier 1 — Niche / Unfamiliar / New IP", 0.78, 0.65),
    2: ("Tier 2 — Recognised Genre Piece / Cult Following", 0.90, 0.82),
    3: ("Tier 3 — Solid Mainstream Touring Title", 1.00, 1.00),
    4: ("Tier 4 — High-Demand Household Brand / Hit Tour", 1.10, 1.22),
    5: ("Tier 5 — Blockbuster Phenomenon / Instant Sell-Out", 1.18, 1.45),
}

AVAILABLE_GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "Custom Model ID...",
]


# ==============================================================================
# 3. STRUCTURED SCHEMA FOR GEMINI AI TITLE ANALYSIS
# ==============================================================================
class ShowTitleAnalysis(BaseModel):
    recommended_archetype: Literal[
        "No. 1 Touring Musical (Blockbuster)",
        "Jukebox / Feel-Good Touring Musical",
        "Annual Pantomime",
        "Touring Drama / Star-Led Play",
        "Set-Text / Classic Drama (Schools Heavy)",
        "One-Nighter: Arena/TV Stand-Up Comedy",
        "One-Nighter: Tribute / Nostalgia / Concert",
        "Family / Children's Branded Adaptations",
        "Opera / Ballet / Classical",
    ] = Field(description="Closest matching regional theatre genre archetype.")
    brand_tier: int = Field(
        ge=1,
        le=5,
        description="Brand recognition tier from 1 (obscure/new) to 5 (mega blockbuster/household name).",
    )
    star_casting_multiplier: float = Field(
        ge=0.85,
        le=1.25,
        description="Multiplier for star casting or marquee talent appeal (1.00 = standard tour cast, 1.15+ = major TV/West End star).",
    )
    bradford_regional_multiplier: float = Field(
        ge=0.85,
        le=1.20,
        description="West Yorkshire / Bradford demographic and cultural affinity multiplier (1.00 = neutral, >1.05 = strong Yorkshire/Northern or local demographic resonance).",
    )
    catchment_saturation_factor: float = Field(
        ge=0.75,
        le=1.05,
        description="Adjustment for West Yorkshire touring saturation (e.g., 0.85 if heavily toured in Leeds/Sheffield/Manchester recently; 1.00+ if rare/premiere).",
    )
    launch_urgency_modifier: float = Field(
        ge=0.80,
        le=1.35,
        description="Modifier for how heavily Friends/Priority members and day-1 fans will front-load bookings at On-Sale.",
    )
    comparable_titles: List[str] = Field(
        description="3 comparable touring shows that follow a similar booking pattern in UK regional receiving theatres."
    )
    executive_summary: str = Field(
        description="Concise, commercially sharp 2-3 sentence assessment of expected booking behaviour in Bradford (Alhambra / St George's Hall catchment)."
    )


def analyze_title_with_gemini(
    api_key: str,
    model_id: str,
    title: str,
    notes: str,
    venue_name: str,
    run_perfs: int,
    sales_weeks: int,
    gp: float,
) -> ShowTitleAnalysis:
    """Calls Gemini Flash to evaluate the show title for Bradford Theatres."""
    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an expert UK Regional Receiving House Theatre Programmer and Box Office Analyst,
    specifically advising Bradford Theatres (Alhambra Theatre, St George's Hall, King's Hall Ilkley, The Studio)
    in West Yorkshire.

    Evaluate the following proposed / upcoming show for initial box office forecasting:
    - Show Title: "{title}"
    - Casting / Producer / Context Notes: "{notes if notes else 'None provided'}"
    - Target Venue: {venue_name}
    - Number of Performances: {run_perfs}
    - Length of Sales Period: {sales_weeks} weeks
    - Gross Potential (GP): £{gp:,.0f}

    Consider:
    1. How strongly this title resonates with the Bradford & West Yorkshire catchment (including drive-time from Huddersfield, Halifax, Keighley, Ilkley, Skipton, and West Leeds).
    2. How heavily Priority/Friends members and group bookers will jump on this title during the first 7 days of On-Sale vs waiting for late reviews/payday impulses.
    3. Whether this title suffers from West Yorkshire catchment saturation (e.g., frequent repeat runs at Leeds Grand, Leeds Playhouse, or Sheffield Lyceum) or has scarcity value.

    Return a structured parameterization matching the schema.
    """

    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ShowTitleAnalysis,
            temperature=0.2,
        ),
    )
    return ShowTitleAnalysis.model_validate_json(response.text)


# ==============================================================================
# 4. SESSION STATE INITIALISATION
# ==============================================================================
DEFAULT_STATE = {
    "venue_choice": "Alhambra Theatre",
    "show_title": "Sister Act",
    "show_notes": "UK Tour, starring TV/West End guest lead",
    "gp_input": 340000.0,
    "seat_mode": "Per-Performance Capacity",
    "per_perf_seats": 1450,
    "total_run_seats": 11600,
    "run_perfs": 8,
    "sales_weeks": 38,
    "archetype": "Jukebox / Feel-Good Touring Musical",
    "brand_tier": 4,
    "star_mult": 1.08,
    "regional_mult": 1.05,
    "saturation_mult": 0.98,
    "launch_urgency": 1.05,
    "ai_analysis_result": None,
    "last_used_model": None,
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


def on_venue_change():
    v_data = BRADFORD_VENUES[st.session_state.venue_choice]
    st.session_state.per_perf_seats = v_data["default_per_perf"]
    st.session_state.run_perfs = v_data["default_run"]
    st.session_state.total_run_seats = v_data["default_per_perf"] * v_data["default_run"]
    st.session_state.gp_input = v_data["default_gp"]
    st.session_state.sales_weeks = v_data["default_weeks"]
    if st.session_state.venue_choice == "St George's Hall":
        st.session_state.archetype = "One-Nighter: Tribute / Nostalgia / Concert"


# ==============================================================================
# 5. CORE MATHEMATICAL FORECASTING ENGINE
# ==============================================================================
def compute_forecast(
    gp: float,
    total_seats: int,
    run_perfs: int,
    sales_weeks: int,
    archetype_name: str,
    brand_tier: int,
    star_mult: float,
    regional_mult: float,
    saturation_mult: float,
    launch_urgency: float,
) -> dict:
    arch = GENRE_ARCHETYPES[archetype_name]
    _, brand_overall_mult, brand_launch_mult = BRAND_TIER_MAP[brand_tier]

    # 1. Implied Base Average Ticket Price (ATP) at 100% GP
    base_atp = gp / max(total_seats, 1)

    # 2. Run-Length Dilution Factor
    raw_dilution = (arch["ref_run"] / max(run_perfs, 1)) ** arch["dilution_exp"]
    run_dilution_factor = float(np.clip(raw_dilution, 0.65, 1.18))

    launch_concentration = float(
        np.clip((arch["ref_run"] / max(run_perfs, 1)) ** (arch["dilution_exp"] * 0.65), 0.75, 1.30)
    )

    # 3. Sales Period Maturity Factor
    window_maturity = float(np.clip(0.54 + 0.125 * math.log(max(sales_weeks, 4)), 0.72, 1.07))
    short_window_compression = float(np.clip((32.0 / max(sales_weeks, 6)) ** 0.14, 0.88, 1.25))

    # 4. Projected Final Settlement (% of GP and £)
    combined_demand_mult = (
        arch["base_gp_pct"]
        * brand_overall_mult
        * star_mult
        * regional_mult
        * saturation_mult
        * run_dilution_factor
        * window_maturity
    )
    final_gp_pct = float(np.clip(combined_demand_mult, 0.15, 0.985))
    final_gross = gp * final_gp_pct

    final_achieved_atp = base_atp * (1.0 + 0.06 * (1.0 - final_gp_pct))
    final_tickets = int(min(total_seats, round(final_gross / max(final_achieved_atp, 1.0))))

    # 5. Estimated On-Sale Opening Value (First 7 Days: Priority + General Launch Week)
    raw_launch_ticket_share = (
        arch["launch_ticket_share"]
        * brand_launch_mult
        * launch_urgency
        * launch_concentration
        * short_window_compression
    )
    launch_share_of_final_tickets = float(np.clip(raw_launch_ticket_share, 0.05, 0.78))
    launch_tickets = int(round(final_tickets * launch_share_of_final_tickets))

    launch_atp = base_atp * arch["early_yield_skew"]
    launch_gross = min(final_gross * 0.85, launch_tickets * launch_atp)
    launch_gp_pct = launch_gross / max(gp, 1.0)

    # 6. Estimated Advance at Curtain-Up (Morning of Performance 1)
    in_run_walkup_share = (1.0 - arch["pre_open_share"]) * (1.0 if run_perfs > 1 else 0.25)
    if final_gp_pct > 0.90:
        in_run_walkup_share *= 0.55

    pre_open_share_effective = float(
        np.clip(1.0 - in_run_walkup_share, launch_gp_pct / max(final_gp_pct, 0.01) + 0.05, 0.99)
    )
    curtain_up_gross = final_gross * pre_open_share_effective
    curtain_up_gp_pct = curtain_up_gross / max(gp, 1.0)
    curtain_up_atp = base_atp * (1.0 + (arch["early_yield_skew"] - 1.0) * 0.35)
    curtain_up_tickets = int(min(final_tickets, round(curtain_up_gross / max(curtain_up_atp, 1.0))))

    # 7. Cumulative Pacing Trajectory Across Campaign Milestones
    milestone_labels = [
        f"On-Sale (W-{sales_weeks})",
        f"W-{int(round(sales_weeks * 0.85))}",
        f"W-{int(round(sales_weeks * 0.70))}",
        f"Mid-Campaign (W-{int(round(sales_weeks * 0.50))})",
        f"W-{int(round(sales_weeks * 0.30))}",
        f"Group/Push (W-{int(round(sales_weeks * 0.15))})",
        "Curtain-Up (Perf 1)",
        "Final Settlement",
    ]

    base_curve = arch["curve_shape"]
    launch_ratio = launch_gross / max(final_gross, 1.0)
    curtain_ratio = curtain_up_gross / max(final_gross, 1.0)

    mid_points_raw = np.array(base_curve[1:6])
    mid_min, mid_max = base_curve[0], base_curve[6]
    mid_norm = (mid_points_raw - mid_min) / max(mid_max - mid_min, 0.01)
    adjusted_mids = launch_ratio + mid_norm * (curtain_ratio - launch_ratio)

    full_ratios = [launch_ratio] + list(adjusted_mids) + [curtain_ratio, 1.00]
    curve_gross = [r * final_gross for r in full_ratios]
    curve_gp_pct = [(g / max(gp, 1.0)) * 100.0 for g in curve_gross]

    return {
        "base_atp": base_atp,
        "launch_atp": launch_atp,
        "run_dilution_factor": run_dilution_factor,
        "window_maturity": window_maturity,
        "launch_gross": launch_gross,
        "launch_gp_pct": launch_gp_pct,
        "launch_tickets": launch_tickets,
        "curtain_up_gross": curtain_up_gross,
        "curtain_up_gp_pct": curtain_up_gp_pct,
        "curtain_up_tickets": curtain_up_tickets,
        "final_gross": final_gross,
        "final_gp_pct": final_gp_pct,
        "final_tickets": final_tickets,
        "milestone_labels": milestone_labels,
        "curve_gross": curve_gross,
        "curve_gp_pct": curve_gp_pct,
    }


# ==============================================================================
# 6. SIDEBAR: GEMINI API KEY, MODEL SELECTOR & VENUE / SHOW INPUTS
# ==============================================================================
with st.sidebar:
    st.title("🎭 Bradford Forecaster")
    st.caption("Initial Box Office Opening & Pacing Estimator")

    # Check if a default key is stored in Streamlit Cloud Secrets
    secret_api_key = ""
    try:
        secret_api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_api_key = ""

    with st.expander("🔑 Gemini AI Configuration", expanded=True):
        gemini_api_key_input = st.text_input(
            "Gemini API Key",
            value="",
            type="password",
            placeholder="Paste AIza... key here (or set in st.secrets)",
            help="Enter your Google AI Studio API key here. If you set GEMINI_API_KEY in Streamlit Cloud secrets, you can leave this blank.",
        )
        effective_api_key = gemini_api_key_input.strip() or secret_api_key

        if secret_api_key and not gemini_api_key_input.strip():
            st.caption("✅ Using API key from Streamlit Secrets")

        selected_model_option = st.selectbox(
            "Gemini Model Version",
            options=AVAILABLE_GEMINI_MODELS,
            index=0,
            help="gemini-3.8-flash is the latest flagship Flash model as of Autumn 2026.",
        )
        if selected_model_option == "Custom Model ID...":
            active_model_id = st.text_input("Enter Custom Model ID", value="gemini-3.8-flash").strip()
        else:
            active_model_id = selected_model_option

        if not GENAI_AVAILABLE:
            st.warning("Install `google-genai` (`pip install google-genai`) to enable AI title scoring.")

    st.divider()
    st.subheader("1. Venue & Show Setup")

    st.selectbox(
        "Bradford Venue Preset",
        options=list(BRADFORD_VENUES.keys()),
        key="venue_choice",
        on_change=on_venue_change,
    )
    st.caption(BRADFORD_VENUES[st.session_state.venue_choice]["desc"])

    st.text_input("Show Title", key="show_title", placeholder="e.g., Hamilton, Fastlove, Sarah Millican")
    st.text_input(
        "Casting / Tour Notes (Optional)",
        key="show_notes",
        placeholder="e.g., Star casting TBC, first time in Bradford",
    )

    if st.button("✨ Analyse Title with Gemini AI", use_container_width=True, type="primary"):
        if not effective_api_key:
            st.error("Please enter your Gemini API Key in the box above.")
        elif not GENAI_AVAILABLE:
            st.error("The `google-genai` package is not installed.")
        elif not st.session_state.show_title.strip():
            st.error("Please enter a Show Title.")
        else:
            with st.spinner(
                f"Analysing '{st.session_state.show_title}' with {active_model_id}..."
            ):
                try:
                    analysis = analyze_title_with_gemini(
                        api_key=effective_api_key,
                        model_id=active_model_id,
                        title=st.session_state.show_title,
                        notes=st.session_state.show_notes,
                        venue_name=st.session_state.venue_choice,
                        run_perfs=st.session_state.run_perfs,
                        sales_weeks=st.session_state.sales_weeks,
                        gp=st.session_state.gp_input,
                    )
                    st.session_state.archetype = analysis.recommended_archetype
                    st.session_state.brand_tier = analysis.brand_tier
                    st.session_state.star_mult = float(analysis.star_casting_multiplier)
                    st.session_state.regional_mult = float(analysis.bradford_regional_multiplier)
                    st.session_state.saturation_mult = float(analysis.catchment_saturation_factor)
                    st.session_state.launch_urgency = float(analysis.launch_urgency_modifier)
                    st.session_state.ai_analysis_result = analysis
                    st.session_state.last_used_model = active_model_id
                    st.success(f"Parameters calibrated via {active_model_id}!")
                except Exception as e:
                    st.error(f"Gemini API Error ({active_model_id}): {e}")

    st.divider()
    st.subheader("2. Commercial & Run Inputs")

    st.number_input(
        "Gross Potential (GP £)",
        min_value=1000.0,
        max_value=5000000.0,
        step=2500.0,
        format="%.2f",
        key="gp_input",
        help="Total net/gross potential built from your house scaling & pricing model across the run.",
    )

    st.number_input(
        "Length of Run (Performances)",
        min_value=1,
        max_value=100,
        step=1,
        key="run_perfs",
    )

    st.radio(
        "Seat Input Mode",
        options=["Per-Performance Capacity", "Total Seats Across Run"],
        key="seat_mode",
        horizontal=True,
    )

    if st.session_state.seat_mode == "Per-Performance Capacity":
        st.number_input(
            "Sellable Seats per Performance",
            min_value=50,
            max_value=3000,
            step=10,
            key="per_perf_seats",
        )
        effective_total_seats = st.session_state.per_perf_seats * st.session_state.run_perfs
        st.caption(f"**Total Run Capacity:** {effective_total_seats:,} seats ({st.session_state.run_perfs} perfs)")
    else:
        st.number_input(
            "Total Sellable Seats (Full Run)",
            min_value=50,
            max_value=250000,
            step=100,
            key="total_run_seats",
        )
        effective_total_seats = st.session_state.total_run_seats
        st.caption(
            f"**Implied Seats / Perf:** {int(round(effective_total_seats / max(st.session_state.run_perfs, 1))):,} seats"
        )

    st.slider(
        "Length of Sales Period (Weeks)",
        min_value=4,
        max_value=65,
        step=1,
        key="sales_weeks",
        help="Weeks between Priority/General On-Sale and Performance 1 Curtain-Up.",
    )


# ==============================================================================
# 7. MAIN PANEL: HEADER, AI INSIGHTS & CALIBRATION SLIDERS
# ==============================================================================
st.markdown(
    f"## 📊 Initial Box Office Opening Forecaster — **{st.session_state.show_title or 'Untitled Show'}**"
)
st.caption(
    f"**Venue:** {st.session_state.venue_choice}  |  "
    f"**Run:** {st.session_state.run_perfs} perf(s) ({effective_total_seats:,} total seats)  |  "
    f"**Gross Potential:** £{st.session_state.gp_input:,.0f}  |  "
    f"**Sales Window:** {st.session_state.sales_weeks} weeks"
)

if st.session_state.ai_analysis_result is not None:
    ai_res: ShowTitleAnalysis = st.session_state.ai_analysis_result
    comps_str = " • ".join(ai_res.comparable_titles)
    model_badge = st.session_state.last_used_model or active_model_id
    st.markdown(
        f"""
        <div class="ai-box">
            <strong>🤖 Gemini Title Intelligence ({st.session_state.venue_choice} — <code>{model_badge}</code>):</strong><br/>
            {ai_res.executive_summary}<br/>
            <small style="color: #38bdf8;"><strong>Historical Pacing Comparables:</strong> {comps_str}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("🎛️ Title Demand & Bradford Catchment Calibration (AI-Populated / Manual Override)", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.selectbox(
            "Genre Pacing Archetype",
            options=list(GENRE_ARCHETYPES.keys()),
            key="archetype",
        )
        st.select_slider(
            "Title / Brand Tier (1–5)",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: BRAND_TIER_MAP[x][0],
            key="brand_tier",
        )
    with c2:
        st.slider(
            "Star Casting / Marquee Pull",
            min_value=0.85,
            max_value=1.25,
            step=0.01,
            key="star_mult",
            help="1.00 = Standard tour cast; 1.15+ = Household TV/West End name driving early surge.",
        )
        st.slider(
            "Bradford / West Yorks Affinity",
            min_value=0.85,
            max_value=1.20,
            step=0.01,
            key="regional_mult",
            help="Adjusts for local West Yorkshire demographic appeal and northern cultural resonance.",
        )
    with c3:
        st.slider(
            "Catchment Saturation Factor",
            min_value=0.75,
            max_value=1.05,
            step=0.01,
            key="saturation_mult",
            help="< 1.00 if the title recently played Leeds Grand, Sheffield, or Bradford within 24–36 months.",
        )
        st.slider(
            "Friends / Priority Launch Urgency",
            min_value=0.80,
            max_value=1.35,
            step=0.01,
            key="launch_urgency",
            help="Controls how aggressively Priority members and core fans front-load Week 1 bookings.",
        )


# ==============================================================================
# 8. RUN FORECAST & DISPLAY PRIMARY KPI CHECKPOINTS
# ==============================================================================
results = compute_forecast(
    gp=st.session_state.gp_input,
    total_seats=effective_total_seats,
    run_perfs=st.session_state.run_perfs,
    sales_weeks=st.session_state.sales_weeks,
    archetype_name=st.session_state.archetype,
    brand_tier=st.session_state.brand_tier,
    star_mult=st.session_state.star_mult,
    regional_mult=st.session_state.regional_mult,
    saturation_mult=st.session_state.saturation_mult,
    launch_urgency=st.session_state.launch_urgency,
)

st.subheader("🎯 Initial Forecasting Checkpoints (Base Case)")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="1. On-Sale Opening (First 7 Days)",
        value=f"£{results['launch_gross']:,.0f}",
        delta=f"{results['launch_gp_pct'] * 100:.1f}% of GP  ({results['launch_tickets']:,} tix)",
    )
    st.caption(
        f"Priority + General Launch Week | Est. Early ATP: **£{results['launch_atp']:.2f}**"
    )

with kpi2:
    st.metric(
        label="2. Opening Night Advance (Curtain-Up)",
        value=f"£{results['curtain_up_gross']:,.0f}",
        delta=f"{results['curtain_up_gp_pct'] * 100:.1f}% of GP  ({results['curtain_up_tickets']:,} tix)",
    )
    st.caption(
        f"Banked before Perf 1 opens | Avg **{int(round(results['curtain_up_tickets'] / max(st.session_state.run_perfs, 1))):,}** seats/perf"
    )

with kpi3:
    st.metric(
        label="3. Projected Final Settlement",
        value=f"£{results['final_gross']:,.0f}",
        delta=f"{results['final_gp_pct'] * 100:.1f}% of GP  ({results['final_tickets']:,} tix)",
    )
    in_run_gross = results["final_gross"] - results["curtain_up_gross"]
    st.caption(
        f"Includes **£{in_run_gross:,.0f}** in-run walk-up | Avg **{int(round(results['final_tickets'] / max(st.session_state.run_perfs, 1))):,}** seats/perf"
    )

with kpi4:
    st.metric(
        label="Pricing & Structural Mechanics",
        value=f"£{results['base_atp']:.2f} Base ATP",
        delta=f"Run Factor: {results['run_dilution_factor']:.2f}x | Window: {results['window_maturity']:.2f}x",
        delta_color="off",
    )
    st.caption(
        f"100% GP = £{st.session_state.gp_input / max(st.session_state.run_perfs, 1):,.0f} per performance"
    )


# ==============================================================================
# 9. VISUALISATION & SCENARIO MATRIX TABS
# ==============================================================================
tab_chart, tab_scenarios, tab_export = st.tabs(
    ["📈 Cumulative Pacing Trajectory", "📋 Low / Base / Stretch Scenario Table", "📥 Export Forecast"]
)

with tab_chart:
    col_plot, col_notes = st.columns([3, 1])

    with col_plot:
        fig = go.Figure()

        upper_curve = [min(st.session_state.gp_input * 0.99, g * 1.10) for g in results["curve_gross"]]
        lower_curve = [g * 0.90 for g in results["curve_gross"]]

        fig.add_trace(
            go.Scatter(
                x=results["milestone_labels"] + results["milestone_labels"][::-1],
                y=upper_curve + lower_curve[::-1],
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.14)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="±10% Confidence Band",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=results["milestone_labels"],
                y=results["curve_gross"],
                mode="lines+markers+text",
                name="Projected Cumulative Gross (£)",
                line=dict(color="#38bdf8", width=4),
                marker=dict(size=10, color="#0284c7", line=dict(width=2, color="#ffffff")),
                text=[f"£{v/1000:,.0f}k ({p:.0f}%)" for v, p in zip(results["curve_gross"], results["curve_gp_pct"])],
                textposition="top center",
                hovertemplate="<b>%{x}</b><br>Projected Gross: £%{y:,.0f}<extra></extra>",
            )
        )

        fig.add_hline(
            y=st.session_state.gp_input,
            line_dash="dash",
            line_color="#f43f5e",
            annotation_text=f"100% Gross Potential (£{st.session_state.gp_input:,.0f})",
            annotation_position="top left",
        )

        fig.update_layout(
            title=f"Projected Box Office Accumulation Curve — {st.session_state.show_title}",
            xaxis_title="Campaign Timeline Milestone",
            yaxis_title="Cumulative Box Office (£)",
            yaxis=dict(tickprefix="£", range=[0, st.session_state.gp_input * 1.12]),
            template="plotly_dark",
            height=460,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_notes:
        st.markdown("#### Pacing Dynamics")
        arch_info = GENRE_ARCHETYPES[st.session_state.archetype]
        st.markdown(
            f"""
            * **Early Yield Skew ({arch_info['early_yield_skew']:.2f}x):** Priority & Week-1 bookers heavily target top-price Stalls and Dress Circle, meaning **£% of GP outpaces % of seats sold** early on.
            * **On-Sale Share of Total Demand:** Estimated at **{(results['launch_gross'] / max(results['final_gross'], 1)) * 100:.1f}%** of the show's eventual final revenue.
            * **In-Run Walk-Up Window:** Estimated at **{(1 - (results['curtain_up_gross'] / max(results['final_gross'], 1))) * 100:.1f}%** of revenue banking after Curtain-Up on Performance 1.
            """
        )
        if arch_info["group_cliff_week"] > 0:
            st.info(
                f"📌 **Group / School Allocation Checkpoint:** Watch **W-{arch_info['group_cliff_week']}** closely—unconfirmed group reservations will either convert to banked cash or release back to general sale here."
            )

with tab_scenarios:
    st.markdown("#### Scenario Planning Matrix (Conservative / Base / Stretch)")

    scenarios = [
        ("Conservative (-10%)", 0.90),
        ("Base Case Forecast", 1.00),
        ("Stretch Target (+10%)", 1.10),
    ]

    rows = []
    for label, mult in scenarios:
        s_launch = min(st.session_state.gp_input * 0.90, results["launch_gross"] * mult)
        s_curtain = min(st.session_state.gp_input * 0.97, results["curtain_up_gross"] * mult)
        s_final = min(st.session_state.gp_input * 0.99, results["final_gross"] * mult)

        rows.append(
            {
                "Scenario": label,
                "On-Sale Opening (W1 £)": f"£{s_launch:,.0f}",
                "On-Sale (% GP)": f"{(s_launch / st.session_state.gp_input) * 100:.1f}%",
                "Curtain-Up Advance (£)": f"£{s_curtain:,.0f}",
                "Curtain-Up (% GP)": f"{(s_curtain / st.session_state.gp_input) * 100:.1f}%",
                "Final Settlement (£)": f"£{s_final:,.0f}",
                "Final (% GP)": f"{(s_final / st.session_state.gp_input) * 100:.1f}%",
                "Est. Avg Seats / Perf": f"{int(round((results['final_tickets'] * mult) / max(st.session_state.run_perfs, 1))):,}",
            }
        )

    df_scenarios = pd.DataFrame(rows)
    st.dataframe(df_scenarios, use_container_width=True, hide_index=True)

with tab_export:
    st.markdown("#### Download Forecast Record for Programming & Finance Packs")
    export_df = pd.DataFrame(
        [
            {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Venue": st.session_state.venue_choice,
                "Show_Title": st.session_state.show_title,
                "Archetype": st.session_state.archetype,
                "Brand_Tier": st.session_state.brand_tier,
                "Gross_Potential_GBP": round(st.session_state.gp_input, 2),
                "Performances": st.session_state.run_perfs,
                "Total_Sellable_Seats": effective_total_seats,
                "Sales_Period_Weeks": st.session_state.sales_weeks,
                "Base_ATP_GBP": round(results["base_atp"], 2),
                "Est_OnSale_Week1_GBP": round(results["launch_gross"], 2),
                "Est_OnSale_Week1_Pct_GP": round(results["launch_gp_pct"] * 100, 1),
                "Est_OnSale_Week1_Tickets": results["launch_tickets"],
                "Est_CurtainUp_Advance_GBP": round(results["curtain_up_gross"], 2),
                "Est_CurtainUp_Advance_Pct_GP": round(results["curtain_up_gp_pct"] * 100, 1),
                "Est_Final_Settlement_GBP": round(results["final_gross"], 2),
                "Est_Final_Settlement_Pct_GP": round(results["final_gp_pct"] * 100, 1),
                "Est_Final_Tickets": results["final_tickets"],
            }
        ]
    )
    st.dataframe(export_df, use_container_width=True, hide_index=True)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Forecast Row as CSV",
        data=csv_bytes,
        file_name=f"bradford_initial_forecast_{st.session_state.show_title.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

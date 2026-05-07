"""
Multimarker Calculator — CURES Recalibration (Streamlit).
Run: streamlit run model3_risk_calculator.py
"""

from __future__ import annotations

import math
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import streamlit as st

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

APP_TITLE = "Multimarker Calculator — CURES Recalibration"
APP_SUBTITLE = "30-Day Mortality Risk Stratification & Attribution in Pulmonary Embolism"

COEF: dict[str, float] = {
    "Intercept": -4.9020,
    "age": 1.0985,
    "cancer": 1.0390,
    "CPD": 0.3955,
    "heart_rate": 0.3229,
    "SBP": 0.5214,
    "SAT": 0.2534,
    "BNP_fp": 0.5338,
    "BNP_fm": 0.8922,
    "cTN_p": 0.1931,
    "cTN_m": 0.0842,
    "dvt": 0.5415,
    "dvt_m": 0.4241,
}

FEATURE_ORDER = [
    "age", "cancer", "CPD", "heart_rate", "SBP", "SAT",
    "BNP_fp", "BNP_fm", "cTN_p", "cTN_m", "dvt", "dvt_m",
]

CLINICAL_KEYS = ("age", "cancer", "CPD", "heart_rate", "SBP", "SAT", "BNP_fp", "cTN_p", "dvt")
MISSING_KEYS = frozenset({"BNP_fm", "cTN_m", "dvt_m"})

DEFAULT_REFERENCE: dict[str, float] = {k: 0.0 for k in FEATURE_ORDER}

FEATURE_LABELS_UI: dict[str, str] = {
    "age": "Age > 80 years",
    "cancer": "History of cancer",
    "CPD": "History of chronic cardiopulmonary disease",
    "heart_rate": "Heart rate ≥ 110 beats/min",
    "SBP": "Systolic blood pressure < 100 mmHg",
    "SAT": "Arterial oxyhemoglobin saturation < 90 %",
    "BNP_fp": "Brain natriuretic peptide > 100 pg/mL",
    "BNP_fm": "Missing indicator: BNP (laboratory not available)",
    "cTN_p": "Cardiac troponin > 0.05 ng/mL",
    "cTN_m": "Missing indicator: cardiac troponin",
    "dvt": "Deep vein thrombosis by CCUS",
    "dvt_m": "Missing indicator: CCUS/DVT information",
}

RISK_TIER_TXT = {
    "Low": "Predicted 30-day probability < 1%",
    "Intermediate-Low": "Predicted probability between 1% and 10%",
    "Intermediate-High": "Predicted probability > 10%",
}

TIER_PHRASE = {
    "Low": "**low-risk** (<1% predicted 30-day mortality)",
    "Intermediate-Low": "**intermediate-low-risk** (1–10% predicted 30-day mortality)",
    "Intermediate-High": "**intermediate-high-risk** (>10% predicted 30-day mortality)",
}

# (accent / left border, soft background) — background hue matches each accent
STRAT_STYLE = {
    "Low": ("#154E82", "#E8F1F8"),
    "Intermediate-Low": ("#E0A829", "#FCF5E5"),
    "Intermediate-High": ("#9A2225", "#F7E8E7"),
}

WF_PREFIX = "wf__"


class PatientFeatures:
    def __init__(
        self,
        age: int,
        cancer: int,
        CPD: int,
        heart_rate: int,
        SBP: int,
        SAT: int,
        BNP_fp: float,
        BNP_fm: int,
        cTN_p: float,
        cTN_m: int,
        dvt: float,
        dvt_m: int,
    ) -> None:
        (
            self.age,
            self.cancer,
            self.CPD,
            self.heart_rate,
            self.SBP,
            self.SAT,
        ) = (
            age,
            cancer,
            CPD,
            heart_rate,
            SBP,
            SAT,
        )
        self.BNP_fp = BNP_fp
        self.BNP_fm = BNP_fm
        self.cTN_p = cTN_p
        self.cTN_m = cTN_m
        self.dvt = dvt
        self.dvt_m = dvt_m

    @classmethod
    def from_inputs(
        cls,
        age: int,
        cancer: int,
        CPD: int,
        heart_rate: int,
        SBP: int,
        SAT: int,
        *,
        bnp_missing: bool,
        bnp_value: int,
        ctn_missing: bool,
        ctn_value: int,
        dvt_missing: bool,
        dvt_value: int,
    ) -> "PatientFeatures":
        if bnp_missing:
            BNP_fp, BNP_fm = 0.0, 1
        else:
            BNP_fp, BNP_fm = float(int(bnp_value)), 0
        if ctn_missing:
            cTN_p, cTN_m = 0.0, 1
        else:
            cTN_p, cTN_m = float(int(ctn_value)), 0
        if dvt_missing:
            dvt, dvt_m = 0.0, 1
        else:
            dvt, dvt_m = float(int(dvt_value)), 0
        return cls(
            int(age),
            int(cancer),
            int(CPD),
            int(heart_rate),
            int(SBP),
            int(SAT),
            BNP_fp,
            BNP_fm,
            cTN_p,
            cTN_m,
            dvt,
            dvt_m,
        )


def logit(pf: PatientFeatures) -> float:
    return (
        COEF["Intercept"]
        + COEF["age"] * pf.age
        + COEF["cancer"] * pf.cancer
        + COEF["CPD"] * pf.CPD
        + COEF["heart_rate"] * pf.heart_rate
        + COEF["SBP"] * pf.SBP
        + COEF["SAT"] * pf.SAT
        + COEF["BNP_fp"] * pf.BNP_fp
        + COEF["BNP_fm"] * pf.BNP_fm
        + COEF["cTN_p"] * pf.cTN_p
        + COEF["cTN_m"] * pf.cTN_m
        + COEF["dvt"] * pf.dvt
        + COEF["dvt_m"] * pf.dvt_m
    )


def absolute_risk(pf: PatientFeatures) -> float:
    z = logit(pf)
    return 1.0 / (1.0 + math.exp(-z))


def risk_stratum(p: float) -> tuple[str, str]:
    if p < 0.01:
        return "Low", RISK_TIER_TXT["Low"]
    if p <= 0.1:
        return "Intermediate-Low", RISK_TIER_TXT["Intermediate-Low"]
    return "Intermediate-High", RISK_TIER_TXT["Intermediate-High"]


def _x_vector(pf: PatientFeatures) -> np.ndarray:
    return np.array(
        [
            pf.age,
            pf.cancer,
            pf.CPD,
            pf.heart_rate,
            pf.SBP,
            pf.SAT,
            pf.BNP_fp,
            pf.BNP_fm,
            pf.cTN_p,
            pf.cTN_m,
            pf.dvt,
            pf.dvt_m,
        ],
        dtype=float,
    )


def is_pure_reference_profile(pf: PatientFeatures) -> bool:
    return bool(np.max(np.abs(_x_vector(pf))) < 1e-12)


def linear_contribs(pf: PatientFeatures) -> list[tuple[str, float, float, float]]:
    beta = np.array([COEF[k] for k in FEATURE_ORDER], dtype=float)
    x = _x_vector(pf)
    ref = np.zeros_like(x)
    d = x - ref
    sh = beta * d
    return [(FEATURE_ORDER[i], float(beta[i]), float(x[i]), float(sh[i])) for i in range(len(FEATURE_ORDER))]


def logit_ref() -> float:
    return float(COEF["Intercept"])


def label_ui(k: str) -> str:
    return FEATURE_LABELS_UI.get(k, k)


def narrative_text(pf: PatientFeatures, p: float, tier: str, contribs: list[tuple[str, float, float, float]]) -> str:
    strat = TIER_PHRASE.get(tier, tier)
    lines = [
        f"This patient's **estimated 30-day mortality risk is {p * 100:.2f}%**, classified as {strat}."
    ]
    if is_pure_reference_profile(pf):
        lines.append(
            "Compared to the **baseline reference** (all listed factors at the negative reference level; "
            "no missing-data flags), this profile matches that reference and **shows no factor-specific attribution** "
            "beyond the intercept."
        )
        return "\n\n".join(lines)

    rows = [c for c in contribs if abs(c[3]) > 1e-9]
    rows.sort(key=lambda t: abs(t[3]), reverse=True)
    if not rows:
        lines.append(
            "Versus the same baseline, **attributable contributions are negligible** on the logit scale."
        )
        return "\n\n".join(lines)

    clin = [c for c in rows if c[0] not in MISSING_KEYS][:8]
    mis = [c for c in rows if c[0] in MISSING_KEYS][:8]
    lines.append(
        "**Attribution vs baseline** (all risk markers absent at reference; missing indicators = 0):"
    )
    if clin:
        parts = [
            f"{label_ui(n)}: **{c:+.3f}** on logit (marker = {int(round(xv))})"
            for n, _b, xv, c in clin
        ]
        lines.append("**Clinical / laboratory:** " + "; ".join(parts) + ".")
    if mis:
        parts = [
            f"{label_ui(n)}: **{c:+.3f}** on logit (indicator = {int(round(xv))})"
            for n, _b, xv, c in mis
        ]
        lines.append("**Missing-data indicators:** " + "; ".join(parts) + ".")
    lines.append(
        "These are additive components on the **logit** scale; **not a substitute** for comprehensive clinical judgment."
    )
    return "\n\n".join(lines)


def color_clinical(v: float) -> str:
    return STRAT_STYLE["Intermediate-High"][0] if v > 0 else STRAT_STYLE["Low"][0]


def build_shap_figure(contribs: list[tuple[str, float, float, float]]) -> plt.Figure:
    """Clinical / laboratory features only (missing-data indicator covariates omitted from the chart)."""
    byk = {c[0]: c for c in contribs}
    clin = [byk[k] for k in CLINICAL_KEYS]

    lab = [textwrap.fill(label_ui(t[0]), width=36) for t in clin]
    vals = [t[3] for t in clin]
    y = np.arange(len(clin))

    vmax = max(abs(v) for v in vals + [1e-9])
    rng = max(vmax * 1.12, 5e-3)

    n1 = len(clin)
    h = max(5.5, n1 * 0.42 + 2.5)
    fig, ax = plt.subplots(figsize=(8.8, h), dpi=120)

    for yi, vt in enumerate(clin):
        v = vt[3]
        if abs(v) <= 1e-12:
            ax.plot(
                [0],
                [yi],
                marker="|",
                ms=13,
                mew=2.8,
                color="#95a5a6",
                linestyle="none",
                clip_on=False,
            )
            continue
        c = color_clinical(v)
        ax.barh(yi, v, height=0.72, left=0.0, color=c, edgecolor="white", linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=8)
    ax.axvline(0, color="0.35", lw=0.9)
    ax.set_xlim(-rng, rng)
    ax.set_xlabel("Contribution to logit (vs all-zero reference)")
    ax.set_title("Clinical / laboratory markers", fontsize=11)
    ax.invert_yaxis()

    plt.tight_layout()
    return fig


def mortality_to_three_third_axis(p: float) -> float:
    """
    Map mortality probability [0,1] to gauge inner scale [0, 100].
    Physically equally spaced thirds correspond to probability bands:
      [0, 1%), [1%, 10%], (10%, 100%].
    """
    x = float(max(0.0, min(1.0, p)))
    span = 100.0 / 3.0
    if x <= 0.01:
        return span * (x / 0.01)
    if x <= 0.1:
        return span + span * ((x - 0.01) / 0.09)
    return 2.0 * span + span * ((x - 0.1) / 0.9)


def build_mortality_gradient_bar(p: float) -> plt.Figure:
    """
    Horizontal strip: three equal thirds (0–1%, 1–10%, 10–100% bands), each third a flat primary color;
    segment past the needle is muted for an “unfilled” track.
    """
    gv = mortality_to_three_third_axis(p)
    t1 = 100.0 / 3.0
    t2 = 200.0 / 3.0

    rgb_low = np.array(to_rgb(STRAT_STYLE["Low"][0]))
    rgb_lm = np.array(to_rgb(STRAT_STYLE["Intermediate-Low"][0]))
    rgb_mh = np.array(to_rgb(STRAT_STYLE["Intermediate-High"][0]))

    nw = 420
    s_axis = np.linspace(0.0, 100.0, nw)
    zm = np.zeros((1, nw, 4), dtype=float)
    for j, s_pos in enumerate(s_axis):
        if s_pos <= t1:
            zm[0, j, :3] = rgb_low
        elif s_pos <= t2:
            zm[0, j, :3] = rgb_lm
        else:
            zm[0, j, :3] = rgb_mh
        zm[0, j, 3] = 1.0

    mute = np.array(to_rgb("#ecf0f1"))
    for j, s_pos in enumerate(s_axis):
        if s_pos <= gv:
            continue
        mix = np.array(zm[0, j, :3]) * 0.38 + mute * 0.62
        zm[0, j, :3] = mix
        zm[0, j, 3] = 1.0

    fig, ax = plt.subplots(figsize=(8.6, 1.72), dpi=120)
    ax.imshow(zm, aspect="auto", extent=[0, 100.0, 0.0, 1.0], interpolation="nearest", origin="lower", zorder=1)
    ax.axvline(gv, color="#1a1a1a", lw=3.8, linestyle="-", alpha=1.0, zorder=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.85)
        spine.set_edgecolor("#000000")

    ax.set_yticks([])
    ax.set_xlim(0.0, 100.0)
    ax.tick_params(axis="x", labelsize=9, colors="#2c3e50")
    ax.set_xticks([0.0, t1, t2, 100.0])
    ax.set_xticklabels(["0", "1", "10", "100"])
    ax.set_title("Predicted 30-day mortality (%) ", fontsize=13, pad=10)
    fig.patch.set_facecolor("white")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def yn(name: str, key: str) -> int:
    v = st.radio(name, ("No", "Yes"), horizontal=True, key=WF_PREFIX + key)
    return 1 if v == "Yes" else 0


def clear_form_state() -> None:
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(WF_PREFIX):
            del st.session_state[k]


def run_app() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")

    st.markdown(f"### {APP_TITLE}")
    st.markdown(
        f'<p style="font-size:16.5px;color:#2c3e50;margin-top:-6px;margin-bottom:4px">{APP_SUBTITLE}</p>',
        unsafe_allow_html=True,
    )

    with st.expander("Help", expanded=False):
        st.markdown(
            """Model source: Recalibration of the Multimarker Calculator

Intended population: East Asian pulmonary embolism patients

Version: 1.1.0"""
        )

    left, right = st.columns([1.02, 1], gap="large")

    with left:
        st.markdown("###### Patient Characteristics")
        bt1, _ = st.columns([1, 5])
        with bt1:
            if st.button("Reset", use_container_width=True):
                clear_form_state()
                st.rerun()

        age = yn("Age > 80 years", "age")
        cancer = yn("History of cancer", "cancer")
        cpd = yn("History of chronic cardiopulmonary disease", "cpd")
        hr = yn("Heart rate ≥ 110 beats/min", "hr")
        sbp = yn("Systolic blood pressure < 100 mmHg", "sbp")
        sat = yn("Arterial oxyhemoglobin saturation < 90 %", "sat")

        st.markdown("**Brain natriuretic peptide > 100 pg/mL**")
        bnp_miss = st.checkbox("Missing", key=WF_PREFIX + "bnp_miss")
        bnp_obs = 0 if bnp_miss else yn("Above threshold (Yes) / not (No)", "bnp_obs")

        st.markdown("**Cardiac troponin > 0.05 ng/mL**")
        ctn_miss = st.checkbox("Missing", key=WF_PREFIX + "ctn_miss")
        ctn_obs = 0 if ctn_miss else yn("Above threshold (Yes) / not (No)", "ctn_obs")

        st.markdown("**Deep vein thrombosis by CCUS**")
        dvt_miss = st.checkbox("Missing", key=WF_PREFIX + "dvt_miss")
        dvt_obs = 0 if dvt_miss else yn("Present / positive per definition", "dvt_obs")

        pf = PatientFeatures.from_inputs(
            age,
            cancer,
            cpd,
            hr,
            sbp,
            sat,
            bnp_missing=bnp_miss,
            bnp_value=bnp_obs,
            ctn_missing=ctn_miss,
            ctn_value=ctn_obs,
            dvt_missing=dvt_miss,
            dvt_value=dvt_obs,
        )

    p = absolute_risk(pf)
    tier, tier_desc = risk_stratum(p)
    ctr = linear_contribs(pf)
    story = narrative_text(pf, p, tier, ctr)

    bar_fig = build_mortality_gradient_bar(p)

    with right:
        st.markdown("###### Predicted 30-Day Mortality Risk")
        st.metric("Probability", f"{p * 100:.2f}%", help="Inverse logit of the linear predictor")
        st.markdown(
            '<span style="font-size:11.5px;color:#7f8c8d">This prediction is for reference only; clinical decisions must incorporate full patient evaluation.</span>',
            unsafe_allow_html=True,
        )
        st.pyplot(bar_fig, use_container_width=True)
        plt.close(bar_fig)

        st.divider()
        st.markdown("###### Risk Stratification")
        border, bg = STRAT_STYLE[tier]
        st.markdown(
            f'<div style="background:{bg};border-left:6px solid {border};padding:12px 14px;border-radius:6px;">'
            f"<strong style='font-size:18px;color:{border}'>{tier}</strong><br/>"
            f"<span style='font-size:13px;color:#2c3e50'>{tier_desc}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Low <1%  |  Intermediate-Low 1–10% (incl. 10%)  |  Intermediate-High >10%"
        )

        st.divider()
        st.markdown("###### Risk Attribution (SHAP)")
        st.caption("Contribution vs reference for clinical markers (missing-data indicators not shown)")

        fig_shap = build_shap_figure(ctr)
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)

        st.divider()
        st.markdown("###### Risk interpretation")
        st.markdown(story)

        with st.expander("Decomposition table", expanded=False):
            tab = []
            for n, _b, xv, s in ctr:
                tab.append(
                    {
                        "feature": label_ui(n),
                        "value": int(round(xv)),
                        "beta": COEF[n],
                        "contrib_logit": s,
                    }
                )
            st.dataframe(tab, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    run_app()

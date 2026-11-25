from datetime import datetime, timezone
import time
from google import genai
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="MEVO Trójmiasto – Dostępność Rowerów i Agent AI",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Spokojny, elegancki granat bez neonów
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0F172A;
            color: #F1F5F9;
        }
        section[data-testid="stSidebar"] {
            background-color: #162033;
            border-right: 1px solid #283852;
        }
        div[data-testid="stMetric"] {
            background-color: #1E293B;
            border: 1px solid #334155;
            padding: 16px 20px;
            border-radius: 12px;
        }
        div[data-testid="stMetric"] label {
            color: #CBD5E1 !important;
            font-size: 0.84rem !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-size: 1.75rem !important;
            font-weight: 700 !important;
        }
        .lead-box {
            background-color: #1E293B;
            border-left: 4px solid #60A5FA;
            border-top: 1px solid #334155;
            border-right: 1px solid #334155;
            border-bottom: 1px solid #334155;
            padding: 14px 18px;
            border-radius: 0 10px 10px 0;
            margin-bottom: 16px;
            color: #F1F5F9;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

NAVY_DARK = "#0F172A"
CARD_BG = "#1E293B"
PLOT_BG = "#162033"
GRID_COLOR = "#283852"

BLUE_SOFT = "#60A5FA"
TEAL_SOFT = "#5EEAD4"
AMBER_SOFT = "#FBBF24"
RED_SOFT = "#F87171"
SLATE_BAR = "#475569"

STATUS_MAP = {
    "Dostępne (1–79%)": TEAL_SOFT,
    "Dużo rowerów (≥80%)": BLUE_SOFT,
    "Brak rowerów (0 szt.)": RED_SOFT,
}


def assign_city(row: pd.Series) -> str:
    lat, lon = row["lat"], row["lon"]
    if 54.27 <= lat <= 54.425 and 18.45 <= lon <= 18.95:
        return "Gdańsk"
    if 54.425 < lat <= 54.475 and 18.50 <= lon <= 18.60:
        return "Sopot"
    if 54.46 <= lat <= 54.58 and 18.38 <= lon <= 18.58:
        return "Gdynia"
    if lat < 54.27 and lon > 18.55:
        return "Pruszcz Gdański / Tczew"
    if lat > 54.55 and lon <= 18.45:
        return "Rumia / Reda / Wejherowo"
    return "Pozostałe gminy"


def assign_zone_type(name: str) -> str:
    label = str(name).lower()
    commuter_keys = ["skm", "pkm", "dworzec", "główny", "wrzeszcz", "oliwa", "politechnika", "uniwersytet", "alchemia", "olivia", "centrum"]
    leisure_keys = ["plaża", "molo", "brzeźno", "jelitkowo", "orłowo", "bulwar", "park", "stogi", "sobieszewo", "nadmorski", "aleja"]
    if any(k in label for k in commuter_keys):
        return "Węzły i uczelnie"
    if any(k in label for k in leisure_keys):
        return "Wybrzeże i rekreacja"
    return "Osiedla / lokalne"


def assign_availability_status(row: pd.Series) -> str:
    if row["num_bikes_available"] == 0:
        return "Brak rowerów (0 szt.)"
    if row["wypelnienie_pct"] >= 80:
        return "Dużo rowerów (≥80%)"
    return "Dostępne (1–79%)"


def assign_battery_bucket(km: float) -> str:
    if km < 20:
        return "Do 20 km"
    if km < 50:
        return "20–50 km"
    return "Powyżej 50 km"


def style_chart(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
        plot_bgcolor=PLOT_BG,
        font={"color": "#F1F5F9", "family": "sans-serif"},
        height=height,
        margin={"t": 20, "b": 24, "l": 12, "r": 12},
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


def build_fallback_answer(
    stations: pd.DataFrame,
    filtered: pd.DataFrame,
    ebikes: pd.DataFrame,
    avg_ebike_km: str,
    timestamp: str,
    err: Exception | None = None,
) -> str:
    best_city = stations.groupby("Miasto")["num_bikes_available"].sum().idxmax()
    best_city_cnt = int(stations.groupby("Miasto")["num_bikes_available"].sum().max())
    top3 = ", ".join(filtered.sort_values("num_bikes_available", ascending=False).head(3)["name"].tolist())
    long_range = int((ebikes["zasieg_km"] >= 50).sum()) if len(ebikes) > 0 else 0

    msg = (
        f"**Podsumowanie z danych na żywo ({timestamp}):**\n\n"
        f"- Najwięcej rowerów na stacjach jest obecnie w obszarze **{best_city}** ({best_city_cnt} szt.).\n"
        f"- Najlepsze stacje pod względem liczby rowerów w wybranym filtrze: **{top3}**.\n"
        f"- E-bike'i z zasięgiem ≥ 50 km: **{long_range}** szt. (średni zasięg: **{avg_ebike_km}**)."
    )
    if err is not None:
        msg += f"\n\n*(Użyto podsumowania lokalnego — chwilowy błąd API: {err})*"
    return msg


@st.cache_data(ttl=60)
def load_mevo_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    gbfs_index = requests.get("https://gbfs.urbansharing.com/rowermevo.pl/gbfs.json", timeout=10).json()
    feeds = {f["name"]: f["url"] for f in gbfs_index["data"]["pl"]["feeds"]}

    stations_info = pd.DataFrame(requests.get(feeds["station_information"], timeout=10).json()["data"]["stations"])
    stations_status = pd.DataFrame(requests.get(feeds["station_status"], timeout=10).json()["data"]["stations"])
    free_bikes = pd.DataFrame(requests.get(feeds["free_bike_status"], timeout=10).json()["data"]["bikes"])

    stations = pd.merge(stations_info, stations_status, on="station_id", how="inner")
    stations["wypelnienie_pct"] = np.where(
        stations["capacity"] > 0,
        (stations["num_bikes_available"] / stations["capacity"] * 100).round(1),
        0.0,
    )
    stations["Miasto"] = stations.apply(assign_city, axis=1)
    stations["Typ_Lokalizacji"] = stations["name"].apply(assign_zone_type)
    stations["Status"] = stations.apply(assign_availability_status, axis=1)

    now_ts = datetime.now(timezone.utc).timestamp()
    if "last_reported" in stations.columns:
        stations["opoznienie_min"] = ((now_ts - stations["last_reported"]) / 60).clip(lower=0).round(1)
    else:
        stations["opoznienie_min"] = np.nan

    bikes = free_bikes[
        free_bikes["lat"].between(53.8, 55.0) & free_bikes["lon"].between(17.5, 19.5)
    ].copy()
    bikes["zasieg_km"] = (bikes["current_range_meters"] / 1000).fillna(0).round(1) if "current_range_meters" in bikes.columns else 0.0
    bikes["Typ"] = bikes["vehicle_type_id"].map({"ebike": "E-bike", "bike": "Klasyczny"}).fillna("Inny") if "vehicle_type_id" in bikes.columns else "Inny"

    meta = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feeds_count": len(feeds),
        "raw_bikes_count": len(free_bikes),
        "gps_outliers_removed": int(len(free_bikes) - len(bikes)),
        "free_floating_count": int(len(bikes)),
    }
    return stations, bikes, meta


stations_df, bikes_df, meta = load_mevo_data()

# Panel boczny
st.sidebar.markdown("### Filtry")
cities = sorted(stations_df["Miasto"].unique())
default_cities = [c for c in ["Gdańsk", "Gdynia", "Sopot"] if c in cities] or cities
selected_cities = st.sidebar.multiselect("Obszar / gminy:", options=cities, default=default_cities)

st.sidebar.markdown("### Rower elektryczny")
min_km = st.sidebar.slider("Minimalny zasięg trasy (km):", 5, 100, 30, step=5)
only_ready = st.sidebar.checkbox("Tylko rowery gotowe do jazdy", value=True)

filtered_stations = stations_df[stations_df["Miasto"].isin(selected_cities)].copy() if selected_cities else stations_df.copy()

filtered_bikes = bikes_df.copy()
if only_ready and {"is_disabled", "is_reserved"}.issubset(filtered_bikes.columns):
    filtered_bikes = filtered_bikes[(~filtered_bikes["is_disabled"]) & (~filtered_bikes["is_reserved"])]

ebikes = filtered_bikes[filtered_bikes["vehicle_type_id"] == "ebike"].copy() if "vehicle_type_id" in filtered_bikes.columns else pd.DataFrame(columns=["zasieg_km"])
ebikes_ready = ebikes[ebikes["zasieg_km"] >= min_km]

# Nagłówek i 4 główne KPI
st.markdown("## MEVO Trójmiasto — Dostępność i Stan Floty")
st.caption(f"Otwarte dane GBFS ({meta['feeds_count']} źródeł) • Odświeżono: {meta['updated_at']}")

total_bikes = int(filtered_stations["num_bikes_available"].sum())
total_docks = int(filtered_stations["num_docks_available"].sum()) if "num_docks_available" in filtered_stations.columns else int(filtered_stations["capacity"].sum() - total_bikes)
total_cap = int(filtered_stations["capacity"].sum())
fill_pct = round((total_bikes / total_cap * 100), 1) if total_cap > 0 else 0.0
active_stations = int((filtered_stations["num_bikes_available"] > 0).sum())
empty_stations = int((filtered_stations["num_bikes_available"] == 0).sum())
avg_km_str = f"{ebikes['zasieg_km'].mean():.1f} km" if len(ebikes) > 0 else "0.0 km"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Dostępne rowery (stacje)", f"{total_bikes:,}".replace(",", " "), f"Wolnych miejsc: {total_docks:,}".replace(",", " "))
k2.metric("Stacje z rowerami", f"{active_stations} / {len(filtered_stations)}", f"Pustych: {empty_stations}", delta_color="inverse")
k3.metric("Średnie wypełnienie", f"{fill_pct}%", f"Pojemność: {total_cap:,}".replace(",", " "))
k4.metric(f"E-bike ≥ {min_km} km", f"{len(ebikes_ready):,}".replace(",", " "), f"Śr. zasięg: {avg_km_str}")

st.write("")

tab1, tab2, tab3 = st.tabs(["Dostępność i Stacje", "Flota i Baterie", "Asystent AI"])

# --- ZAKŁADKA 1: DOSTĘPNOŚĆ I STACJE ---
with tab1:
    if len(filtered_stations) > 0:
        top_st = filtered_stations.sort_values("num_bikes_available", ascending=False).iloc[0]
        median_lag = filtered_stations["opoznienie_min"].median()
        lag_txt = f" • Mediana opóźnienia sygnału stacji: {median_lag:.1f} min" if pd.notna(median_lag) else ""
        st.markdown(
            f"<div class='lead-box'>Najwięcej rowerów stoi obecnie na stacji <b>{top_st['name']}</b> "
            f"({int(top_st['num_bikes_available'])} szt.). W wybranym obszarze <b>{empty_stations}</b> stacji ma 0 rowerów{lag_txt}.</div>",
            unsafe_allow_html=True,
        )

    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.markdown("#### TOP 10 stacji — liczba dostępnych rowerów")
        top10 = filtered_stations.sort_values("num_bikes_available", ascending=False).head(10)
        fig_top = px.bar(
            top10.iloc[::-1],
            x="num_bikes_available",
            y="name",
            orientation="h",
            text="num_bikes_available",
            color_discrete_sequence=[BLUE_SOFT],
            labels={"num_bikes_available": "Dostępne rowery", "name": ""},
        )
        fig_top.update_traces(textposition="outside", marker_color=BLUE_SOFT)
        style_chart(fig_top, height=380)
        st.plotly_chart(fig_top, use_container_width=True)

    with c_right:
        st.markdown("#### Rowery vs wolne miejsca na oddanie (wg obszaru)")
        city_df = (
            stations_df.groupby("Miasto", as_index=False)
            .agg(
                Rowery=("num_bikes_available", "sum"),
                Wolne_miejsca=("num_docks_available", "sum") if "num_docks_available" in stations_df.columns else ("capacity", lambda s: 0),
            )
            .sort_values("Rowery", ascending=True)
        )
        fig_city = go.Figure()
        fig_city.add_trace(go.Bar(y=city_df["Miasto"], x=city_df["Rowery"], name="Dostępne rowery", orientation="h", marker_color=BLUE_SOFT))
        fig_city.add_trace(go.Bar(y=city_df["Miasto"], x=city_df["Wolne_miejsca"], name="Wolne miejsca", orientation="h", marker_color=SLATE_BAR))
        style_chart(fig_city, height=380)
        fig_city.update_layout(barmode="stack", legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1})
        st.plotly_chart(fig_city, use_container_width=True)

    g_left, g_right = st.columns([1.3, 1])

    with g_left:
        st.markdown("#### Rozmieszczenie stacji (współrzędne GPS)")
        fig_geo = px.scatter(
            filtered_stations,
            x="lon",
            y="lat",
            size="num_bikes_available",
            color="Status",
            hover_name="name",
            hover_data={"Miasto": True, "num_bikes_available": True, "capacity": True, "lon": False, "lat": False},
            color_discrete_map=STATUS_MAP,
            labels={"lon": "", "lat": ""},
        )
        style_chart(fig_geo, height=400)
        fig_geo.update_layout(
            xaxis={"showticklabels": False},
            yaxis={"showticklabels": False, "scaleanchor": "x", "scaleratio": 1.6},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    with g_right:
        st.markdown("#### Podział rowerów wg charakteru okolicy")
        zone_df = (
            filtered_stations.groupby("Typ_Lokalizacji", as_index=False)
            .agg(Rowery=("num_bikes_available", "sum"), Stacje=("station_id", "count"))
        )
        fig_zone = px.pie(zone_df, values="Rowery", names="Typ_Lokalizacji", hole=0.55, color_discrete_sequence=[BLUE_SOFT, TEAL_SOFT, AMBER_SOFT])
        fig_zone.update_traces(textposition="inside", textinfo="percent+label")
        style_chart(fig_zone, height=250)
        fig_zone.update_layout(showlegend=False, margin={"t": 8, "b": 8, "l": 8, "r": 8})
        st.plotly_chart(fig_zone, use_container_width=True)

        st.dataframe(
            zone_df.rename(columns={"Typ_Lokalizacji": "Typ okolicy", "Rowery": "Dostępne rowery", "Stacje": "Stacje"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Lista stacji")
    mode = st.radio(
        "Widok tabeli:",
        ["Najwięcej rowerów", "Puste stacje (0)", "Wyszukiwarka"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode == "Najwięcej rowerów":
        tbl = filtered_stations.sort_values("num_bikes_available", ascending=False).head(15)
    elif mode == "Puste stacje (0)":
        tbl = filtered_stations[filtered_stations["num_bikes_available"] == 0].sort_values("capacity", ascending=False)
    else:
        q = st.text_input("Szukaj po nazwie stacji:", placeholder="np. Wrzeszcz, Molo, SKM")
        tbl = filtered_stations.sort_values("num_bikes_available", ascending=False)
        if q:
            tbl = tbl[tbl["name"].str.contains(q, case=False, na=False)]

    st.dataframe(
        tbl[["name", "Miasto", "Typ_Lokalizacji", "num_bikes_available", "capacity", "wypelnienie_pct", "opoznienie_min"]].rename(
            columns={
                "name": "Stacja",
                "Typ_Lokalizacji": "Okolica",
                "num_bikes_available": "Rowery",
                "capacity": "Pojemność",
                "wypelnienie_pct": "Wypełnienie (%)",
                "opoznienie_min": "Opóźnienie (min)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# --- ZAKŁADKA 2: FLOTA I BATERIE ---
with tab2:
    f1, f2 = st.columns(2)

    with f1:
        st.markdown("#### Rowery poza stacjami (free-floating) wg typu")
        if len(bikes_df) > 0:
            t_df = bikes_df.groupby("Typ", as_index=False).size().rename(columns={"size": "Liczba"})
            fig_t = px.pie(t_df, values="Liczba", names="Typ", hole=0.55, color_discrete_sequence=[TEAL_SOFT, BLUE_SOFT])
            fig_t.update_traces(textinfo="value+percent")
            style_chart(fig_t, height=300)
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("Brak rekordów w feedzie free_bike_status.")

    with f2:
        st.markdown("#### Jakość danych GPS (kontrola anomalii)")
        st.caption(
            f"Surowych rekordów z API: **{meta['raw_bikes_count']}** • "
            f"Odfiltrowano poza regionem: **{meta['gps_outliers_removed']}** • "
            f"Poprawnych w regionie: **{meta['free_floating_count']}**."
        )
        if {"is_disabled", "is_reserved"}.issubset(bikes_df.columns):
            disabled_cnt = int(bikes_df["is_disabled"].sum())
            reserved_cnt = int(bikes_df["is_reserved"].sum())
            ready_cnt = int(len(bikes_df) - disabled_cnt - reserved_cnt)
            st_df = pd.DataFrame({
                "Status": ["Gotowe", "Zarezerwowane", "Wyłączone"],
                "Liczba": [max(ready_cnt, 0), reserved_cnt, disabled_cnt],
            })
            fig_s = px.bar(st_df, x="Status", y="Liczba", text="Liczba", color_discrete_sequence=[TEAL_SOFT])
            fig_s.update_traces(textposition="outside", marker_color=TEAL_SOFT)
            style_chart(fig_s, height=250)
            st.plotly_chart(fig_s, use_container_width=True)

    st.markdown(f"#### Rozkład zasięgu baterii e-bike'ów (próg: ≥ {min_km} km)")
    if len(ebikes) > 0:
        ebikes["Koszyk"] = ebikes["zasieg_km"].apply(assign_battery_bucket)
        b1, b2 = st.columns([1.3, 1])

        with b1:
            fig_h = px.histogram(
                ebikes,
                x="zasieg_km",
                nbins=20,
                labels={"zasieg_km": "Szacowany zasięg (km)", "count": "Liczba e-bike'ów"},
            )
            fig_h.update_traces(marker_color=BLUE_SOFT)
            fig_h.add_vline(x=min_km, line_dash="dash", line_color=RED_SOFT, annotation_text=f"{min_km} km ({len(ebikes_ready)} szt.)")
            style_chart(fig_h, height=340)
            fig_h.update_layout(bargap=0.1, yaxis_title="Liczba rowerów")
            st.plotly_chart(fig_h, use_container_width=True)

        with b2:
            b_df = ebikes.groupby("Koszyk", as_index=False).size().rename(columns={"size": "Liczba"})
            fig_b = px.bar(b_df, x="Koszyk", y="Liczba", text="Liczba", labels={"Koszyk": "Przedział zasięgu", "Liczba": "Liczba rowerów"})
            fig_b.update_traces(textposition="outside", marker_color=AMBER_SOFT)
            style_chart(fig_b, height=340)
            st.plotly_chart(fig_b, use_container_width=True)
    else:
        st.info("Brak danych o zasięgu e-bike'ów dla bieżących filtrów.")

# --- ZAKŁADKA 3: ASYSTENT AI ---
with tab3:
    st.markdown("#### Zapytaj asystenta o bieżące dane")
    st.caption("Odpowiedź generowana na podstawie aktualnych statystyk ze stacji i floty.")

    top_txt = filtered_stations.sort_values("num_bikes_available", ascending=False).head(15)[
        ["name", "Miasto", "num_bikes_available", "capacity", "wypelnienie_pct"]
    ].to_string(index=False)

    city_txt = (
        stations_df.groupby("Miasto", as_index=False)
        .agg(Rowery=("num_bikes_available", "sum"), Pojemnosc=("capacity", "sum"), Puste=("num_bikes_available", lambda s: int((s == 0).sum())))
        .to_string(index=False)
    )

    presets = [
        "Gdzie jest teraz najwięcej rowerów na stacjach w wybranym obszarze?",
        "Jak wygląda sytuacja z bateriami e-bike'ów na dłuższą trasę (≥ 50 km)?",
        "Które obszary mają najwięcej pustych stacji?",
    ]
    chosen = st.selectbox("Przykładowe pytanie:", presets)
    question = st.text_input("Twoje pytanie:", value=chosen)

    if st.button("Wyślij pytanie", type="primary"):
        api_key = st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None
        if api_key:
            with st.spinner("Analizuję dane..."):
                client = genai.Client(api_key=api_key)
                prompt = (
                    f"Jesteś analitykiem danych systemu rowerowego MEVO. Odpowiedz krótko i rzeczowo po polsku "
                    f"wyłącznie na podstawie danych z {meta['updated_at']}:\n\n"
                    f"MIASTA:\n{city_txt}\n\nTOP 15 STACJI:\n{top_txt}\n\n"
                    f"E-BIKE: liczba={len(ebikes)}, średni zasięg={avg_km_str}, >=50 km={int((ebikes['zasieg_km'] >= 50).sum()) if len(ebikes) > 0 else 0}\n\n"
                    f"PYTANIE: {question}"
                )
                resp = None
                last_err = None
                for model_name in ["gemini-3.8-flash", "gemini-2.5-flash"]:
                    for _ in range(2):
                        try:
                            resp = client.models.generate_content(model=model_name, contents=prompt)
                            break
                        except Exception as err:
                            last_err = err
                            if "503" in str(err) or "UNAVAILABLE" in str(err):
                                time.sleep(1.2)
                            else:
                                break
                    if resp is not None:
                        break

                if resp is not None and getattr(resp, "text", None):
                    st.markdown(f"<div class='lead-box'>{resp.text}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div class='lead-box'>{build_fallback_answer(stations_df, filtered_stations, ebikes, avg_km_str, meta['updated_at'], last_err)}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                f"<div class='lead-box'>{build_fallback_answer(stations_df, filtered_stations, ebikes, avg_km_str, meta['updated_at'])}</div>",
                unsafe_allow_html=True,
            )

st.divider()
st.caption("Źródło: Otwarte Dane MEVO (GBFS) • Projekt analityczny.")

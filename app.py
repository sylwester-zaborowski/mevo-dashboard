from datetime import datetime
import time
from google import genai
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="MEVO Trójmiasto – Analiza Dostępności i Agent AI",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Granatowy motyw BI
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0B132B;
            color: #F1F5F9;
        }
        section[data-testid="stSidebar"] {
            background-color: #111D3A;
            border-right: 1px solid #26385E;
        }
        div[data-testid="stMetric"] {
            background-color: #1C2541;
            border: 1px solid #3A506B;
            padding: 16px 20px;
            border-radius: 12px;
        }
        div[data-testid="stMetric"] label {
            color: #94A3B8 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-size: 1.75rem !important;
            font-weight: 700 !important;
        }
        .summary-card {
            background-color: #1C2541;
            border-left: 4px solid #38BDF8;
            border-top: 1px solid #26385E;
            border-right: 1px solid #26385E;
            border-bottom: 1px solid #26385E;
            padding: 14px 18px;
            border-radius: 0 10px 10px 0;
            margin-bottom: 16px;
            color: #E2E8F0;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

NAVY_SEQ = ["#38BDF8", "#2DD4BF", "#60A5FA", "#FBBF24", "#FB7185"]
STATUS_COLORS = {
    "Dużo rowerów (≥80%)": "#38BDF8",
    "Dostępne (1–79%)": "#2DD4BF",
    "Brak rowerów (0 szt.)": "#FB7185",
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
        return "Węzły przesiadkowe i uczelnie"
    if any(k in label for k in leisure_keys):
        return "Strefa nadmorska i rekreacja"
    return "Dzielnice mieszkaniowe"


def assign_availability_status(row: pd.Series) -> str:
    if row["num_bikes_available"] == 0:
        return "Brak rowerów (0 szt.)"
    if row["wypelnienie_pct"] >= 80:
        return "Dużo rowerów (≥80%)"
    return "Dostępne (1–79%)"


def assign_battery_bucket(km: float) -> str:
    if km < 20:
        return "1. Krótka trasa (< 20 km)"
    if km < 50:
        return "2. Średni dystans (20–50 km)"
    return "3. Długa wycieczka (> 50 km)"


def apply_navy_theme(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1C2541",
        plot_bgcolor="#131C31",
        font={"color": "#E2E8F0"},
        height=height,
        margin={"t": 24, "b": 24, "l": 12, "r": 12},
    )
    return fig


def build_fallback_summary(
    stations: pd.DataFrame,
    filtered: pd.DataFrame,
    ebikes: pd.DataFrame,
    avg_ebike_km: str,
    timestamp: str,
    error_detail: Exception | None = None,
) -> str:
    best_city = stations.groupby("Miasto")["num_bikes_available"].sum().idxmax()
    best_city_cnt = int(stations.groupby("Miasto")["num_bikes_available"].sum().max())
    top3_names = ", ".join(filtered.sort_values("num_bikes_available", ascending=False).head(3)["name"].tolist())
    long_range_ebikes = int((ebikes["zasieg_km"] >= 50).sum()) if len(ebikes) > 0 else 0

    base = f"""🤖 **Podsumowanie Agenta (stan na {timestamp}):**

* **Najwięcej rowerów:** Liderem dostępności na stacjach jest obecnie **{best_city}** ({best_city_cnt} szt.).
* **TOP 3 stacje na start:** {top3_names}.
* **Baterie e-bike:** Dostępnych jest **{long_range_ebikes}** rowerów elektrycznych z zasięgiem powyżej 50 km (średni zasięg: **{avg_ebike_km}**)."""

    if error_detail is not None:
        return f"{base}\n\n*(Tryb zapasowy z powodu chwilowego błędu API: {error_detail})*"
    return f"{base}\n\n*(Dodaj `GEMINI_API_KEY` w zakładce Secrets na Streamlit Cloud, aby odblokować swobodne pytania do modelu).* "


@st.cache_data(ttl=60)
def load_mevo_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
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

    bikes = free_bikes[
        free_bikes["lat"].between(53.8, 55.0) & free_bikes["lon"].between(17.5, 19.5)
    ].copy()
    bikes["zasieg_km"] = (bikes["current_range_meters"] / 1000).fillna(0).round(1) if "current_range_meters" in bikes.columns else 0.0

    return stations, bikes, datetime.now().strftime("%Y-%m-%d %H:%M:%S")


stations_df, bikes_df, updated_at = load_mevo_data()

# Panel boczny
st.sidebar.title("🎛️ Filtry analizy")
available_cities = sorted(stations_df["Miasto"].unique())
default_cities = [c for c in ["Gdańsk", "Gdynia", "Sopot"] if c in available_cities] or available_cities
selected_cities = st.sidebar.multiselect("Wybierz miasta / obszar:", options=available_cities, default=default_cities)

st.sidebar.divider()
st.sidebar.subheader("🚴‍♂️ Filtr dla rowerzysty")
min_range = st.sidebar.slider("Jaki dystans chcesz przejechać (km)?", 5, 100, 30, step=5)
only_working = st.sidebar.checkbox("Pokaż tylko sprawne rowery", value=True)

filtered_stations = stations_df[stations_df["Miasto"].isin(selected_cities)].copy() if selected_cities else stations_df.copy()

filtered_bikes = bikes_df.copy()
if only_working and {"is_disabled", "is_reserved"}.issubset(filtered_bikes.columns):
    filtered_bikes = filtered_bikes[(~filtered_bikes["is_disabled"]) & (~filtered_bikes["is_reserved"])]

ebikes = filtered_bikes[filtered_bikes["vehicle_type_id"] == "ebike"].copy() if "vehicle_type_id" in filtered_bikes.columns else pd.DataFrame(columns=["zasieg_km"])
ebikes_ready = ebikes[ebikes["zasieg_km"] >= min_range]

# Nagłówek i KPI
st.title("🚲 MEVO Trójmiasto – Analiza Dostępności & Agent AI")
st.markdown(
    f"Analiza dostępności rowerów miejskich, obłożenia stacji, baterii oraz asystent AI oparty na otwartych danych MEVO. "
    f"• *Aktualizacja:* `{updated_at}`"
)

total_bikes_on_stations = int(filtered_stations["num_bikes_available"].sum())
total_capacity = int(filtered_stations["capacity"].sum())
avg_fill = round((total_bikes_on_stations / total_capacity * 100), 1) if total_capacity > 0 else 0.0
stations_with_bikes = int((filtered_stations["num_bikes_available"] > 0).sum())
empty_stations = int((filtered_stations["num_bikes_available"] == 0).sum())
avg_ebike_km = f"{ebikes['zasieg_km'].mean():.1f} km" if len(ebikes) > 0 else "0.0 km"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rowery na stacjach", f"{total_bikes_on_stations:,}".replace(",", " "), f"z {total_capacity:,} miejsc postojowych".replace(",", " "))
c2.metric("Stacje z dostępnym rowerem", f"{stations_with_bikes} z {len(filtered_stations)}", f"{empty_stations} stacji pustych", delta_color="inverse")
c3.metric("Średnie wypełnienie stacji", f"{avg_fill}%", "udział zajętych stojaków")
c4.metric(f"E-bike'i gotowe na ≥ {min_range} km", f"{len(ebikes_ready):,}".replace(",", " "), f"średni zasięg: {avg_ebike_km}")

st.write("")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dostępność i TOP Stacje",
    "📍 Rozkład w Trójmieście",
    "🔋 Bateria i Zasięg",
    "🤖 Asystent AI (Agent Danych)",
])

with tab1:
    if len(filtered_stations) > 0:
        best_station = filtered_stations.sort_values("num_bikes_available", ascending=False).iloc[0]
        st.markdown(
            f"""<div class="summary-card">
            <b>Wnioski z bieżących danych:</b> W wybranym obszarze dostępnych jest <b>{total_bikes_on_stations}</b> rowerów na <b>{stations_with_bikes}</b> stacjach.
            Najwięcej rowerów stoi na stacji <b>{best_station['name']}</b> ({int(best_station['num_bikes_available'])} szt.).
            Na <b>{empty_stations}</b> stacjach nie ma obecnie żadnego wolnego roweru.
            </div>""",
            unsafe_allow_html=True,
        )

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("🏆 TOP 10 stacji pod względem liczby rowerów")
        top10 = filtered_stations.sort_values("num_bikes_available", ascending=False).head(10)
        fig_top = px.bar(
            top10.iloc[::-1],
            x="num_bikes_available",
            y="name",
            orientation="h",
            text="num_bikes_available",
            color="wypelnienie_pct",
            color_continuous_scale=["#1D4ED8", "#38BDF8", "#2DD4BF"],
            labels={"num_bikes_available": "Liczba dostępnych rowerów", "name": "", "wypelnienie_pct": "Wypełnienie (%)"},
        )
        fig_top.update_traces(textposition="outside")
        apply_navy_theme(fig_top, height=430)
        fig_top.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_top, use_container_width=True)

    with col_right:
        st.subheader("🏙️ Porównanie miast (Rowery vs Wolne miejsca)")
        city_summary = (
            stations_df.groupby("Miasto", as_index=False)
            .agg(Dostepne=("num_bikes_available", "sum"), Wolne_Miejsca=("num_docks_available", "sum"))
            .sort_values("Dostepne", ascending=True)
        )
        fig_cities = go.Figure()
        fig_cities.add_trace(
            go.Bar(
                y=city_summary["Miasto"],
                x=city_summary["Dostepne"],
                name="Dostępne rowery",
                orientation="h",
                marker_color="#38BDF8",
                text=city_summary["Dostepne"],
                textposition="auto",
            )
        )
        fig_cities.add_trace(
            go.Bar(
                y=city_summary["Miasto"],
                x=city_summary["Wolne_Miejsca"],
                name="Wolne stojaki",
                orientation="h",
                marker_color="#2B3A5E",
            )
        )
        apply_navy_theme(fig_cities, height=430)
        fig_cities.update_layout(
            barmode="stack",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        st.plotly_chart(fig_cities, use_container_width=True)

    st.subheader("📋 Przegląd stacji")
    view_option = st.radio(
        "Pokaż stacje:",
        ["Najwięcej dostępnych rowerów", "Puste stacje (0 rowerów)", "Wszystkie stacje (wyszukiwarka)"],
        horizontal=True,
    )

    if view_option == "Najwięcej dostępnych rowerów":
        display_df = filtered_stations.sort_values("num_bikes_available", ascending=False).head(15)
    elif view_option == "Puste stacje (0 rowerów)":
        display_df = filtered_stations[filtered_stations["num_bikes_available"] == 0].sort_values("capacity", ascending=False)
    else:
        search_txt = st.text_input("Wpisz nazwę ulicy lub przystanku (np. Wrzeszcz, Oliwa, Molo):")
        display_df = filtered_stations.sort_values("num_bikes_available", ascending=False)
        if search_txt:
            display_df = display_df[display_df["name"].str.contains(search_txt, case=False, na=False)]

    st.dataframe(
        display_df[["name", "Miasto", "Typ_Lokalizacji", "num_bikes_available", "capacity", "wypelnienie_pct", "Status"]].rename(
            columns={
                "name": "Stacja",
                "Typ_Lokalizacji": "Typ okolicy",
                "num_bikes_available": "Dostępne rowery",
                "capacity": "Wszystkie stojaki",
                "wypelnienie_pct": "Wypełnienie (%)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    g1, g2 = st.columns([1.3, 1])

    with g1:
        st.subheader("📍 Rozkład przestrzenny stacji w Metropolii")
        fig_geo = px.scatter(
            filtered_stations,
            x="lon",
            y="lat",
            size="num_bikes_available",
            color="Status",
            hover_name="name",
            hover_data={"Miasto": True, "num_bikes_available": True, "capacity": True, "lon": False, "lat": False},
            color_discrete_map=STATUS_COLORS,
            labels={"lon": "Długość geograficzna", "lat": "Szerokość geograficzna"},
        )
        apply_navy_theme(fig_geo, height=460)
        fig_geo.update_layout(
            xaxis={"showgrid": True, "gridcolor": "#26385E", "showticklabels": False, "title": "Zachód ← → Wschód"},
            yaxis={"showgrid": True, "gridcolor": "#26385E", "showticklabels": False, "title": "Południe ← → Północ", "scaleanchor": "x", "scaleratio": 1.6},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    with g2:
        st.subheader("🏘️ Dostępność wg typu okolicy")
        zone_agg = (
            filtered_stations.groupby("Typ_Lokalizacji", as_index=False)
            .agg(Rowery=("num_bikes_available", "sum"), Stacje=("station_id", "count"), Srednio_na_stacje=("num_bikes_available", "mean"))
        )
        zone_agg["Srednio_na_stacje"] = zone_agg["Srednio_na_stacje"].round(1)

        fig_zone = px.pie(
            zone_agg,
            values="Rowery",
            names="Typ_Lokalizacji",
            hole=0.55,
            color_discrete_sequence=NAVY_SEQ,
        )
        fig_zone.update_traces(textposition="inside", textinfo="percent+label")
        apply_navy_theme(fig_zone, height=290)
        fig_zone.update_layout(showlegend=False)
        st.plotly_chart(fig_zone, use_container_width=True)

        st.dataframe(
            zone_agg.rename(
                columns={
                    "Typ_Lokalizacji": "Typ okolicy",
                    "Rowery": "Dostępne rowery",
                    "Stacje": "Liczba stacji",
                    "Srednio_na_stacje": "Śr. rowerów / stację",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab3:
    st.subheader(f"🔋 Ile rowerów elektrycznych wystarczy na trasę ≥ {min_range} km?")

    if len(ebikes) > 0:
        ebikes["Kategoria_Zasiegu"] = ebikes["zasieg_km"].apply(assign_battery_bucket)
        b1, b2 = st.columns([1.3, 1])

        with b1:
            fig_hist = px.histogram(
                ebikes,
                x="zasieg_km",
                nbins=20,
                color_discrete_sequence=["#38BDF8"],
                labels={"zasieg_km": "Szacowany zasięg baterii (km)", "count": "Liczba rowerów"},
            )
            fig_hist.add_vline(x=min_range, line_dash="dash", line_color="#FB7185", annotation_text=f"Twój cel: {min_range} km ({len(ebikes_ready)} szt.)")
            apply_navy_theme(fig_hist, height=400)
            fig_hist.update_layout(bargap=0.1, yaxis_title="Liczba rowerów elektrycznych")
            st.plotly_chart(fig_hist, use_container_width=True)

        with b2:
            bucket_df = ebikes.groupby("Kategoria_Zasiegu", as_index=False).size().rename(columns={"size": "Liczba_Rowerow"})
            fig_bucket = px.bar(
                bucket_df,
                x="Kategoria_Zasiegu",
                y="Liczba_Rowerow",
                text="Liczba_Rowerow",
                color="Kategoria_Zasiegu",
                color_discrete_sequence=["#FB7185", "#FBBF24", "#2DD4BF"],
                labels={"Kategoria_Zasiegu": "Gotowość baterii", "Liczba_Rowerow": "Liczba rowerów"},
            )
            fig_bucket.update_traces(textposition="outside")
            apply_navy_theme(fig_bucket, height=400)
            fig_bucket.update_layout(showlegend=False)
            st.plotly_chart(fig_bucket, use_container_width=True)

        pct_ready = round(len(ebikes_ready) / len(ebikes) * 100, 1)
        st.info(
            f"🚴‍♂️ **Okiem rowerzysty:** Jako osoba, która regularnie jeździ na rowerze po Trójmieście, wiem, że sama obecność roweru na stacji to połowa sukcesu — liczy się stan baterii. "
            f"Spośród **{len(ebikes)}** e-bike'ów w bieżącym zestawieniu, **{len(ebikes_ready)}** (**{pct_ready}%**) ma zasięg co najmniej **{min_range} km**."
        )
    else:
        st.info("Brak danych o zasięgu e-bike'ów dla wybranych filtrów.")

with tab4:
    st.subheader("🤖 Asystent AI – Analiza bieżących danych w języku naturalnym")
    st.markdown("Agent odpowiada na podstawie aktualnego stanu stacji i floty MEVO.")

    top_stations_txt = filtered_stations.sort_values("num_bikes_available", ascending=False).head(15)[
        ["name", "Miasto", "num_bikes_available", "capacity", "wypelnienie_pct"]
    ].to_string(index=False)

    city_stats_txt = (
        stations_df.groupby("Miasto", as_index=False)
        .agg(
            Dostepne=("num_bikes_available", "sum"),
            Pojemnosc=("capacity", "sum"),
            Puste_Stacje=("num_bikes_available", lambda s: int((s == 0).sum())),
        )
        .to_string(index=False)
    )

    prompt_options = [
        "Podsumuj krótko aktualną dostępność rowerów i wskaż 3 najlepsze miejsca na start wycieczki.",
        "Które obszary mają teraz najlepsze wypełnienie stacji, a gdzie jest najwięcej pustych stacji?",
        "Jak wygląda gotowość baterii e-bike'ów na dłuższą trasę (powyżej 50 km)?",
    ]
    selected_prompt = st.selectbox("Wybierz gotowe pytanie lub wpisz własne:", prompt_options)
    user_question = st.text_input("Pytanie do Agenta AI:", value=selected_prompt)

    if st.button("🚀 Zapytaj Agenta AI", type="primary"):
        api_key = st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None

        if api_key:
            with st.spinner("Agent analizuje aktualne dane..."):
                client = genai.Client(api_key=api_key)
                context_prompt = f"""Jesteś analitykiem danych systemu rowerowego MEVO w Trójmieście.
Odpowiedz zwięźle i konkretnie po polsku na pytanie wyłącznie na podstawie danych z godziny {updated_at}:

STATYSTYKI OBSZARÓW:
{city_stats_txt}

TOP 15 STACJI:
{top_stations_txt}

BATERIE E-BIKE:
- Liczba e-bike'ów w zestawieniu: {len(ebikes)}
- Średni zasięg: {avg_ebike_km}
- Zasięg >= 50 km: {int((ebikes['zasieg_km'] >= 50).sum()) if len(ebikes) > 0 else 0}

PYTANIE: {user_question}"""

                response = None
                last_error = None
                for model_name in ["gemini-3.8-flash", "gemini-2.5-flash"]:
                    for _ in range(2):
                        try:
                            response = client.models.generate_content(model=model_name, contents=context_prompt)
                            break
                        except Exception as err:
                            last_error = err
                            if "503" in str(err) or "UNAVAILABLE" in str(err):
                                time.sleep(1.2)
                            else:
                                break
                    if response is not None:
                        break

                if response is not None and getattr(response, "text", None):
                    st.success("**Odpowiedź Agenta AI:**")
                    st.write(response.text)
                else:
                    st.warning("Model jest chwilowo przeciążony — wyświetlam podsumowanie z bieżących danych:")
                    st.info(build_fallback_summary(stations_df, filtered_stations, ebikes, avg_ebike_km, updated_at, last_error))
        else:
            st.info(build_fallback_summary(stations_df, filtered_stations, ebikes, avg_ebike_km, updated_at))

st.divider()
st.caption("Autor: Sylwester Zaborowski | Źródło danych: Otwarte Dane Systemu Roweru Metropolitalnego MEVO (GBFS) | Projekt portfolio Data Analytics.")

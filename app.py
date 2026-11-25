import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime
from google import genai

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="MEVO Trójmiasto – Pełna Analiza Otwartych Danych",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLIZACJA ---
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"] label {
        color: #8B949E !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F0F6FC !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    .summary-card {
        background-color: #161B22;
        border-left: 4px solid #2A9D8F;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
        color: #E6EDF3;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# --- POBIERANIE WSZYSTKICH DOSTĘPNYCH DANYCH Z API GBFS ---
@st.cache_data(ttl=60)
def load_all_mevo_feeds():
    gbfs_url = "https://gbfs.urbansharing.com/rowermevo.pl/gbfs.json"
    gbfs_index = requests.get(gbfs_url, timeout=10).json()
    feeds = {f["name"]: f["url"] for f in gbfs_index["data"]["pl"]["feeds"]}

    raw_feeds = {}
    for feed_name, feed_url in feeds.items():
        try:
            raw_feeds[feed_name] = requests.get(feed_url, timeout=10).json()
        except Exception:
            raw_feeds[feed_name] = None

    # 1. Stacje (informacje + status)
    stations_info = pd.DataFrame(raw_feeds["station_information"]["data"]["stations"])
    stations_status = pd.DataFrame(raw_feeds["station_status"]["data"]["stations"])
    stations = pd.merge(stations_info, stations_status, on="station_id", how="inner")

    def extract_station_vehicle_counts(row):
        v_list = row.get("vehicle_types_available", [])
        ebike_cnt, bike_cnt = 0, 0
        if isinstance(v_list, list):
            for item in v_list:
                if item.get("vehicle_type_id") == "ebike":
                    ebike_cnt = item.get("count", 0)
                elif item.get("vehicle_type_id") == "bike":
                    bike_cnt = item.get("count", 0)
        return pd.Series([ebike_cnt, bike_cnt])

    if "vehicle_types_available" in stations.columns:
        stations[["ebike_na_stacji", "klasyczne_na_stacji"]] = stations.apply(extract_station_vehicle_counts, axis=1)
    else:
        stations["ebike_na_stacji"] = 0
        stations["klasyczne_na_stacji"] = stations["num_bikes_available"]

    stations["wypelnienie_pct"] = np.where(
        stations["capacity"] > 0,
        (stations["num_bikes_available"] / stations["capacity"] * 100).round(1),
        0
    )

    if "last_reported" in stations.columns:
        stations["ostatni_raport"] = pd.to_datetime(stations["last_reported"], unit="s", utc=True).dt.tz_convert("Europe/Warsaw").dt.strftime("%H:%M:%S")
    else:
        stations["ostatni_raport"] = "-"

    # Przypisanie miasta na podstawie współrzędnych GPS
    def assign_city(row):
        lat, lon = row["lat"], row["lon"]
        if 54.27 <= lat <= 54.425 and 18.45 <= lon <= 18.95:
            return "Gdańsk"
        elif 54.425 < lat <= 54.475 and 18.50 <= lon <= 18.60:
            return "Sopot"
        elif 54.46 <= lat <= 54.58 and 18.38 <= lon <= 18.58:
            return "Gdynia"
        elif lat < 54.27 and lon > 18.55:
            return "Pruszcz Gdański / Tczew"
        elif lat > 54.55 and lon <= 18.45:
            return "Rumia / Reda / Wejherowo"
        else:
            return "Pozostałe gminy"

    stations["Miasto"] = stations.apply(assign_city, axis=1)

    # Kategoryzacja typu okolicy stacji
    def assign_zone_type(name):
        n = str(name).lower()
        if any(k in n for k in ["skm", "pkm", "dworzec", "główny", "wrzeszcz", "oliwa", "politechnika", "uniwersytet", "alchemia", "olivia", "centrum"]):
            return "Węzły przesiadkowe i uczelnie"
        elif any(k in n for k in ["plaża", "molo", "brzeźno", "jelitkowo", "orłowo", "bulwar", "park", "stogi", "sobieszewo", "nadmorski", "aleja"]):
            return "Strefa nadmorska i rekreacja"
        else:
            return "Dzielnice mieszkaniowe"

    stations["Typ_Lokalizacji"] = stations["name"].apply(assign_zone_type)

    def availability_status(row):
        if row["num_bikes_available"] == 0:
            return "Brak rowerów (0 szt.)"
        elif row["wypelnienie_pct"] >= 80:
            return "Dużo rowerów (≥80%)"
        else:
            return "Dostępne (1–79%)"

    stations["Status"] = stations.apply(availability_status, axis=1)

    # 2. Pojedyncze rowery poza stacjami (free_bike_status)
    free_bikes_raw = pd.DataFrame(raw_feeds["free_bike_status"]["data"]["bikes"])
    valid_mask = (free_bikes_raw["lat"].between(53.8, 55.0)) & (free_bikes_raw["lon"].between(17.5, 19.5))
    anomalies_count = int((~valid_mask).sum())

    bikes = free_bikes_raw[valid_mask].copy()
    bikes["zasieg_km"] = (bikes["current_range_meters"] / 1000).fillna(0).round(1) if "current_range_meters" in bikes.columns else 0.0
    bikes["Typ_Roweru"] = bikes["vehicle_type_id"].map({"ebike": "Elektryczny (E-Bike)", "bike": "Klasyczny"}).fillna("Inny") if "vehicle_type_id" in bikes.columns else "Inny"

    # Bezpieczne sprawdzenie czy w API istnieje kolumna station_id
    if "station_id" in bikes.columns:
        bikes["Gdzie_Stoi"] = np.where(bikes["station_id"].notna() & (bikes["station_id"] != ""), "Na stacji", "Poza stacją (Free-floating)")
    else:
        bikes["Gdzie_Stoi"] = "Poza stacją (Free-floating)"

    def bike_tech_status(row):
        if row.get("is_disabled"):
            return "Wyłączony / Serwis"
        elif row.get("is_reserved"):
            return "Zarezerwowany"
        else:
            return "Gotowy do jazdy"

    bikes["Status_Techniczny"] = bikes.apply(bike_tech_status, axis=1)

    # 3. Dodatkowe metadane z pozostałych plików GBFS
    sys_info = raw_feeds.get("system_information", {}).get("data", {}) if raw_feeds.get("system_information") else {}
    veh_types = pd.DataFrame(raw_feeds.get("vehicle_types", {}).get("data", {}).get("vehicle_types", [])) if raw_feeds.get("vehicle_types") else pd.DataFrame()
    pricing_plans = pd.DataFrame(raw_feeds.get("system_pricing_plans", {}).get("data", {}).get("plans", [])) if raw_feeds.get("system_pricing_plans") else pd.DataFrame()
    alerts = pd.DataFrame(raw_feeds.get("system_alerts", {}).get("data", {}).get("alerts", [])) if raw_feeds.get("system_alerts") else pd.DataFrame()

    meta = {
        "feeds_list": list(feeds.keys()),
        "anomalies_count": anomalies_count,
        "raw_bikes_total": len(free_bikes_raw),
        "sys_info": sys_info,
        "veh_types": veh_types,
        "pricing_plans": pricing_plans,
        "alerts": alerts,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return stations, bikes, meta

stations_df, bikes_df, meta = load_all_mevo_feeds()

# --- PANEL BOCZNY (FILTRY) ---
st.sidebar.title("🎛️ Filtry analizy")

available_cities = sorted(stations_df["Miasto"].unique())
default_cities = [c for c in ["Gdańsk", "Gdynia", "Sopot"] if c in available_cities] or available_cities

selected_cities = st.sidebar.multiselect(
    "Wybierz miasta / obszar:",
    options=available_cities,
    default=default_cities
)

st.sidebar.divider()
st.sidebar.subheader("🚴‍♂️ Filtr dla rowerzysty")
min_range = st.sidebar.slider("Planowany dystans trasy (km):", 5, 100, 30, step=5)
only_working = st.sidebar.checkbox("Uwzględnij tylko sprawne rowery", value=True)

filtered_stations = stations_df[stations_df["Miasto"].isin(selected_cities)].copy() if selected_cities else stations_df.copy()
filtered_bikes = bikes_df[bikes_df["Status_Techniczny"] == "Gotowy do jazdy"].copy() if only_working else bikes_df.copy()

ebikes = filtered_bikes[filtered_bikes["vehicle_type_id"] == "ebike"].copy() if "vehicle_type_id" in filtered_bikes.columns else pd.DataFrame(columns=["zasieg_km"])
ebikes_ready = ebikes[ebikes["zasieg_km"] >= min_range]

# --- NAGŁÓWEK ---
st.title("🚲 MEVO Trójmiasto – Pełna Analiza Dostępności Rowerów")
st.markdown(
    f"Kompleksowy podgląd otwartych danych MEVO ({len(meta['feeds_list'])} aktywnych źródeł GBFS): "
    f"stacje, pojazdy, baterie oraz asystent AI • *Aktualizacja:* `{meta['updated_at']}`"
)

# --- 4 GŁÓWNE KARTY KPI ---
total_bikes_on_stations = int(filtered_stations["num_bikes_available"].sum())
total_capacity = int(filtered_stations["capacity"].sum())
avg_fill = round((total_bikes_on_stations / total_capacity * 100), 1) if total_capacity > 0 else 0
stations_with_bikes = len(filtered_stations[filtered_stations["num_bikes_available"] > 0])
empty_stations = len(filtered_stations[filtered_stations["num_bikes_available"] == 0])
avg_ebike_km = f"{ebikes['zasieg_km'].mean():.1f} km" if len(ebikes) > 0 else "0.0 km"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rowery na stacjach", f"{total_bikes_on_stations:,}".replace(",", " "), f"z {total_capacity:,} miejsc postojowych".replace(",", " "))
c2.metric("Stacje z dostępnym rowerem", f"{stations_with_bikes} z {len(filtered_stations)}", f"{empty_stations} stacji pustych", delta_color="inverse")
c3.metric("Średnie wypełnienie stacji", f"{avg_fill}%", "udział zajętych stojaków")
c4.metric(f"E-bike'i gotowe na ≥ {min_range} km", f"{len(ebikes_ready):,}".replace(",", " "), f"średni zasięg: {avg_ebike_km}")

st.write("")

# --- 4 ZAKŁADKI ANALITYCZNE ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Stacje, Mapa i TOP Lokalizacje",
    "🚲 Struktura Floty & Baterie",
    "📡 Pełny Zrzut z API (Metadane)",
    "🤖 Asystent AI (Zapytaj o dane)"
])

# ==============================================================================
# ZAKŁADKA 1: STACJE, MAPA I TOP LOKALIZACJE
# ==============================================================================
with tab1:
    best_station = filtered_stations.sort_values("num_bikes_available", ascending=False).iloc[0] if len(filtered_stations) > 0 else None
    if best_station is not None:
        st.markdown(
            f"""<div class="summary-card">
            <b>Szybkie podsumowanie:</b> W wybranym obszarze dostępnych jest <b>{total_bikes_on_stations}</b> rowerów na <b>{stations_with_bikes}</b> stacjach. 
            Najwięcej rowerów stoi obecnie na stacji <b>{best_station['name']}</b> ({int(best_station['num_bikes_available'])} szt.). 
            Na <b>{empty_stations}</b> stacjach nie ma obecnie żadnego wolnego roweru.
            </div>""",
            unsafe_allow_html=True
        )

    g1, g2 = st.columns([1.4, 1])
    with g1:
        st.subheader("🗺️ Interaktywna mapa stacji w Metropolii")
        map_df = filtered_stations.copy()

        def get_point_color(row):
            if row["num_bikes_available"] == 0:
                return [230, 57, 70, 210]    # Czerwony (brak rowerów)
            elif row["wypelnienie_pct"] >= 80:
                return [0, 180, 216, 220]    # Niebieski (dużo rowerów)
            else:
                return [42, 157, 143, 210]   # Szmaragdowy (dostępne)

        map_df["color"] = map_df.apply(get_point_color, axis=1)
        map_df["radius"] = np.clip(map_df["num_bikes_available"] * 18 + 60, 60, 400)

        view_state = pdk.ViewState(
            latitude=float(map_df["lat"].mean()) if len(map_df) > 0 else 54.42,
            longitude=float(map_df["lon"].mean()) if len(map_df) > 0 else 18.58,
            zoom=10.2,
            pitch=0
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["lon", "lat"],
            get_color="color",
            get_radius="radius",
            pickable=True,
            auto_highlight=True
        )

        tooltip = {
            "html": "<b>{name}</b><br/>"
                    "Miasto: {Miasto}<br/>"
                    "Dostępne rowery: <b>{num_bikes_available}</b> / {capacity} ({wypelnienie_pct}%)<br/>"
                    "⚡ E-Bike: {ebike_na_stacji} | 🚲 Klasyczne: {klasyczne_na_stacji}",
            "style": {"backgroundColor": "#161B22", "color": "#F0F6FC", "border": "1px solid #30363D"}
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
            ),
            use_container_width=True
        )
        st.caption("🟢 Dostępne (1–79%)  |  🔵 Dużo rowerów (≥80%)  |  🔴 Pusta stacja (0 szt.) — najedź kursorem na punkt, aby zobaczyć szczegóły.")

    with g2:
        st.subheader("🏘️ Dostępność wg typu okolicy")
        zone_agg = filtered_stations.groupby("Typ_Lokalizacji", as_index=False).agg(
            Rowery=("num_bikes_available", "sum"),
            Stacje=("station_id", "count"),
            Srednio=("num_bikes_available", "mean")
        )
        zone_agg["Srednio"] = zone_agg["Srednio"].round(1)
        fig_zone = px.pie(
            zone_agg,
            values="Rowery",
            names="Typ_Lokalizacji",
            hole=0.55,
            color_discrete_sequence=["#2A9D8F", "#00B4D8", "#F4A261"]
        )
        fig_zone.update_traces(textposition="inside", textinfo="percent+label")
        fig_zone.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=290,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_zone, use_container_width=True)
        st.dataframe(
            zone_agg.rename(columns={"Typ_Lokalizacji": "Typ okolicy", "Rowery": "Rowery", "Stacje": "Stacje", "Srednio": "Śr. / stację"}),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("🏆 TOP 10 stacji z największą liczbą rowerów")
        top10 = filtered_stations.sort_values("num_bikes_available", ascending=False).head(10)
        fig_top = px.bar(
            top10.iloc[::-1],
            x="num_bikes_available",
            y="name",
            orientation="h",
            text="num_bikes_available",
            color="wypelnienie_pct",
            color_continuous_scale="Tealgrn",
            labels={"num_bikes_available": "Liczba dostępnych rowerów", "name": "", "wypelnienie_pct": "Wypełnienie (%)"}
        )
        fig_top.update_traces(textposition="outside")
        fig_top.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            coloraxis_showscale=False,
            margin=dict(t=10, b=20, l=10, r=30)
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col_right:
        st.subheader("🏙️ Porównanie miast (Rowery vs Wolne miejsca)")
        city_summary = stations_df.groupby("Miasto", as_index=False).agg(
            Dostepne=("num_bikes_available", "sum"),
            Wolne_Miejsca=("num_docks_available", "sum")
        ).sort_values("Dostepne", ascending=True)

        fig_cities = go.Figure()
        fig_cities.add_trace(go.Bar(
            y=city_summary["Miasto"], x=city_summary["Dostepne"],
            name="Dostępne rowery", orientation="h", marker_color="#2A9D8F",
            text=city_summary["Dostepne"], textposition="auto"
        ))
        fig_cities.add_trace(go.Bar(
            y=city_summary["Miasto"], x=city_summary["Wolne_Miejsca"],
            name="Wolne stojaki", orientation="h", marker_color="#30363D"
        ))
        fig_cities.update_layout(
            barmode="stack",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(t=10, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cities, use_container_width=True)

    st.subheader("📋 Szczegółowa tabela stacji (z podziałem na E-Bike i Klasyczne)")
    view_option = st.radio(
        "Filtr widoku:",
        ["Najwięcej dostępnych rowerów", "Puste stacje (0 rowerów)", "Wyszukaj konkretną stację"],
        horizontal=True
    )

    if view_option == "Najwięcej dostępnych rowerów":
        display_df = filtered_stations.sort_values("num_bikes_available", ascending=False).head(15)
    elif view_option == "Puste stacje (0 rowerów)":
        display_df = filtered_stations[filtered_stations["num_bikes_available"] == 0].sort_values("capacity", ascending=False)
    else:
        search_txt = st.text_input("Wpisz nazwę stacji lub ulicy (np. Wrzeszcz, Oliwa, Molo, SKM):")
        display_df = filtered_stations.sort_values("num_bikes_available", ascending=False)
        if search_txt:
            display_df = display_df[display_df["name"].str.contains(search_txt, case=False, na=False)]

    st.dataframe(
        display_df[["station_id", "name", "Miasto", "num_bikes_available", "ebike_na_stacji", "klasyczne_na_stacji", "capacity", "wypelnienie_pct", "ostatni_raport"]].rename(columns={
            "station_id": "ID",
            "name": "Nazwa stacji",
            "num_bikes_available": "Wszystkie rowery",
            "ebike_na_stacji": "E-Bike",
            "klasyczne_na_stacji": "Klasyczne",
            "capacity": "Pojemność",
            "wypelnienie_pct": "Wypełnienie (%)",
            "ostatni_raport": "Ostatni sygnał"
        }),
        use_container_width=True,
        hide_index=True
    )

# ==============================================================================
# ZAKŁADKA 2: STRUKTURA FLOTY & BATERIE
# ==============================================================================
with tab2:
    st.subheader("🚲 Pełny obraz floty widocznej w API (`free_bike_status`)")
    st.markdown("Analiza rowerów raportujących swoją indywidualną pozycję GPS poza stacjami (*free-floating*) oraz porównanie z rowerami przypisanymi do stacji.")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("**1. Typ roweru (Elektryczny vs Klasyczny)**")
        type_df = bikes_df.groupby("Typ_Roweru", as_index=False).size().rename(columns={"size": "Liczba"})
        fig_t = px.pie(type_df, values="Liczba", names="Typ_Roweru", hole=0.5, color_discrete_sequence=["#2A9D8F", "#00B4D8"])
        fig_t.update_traces(textinfo="value+percent")
        fig_t.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_t, use_container_width=True)

    with f2:
        st.markdown("**2. Status techniczny rowerów (`free_bike_status`)**")
        stat_df = bikes_df.groupby("Status_Techniczny", as_index=False).size().rename(columns={"size": "Liczba"})
        fig_s = px.pie(stat_df, values="Liczba", names="Status_Techniczny", hole=0.5, color_discrete_sequence=["#2A9D8F", "#F4A261", "#E63946"])
        fig_s.update_traces(textinfo="value+percent")
        fig_s.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_s, use_container_width=True)

    with f3:
        st.markdown("**3. Rozmieszczenie w systemie (Stacje vs Ulica)**")
        all_stations_bikes = int(stations_df["num_bikes_available"].sum())
        free_floating_bikes = len(bikes_df[bikes_df["Status_Techniczny"] == "Gotowy do jazdy"])
        loc_df = pd.DataFrame({
            "Lokalizacja": ["Na stacjach MEVO", "Poza stacjami (Free-floating)"],
            "Liczba": [all_stations_bikes, free_floating_bikes]
        })
        fig_l = px.pie(loc_df, values="Liczba", names="Lokalizacja", hole=0.5, color_discrete_sequence=["#00B4D8", "#9B5DE5"])
        fig_l.update_traces(textinfo="value+percent")
        fig_l.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_l, use_container_width=True)

    st.divider()
    st.subheader(f"🔋 Rozkład zasięgu baterii w e-bike'ach (Twój próg: ≥ {min_range} km)")

    if len(ebikes) > 0:
        def battery_bucket(km):
            if km < 20:
                return "1. Krótka trasa (< 20 km)"
            elif km < 50:
                return "2. Średni dystans (20–50 km)"
            else:
                return "3. Długa wycieczka (> 50 km)"

        ebikes["Kategoria_Zasiegu"] = ebikes["zasieg_km"].apply(battery_bucket)

        b1, b2 = st.columns([1.3, 1])
        with b1:
            fig_hist = px.histogram(
                ebikes, x="zasieg_km", nbins=20, color_discrete_sequence=["#2A9D8F"],
                labels={"zasieg_km": "Szacowany zasięg baterii (km)", "count": "Liczba rowerów"}
            )
            fig_hist.add_vline(x=min_range, line_dash="dash", line_color="#E63946", annotation_text=f"Cel: {min_range} km ({len(ebikes_ready)} szt.)")
            fig_hist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=360, bargap=0.1, yaxis_title="Liczba e-bike'ów", margin=dict(t=20, b=20, l=10, r=10))
            st.plotly_chart(fig_hist, use_container_width=True)

        with b2:
            bucket_df = ebikes.groupby("Kategoria_Zasiegu", as_index=False).size().rename(columns={"size": "Liczba_Rowerow"})
            fig_bucket = px.bar(
                bucket_df, x="Kategoria_Zasiegu", y="Liczba_Rowerow", text="Liczba_Rowerow",
                color="Kategoria_Zasiegu", color_discrete_sequence=["#E63946", "#F4A261", "#2A9D8F"],
                labels={"Kategoria_Zasiegu": "Gotowość baterii", "Liczba_Rowerow": "Liczba rowerów"}
            )
            fig_bucket.update_traces(textposition="outside")
            fig_bucket.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=360, margin=dict(t=20, b=20, l=10, r=10))
            st.plotly_chart(fig_bucket, use_container_width=True)
    else:
        st.info("Brak danych o zasięgu e-bike'ów w bieżącym zrzucie `free_bike_status`.")

# ==============================================================================
# ZAKŁADKA 3: PEŁNY ZRZUT Z API (METADANE, CENNIKI, ALERTY, JAKOŚĆ DANYCH)
# ==============================================================================
with tab3:
    st.subheader("📡 Wszystko, co udostępnia otwarte API MEVO (GBFS)")
    st.markdown(
        f"Aplikacja łączy się z głównym indeksem `gbfs.json` i pobiera wszystkie dostępne zbiory: "
        f"`{', '.join(meta['feeds_list'])}`."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Surowe rekordy w `free_bike_status`", meta["raw_bikes_total"])
    m2.metric("Odfiltrowane anomalie GPS", meta["anomalies_count"], "poza obszarem Pomorza")
    m3.metric("Liczba wszystkich stacji w bazie", len(stations_df))

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### ℹ️ Informacje o systemie (`system_information`)")
        if meta["sys_info"]:
            sys_table = pd.DataFrame(list(meta["sys_info"].items()), columns=["Parametr API", "Wartość"])
            st.dataframe(sys_table, use_container_width=True, hide_index=True)

        st.markdown("#### 🚲 Specyfikacja typów pojazdów (`vehicle_types`)")
        if not meta["veh_types"].empty:
            st.dataframe(meta["veh_types"], use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### 📋 Oficjalne plany taryfowe w API (`system_pricing_plans`)")
        if not meta["pricing_plans"].empty:
            cols_to_show = [c for c in ["plan_id", "name", "price", "currency", "description"] if c in meta["pricing_plans"].columns]
            st.dataframe(meta["pricing_plans"][cols_to_show], use_container_width=True, hide_index=True)
        else:
            st.info("Brak aktywnych planów taryfowych w pliku `system_pricing_plans.json`.")

        st.markdown("#### ⚠️ Komunikaty i alerty systemowe (`system_alerts`)")
        if not meta["alerts"].empty:
            st.dataframe(meta["alerts"], use_container_width=True, hide_index=True)
        else:
            st.success("Brak aktywnych alertów awarii w feedzie `system_alerts.json`.")

    with st.expander("🔍 Podgląd surowej tabeli pojedynczych rowerów (pierwsze 50 rekordów z API)"):
        st.dataframe(bikes_df.head(50), use_container_width=True, hide_index=True)

# ==============================================================================
# ZAKŁADKA 4: AGENT AI (ANALITYK DANYCH NA ŻYWO)
# ==============================================================================
with tab4:
    st.subheader("🤖 Asystent AI – Zadaj pytanie o bieżącą sytuację w sieci MEVO")
    st.markdown("Agent analizuje aktualny zrzut danych ze wszystkich stacji i rowerów i odpowiada prostym językiem.")

    top_stations_txt = filtered_stations.sort_values("num_bikes_available", ascending=False).head(15)[
        ["name", "Miasto", "num_bikes_available", "ebike_na_stacji", "klasyczne_na_stacji", "capacity"]
    ].to_string(index=False)

    city_stats_txt = stations_df.groupby("Miasto", as_index=False).agg(
        Dostepne=("num_bikes_available", "sum"),
        Pojemnosc=("capacity", "sum"),
        Puste_Stacje=("num_bikes_available", lambda s: (s == 0).sum())
    ).to_string(index=False)

    prompt_options = [
        "Podsumuj krótko aktualną dostępność rowerów w Trójmieście i wskaż 3 najlepsze miejsca na start wycieczki.",
        "Które miasta mają teraz najlepsze wypełnienie stacji, a gdzie brakuje rowerów?",
        "Ile jest teraz dostępnych e-bike'ów vs rowerów klasycznych i czy wystarczy baterii na dłuższą trasę?"
    ]
    selected_prompt = st.selectbox("Wybierz gotowe pytanie lub wpisz własne poniżej:", prompt_options)
    user_question = st.text_input("Twoje pytanie do Agenta:", value=selected_prompt)

    if st.button("🚀 Zapytaj Agenta AI", type="primary"):
        api_key = st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None

        if api_key:
            with st.spinner("Agent analizuje aktualne dane z API MEVO..."):
                try:
                    client = genai.Client(api_key=api_key)
                    context_prompt = f"""
                    Jesteś pomocnym analitykiem danych systemu rowerowego MEVO w Trójmieście.
                    Odpowiedz prostym, zrozumiałym językiem po polsku na pytanie użytkownika wyłącznie na podstawie poniższych danych z godziny {meta['updated_at']}:

                    STATYSTYKI MIAST:
                    {city_stats_txt}

                    TOP 15 STACJI:
                    {top_stations_txt}

                    FLOTA ROWEROWA (poza stacjami / free_bike_status):
                    - Sprawne rowery w free_bike_status: {len(filtered_bikes)} (w tym e-bike: {len(ebikes)}, klasyczne: {len(filtered_bikes) - len(ebikes)})
                    - Średni zasięg e-bike'ów: {avg_ebike_km}
                    - E-bike'i z zasięgiem >= 50 km: {len(ebikes[ebikes['zasieg_km'] >= 50])}

                    PYTANIE UŻYTKOWNIKA: {user_question}
                    """
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=context_prompt,
                    )
                    st.success("**Odpowiedź Agenta AI:**")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Błąd połączenia z API modelu: {e}")
        else:
            best_city = stations_df.groupby("Miasto")["num_bikes_available"].sum().idxmax()
            best_city_cnt = int(stations_df.groupby("Miasto")["num_bikes_available"].sum().max())
            top3_names = ", ".join(filtered_stations.sort_values("num_bikes_available", ascending=False).head(3)["name"].tolist())
            long_range_ebikes = len(ebikes[ebikes["zasieg_km"] >= 50])

            st.info(
                f"🤖 **Podsumowanie Agenta (stan na {meta['updated_at']}):**\n\n"
                f"* **Najwięcej rowerów:** Obecnie liderem jest **{best_city}** ({best_city_cnt} dostępnych rowerów na stacjach).\n"
                f"* **TOP 3 stacje na start:** {top3_names}.\

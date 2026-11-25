import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="MEVO Trójmiasto – Analiza Dostępności Rowerów",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLIZACJA (ESTETYCZNY WYGLĄD DASHBOARDU) ---
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

# --- POBIERANIE I CZYSZCZENIE DANYCH (ETL) ---
@st.cache_data(ttl=60)
def load_mevo_data():
    gbfs_index = requests.get("https://gbfs.urbansharing.com/rowermevo.pl/gbfs.json", timeout=10).json()
    feeds = {f["name"]: f["url"] for f in gbfs_index["data"]["pl"]["feeds"]}

    stations_info = pd.DataFrame(requests.get(feeds["station_information"], timeout=10).json()["data"]["stations"])
    stations_status = pd.DataFrame(requests.get(feeds["station_status"], timeout=10).json()["data"]["stations"])
    free_bikes = pd.DataFrame(requests.get(feeds["free_bike_status"], timeout=10).json()["data"]["bikes"])

    # Połączenie informacji o stacjach z ich aktualnym statusem
    stations = pd.merge(stations_info, stations_status, on="station_id", how="inner")
    stations["wypelnienie_pct"] = np.where(
        stations["capacity"] > 0,
        (stations["num_bikes_available"] / stations["capacity"] * 100).round(1),
        0
    )

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

    # Prosty status dostępności na stacji
    def availability_status(row):
        if row["num_bikes_available"] == 0:
            return "Brak rowerów (0 szt.)"
        elif row["wypelnienie_pct"] >= 80:
            return "Dużo rowerów (≥80%)"
        else:
            return "Dostępne (1–79%)"

    stations["Status"] = stations.apply(availability_status, axis=1)

    # Czyszczenie danych o rowerach (usunięcie błędnych odczytów GPS spoza regionu)
    bikes = free_bikes[
        (free_bikes["lat"].between(53.8, 55.0)) &
        (free_bikes["lon"].between(17.5, 19.5))
    ].copy()
    bikes["zasieg_km"] = (bikes["current_range_meters"] / 1000).fillna(0).round(1)

    return stations, bikes, datetime.now().strftime("%Y-%m-%d %H:%M:%S")

stations_df, bikes_df, updated_at = load_mevo_data()

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
min_range = st.sidebar.slider("Jaki dystans chcesz przejechać (km)?", 5, 100, 30, step=5)
only_working = st.sidebar.checkbox("Pokaż tylko sprawne rowery", value=True)

# Filtrowanie stacji
filtered_stations = stations_df[stations_df["Miasto"].isin(selected_cities)].copy() if selected_cities else stations_df.copy()

# Filtrowanie rowerów
filtered_bikes = bikes_df.copy()
if only_working:
    filtered_bikes = filtered_bikes[(~filtered_bikes["is_disabled"]) & (~filtered_bikes["is_reserved"])]

ebikes = filtered_bikes[filtered_bikes["vehicle_type_id"] == "ebike"].copy()
ebikes_ready = ebikes[ebikes["zasieg_km"] >= min_range]

# --- NAGŁÓWEK ---
st.title("🚲 MEVO Trójmiasto – Gdzie najłatwiej złapać rower?")
st.markdown(
    f"Analiza dostępności rowerów miejskich, obłożenia stacji oraz stanu baterii na podstawie otwartych danych MEVO. "
    f"• *Aktualizacja na żywo:* `{updated_at}`"
)

# --- 4 GŁÓWNE KARTY KPI ---
total_bikes_on_stations = int(filtered_stations["num_bikes_available"].sum())
total_capacity = int(filtered_stations["capacity"].sum())
avg_fill = round((total_bikes_on_stations / total_capacity * 100), 1) if total_capacity > 0 else 0
stations_with_bikes = len(filtered_stations[filtered_stations["num_bikes_available"] > 0])
empty_stations = len(filtered_stations[filtered_stations["num_bikes_available"] == 0])

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Rowery na stacjach",
    f"{total_bikes_on_stations:,}".replace(",", " "),
    f"z {total_capacity:,} miejsc postojowych".replace(",", " ")
)
c2.metric(
    "Stacje z dostępnym rowerem",
    f"{stations_with_bikes} z {len(filtered_stations)}",
    f"{empty_stations} stacji aktualnie pustych",
    delta_color="inverse"
)
c3.metric(
    "Średnie wypełnienie stacji",
    f"{avg_fill}%",
    "udział zajętych stojaków"
)
c4.metric(
    f"E-bike'i gotowe na ≥ {min_range} km",
    f"{len(ebikes_ready):,}".replace(",", " "),
    f"średni zasięg: {ebikes['zasieg_km'].mean():.1f} km"
)

st.write("")

# --- ZAKŁADKI ---
tab1, tab2, tab3 = st.tabs([
    "📊 Dostępność i TOP Stacje",
    "📍 Rozkład w Trójmieście",
    "🔋 Bateria i Zasięg (Okiem Rowerzysty)"
])

# ==============================================================================
# ZAKŁADKA 1: DOSTĘPNOŚĆ I TOP STACJE
# ==============================================================================
with tab1:
    # Krótkie, zrozumiałe podsumowanie wniosków z danych
    best_station = filtered_stations.sort_values("num_bikes_available", ascending=False).iloc[0] if len(filtered_stations) > 0 else None
    if best_station is not None:
        st.markdown(
            f"""<div class="summary-card">
            <b>Wnioski z bieżących danych:</b> W wybranym obszarze dostępnych jest <b>{total_bikes_on_stations}</b> rowerów na <b>{stations_with_bikes}</b> stacjach. 
            Najwięcej rowerów stoi obecnie na stacji <b>{best_station['name']}</b> ({int(best_station['num_bikes_available'])} szt.). 
            Na <b>{empty_stations}</b> stacjach nie ma w tej chwili żadnego wolnego roweru.
            </div>""",
            unsafe_allow_html=True
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
            color_continuous_scale="Tealgrn",
            labels={"num_bikes_available": "Liczba dostępnych rowerów", "name": "", "wypelnienie_pct": "Wypełnienie (%)"}
        )
        fig_top.update_traces(textposition="outside")
        fig_top.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=430,
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
            height=430,
            margin=dict(t=10, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cities, use_container_width=True)

    # Interaktywna tabela stacji
    st.subheader("📋 Przegląd stacji – filtruj według potrzeb")
    view_option = st.radio(
        "Pokaż stacje:",
        ["Najwięcej dostępnych rowerów", "Puste stacje (0 rowerów)", "Wszystkie stacje (wyszukiwarka)"],
        horizontal=True
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
        display_df[["name", "Miasto", "Typ_Lokalizacji", "num_bikes_available", "capacity", "wypelnienie_pct", "Status"]].rename(columns={
            "name": "Stacja",
            "Typ_Lokalizacji": "Typ okolicy",
            "num_bikes_available": "Dostępne rowery",
            "capacity": "Wszystkie stojaki",
            "wypelnienie_pct": "Wypełnienie (%)"
        }),
        use_container_width=True,
        hide_index=True
    )

# ==============================================================================
# ZAKŁADKA 2: ROZKŁAD PRZESTRZENNY I TYP OKOLICY
# ==============================================================================
with tab2:
    g1, g2 = st.columns([1.3, 1])

    with g1:
        st.subheader("📍 Mapa punktowa stacji w Metropolii")
        fig_geo = px.scatter(
            filtered_stations,
            x="lon",
            y="lat",
            size="num_bikes_available",
            color="Status",
            hover_name="name",
            hover_data={"Miasto": True, "num_bikes_available": True, "capacity": True, "lon": False, "lat": False},
            color_discrete_map={
                "Dużo rowerów (≥80%)": "#00B4D8",
                "Dostępne (1–79%)": "#2A9D8F",
                "Brak rowerów (0 szt.)": "#E63946"
            },
            labels={"lon": "Długość geograficzna", "lat": "Szerokość geograficzna"}
        )
        fig_geo.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#161B22",
            height=460,
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis=dict(showgrid=True, gridcolor="#30363D", showticklabels=False, title="Zachód ← → Wschód"),
            yaxis=dict(showgrid=True, gridcolor="#30363D", showticklabels=False, title="Południe ← → Północ", scaleanchor="x", scaleratio=1.6),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    with g2:
        st.subheader("🏘️ Gdzie stoją rowery? (Podział wg okolicy)")
        zone_agg = filtered_stations.groupby("Typ_Lokalizacji", as_index=False).agg(
            Rowery=("num_bikes_available", "sum"),
            Stacje=("station_id", "count"),
            Srednio_na_stacje=("num_bikes_available", "mean")
        )
        zone_agg["Srednio_na_stacje"] = zone_agg["Srednio_na_stacje"].round(1)

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
            height=300,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_zone, use_container_width=True)

        st.dataframe(
            zone_agg.rename(columns={
                "Typ_Lokalizacji": "Typ okolicy",
                "Rowery": "Dostępne rowery",
                "Stacje": "Liczba stacji",
                "Srednio_na_stacje": "Śr. rowerów / stację"
            }),
            use_container_width=True,
            hide_index=True
        )

# ==============================================================================
# ZAKŁADKA 3: BATERIA I PERSPEKTYWA ROWERZYSTY
# ==============================================================================
with tab3:
    st.subheader(f"🔋 Ile rowerów elektrycznych wystarczy na trasę ≥ {min_range} km?")

    # Kategoryzacja zasięgu baterii na zrozumiałe przedziały
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
            ebikes,
            x="zasieg_km",
            nbins=20,
            color_discrete_sequence=["#2A9D8F"],
            labels={"zasieg_km": "Szacowany zasięg baterii (km)", "count": "Liczba rowerów"}
        )
        fig_hist.add_vline(
            x=min_range,
            line_dash="dash",
            line_color="#E63946",
            annotation_text=f"Twój cel: {min_range} km ({len(ebikes_ready)} szt.)"
        )
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            bargap=0.1,
            yaxis_title="Liczba rowerów elektrycznych",
            margin=dict(t=20, b=20, l=10, r=10)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with b2:
        bucket_df = ebikes.groupby("Kategoria_Zasiegu", as_index=False).size().rename(columns={"size": "Liczba_Rowerow"})
        fig_bucket = px.bar(
            bucket_df,
            x="Kategoria_Zasiegu",
            y="Liczba_Rowerow",
            text="Liczba_Rowerow",
            color="Kategoria_Zasiegu",
            color_discrete_sequence=["#E63946", "#F4A261", "#2A9D8F"],
            labels={"Kategoria_Zasiegu": "Gotowość baterii", "Liczba_Rowerow": "Liczba rowerów"}
        )
        fig_bucket.update_traces(textposition="outside")
        fig_bucket.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=400,
            margin=dict(t=20, b=20, l=10, r=10)
        )
        st.plotly_chart(fig_bucket, use_container_width=True)

    pct_ready = round(len(ebikes_ready) / max(len(ebikes), 1) * 100, 1)
    st.info(
        f"🚴‍♂️ **Okiem rowerzysty:** Jako osoba, która regularnie jeździ na rowerze po Trójmieście, wiem, że sama obecność roweru na stacji to połowa sukcesu — liczy się stan baterii. "
        f"W tej chwili spośród **{len(ebikes)}** sprawnych e-bike'ów w systemie, **{len(ebikes_ready)}** (**{pct_ready}%**) ma wystarczający zapas energii na przejechanie minimum **{min_range} km**."
    )

# --- STOPKA ---
st.divider()
st.caption("Autor: Sylwester Zaborowski | Źródło danych: Otwarte Dane Systemu Roweru Metropolitalnego MEVO (GBFS) | Projekt portfolio Data Analytics.")

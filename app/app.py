"""
app/app.py
Phase 7 — Streamlit Query Interface

Interactive web app to explore extracted biodiversity entities.
Query by species, behavior, or habitat across all 29,981 observations.

Run:
    cd biodiversity-ner
    streamlit run app/app.py
"""

import json
import pandas as pd
import streamlit as st
from pathlib import Path
from collections import Counter

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Florida Biodiversity NER Explorer",
    page_icon="🦝",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_triplets():
    path = RESULTS / "triplets.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Parse list columns
    for col in ['behaviors', 'habitats']:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
            )
    return df


@st.cache_data
def load_predictions():
    path = RESULTS / "baseline_predictions.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_stats():
    path = RESULTS / "baseline_stats.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/"
                 "INaturalist_logo.svg/200px-INaturalist_logo.svg.png", width=120)
st.sidebar.title("🌿 Biodiversity NER")
st.sidebar.markdown("""
**Florida Mammal Observation Explorer**

NER pipeline over 29,981 iNaturalist observations.
Extracts: `SPECIES` · `BEHAVIOR` · `HABITAT`
""")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🔍 Query Species",
    "📊 Entity Statistics",
    "🗺️ Observation Map",
    "📄 Raw Predictions",
])

# ── Load ──────────────────────────────────────────────────────────────────────
df_triplets   = load_triplets()
predictions   = load_predictions()
stats         = load_stats()

# ── Overview page ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🦝 Florida Biodiversity NER Explorer")
    st.markdown("""
    This application demonstrates a **Named Entity Recognition (NER) pipeline**
    trained on real iNaturalist citizen science observations from Florida (2023–2024).

    The pipeline extracts three entity types from naturalist observation notes:
    - 🐾 **SPECIES** — animal species mentioned (common and scientific names)
    - 🏃 **BEHAVIOR** — observed behaviors and actions
    - 🌿 **HABITAT** — location and habitat types
    """)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Observations", "29,981")
    with col2:
        st.metric("Sentences Processed", "35,015")
    with col3:
        st.metric("Entities Extracted",
                  f"{stats.get('total_entities', 0):,}")
    with col4:
        st.metric("Structured Triplets",
                  f"{stats.get('total_triplets', 0):,}")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🐾 Top Species")
        if stats.get('top_species'):
            sp_df = pd.DataFrame(stats['top_species'], columns=['Species', 'Count'])
            st.dataframe(sp_df.head(10), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("🏃 Top Behaviors")
        if stats.get('top_behaviors'):
            bh_df = pd.DataFrame(stats['top_behaviors'], columns=['Behavior', 'Count'])
            st.dataframe(bh_df.head(10), use_container_width=True, hide_index=True)

    with col3:
        st.subheader("🌿 Top Habitats")
        if stats.get('top_habitats'):
            hb_df = pd.DataFrame(stats['top_habitats'], columns=['Habitat', 'Count'])
            st.dataframe(hb_df.head(10), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Pipeline Architecture")
    st.code("""
iNaturalist CSV (29,981 observations)
        │
        ▼
Preprocessing & sentence tokenization
        │
        ▼
Rule-based NER (SPECIES / BEHAVIOR / HABITAT)
        │
        ▼
Manual annotation → 300 gold-standard sentences
        │
        ▼
BERT fine-tuning (bert-base-cased)
        │
        ▼
Structured triplet database (57,299 entries)
        │
        ▼
Query interface (this app)
    """, language="text")


# ── Query species page ────────────────────────────────────────────────────────
elif page == "🔍 Query Species":
    st.title("🔍 Query by Species")

    if df_triplets.empty:
        st.warning("No triplet data found. Run the pipeline first.")
    else:
        # Get unique species
        all_species = sorted(df_triplets['species'].str.lower().unique())

        query = st.text_input(
            "Enter a species name:",
            placeholder="e.g. raccoon, bobcat, manatee, white-tailed deer",
        )

        if query:
            # Filter
            mask = df_triplets['species'].str.lower().str.contains(
                query.lower(), na=False)
            results = df_triplets[mask]

            if results.empty:
                st.warning(f"No results for '{query}'. Try a different name.")
            else:
                st.success(f"Found **{len(results):,}** observations for '{query}'")

                # Behavior summary
                all_behaviors = [b for behaviors in results['behaviors']
                                 for b in (behaviors if isinstance(behaviors, list)
                                           else [])]
                all_habitats  = [h for habitats  in results['habitats']
                                 for h in (habitats  if isinstance(habitats,  list)
                                           else [])]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Observations", len(results))
                with col2:
                    st.metric("Behaviors recorded", len(set(all_behaviors)))
                with col3:
                    st.metric("Habitats recorded", len(set(all_habitats)))

                tab1, tab2, tab3 = st.tabs(["Behaviors", "Habitats", "Raw Evidence"])

                with tab1:
                    if all_behaviors:
                        beh_counts = Counter(b.lower() for b in all_behaviors)
                        beh_df = pd.DataFrame(beh_counts.most_common(15),
                                              columns=['Behavior', 'Count'])
                        st.bar_chart(beh_df.set_index('Behavior'))
                    else:
                        st.info("No behaviors recorded for this species.")

                with tab2:
                    if all_habitats:
                        hab_counts = Counter(h.lower() for h in all_habitats)
                        hab_df = pd.DataFrame(hab_counts.most_common(15),
                                              columns=['Habitat', 'Count'])
                        st.bar_chart(hab_df.set_index('Habitat'))
                    else:
                        st.info("No habitats recorded for this species.")

                with tab3:
                    st.subheader("Source observations")
                    display_cols = ['species', 'source_text',
                                    'place_guess', 'observation_id']
                    display_cols = [c for c in display_cols if c in results.columns]
                    st.dataframe(
                        results[display_cols].head(50),
                        use_container_width=True,
                        hide_index=True,
                    )

        else:
            # Show all species overview
            st.subheader("All species in dataset")
            sp_counts = df_triplets['species'].str.lower().value_counts()
            sp_df = pd.DataFrame({
                'Species': sp_counts.index,
                'Observations': sp_counts.values,
            }).head(20)
            st.bar_chart(sp_df.set_index('Species'))


# ── Entity statistics page ────────────────────────────────────────────────────
elif page == "📊 Entity Statistics":
    st.title("📊 Entity Statistics")

    if not predictions:
        st.warning("No predictions found. Run the pipeline first.")
    else:
        all_ents   = [e for p in predictions for e in p['entities']]
        sp_ents    = [e for e in all_ents if e['label'] == 'SPECIES']
        beh_ents   = [e for e in all_ents if e['label'] == 'BEHAVIOR']
        hab_ents   = [e for e in all_ents if e['label'] == 'HABITAT']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("SPECIES entities",  f"{len(sp_ents):,}")
        with col2:
            st.metric("BEHAVIOR entities", f"{len(beh_ents):,}")
        with col3:
            st.metric("HABITAT entities",  f"{len(hab_ents):,}")

        tab1, tab2, tab3 = st.tabs(["SPECIES", "BEHAVIOR", "HABITAT"])

        with tab1:
            sp_counts = Counter(e['text'].lower() for e in sp_ents)
            sp_df = pd.DataFrame(sp_counts.most_common(20),
                                 columns=['Species', 'Count'])
            st.bar_chart(sp_df.set_index('Species'))
            st.dataframe(sp_df, use_container_width=True, hide_index=True)

        with tab2:
            bh_counts = Counter(e['text'].lower() for e in beh_ents)
            bh_df = pd.DataFrame(bh_counts.most_common(20),
                                 columns=['Behavior', 'Count'])
            st.bar_chart(bh_df.set_index('Behavior'))
            st.dataframe(bh_df, use_container_width=True, hide_index=True)

        with tab3:
            hb_counts = Counter(e['text'].lower() for e in hab_ents)
            hb_df = pd.DataFrame(hb_counts.most_common(20),
                                 columns=['Habitat', 'Count'])
            st.bar_chart(hb_df.set_index('Habitat'))
            st.dataframe(hb_df, use_container_width=True, hide_index=True)


# ── Map page ──────────────────────────────────────────────────────────────────
elif page == "🗺️ Observation Map":
    st.title("🗺️ Observation Map")
    st.markdown("Geographic distribution of mammal observations across Florida.")

    if df_triplets.empty:
        st.warning("No triplet data found.")
    else:
        map_df = df_triplets[
            df_triplets['latitude'].notna() &
            df_triplets['longitude'].notna()
        ][['latitude', 'longitude', 'species', 'place_guess']].copy()

        map_df = map_df.rename(columns={
            'latitude': 'lat', 'longitude': 'lon'})

        # Optional species filter
        species_options = ['All'] + sorted(
            df_triplets['species'].str.lower().unique().tolist())
        selected_species = st.selectbox("Filter by species:", species_options)

        if selected_species != 'All':
            mask   = df_triplets['species'].str.lower() == selected_species
            map_df = df_triplets[mask & df_triplets['latitude'].notna()][
                ['latitude','longitude','species','place_guess']].rename(
                columns={'latitude':'lat','longitude':'lon'})

        st.metric("Observations on map", len(map_df))

        if not map_df.empty:
            st.map(map_df[['lat', 'lon']], zoom=6)
        else:
            st.info("No coordinates available for this filter.")


# ── Raw predictions page ──────────────────────────────────────────────────────
elif page == "📄 Raw Predictions":
    st.title("📄 Raw NER Predictions")
    st.markdown("Browse the raw sentence-level NER output.")

    if not predictions:
        st.warning("No predictions found. Run the pipeline first.")
    else:
        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            min_entities = st.slider("Minimum entities per sentence", 1, 10, 2)
        with col2:
            entity_filter = st.multiselect(
                "Show sentences with entity type:",
                ["SPECIES", "BEHAVIOR", "HABITAT"],
                default=["SPECIES", "BEHAVIOR"],
            )

        # Filter predictions
        filtered = [
            p for p in predictions
            if p['n_entities'] >= min_entities
            and any(e['label'] in entity_filter for e in p['entities'])
        ]

        st.info(f"Showing {min(50, len(filtered))} of {len(filtered):,} matching sentences")

        for pred in filtered[:50]:
            with st.expander(f"[{pred.get('common_name','')}] "
                             f"{pred['text'][:80]}..."):
                st.markdown(f"**Full text:** {pred['text']}")
                st.markdown(f"**Place:** {pred.get('place_guess','')}")
                st.markdown("**Entities:**")
                for ent in pred['entities']:
                    color = {"SPECIES":"🟢","BEHAVIOR":"🟣","HABITAT":"🟠"}.get(
                        ent['label'], "⚪")
                    st.markdown(f"  {color} `[{ent['label']}]` **{ent['text']}**")


# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("""
**Data:** iNaturalist Florida Mammals 2023–2024  
**Model:** Rule-based NER + BERT fine-tuned  
**Entities:** SPECIES · BEHAVIOR · HABITAT  
""")

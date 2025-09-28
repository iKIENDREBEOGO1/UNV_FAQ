# app_streamlit.py — UI améliorée avec analytics et tableau de bord
import yaml
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import json
from pathlib import Path
from collections import Counter

from src.loader import load_faq, load_json
from src.ner import RegexNER
from src.retriever import Retriever
from src.intent import classify_intent
from src.templates import TemplateManager

st.set_page_config(page_title="UV-BF FAQ Chatbot", page_icon="🎓", layout="wide")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "chatbot_analytics.db"

# --------- CSS amélioré avec design professionnel ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --primary-green: #1FAA4B;
  --dark-green: #168A3C;
  --light-green: #E8F5E8;
  --text-dark: #1a202c;
  --text-medium: #4a5568;
  --text-light: #718096;
  --bg-main: #fafafa;
  --bg-white: #ffffff;
  --border-light: #e2e8f0;
  --shadow-soft: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-card: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

* {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg-main);
  color: var(--text-dark);
}

/* Navigation stylée */
.nav-container {
  background: var(--bg-white);
  padding: 20px 24px;
  margin-bottom: 32px;
  border-bottom: 1px solid var(--border-light);
  box-shadow: var(--shadow-soft);
}

.custom-tabs {
  display: flex;
  justify-content: center;
  gap: 8px;
  margin: 0 auto;
  max-width: 400px;
}

.tab-button {
  flex: 1;
  padding: 12px 20px;
  border: 2px solid var(--border-light);
  border-radius: 25px;
  background: var(--bg-white);
  color: var(--text-medium);
  font-weight: 500;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.3s ease;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.tab-button:hover {
  border-color: var(--primary-green);
  background: var(--light-green);
  transform: translateY(-1px);
}

.tab-button.active {
  background: linear-gradient(135deg, var(--primary-green) 0%, var(--dark-green) 100%);
  border-color: var(--primary-green);
  color: white;
  box-shadow: 0 4px 12px rgba(31, 170, 75, 0.3);
}

.tab-icon {
  font-size: 16px;
}

/* Masquer les radio buttons par défaut de Streamlit */
.stRadio > div {
  display: none !important;
}

/* Titre principal */
.main-title {
  text-align: center;
  margin: 32px auto 40px auto;
  max-width: 800px;
}

.main-title h1 {
  font-size: 48px;
  font-weight: 700;
  color: var(--text-dark);
  margin-bottom: 12px;
  background: linear-gradient(135deg, var(--text-dark) 0%, var(--primary-green) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.main-title .subtitle {
  font-size: 18px;
  color: var(--text-medium);
  font-weight: 400;
  margin-bottom: 8px;
}

.title-accent {
  width: 60px;
  height: 4px;
  background: linear-gradient(90deg, var(--primary-green), var(--dark-green));
  margin: 16px auto 0 auto;
  border-radius: 2px;
}

/* Zone de saisie simplifiée */
.chat-container {
  max-width: 800px;
  margin: 0 auto 40px auto;
  padding: 0 20px;
}

/* Style des inputs Streamlit */
.stTextInput > div > div > input {
  font-size: 16px !important;
  padding: 12px 16px !important;
  border: 2px solid var(--border-light) !important;
  border-radius: 12px !important;
  background: var(--bg-white) !important;
  transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus {
  border-color: var(--primary-green) !important;
  box-shadow: 0 0 0 3px rgba(31, 170, 75, 0.1) !important;
  outline: none !important;
}

.stButton > button {
  background: linear-gradient(135deg, var(--primary-green) 0%, var(--dark-green) 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 12px 32px !important;
  font-weight: 600 !important;
  font-size: 16px !important;
  transition: all 0.3s ease !important;
  box-shadow: var(--shadow-soft) !important;
}

.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 15px -3px rgba(31, 170, 75, 0.3) !important;
}

/* Bulles de conversation */
.message-bubble {
  margin: 16px 0;
  padding: 18px 24px;
  border-radius: 18px;
  box-shadow: var(--shadow-soft);
  animation: fadeInUp 0.3s ease;
}

.message-bubble.user {
  background: linear-gradient(135deg, var(--primary-green) 0%, var(--dark-green) 100%);
  color: white;
  margin-left: 80px;
  border-bottom-right-radius: 6px;
}

.message-bubble.bot {
  background: var(--bg-white);
  color: var(--text-dark);
  margin-right: 80px;
  border: 1px solid var(--border-light);
  border-bottom-left-radius: 6px;
}

.message-header {
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 8px;
  opacity: 0.9;
}

.message-text {
  line-height: 1.6;
  font-size: 15px;
}

/* Système de feedback */
.feedback-container {
  display: flex;
  gap: 8px;
  margin-top: 12px;
  justify-content: flex-start;
}

.feedback-btn {
  background: transparent;
  border: 1px solid var(--border-light);
  border-radius: 20px;
  padding: 6px 12px;
  cursor: pointer;
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 4px;
  transition: all 0.2s ease;
  color: var(--text-medium);
}

.feedback-btn:hover {
  background: var(--light-green);
  border-color: var(--primary-green);
}

.feedback-btn.active-like {
  background: var(--primary-green);
  border-color: var(--primary-green);
  color: white;
}

.feedback-btn.active-dislike {
  background: #dc3545;
  border-color: #dc3545;
  color: white;
}

/* Analytics Cards */
.metric-card {
  background: var(--bg-white);
  padding: 20px;
  border-radius: 12px;
  box-shadow: var(--shadow-soft);
  border: 1px solid var(--border-light);
}

/* Animations */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Responsive */
@media (max-width: 768px) {
  .main-title h1 {
    font-size: 36px;
  }
  
  .message-bubble.user {
    margin-left: 20px;
  }
  
  .message-bubble.bot {
    margin-right: 20px;
  }
}

/* Masquer les éléments Streamlit indésirables */
.stDeployButton {display: none;}
header[data-testid="stHeader"] {display: none;}
.stMainBlockContainer {padding-top: 0;}
</style>
""", unsafe_allow_html=True)

# --------- Fonctions Analytics ----------
def init_analytics_db():
    """Initialise la base de données analytics"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            response TEXT,
            entities TEXT,
            intent TEXT,
            confidence_score REAL,
            feedback TEXT,
            session_id TEXT,
            response_time REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_interaction(query, response, entities, intent, confidence_score, response_time):
    """Enregistre une interaction dans la base de données"""
    conn = sqlite3.connect(DB_PATH)
    session_id = st.session_state.get("session_id", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    conn.execute('''
        INSERT INTO interactions (timestamp, query, response, entities, intent, confidence_score, session_id, response_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        query,
        response,
        json.dumps(entities),
        intent,
        confidence_score,
        session_id,
        response_time
    ))
    conn.commit()
    conn.close()

def update_feedback(interaction_id, feedback_type):
    """Met à jour le feedback d'une interaction"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE interactions SET feedback = ? WHERE id = ?', (feedback_type, interaction_id))
    conn.commit()
    conn.close()

def get_analytics_data():
    """Récupère les données analytics de la base"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM interactions ORDER BY timestamp DESC', conn)
    conn.close()
    return df

def calculate_metrics(df):
    """Calcule les métriques principales"""
    if df.empty:
        return {
            "total_questions": 0,
            "satisfaction_rate": 0,
            "avg_response_time": 0,
            "unique_sessions": 0
        }
    
    total_questions = len(df)
    
    # Taux de satisfaction
    feedback_df = df[df['feedback'].notna()]
    satisfaction_rate = 0
    if not feedback_df.empty:
        likes = len(feedback_df[feedback_df['feedback'] == 'like'])
        total_feedback = len(feedback_df)
        satisfaction_rate = (likes / total_feedback) * 100 if total_feedback > 0 else 0
    
    # Temps de réponse moyen
    avg_response_time = df['response_time'].mean() if 'response_time' in df.columns else 0
    
    # Sessions uniques
    unique_sessions = df['session_id'].nunique()
    
    return {
        "total_questions": total_questions,
        "satisfaction_rate": satisfaction_rate,
        "avg_response_time": avg_response_time,
        "unique_sessions": unique_sessions
    }

# --------- Configuration et pipeline ----------
init_analytics_db()

with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

faq = load_faq(cfg["data"]["faq_csv"])
ner_schema = load_json(cfg["data"]["ner_json"])
templates = load_json(cfg["data"]["templates_json"])

retr_cfg = cfg.get("retriever", {})
nr = retr_cfg.get("ngram_range", (1,2))
if isinstance(nr, list): nr = (int(nr[0]), int(nr[1]))
top_k = int(retr_cfg.get("top_k", 3))

ner = RegexNER(ner_schema)
retr = Retriever(faq["index_text"], ngram_range=nr, min_df=retr_cfg.get("min_df", 1), max_df=retr_cfg.get("max_df", 0.95))
tm = TemplateManager(templates)

def answer(query: str) -> tuple:
    """Retourne la réponse et les métadonnées pour analytics (retrieval-first)"""
    start_time = datetime.now()

    # 1) Extraction d'entités (NER)
    ents = ner.extract(query)

    # 2) Recherche FAQ (retriever)
    hits = retr.search(query, top_k=top_k)

    if not hits:
        response = "Désolé, je n'ai pas trouvé d'information pertinente."
        intent_final = "info_generale_uvbf"   # catégorie par défaut si rien trouvé
        confidence_score = 0.0
    else:
        # meilleur candidat
        idx, score = hits[0]
        row = faq.iloc[idx]

        # Catégorie vraie issue du CSV (sera utilisée comme 'intent' pour l'analytics et les templates)
        intent_final = str(row.get("categorie", "")).strip() or "info_generale_uvbf"
        confidence_score = float(score)

        # 3) Option templates : on tente un rendu avec la catégorie trouvée
        #    Si pas de template ou info manquante -> on renvoie la réponse CSV
        rendered = tm.render(intent_final, ents)
        if rendered and (not rendered.get("need_more_info")) and rendered.get("text"):
            response = rendered["text"]
        else:
            # Réponse brute de la FAQ (CSV simplifié)
            response = f"**{row['question']}**\n\n{row['reponse']}"

    # 4) Logging
    response_time = (datetime.now() - start_time).total_seconds()
    log_interaction(query, response, ents, intent_final, confidence_score, response_time)

    return response, ents, intent_final, confidence_score


def handle_feedback(interaction_id, feedback_type):
    """Gère le feedback et met à jour la base de données"""
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    
    current_feedback = st.session_state.feedback.get(interaction_id)
    if current_feedback == feedback_type:
        del st.session_state.feedback[interaction_id]
        update_feedback(interaction_id, None)
    else:
        st.session_state.feedback[interaction_id] = feedback_type
        update_feedback(interaction_id, feedback_type)

# --------- Interface utilisateur avec navigation améliorée ----------

# Navigation avec onglets personnalisés fonctionnels
st.markdown('<div class="nav-container">', unsafe_allow_html=True)

# Initialisation de l'état de navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "chatbot"

# Création des boutons d'onglets
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    subcol1, subcol2 = st.columns(2)
    
    with subcol1:
        if st.button(
            "💬 Chatbot",
            key="tab_chatbot",
            use_container_width=True,
            type="primary" if st.session_state.current_page == "chatbot" else "secondary"
        ):
            st.session_state.current_page = "chatbot"
            st.rerun()
    
    with subcol2:
        if st.button(
            "📊 Analytics", 
            key="tab_analytics",
            use_container_width=True,
            type="primary" if st.session_state.current_page == "analytics" else "secondary"
        ):
            st.session_state.current_page = "analytics"
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Définir la page courante
page = "💬 Chatbot" if st.session_state.current_page == "chatbot" else "📊 Tableau de bord"

# Initialisation des states
if "history" not in st.session_state: 
    st.session_state.history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "interaction_ids" not in st.session_state:
    st.session_state.interaction_ids = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# --------- PAGE CHATBOT ----------
if page == "💬 Chatbot":
    # Titre principal
    st.markdown("""
    <div class="main-title">
        <h1>🎓 UV-BF FAQ Chatbot</h1>
        <div class="subtitle">Posez votre question et obtenez une réponse instantanée</div>
        <div class="title-accent"></div>
    </div>
    """, unsafe_allow_html=True)

    # Zone de chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Formulaire de saisie
    with st.form("chat", clear_on_submit=True):
        query = st.text_input(
            label="Votre question",
            placeholder="Ex: Frais Licence S2 ? • Accès plateforme • Mot de passe oublié",
            label_visibility="collapsed"
        )
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            submitted = st.form_submit_button("✨ Envoyer", use_container_width=True)
        
        if submitted and query.strip():
            response, entities, intent, confidence = answer(query.strip())
            
            # Récupérer l'ID de la dernière interaction
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.execute('SELECT id FROM interactions ORDER BY id DESC LIMIT 1')
            interaction_id = cursor.fetchone()[0]
            conn.close()
            
            st.session_state.history.append({
                "role": "user", 
                "text": query.strip(),
                "interaction_id": None
            })
            st.session_state.history.append({
                "role": "bot", 
                "text": response,
                "entities": entities,
                "intent": intent,
                "confidence": confidence,
                "interaction_id": interaction_id
            })
            st.rerun()

    # Affichage de l'historique
    for message_idx, message in enumerate(st.session_state.history):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="message-bubble user">
                <div class="message-header">Vous</div>
                <div class="message-text">{message["text"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-bubble bot">
                <div class="message-header">Assistant UV-BF</div>
                <div class="message-text">{message["text"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Boutons de feedback
            if message.get("interaction_id"):
                interaction_id = message["interaction_id"]
                current_feedback = st.session_state.feedback.get(interaction_id, None)
                
                col1, col2, col3 = st.columns([1, 1, 10])
                
                with col1:
                    if st.button("👍", key=f"like_{interaction_id}", help="Réponse utile"):
                        handle_feedback(interaction_id, "like")
                        st.rerun()
                
                with col2:
                    if st.button("👎", key=f"dislike_{interaction_id}", help="Réponse non utile"):
                        handle_feedback(interaction_id, "dislike")
                        st.rerun()
                
              

    st.markdown('</div>', unsafe_allow_html=True)

# --------- PAGE TABLEAU DE BORD ----------
elif page == "📊 Tableau de bord":
    st.markdown("""
    <div class="main-title">
        <h1>📊 Analytics Dashboard</h1>
        <div class="subtitle">Analyse des performances du chatbot</div>
        <div class="title-accent"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Récupération des données
    df = get_analytics_data()
    metrics = calculate_metrics(df)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Questions totales", 
            metrics["total_questions"],
            help="Nombre total de questions posées"
        )
    
    with col2:
        st.metric(
            "Taux de satisfaction", 
            f"{metrics['satisfaction_rate']:.1f}%",
            help="Pourcentage de likes par rapport aux feedbacks"
        )
    
    with col3:
        st.metric(
            "Temps de réponse moyen", 
            f"{metrics['avg_response_time']:.2f}s",
            help="Temps moyen de traitement des questions"
        )
    
    with col4:
        st.metric(
            "Sessions uniques", 
            metrics["unique_sessions"],
            help="Nombre de sessions utilisateurs distinctes"
        )
    
    if not df.empty:
        st.markdown("---")
        
        # Graphiques en deux colonnes
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Volume de questions")
            # Questions par jour
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_stats = df.groupby('date').size().reset_index(name='questions')
            
            fig_volume = px.line(
                daily_stats, 
                x='date', 
                y='questions',
                title="Questions par jour",
                color_discrete_sequence=['#1FAA4B']
            )
            fig_volume.update_layout(showlegend=False)
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Distribution des intentions")
            intent_counts = df['intent'].value_counts()
            
            fig_intents = px.pie(
                values=intent_counts.values, 
                names=intent_counts.index,
                title="Types de questions les plus fréquents"
            )
            st.plotly_chart(fig_intents, use_container_width=True)
        
        # Section entités
        st.subheader("🏷️ Entités les plus extraites")
        all_entities = []
        for entities_str in df['entities'].dropna():
            try:
                entities_dict = json.loads(entities_str)
                for entity_type, values in entities_dict.items():
                    all_entities.extend(values)
            except:
                continue
        
        if all_entities:
            entity_counts = Counter(all_entities).most_common(10)
            entities_df = pd.DataFrame(entity_counts, columns=['Entité', 'Fréquence'])
            
            fig_entities = px.bar(
                entities_df, 
                x='Fréquence', 
                y='Entité',
                orientation='h',
                title="Top 10 des entités mentionnées",
                color_discrete_sequence=['#1FAA4B']
            )
            st.plotly_chart(fig_entities, use_container_width=True)
        
        # Tableau des dernières interactions
        st.subheader("💬 Dernières interactions")
        recent_df = df.head(10)[['timestamp', 'query', 'intent', 'confidence_score', 'feedback']]
        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%d/%m/%Y %H:%M')
        recent_df = recent_df.rename(columns={
            'timestamp': 'Date/Heure',
            'query': 'Question',
            'intent': 'Intention',
            'confidence_score': 'Confiance',
            'feedback': 'Feedback'
        })
        st.dataframe(recent_df, use_container_width=True)
        
        # Section export
        st.markdown("---")
        st.subheader("📥 Export des données")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Télécharger CSV",
                data=csv,
                file_name=f"analytics_uvbf_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.button("🗑️ Réinitialiser les données"):
                conn = sqlite3.connect(DB_PATH)
                conn.execute('DELETE FROM interactions')
                conn.commit()
                conn.close()
                st.success("Données supprimées avec succès!")
                st.rerun()
    
    else:
        st.info("🤖 Aucune donnée disponible. Utilisez d'abord le chatbot pour générer des analytics!")

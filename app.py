"""
KnowMap with AI - Enterprise Knowledge Graph and Semantic Intelligence Platform
Production-Ready Flask Backend with SQLite Persistence, JWT Authentication,
spaCy NLP Triple Extraction, Sentence-Transformers Semantic Search, and Vis.js Graph Visualization.
"""

import os
import sys
import io
import csv
import time
import json
import html
import sqlite3
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import Flask, request, jsonify, render_template_string, redirect, url_for, Response
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import numpy as np

# -------------------- CONFIGURATION --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(BASE_DIR, "users.db")
DB_PATH = os.environ.get("DATABASE_PATH", DEFAULT_DB)

# Ensure database directory exists if a subpath is given
db_dir = os.path.dirname(DB_PATH)
if db_dir and not os.path.exists(db_dir):
    os.makedirs(db_dir, exist_ok=True)

SECRET_KEY = os.environ.get("SECRET_KEY", "knowmap-ai-secure-secret-key-2026")
JWT_ALGO = "HS256"
JWT_EXP_HOURS = 24

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
CORS(app)

# -------------------- AI / NLP MODELS (LAZY LOADED) --------------------
ENABLE_AI_MODELS = os.environ.get("ENABLE_AI_MODELS", "true").lower() != "false"
_nlp = None
_embed_model = None

def get_nlp():
    global _nlp
    if not ENABLE_AI_MODELS:
        return None
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"[WARN] Failed to load spaCy model: {e}")
            _nlp = False
    return _nlp if _nlp is not False else None

def get_embed_model():
    global _embed_model
    if not ENABLE_AI_MODELS:
        return None
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[WARN] Failed to load SentenceTransformer: {e}")
            _embed_model = False
    return _embed_model if _embed_model is not False else None


# -------------------- DATABASE MANAGEMENT --------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()

    # 1. Users Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT,
        password_hash TEXT NOT NULL,
        created_at TEXT
    )""")

    # 2. Knowledge Graph Triples Table (Persistent Storage)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS triples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT NOT NULL,
        relation TEXT NOT NULL,
        object TEXT NOT NULL,
        source TEXT DEFAULT 'upload',
        created_at TEXT,
        UNIQUE(subject, relation, object)
    )""")

    # 3. Feedback Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        rating INTEGER NOT NULL,
        comments TEXT,
        timestamp TEXT
    )""")

    # 4. Pipeline Metrics Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pipeline_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        day_index INTEGER NOT NULL,
        count INTEGER DEFAULT 0,
        recorded_at TEXT
    )""")

    conn.commit()

    # Auto-seed sample triples if empty
    cur.execute("SELECT COUNT(*) FROM triples")
    if cur.fetchone()[0] == 0:
        csv_sample_path = os.path.join(BASE_DIR, "semanticapp_sample_triples.csv")
        if os.path.exists(csv_sample_path):
            try:
                with open(csv_sample_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)  # Skip header
                    now_str = datetime.now(timezone.utc).isoformat()
                    for row in reader:
                        if len(row) >= 3:
                            s, r, o = row[0].strip(), row[1].strip(), row[2].strip()
                            if s and r and o and s.lower() != "subject":
                                cur.execute("""
                                INSERT OR IGNORE INTO triples (subject, relation, object, source, created_at)
                                VALUES (?, ?, ?, 'sample_seed', ?)
                                """, (s, r, o, now_str))
                conn.commit()
                print("[INFO] Seeded database with sample knowledge graph triples.")
            except Exception as e:
                print(f"[WARN] Failed to seed sample triples: {e}")

    conn.close()

# Initialize DB on module import
init_db()


# -------------------- AUTHENTICATION HELPERS --------------------
def create_token(username):
    payload = {
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXP_HOURS),
        "iat": datetime.now(timezone.utc)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGO)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
        elif "x-access-token" in request.headers:
            token = request.headers["x-access-token"].strip()

        if not token:
            return jsonify({"error": "Authentication token missing. Please log in."}), 401

        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGO])
            current_user = data.get("username")
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Session token expired. Please log in again."}), 401
        except Exception:
            return jsonify({"error": "Invalid authentication token. Please log in again."}), 401

        return f(current_user, *args, **kwargs)
    return decorated

def get_optional_user():
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGO])
            return data.get("username", "Anonymous")
        except Exception:
            pass
    return "Anonymous"

def bump_processing(n=1):
    weekday = datetime.now(timezone.utc).weekday()
    now_str = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO pipeline_metrics (day_index, count, recorded_at)
    VALUES (?, ?, ?)
    """, (weekday, n, now_str))
    conn.commit()
    conn.close()


# -------------------- NLP & GRAPH LOGIC --------------------
def extract_triples_from_text(text):
    """
    Extracts Subject-Predicate-Object (SPO) triples from natural language.
    Uses spaCy dependency parsing with graceful heuristic fallback.
    """
    nlp = get_nlp()
    triples = []
    text = text.strip()
    if not text:
        return triples

    if nlp is not None:
        doc = nlp(text)
        for sent in doc.sents:
            verbs = [tok for tok in sent if tok.pos_ in ("VERB", "AUX") or tok.dep_ == "ROOT"]
            for verb in verbs:
                subjects = [tok for tok in verb.lefts if tok.dep_ in ("nsubj", "nsubjpass", "csubj")]
                objects = [tok for tok in verb.rights if tok.dep_ in ("dobj", "attr", "obj", "acomp")]
                prep_objects = []
                for prep in [tok for tok in verb.rights if tok.dep_ == "prep"]:
                    for pobj in [tok for tok in prep.rights if tok.dep_ == "pobj"]:
                        prep_objects.append((prep.text, pobj))

                def get_phrase(tok):
                    words = [t.text for t in tok.subtree if t.dep_ not in ("punct", "relcl") and not t.is_punct]
                    res = " ".join(words).strip()
                    return res if res else tok.text

                for s in subjects:
                    s_text = get_phrase(s)
                    v_text = verb.lemma_ if verb.lemma_ not in ("be", "have") else verb.text
                    for o in objects:
                        o_text = get_phrase(o)
                        if s_text and o_text and s_text.lower() != o_text.lower():
                            triples.append({"subject": s_text, "relation": v_text, "object": o_text})
                    for prep_w, po in prep_objects:
                        po_text = get_phrase(po)
                        if s_text and po_text:
                            triples.append({"subject": s_text, "relation": f"{v_text} {prep_w}", "object": po_text})

        # Fallback if dependency parse produced no triples
        if not triples:
            noun_chunks = [c.text.strip() for c in doc.noun_chunks if c.text.strip()]
            root_verb = doc[0].sent.root.lemma_ if doc[0].sent.root.pos_ in ("VERB", "AUX") else "relates_to"
            if len(noun_chunks) >= 2:
                for i in range(len(noun_chunks) - 1):
                    triples.append({
                        "subject": noun_chunks[i],
                        "relation": root_verb,
                        "object": noun_chunks[i + 1]
                    })
    else:
        # Resilient rule-based extractor
        words = text.split()
        if len(words) >= 3:
            triples.append({
                "subject": words[0],
                "relation": "related_to",
                "object": " ".join(words[1:])
            })

    return triples


def lookup_graph_by_keyword(text):
    """
    When NLP extraction returns 0 triples (e.g. input is a short noun phrase
    like 'Genetic medicine'), fall back to searching existing database triples
    whose subject or object text contains the keywords.
    Returns matched triples list and a flag indicating it was a keyword lookup.
    """
    keywords = [w.strip().lower() for w in text.replace(",", " ").split() if len(w.strip()) > 2]
    if not keywords:
        return [], False

    conn = get_db()
    cur = conn.cursor()

    # Build a LIKE query for each keyword against both subject and object columns
    conditions = []
    params = []
    for kw in keywords:
        conditions.append("(LOWER(subject) LIKE ? OR LOWER(object) LIKE ? OR LOWER(relation) LIKE ?)")
        params.extend([f"%{kw}%", f"%{kw}%", f"%{kw}%"])

    where_clause = " OR ".join(conditions)
    cur.execute(f"""
        SELECT subject, relation, object FROM triples
        WHERE {where_clause}
        ORDER BY id DESC LIMIT 40
    """, params)
    rows = cur.fetchall()
    conn.close()

    triples = [{"subject": r[0], "relation": r[1], "object": r[2]} for r in rows]
    return triples, len(triples) > 0


# -------------------- PROFESSIONAL DESIGN SYSTEM (HTML / CSS / JS) --------------------
BASE_HEAD = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KnowMap AI - Enterprise Knowledge Intelligence</title>
  
  <!-- Modern Typography -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  
  <!-- Bootstrap 5.3 & Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
  
  <!-- Vis.js Network 9.1.2 -->
  <link href="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.css" rel="stylesheet">

  <style>
    :root {
      --primary: #4f46e5;
      --primary-hover: #4338ca;
      --primary-light: #eef2ff;
      --primary-glow: rgba(79, 70, 229, 0.15);
      --success: #10b981;
      --success-light: #ecfdf5;
      --warning: #f59e0b;
      --info: #0284c7;
      --slate-900: #0f172a;
      --slate-800: #1e293b;
      --slate-700: #334155;
      --slate-600: #475569;
      --slate-500: #64748b;
      --slate-400: #94a3b8;
      --slate-200: #e2e8f0;
      --slate-100: #f1f5f9;
      --slate-50: #f8fafc;
    }

    body {
      background-color: var(--slate-50);
      color: var(--slate-800);
      font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      -webkit-font-smoothing: antialiased;
    }

    .font-mono {
      font-family: 'JetBrains Mono', monospace;
    }

    /* Glassmorphic Navbar */
    .navbar-custom {
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--slate-200);
      transition: all 0.2s ease;
    }

    .brand-badge {
      background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
      color: white;
      width: 38px;
      height: 38px;
      border-radius: 10px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      box-shadow: 0 4px 12px rgba(79, 70, 229, 0.28);
    }

    .nav-link-custom {
      font-weight: 500;
      color: var(--slate-600);
      padding: 8px 14px;
      border-radius: 8px;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      text-decoration: none;
    }
    .nav-link-custom:hover {
      color: var(--primary);
      background-color: var(--primary-light);
    }
    .nav-link-custom.active {
      color: var(--primary);
      background-color: var(--primary-light);
      font-weight: 600;
    }

    /* Cards */
    .card-custom {
      background: #ffffff;
      border: 1px solid var(--slate-200);
      border-radius: 16px;
      box-shadow: 0 2px 4px rgba(15, 23, 42, 0.02), 0 8px 16px rgba(15, 23, 42, 0.03);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card-custom-hover:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 12px rgba(15, 23, 42, 0.04), 0 16px 28px rgba(15, 23, 42, 0.06);
    }

    /* Graph Canvas & Controls */
    .graph-wrapper {
      position: relative;
      background-color: #ffffff;
      background-image: radial-gradient(var(--slate-200) 1px, transparent 1px);
      background-size: 20px 20px;
      border-radius: 14px;
      border: 1px solid var(--slate-200);
      overflow: hidden;
    }

    #graph {
      height: 500px;
      width: 100%;
    }

    .graph-toolbar {
      position: absolute;
      top: 14px;
      right: 14px;
      z-index: 10;
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(8px);
      border: 1px solid var(--slate-200);
      border-radius: 10px;
      padding: 4px;
      display: flex;
      gap: 4px;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
    }

    .graph-btn {
      background: transparent;
      border: none;
      width: 32px;
      height: 32px;
      border-radius: 6px;
      color: var(--slate-600);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .graph-btn:hover {
      background: var(--slate-100);
      color: var(--primary);
    }

    /* Live Pulse Status */
    .status-dot {
      width: 8px;
      height: 8px;
      background-color: #10b981;
      border-radius: 50%;
      display: inline-block;
      box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
      animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
      0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
      70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
      100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Dropzone */
    .dropzone-box {
      border: 2px dashed #cbd5e1;
      background-color: #f8fafc;
      border-radius: 12px;
      padding: 30px 16px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .dropzone-box:hover, .dropzone-box.dragover {
      border-color: var(--primary);
      background-color: var(--primary-light);
    }

    /* Star Rating */
    .star-btn {
      cursor: pointer;
      font-size: 28px;
      color: #cbd5e1;
      transition: all 0.15s ease;
    }
    .star-btn:hover, .star-btn.active {
      color: #f59e0b;
      transform: scale(1.1);
    }

    /* Prompt Pills */
    .prompt-pill {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 5px 12px;
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 500;
      color: var(--slate-600);
      cursor: pointer;
      transition: all 0.15s ease;
      user-select: none;
    }
    .prompt-pill:hover {
      background: var(--primary-light);
      color: var(--primary);
      border-color: #c7d2fe;
    }

    /* Toast Container */
    #toastContainer {
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 1055;
    }

    .btn-primary-custom {
      background-color: var(--primary);
      border-color: var(--primary);
      color: #ffffff;
      font-weight: 600;
      border-radius: 10px;
      padding: 9px 18px;
      box-shadow: 0 2px 4px rgba(79, 70, 229, 0.2);
      transition: all 0.2s ease;
    }
    .btn-primary-custom:hover {
      background-color: var(--primary-hover);
      border-color: var(--primary-hover);
      box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3);
      color: #ffffff;
    }
  </style>
</head>
<body>

<div id="toastContainer"></div>
"""

NAVBAR = """
<nav class="navbar navbar-expand-lg navbar-custom sticky-top">
  <div class="container">
    <a class="navbar-brand d-flex align-items-center gap-2 fw-bold text-dark text-decoration-none" href="/">
      <span class="brand-badge"><i class="bi bi-diagram-3-fill"></i></span>
      <div>
        <span class="fs-5 tracking-tight">KnowMap</span>
        <span class="badge bg-indigo-subtle text-primary border border-primary-subtle rounded-pill ms-1 font-mono" style="font-size: 10px;">AI 2.0</span>
      </div>
    </a>
    
    <button class="navbar-toggler border-0" type="button" data-bs-toggle="collapse" data-bs-target="#navContent">
      <span class="navbar-toggler-icon"></span>
    </button>
    
    <div class="collapse navbar-collapse" id="navContent">
      <ul class="navbar-nav me-auto mb-2 mb-lg-0 ms-lg-3 gap-1">
        <li class="nav-item">
          <a class="nav-link-custom" href="/upload"><i class="bi bi-cloud-arrow-up"></i> Ingest Data</a>
        </li>
        <li class="nav-item">
          <a class="nav-link-custom" href="/semantic"><i class="bi bi-search"></i> Semantic Search</a>
        </li>
        <li class="nav-item">
          <a class="nav-link-custom" href="/dashboard"><i class="bi bi-speedometer2"></i> Analytics</a>
        </li>
      </ul>

      <div class="d-flex align-items-center gap-3">
        <div class="d-none d-xl-flex align-items-center gap-2 px-3 py-1 bg-light rounded-pill border small text-muted">
          <span class="status-dot"></span>
          <span>Core Engine Online</span>
        </div>

        <div id="navUser" class="d-flex align-items-center gap-2">
          <a class="btn btn-sm btn-outline-secondary px-3" href="/login">Login</a>
          <a class="btn btn-sm btn-primary-custom px-3" href="/login#signup">Sign Up</a>
        </div>
      </div>
    </div>
  </div>
</nav>

<script>
function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

function showToast(message, type = 'success') {
  const toastContainer = document.getElementById('toastContainer');
  if (!toastContainer) return;
  const id = 'toast_' + Date.now();
  const icon = type === 'success' ? 'bi-check-circle-fill text-success' : (type === 'danger' ? 'bi-exclamation-triangle-fill text-danger' : 'bi-info-circle-fill text-primary');
  const html = `
    <div id="${id}" class="toast align-items-center show shadow-lg border mb-2" role="alert" style="background:#ffffff; border-radius:12px; min-width: 290px;">
      <div class="d-flex p-3 align-items-center">
        <i class="bi ${icon} fs-5 me-2"></i>
        <div class="toast-body p-0 flex-grow-1 small fw-semibold text-dark">${escapeHtml(message)}</div>
        <button type="button" class="btn-close ms-2" onclick="document.getElementById('${id}').remove()"></button>
      </div>
    </div>`;
  toastContainer.insertAdjacentHTML('beforeend', html);
  setTimeout(() => {
    const el = document.getElementById(id);
    if (el) el.remove();
  }, 4500);
}

function decodeJWT(token) {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    return JSON.parse(decodeURIComponent(escape(window.atob(base64))));
  } catch (e) {
    return null;
  }
}

function updateNavbar() {
  const token = localStorage.getItem('jwt');
  const navUser = document.getElementById('navUser');
  if (token) {
    const data = decodeJWT(token);
    if (data && data.exp && data.exp * 1000 > Date.now()) {
      const username = data.username || 'User';
      const initial = username.charAt(0).toUpperCase();
      navUser.innerHTML = `
        <div class="d-flex align-items-center gap-2">
          <div class="rounded-circle bg-primary text-white d-flex align-items-center justify-content-center fw-bold" style="width:34px;height:34px;font-size:14px;">
            ${initial}
          </div>
          <span class="fw-semibold small d-none d-sm-inline">${escapeHtml(username)}</span>
          <button class="btn btn-sm btn-outline-danger ms-2 px-2 py-1" onclick="logout()" title="Logout">
            <i class="bi bi-box-arrow-right"></i>
          </button>
        </div>
      `;
      return;
    } else {
      localStorage.removeItem('jwt');
    }
  }
  navUser.innerHTML = `
    <a class="btn btn-sm btn-outline-secondary px-3" href="/login">Login</a>
    <a class="btn btn-sm btn-primary-custom px-3" href="/login">Sign Up</a>
  `;
}

function logout() {
  localStorage.removeItem('jwt');
  showToast('Logged out successfully', 'info');
  setTimeout(() => { window.location.href = '/login'; }, 600);
}

function authFetch(url, options = {}) {
  options.headers = options.headers || {};
  const token = localStorage.getItem('jwt');
  if (token) {
    if (options.headers instanceof Headers) {
      options.headers.set('Authorization', 'Bearer ' + token);
    } else {
      options.headers['Authorization'] = 'Bearer ' + token;
    }
  }
  return fetch(url, options);
}

document.addEventListener('DOMContentLoaded', updateNavbar);
</script>
<div class="container my-4">
"""

BASE_FOOT = """
</div>

<footer class="py-4 border-top bg-white mt-5">
  <div class="container d-flex flex-column flex-md-row justify-content-between align-items-center gap-2 small text-muted">
    <div>
      <span class="fw-bold text-dark">KnowMap with AI</span> • Enterprise Knowledge Graph & Semantic Discovery
    </div>
    <div class="d-flex gap-3">
      <span><i class="bi bi-database me-1"></i>SQLite Persistent</span>
      <span><i class="bi bi-cpu me-1"></i>spaCy 3.8 + MiniLM</span>
      <span><i class="bi bi-shield-check me-1"></i>JWT Protected</span>
    </div>
  </div>
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# -------------------- FRONT PAGE --------------------
FRONT_HTML = BASE_HEAD + NAVBAR + """
<div class="py-5 text-center position-relative">
  <div class="d-inline-flex align-items-center gap-2 px-3 py-1 bg-white border border-primary-subtle text-primary rounded-pill mb-3 small fw-semibold shadow-sm">
    <i class="bi bi-stars"></i> Next-Gen Semantic Knowledge Discovery Platform
  </div>
  <h1 class="fw-extrabold display-4 tracking-tight text-dark mb-3 mx-auto" style="max-width: 820px; font-weight: 800;">
    Turn Unstructured Information into <span style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Intelligent Connected Graphs</span>
  </h1>
  <p class="lead text-secondary mb-4 mx-auto" style="max-width: 680px; font-size: 1.15rem;">
    Upload custom triples, automatically extract Subject-Predicate-Object relations from text with dependency parsing, and explore connected entity subgraphs using Sentence Transformers.
  </p>
  
  <div class="d-flex justify-content-center gap-3 flex-wrap">
    <a href="/upload" class="btn btn-primary-custom btn-lg px-4 shadow-sm">
      <i class="bi bi-cloud-arrow-up me-2"></i>Ingest & Preview Graph
    </a>
    <a href="/semantic" class="btn btn-outline-secondary btn-lg px-4 bg-white shadow-sm">
      <i class="bi bi-search me-2"></i>Try Semantic Search
    </a>
    <a href="/dashboard" class="btn btn-light btn-lg px-4 border shadow-sm text-dark">
      <i class="bi bi-speedometer2 me-2"></i>Live Dashboard
    </a>
  </div>
</div>

<div class="row g-4 my-2">
  <div class="col-md-4">
    <div class="card card-custom card-custom-hover p-4 h-100">
      <div class="brand-badge mb-3" style="background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);">
        <i class="bi bi-file-earmark-spreadsheet"></i>
      </div>
      <h5 class="fw-bold mb-2">Automated CSV Ingestion</h5>
      <p class="text-muted small mb-0">
        Streamline batch CSV ingestion with intelligent header detection, deduplication, and persistent SQLite storage.
      </p>
    </div>
  </div>

  <div class="col-md-4">
    <div class="card card-custom card-custom-hover p-4 h-100">
      <div class="brand-badge mb-3" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
        <i class="bi bi-diagram-3"></i>
      </div>
      <h5 class="fw-bold mb-2">Dependency NLP Extraction</h5>
      <p class="text-muted small mb-0">
        Deep Subject-Predicate-Object relationship parsing built on top of spaCy syntax trees with prepositional reasoning.
      </p>
    </div>
  </div>

  <div class="col-md-4">
    <div class="card card-custom card-custom-hover p-4 h-100">
      <div class="brand-badge mb-3" style="background: linear-gradient(135deg, #7c3aed 0%, #9333ea 100%);">
        <i class="bi bi-cpu"></i>
      </div>
      <h5 class="fw-bold mb-2">Vector Semantic Retrieval</h5>
      <p class="text-muted small mb-0">
        High-dimensional embedding similarity with Sentence Transformers (`all-MiniLM-L6-v2`) returning authentic connected subgraphs.
      </p>
    </div>
  </div>
</div>

<div class="card card-custom p-4 mt-4 bg-white">
  <div class="d-flex flex-column flex-md-row justify-content-between align-items-md-center gap-3 pb-3 border-bottom">
    <div>
      <h5 class="fw-bold mb-1">Interactive Knowledge Explorer</h5>
      <div class="text-muted small">Real-time preview of the current knowledge base seeded in your persistent SQLite store</div>
    </div>
    <div class="d-flex gap-2">
      <a href="/semantic" class="btn btn-sm btn-outline-primary px-3">Open Deep Query</a>
    </div>
  </div>
  <div class="graph-wrapper mt-3">
    <div class="graph-toolbar">
      <button class="graph-btn" onclick="graphZoomIn()" title="Zoom In"><i class="bi bi-plus-lg"></i></button>
      <button class="graph-btn" onclick="graphZoomOut()" title="Zoom Out"><i class="bi bi-dash-lg"></i></button>
      <button class="graph-btn" onclick="graphFit()" title="Fit to Screen"><i class="bi bi-aspect-ratio"></i></button>
      <button class="graph-btn" id="physicsToggleBtn" onclick="graphTogglePhysics()" title="Toggle Physics"><i class="bi bi-lightning-charge"></i></button>
      <button class="graph-btn" onclick="graphExportPNG()" title="Export PNG"><i class="bi bi-camera"></i></button>
    </div>
    <div id="graph" style="height: 420px;"></div>
  </div>
</div>

<script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
<script>
let network = null;
let physicsEnabled = true;

function drawGraph(triples) {
  const container = document.getElementById('graph');
  if (!triples || triples.length === 0) {
    container.innerHTML = '<div class="text-muted text-center p-5">No knowledge triples found.</div>';
    return;
  }
  const nodes = [];
  const edges = [];
  const seenNodes = new Set();

  const palette = [
    { bg: '#e0e7ff', border: '#6366f1', text: '#312e81' },
    { bg: '#dcfce7', border: '#10b981', text: '#064e3b' },
    { bg: '#e0f2fe', border: '#0284c7', text: '#0c4a6e' },
    { bg: '#fef3c7', border: '#f59e0b', text: '#78350f' },
    { bg: '#f3e8ff', border: '#a855f7', text: '#581c87' }
  ];

  function getColor(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) hash = (hash << 5) - hash + str.charCodeAt(i);
    return palette[Math.abs(hash) % palette.length];
  }

  triples.forEach(t => {
    const s = String(t.subject).trim();
    const o = String(t.object).trim();
    const r = String(t.relation).trim();
    if (s && !seenNodes.has(s)) {
      const c = getColor(s);
      nodes.push({ id: s, label: s, color: { background: c.bg, border: c.border }, font: { color: c.text, face: 'Plus Jakarta Sans', size: 12 } });
      seenNodes.add(s);
    }
    if (o && !seenNodes.has(o)) {
      const c = getColor(o);
      nodes.push({ id: o, label: o, color: { background: c.bg, border: c.border }, font: { color: c.text, face: 'Plus Jakarta Sans', size: 12 } });
      seenNodes.add(o);
    }
    if (s && o) {
      edges.push({
        from: s, to: o, label: r,
        arrows: { to: { enabled: true, scaleFactor: 0.8 } },
        color: { color: '#94a3b8', highlight: '#4f46e5' },
        font: { face: 'Plus Jakarta Sans', size: 10, align: 'middle', background: '#ffffff', strokeWidth: 2, strokeColor: '#ffffff' }
      });
    }
  });

  const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
  const options = {
    nodes: { shape: 'box', margin: 8, borderRadius: 6, shadow: { enabled: true, color: 'rgba(0,0,0,0.04)', size: 4 } },
    physics: { stabilization: { iterations: 100 }, barnesHut: { gravitationalConstant: -2600, springLength: 100 } }
  };
  network = new vis.Network(container, data, options);
}

function graphZoomIn() { if (network) network.moveTo({ scale: network.getScale() * 1.25 }); }
function graphZoomOut() { if (network) network.moveTo({ scale: network.getScale() * 0.8 }); }
function graphFit() { if (network) network.fit({ animation: { duration: 400 } }); }
function graphTogglePhysics() {
  if (network) {
    physicsEnabled = !physicsEnabled;
    network.setOptions({ physics: { enabled: physicsEnabled } });
    document.getElementById('physicsToggleBtn').classList.toggle('text-primary', physicsEnabled);
  }
}
function graphExportPNG() {
  const canvas = document.querySelector('#graph canvas');
  if (canvas) {
    const a = document.createElement('a'); a.href = canvas.toDataURL('image/png'); a.download = 'knowmap_graph.png'; a.click();
  }
}

fetch('/recent_triples').then(r=>r.json()).then(d=>{ if(d.triples) drawGraph(d.triples.slice(0, 30)); });
</script>
""" + BASE_FOOT

# -------------------- LOGIN / SIGNUP PAGE --------------------
LOGIN_HTML = BASE_HEAD + NAVBAR + """
<div class="row justify-content-center my-5">
  <div class="col-md-5 col-lg-4">
    <div class="card card-custom p-4 shadow-sm">
      
      <div class="d-flex justify-content-center mb-3">
        <span class="brand-badge"><i class="bi bi-shield-lock-fill"></i></span>
      </div>

      <!-- Auth Toggle Pills -->
      <div class="d-flex p-1 bg-light rounded-pill mb-4 border">
        <button id="pillLogin" class="btn btn-sm w-50 rounded-pill fw-semibold bg-white text-dark shadow-sm">Sign In</button>
        <button id="pillSignup" class="btn btn-sm w-50 rounded-pill fw-semibold text-muted">Create Account</button>
      </div>

      <!-- Login Section -->
      <div id="loginSection">
        <div id="loginAlert" class="alert alert-danger d-none py-2 small rounded-3"></div>
        <div class="mb-3">
          <label class="form-label small fw-semibold text-secondary">Username</label>
          <div class="input-group">
            <span class="input-group-text bg-white border-end-0 text-muted"><i class="bi bi-person"></i></span>
            <input id="loginUsername" class="form-control border-start-0" placeholder="e.g. jashwankarthi">
          </div>
        </div>
        <div class="mb-3">
          <label class="form-label small fw-semibold text-secondary">Password</label>
          <div class="input-group">
            <span class="input-group-text bg-white border-end-0 text-muted"><i class="bi bi-key"></i></span>
            <input id="loginPassword" type="password" class="form-control border-start-0" placeholder="••••••••">
          </div>
        </div>
        <button id="loginBtn" class="btn btn-primary-custom w-100 py-2 fw-semibold">
          <i class="bi bi-box-arrow-in-right me-1"></i> Sign In to KnowMap
        </button>

        <button class="btn btn-sm btn-outline-secondary w-100 mt-3 py-1 small rounded-3" onclick="fillDemoUser()">
          <i class="bi bi-magic me-1"></i> Quick Fill Existing User
        </button>
      </div>

      <!-- Signup Section -->
      <div id="signupSection" style="display:none;">
        <div id="signupAlert" class="alert alert-danger d-none py-2 small rounded-3"></div>
        <div class="mb-3">
          <label class="form-label small fw-semibold text-secondary">Username</label>
          <div class="input-group">
            <span class="input-group-text bg-white border-end-0 text-muted"><i class="bi bi-person"></i></span>
            <input id="signupUsername" class="form-control border-start-0" placeholder="Choose a username">
          </div>
        </div>
        <div class="mb-3">
          <label class="form-label small fw-semibold text-secondary">Email Address</label>
          <div class="input-group">
            <span class="input-group-text bg-white border-end-0 text-muted"><i class="bi bi-envelope"></i></span>
            <input id="signupEmail" type="email" class="form-control border-start-0" placeholder="name@domain.com">
          </div>
        </div>
        <div class="mb-3">
          <label class="form-label small fw-semibold text-secondary">Password</label>
          <div class="input-group">
            <span class="input-group-text bg-white border-end-0 text-muted"><i class="bi bi-shield-check"></i></span>
            <input id="signupPassword" type="password" class="form-control border-start-0" placeholder="Create a password">
          </div>
        </div>
        <button id="signupBtn" class="btn btn-success w-100 py-2 fw-semibold rounded-3 shadow-sm">
          <i class="bi bi-person-plus me-1"></i> Create Account
        </button>
      </div>

    </div>
  </div>
</div>

<script>
const pillLogin = document.getElementById('pillLogin');
const pillSignup = document.getElementById('pillSignup');
const loginSection = document.getElementById('loginSection');
const signupSection = document.getElementById('signupSection');
const loginAlert = document.getElementById('loginAlert');
const signupAlert = document.getElementById('signupAlert');

function showLoginTab() {
  pillLogin.className = 'btn btn-sm w-50 rounded-pill fw-semibold bg-white text-dark shadow-sm';
  pillSignup.className = 'btn btn-sm w-50 rounded-pill fw-semibold text-muted';
  loginSection.style.display = 'block';
  signupSection.style.display = 'none';
}

function showSignupTab() {
  pillSignup.className = 'btn btn-sm w-50 rounded-pill fw-semibold bg-white text-dark shadow-sm';
  pillLogin.className = 'btn btn-sm w-50 rounded-pill fw-semibold text-muted';
  loginSection.style.display = 'none';
  signupSection.style.display = 'block';
}

pillLogin.onclick = showLoginTab;
pillSignup.onclick = showSignupTab;

if (window.location.hash === '#signup') {
  showSignupTab();
}

function fillDemoUser() {
  document.getElementById('loginUsername').value = 'jashwankarthi';
  document.getElementById('loginPassword').value = '123456';
  showToast('Demo username populated. Enter your password or create a new test account.', 'info');
}

document.getElementById('loginBtn').onclick = async () => {
  loginAlert.classList.add('d-none');
  const u = document.getElementById('loginUsername').value.trim();
  const p = document.getElementById('loginPassword').value;
  if (!u || !p) {
    loginAlert.textContent = 'Please enter both username and password.';
    loginAlert.classList.remove('d-none');
    return;
  }

  try {
    const res = await fetch('/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: u, password: p })
    });
    const data = await res.json();
    if (res.ok) {
      localStorage.setItem('jwt', data.token);
      showToast('Login successful! Redirecting...', 'success');
      setTimeout(() => { window.location.href = '/upload'; }, 800);
    } else {
      loginAlert.textContent = data.error || 'Invalid credentials';
      loginAlert.classList.remove('d-none');
    }
  } catch (err) {
    loginAlert.textContent = 'Unable to reach backend service.';
    loginAlert.classList.remove('d-none');
  }
};

document.getElementById('signupBtn').onclick = async () => {
  signupAlert.classList.add('d-none');
  const u = document.getElementById('signupUsername').value.trim();
  const e = document.getElementById('signupEmail').value.trim();
  const p = document.getElementById('signupPassword').value;
  if (!u || !e || !p) {
    signupAlert.textContent = 'Please fill out all fields.';
    signupAlert.classList.remove('d-none');
    return;
  }

  try {
    const res = await fetch('/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: u, email: e, password: p })
    });
    const data = await res.json();
    if (res.ok) {
      showToast('Account registered! Please sign in with your credentials.', 'success');
      showLoginTab();
      document.getElementById('loginUsername').value = u;
    } else {
      signupAlert.textContent = data.error || 'Signup failed';
      signupAlert.classList.remove('d-none');
    }
  } catch (err) {
    signupAlert.textContent = 'Server communication error.';
    signupAlert.classList.remove('d-none');
  }
};
</script>
""" + BASE_FOOT

# -------------------- UPLOAD & INGESTION PAGE --------------------
UPLOAD_HTML = BASE_HEAD + NAVBAR + """
<div class="d-flex flex-column flex-md-row justify-content-between align-items-md-center gap-2 mb-4">
  <div>
    <h4 class="fw-bold mb-1">Knowledge Ingestion & Graph Builder</h4>
    <p class="text-muted small mb-0">Upload CSV datasets or extract entity relations from arbitrary sentences.</p>
  </div>
  <div class="d-flex align-items-center gap-2">
    <span class="badge bg-light text-secondary border px-3 py-2 rounded-pill font-mono">
      <i class="bi bi-shield-check text-success me-1"></i>SQLite Synced
    </span>
  </div>
</div>

<div class="row g-4">
  <!-- Left Column: Ingestion Methods -->
  <div class="col-lg-5">
    
    <!-- CSV Card -->
    <div class="card card-custom p-4 mb-4">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <h6 class="fw-bold mb-0"><i class="bi bi-filetype-csv text-primary me-2"></i>Batch CSV Dataset</h6>
        <span class="badge bg-primary-subtle text-primary" style="font-size: 11px;">subject, relation, object</span>
      </div>
      <p class="text-muted small mb-3">Header row is automatically detected and skipped.</p>

      <div id="dropzone" class="dropzone-box mb-3">
        <i class="bi bi-cloud-arrow-up text-primary fs-2"></i>
        <div class="fw-semibold mt-1">Drag & drop your CSV file here</div>
        <div class="small text-muted">or click to browse from your device</div>
        <input type="file" id="csvFile" class="d-none" accept=".csv">
      </div>

      <div id="fileInfo" class="d-none alert alert-light border small py-2 px-3 mb-3 d-flex justify-content-between align-items-center">
        <span id="fileName" class="text-truncate fw-semibold"></span>
        <button class="btn btn-sm text-danger p-0" onclick="clearFile()"><i class="bi bi-x-circle"></i></button>
      </div>

      <button id="uploadBtn" class="btn btn-primary-custom w-100 fw-semibold">
        <i class="bi bi-upload me-1"></i> Upload & Ingest Triples
      </button>
      <a href="/download_sample_csv" class="btn btn-sm btn-outline-secondary w-100 mt-2 py-2 rounded-3 text-decoration-none">
        <i class="bi bi-download me-1"></i> Download Sample CSV Dataset
      </a>
      <div id="uploadStatus" class="small mt-2"></div>
    </div>

    <!-- Sentence NLP Card -->
    <div class="card card-custom p-4">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <h6 class="fw-bold mb-0"><i class="bi bi-cpu text-success me-2"></i>Natural Language Extraction</h6>
        <span class="badge bg-success-subtle text-success" style="font-size: 11px;">spaCy Dependency</span>
      </div>
      <p class="text-muted small mb-2">Extract Subject-Predicate-Object relations directly from natural text.</p>

      <!-- Quick Sample Prompt Pills -->
      <div class="d-flex gap-1 flex-wrap mb-2">
        <span class="prompt-pill" onclick="setPrompt('Albert Einstein developed the theory of relativity in 1905.')">🔬 Einstein</span>
        <span class="prompt-pill" onclick="setPrompt('Marie Curie discovered radium and won two Nobel Prizes.')">☢️ Curie</span>
        <span class="prompt-pill" onclick="setPrompt('Alan Turing contributed to computer science and artificial intelligence.')">💻 Turing</span>
        <span class="prompt-pill" onclick="setPrompt('Python is a programming language widely used for machine learning.')">🐍 Python</span>
      </div>

      <textarea id="sentence" rows="3" class="form-control mb-3 small" placeholder="Type or paste any English sentence..."></textarea>
      
      <button id="analyzeBtn" class="btn btn-success w-100 fw-semibold rounded-3 shadow-sm">
        <i class="bi bi-gear-wide-connected me-1"></i> Extract & Add to Knowledge Graph
      </button>
      <div id="analyzeStatus" class="small mt-2"></div>
    </div>

  </div>

  <!-- Right Column: Live Graph Preview & KPI Tiles -->
  <div class="col-lg-7">
    <div class="card card-custom p-4 mb-4">
      <div class="d-flex justify-content-between align-items-center mb-3">
        <div>
          <h6 class="fw-bold mb-0">Graph Viewport</h6>
          <span class="text-muted small">Interactive force-directed visualization</span>
        </div>
        <span id="activeGraphCount" class="badge bg-light text-secondary border small">0 nodes</span>
      </div>

      <div class="graph-wrapper">
        <div class="graph-toolbar">
          <button class="graph-btn" onclick="graphZoomIn()" title="Zoom In"><i class="bi bi-plus-lg"></i></button>
          <button class="graph-btn" onclick="graphZoomOut()" title="Zoom Out"><i class="bi bi-dash-lg"></i></button>
          <button class="graph-btn" onclick="graphFit()" title="Fit to Screen"><i class="bi bi-aspect-ratio"></i></button>
          <button class="graph-btn" id="physicsToggleBtn" onclick="graphTogglePhysics()" title="Toggle Physics"><i class="bi bi-lightning-charge"></i></button>
          <button class="graph-btn" onclick="graphExportPNG()" title="Export Image"><i class="bi bi-camera"></i></button>
        </div>
        <div id="graph"></div>
      </div>
    </div>

    <!-- Quick Metrics Ribbon -->
    <div class="row g-3">
      <div class="col-4">
        <div class="card card-custom p-3 text-center">
          <div class="text-muted small">Files Uploaded</div>
          <div id="filesUploaded" class="fs-4 fw-bold text-primary mt-1">0</div>
        </div>
      </div>
      <div class="col-4">
        <div class="card card-custom p-3 text-center">
          <div class="text-muted small">Total Triples</div>
          <div id="triplesTotal" class="fs-4 fw-bold text-success mt-1">0</div>
        </div>
      </div>
      <div class="col-4">
        <div class="card card-custom p-3 text-center">
          <div class="text-muted small">Unique Entities</div>
          <div id="uniqueNodes" class="fs-4 fw-bold text-info mt-1">0</div>
        </div>
      </div>
    </div>

  </div>
</div>

<script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
<script>
let network = null;
let physicsEnabled = true;

const dropzone = document.getElementById('dropzone');
const csvFileInput = document.getElementById('csvFile');
const fileInfo = document.getElementById('fileInfo');
const fileNameSpan = document.getElementById('fileName');

dropzone.onclick = () => csvFileInput.click();

dropzone.ondragover = (e) => { e.preventDefault(); dropzone.classList.add('dragover'); };
dropzone.ondragleave = () => dropzone.classList.remove('dragover');
dropzone.ondrop = (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    csvFileInput.files = e.dataTransfer.files;
    showSelectedFile();
  }
};

csvFileInput.onchange = showSelectedFile;

function showSelectedFile() {
  if (csvFileInput.files[0]) {
    fileNameSpan.textContent = csvFileInput.files[0].name + ' (' + Math.round(csvFileInput.files[0].size / 1024) + ' KB)';
    fileInfo.classList.remove('d-none');
    dropzone.classList.add('d-none');
  }
}

function clearFile() {
  csvFileInput.value = '';
  fileInfo.classList.add('d-none');
  dropzone.classList.remove('d-none');
}

function setPrompt(text) {
  document.getElementById('sentence').value = text;
}

function drawGraph(triples) {
  const container = document.getElementById('graph');
  if (!triples || triples.length === 0) {
    container.innerHTML = `
      <div class="d-flex flex-column align-items-center justify-content-center h-100 text-muted p-5">
        <i class="bi bi-diagram-2 fs-1 text-secondary opacity-50 mb-2"></i>
        <div class="fw-semibold">No knowledge graph data</div>
        <div class="small">Upload a CSV dataset or extract a sentence to render graph nodes.</div>
      </div>`;
    document.getElementById('activeGraphCount').textContent = '0 nodes';
    return;
  }

  const nodes = [];
  const edges = [];
  const seenNodes = new Set();

  const palette = [
    { bg: '#e0e7ff', border: '#6366f1', text: '#312e81' },
    { bg: '#dcfce7', border: '#10b981', text: '#064e3b' },
    { bg: '#e0f2fe', border: '#0284c7', text: '#0c4a6e' },
    { bg: '#fef3c7', border: '#f59e0b', text: '#78350f' },
    { bg: '#f3e8ff', border: '#a855f7', text: '#581c87' }
  ];

  function getColor(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) hash = (hash << 5) - hash + str.charCodeAt(i);
    return palette[Math.abs(hash) % palette.length];
  }

  triples.forEach(t => {
    const s = String(t.subject).trim();
    const o = String(t.object).trim();
    const r = String(t.relation).trim();
    if (s && !seenNodes.has(s)) {
      const c = getColor(s);
      nodes.push({ id: s, label: s, color: { background: c.bg, border: c.border }, font: { color: c.text, face: 'Plus Jakarta Sans', size: 12 } });
      seenNodes.add(s);
    }
    if (o && !seenNodes.has(o)) {
      const c = getColor(o);
      nodes.push({ id: o, label: o, color: { background: c.bg, border: c.border }, font: { color: c.text, face: 'Plus Jakarta Sans', size: 12 } });
      seenNodes.add(o);
    }
    if (s && o) {
      edges.push({
        from: s, to: o, label: r,
        arrows: { to: { enabled: true, scaleFactor: 0.8 } },
        color: { color: '#94a3b8', highlight: '#4f46e5' },
        font: { face: 'Plus Jakarta Sans', size: 10, align: 'middle', background: '#ffffff', strokeWidth: 2, strokeColor: '#ffffff' },
        smooth: { type: 'cubicBezier', roundness: 0.3 }
      });
    }
  });

  document.getElementById('activeGraphCount').textContent = nodes.length + ' entities';

  const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
  const options = {
    nodes: { shape: 'box', margin: 9, borderRadius: 6, shadow: { enabled: true, color: 'rgba(0,0,0,0.03)', size: 4 } },
    physics: { stabilization: { iterations: 120 }, barnesHut: { gravitationalConstant: -2800, springLength: 110 } }
  };
  network = new vis.Network(container, data, options);
}

function graphZoomIn() { if (network) network.moveTo({ scale: network.getScale() * 1.25 }); }
function graphZoomOut() { if (network) network.moveTo({ scale: network.getScale() * 0.8 }); }
function graphFit() { if (network) network.fit({ animation: { duration: 400 } }); }
function graphTogglePhysics() {
  if (network) {
    physicsEnabled = !physicsEnabled;
    network.setOptions({ physics: { enabled: physicsEnabled } });
    document.getElementById('physicsToggleBtn').classList.toggle('text-primary', physicsEnabled);
  }
}
function graphExportPNG() {
  const canvas = document.querySelector('#graph canvas');
  if (canvas) {
    const a = document.createElement('a'); a.href = canvas.toDataURL('image/png'); a.download = 'knowmap_graph.png'; a.click();
  }
}

document.getElementById('uploadBtn').onclick = async () => {
  const statusDiv = document.getElementById('uploadStatus');
  if (!csvFileInput.files[0]) return showToast('Please select a CSV file first.', 'danger');

  const fd = new FormData();
  fd.append('file', csvFileInput.files[0]);
  statusDiv.innerHTML = '<span class="text-primary"><i class="bi bi-hourglass-split me-1"></i>Ingesting CSV triples...</span>';

  try {
    const res = await authFetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (res.ok) {
      statusDiv.innerHTML = `<span class="text-success fw-semibold"><i class="bi bi-check-circle me-1"></i>Ingested ${data.triples_added} triples.</span>`;
      showToast(`Successfully ingested ${data.triples_added} triples!`, 'success');
      drawGraph(data.triples);
      refreshStats();
      clearFile();
    } else {
      statusDiv.innerHTML = `<span class="text-danger">${escapeHtml(data.error || 'Upload failed')}</span>`;
      showToast(data.error || 'Upload error', 'danger');
    }
  } catch (err) {
    statusDiv.innerHTML = '<span class="text-danger">Network error during upload.</span>';
  }
};

document.getElementById('analyzeBtn').onclick = async () => {
  const text = document.getElementById('sentence').value.trim();
  const statusDiv = document.getElementById('analyzeStatus');
  if (!text) return showToast('Please type a sentence to analyze.', 'danger');

  statusDiv.innerHTML = '<span class="text-primary"><i class="bi bi-hourglass-split me-1"></i>Running NLP dependency extraction...</span>';
  try {
    const res = await authFetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    if (res.ok) {
      const count = data.triples ? data.triples.length : 0;
      const mode = data.mode || 'nlp_extract';
      const message = data.message || '';

      if (count === 0) {
        statusDiv.innerHTML = `
          <div class="alert alert-warning border rounded-3 py-2 px-3 mt-1 mb-0 small">
            <i class="bi bi-info-circle me-1"></i>
            <strong>No triples extracted.</strong><br>
            ${escapeHtml(message)}<br>
            <span class="text-muted">Tip: Enter a full sentence like <em>"Gene editing cures genetic disorders"</em>, or use <a href="/semantic" class="fw-semibold">Semantic Search</a> for keyword queries.</span>
          </div>`;
        showToast('No triples found. Try a full sentence.', 'info');
      } else if (mode === 'graph_lookup') {
        statusDiv.innerHTML = `<span class="text-info fw-semibold"><i class="bi bi-database me-1"></i>Found ${count} related triple(s) from knowledge graph.</span>`;
        showToast(`Found ${count} related triples via keyword lookup!`, 'info');
        drawGraph(data.triples);
        refreshStats();
      } else {
        statusDiv.innerHTML = `<span class="text-success fw-semibold"><i class="bi bi-check-circle me-1"></i>Extracted ${count} triple(s) via NLP.</span>`;
        showToast(`Extracted ${count} knowledge triples!`, 'success');
        drawGraph(data.triples);
        refreshStats();
      }
    } else {
      statusDiv.innerHTML = `<span class="text-danger">${escapeHtml(data.error || 'Extraction failed')}</span>`;
    }
  } catch (err) {
    statusDiv.innerHTML = '<span class="text-danger">Failed to process sentence.</span>';
  }
};

async function refreshStats() {
  try {
    const res = await fetch('/stats');
    const s = await res.json();
    document.getElementById('filesUploaded').textContent = s.files_uploaded;
    document.getElementById('triplesTotal').textContent = s.triples_total;
    document.getElementById('uniqueNodes').textContent = s.unique_nodes;
  } catch (e) {}
}

async function loadInitial() {
  await refreshStats();
  try {
    const res = await fetch('/recent_triples');
    const d = await res.json();
    if (d.triples && d.triples.length > 0) {
      drawGraph(d.triples.slice(0, 30));
    }
  } catch (e) {}
}
loadInitial();
</script>
""" + BASE_FOOT

# -------------------- SEMANTIC SEARCH PAGE --------------------
SEMANTIC_HTML = BASE_HEAD + NAVBAR + """
<div class="d-flex flex-column flex-md-row justify-content-between align-items-md-center gap-2 mb-4">
  <div>
    <h4 class="fw-bold mb-1">Vector Semantic Search</h4>
    <p class="text-muted small mb-0">Compute high-dimensional similarity over entity nodes and retrieve true connected knowledge subgraphs.</p>
  </div>
  <div class="d-flex align-items-center gap-2">
    <span class="badge bg-indigo-subtle text-primary border border-primary-subtle px-3 py-2 rounded-pill font-mono">
      Model: all-MiniLM-L6-v2
    </span>
  </div>
</div>

<!-- Search Input Card -->
<div class="card card-custom p-3 mb-4 shadow-sm">
  <div class="input-group">
    <span class="input-group-text bg-white border-end-0 text-muted"><i class="bi bi-search"></i></span>
    <input id="query" class="form-control border-start-0 py-2" placeholder="Search concepts (e.g., 'Relativity physics', 'Computer science', 'Laws of motion')..." autocomplete="off">
    <button id="searchBtn" class="btn btn-primary-custom px-4 fw-semibold">
      <i class="bi bi-lightning-charge-fill me-1"></i> Search
    </button>
  </div>

  <!-- Search Suggestion Chips -->
  <div class="d-flex align-items-center gap-2 mt-2 pt-2 border-top flex-wrap">
    <span class="text-muted small fw-semibold">Suggestions:</span>
    <span class="prompt-pill" onclick="runSearch('Relativity space time')">🌌 Relativity</span>
    <span class="prompt-pill" onclick="runSearch('Artificial intelligence machine learning')">🤖 AI & ML</span>
    <span class="prompt-pill" onclick="runSearch('Chemical element radium')">🔬 Radium</span>
    <span class="prompt-pill" onclick="runSearch('Computer Science')">💾 Computer Science</span>
  </div>
</div>

<div class="row g-4">
  <!-- Left Column: Knowledge Subgraph -->
  <div class="col-lg-7">
    <div class="card card-custom p-4">
      <div class="d-flex justify-content-between align-items-center mb-3">
        <div>
          <h6 class="fw-bold mb-0">Connected Knowledge Subgraph</h6>
          <span id="searchStatus" class="text-muted small">Enter a query above to explore matches.</span>
        </div>
      </div>

      <div class="graph-wrapper">
        <div class="graph-toolbar">
          <button class="graph-btn" onclick="graphZoomIn()" title="Zoom In"><i class="bi bi-plus-lg"></i></button>
          <button class="graph-btn" onclick="graphZoomOut()" title="Zoom Out"><i class="bi bi-dash-lg"></i></button>
          <button class="graph-btn" onclick="graphFit()" title="Fit to Screen"><i class="bi bi-aspect-ratio"></i></button>
          <button class="graph-btn" id="physicsToggleBtn" onclick="graphTogglePhysics()" title="Toggle Physics"><i class="bi bi-lightning-charge"></i></button>
          <button class="graph-btn" onclick="graphExportPNG()" title="Export PNG"><i class="bi bi-camera"></i></button>
        </div>
        <div id="graph"></div>
      </div>
    </div>
  </div>

  <!-- Right Column: Ranked Matches & Triples -->
  <div class="col-lg-5">
    <!-- Top Ranked Concepts -->
    <div class="card card-custom p-4 mb-4">
      <h6 class="fw-bold mb-1">Top Semantic Entity Matches</h6>
      <p class="text-muted small mb-3">Click any entity below to view its direct relationships.</p>
      <div id="nodeList" class="list-group list-group-flush small" style="max-height: 220px; overflow-y: auto;">
        <div class="text-muted p-2 text-center">Execute a search to rank nodes by vector similarity.</div>
      </div>
    </div>

    <!-- Connected Triples in Subgraph -->
    <div class="card card-custom p-4">
      <h6 class="fw-bold mb-1">Connected Triples in Subgraph</h6>
      <p class="text-muted small mb-2">Real database edges connecting matched concepts.</p>
      <div id="triplesList" class="small" style="max-height: 220px; overflow-y: auto;">
        <div class="text-muted text-center p-3">No subgraph active.</div>
      </div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
<script>
let network = null;
let physicsEnabled = true;

function runSearch(text) {
  document.getElementById('query').value = text;
  document.getElementById('searchBtn').click();
}

document.getElementById('query').addEventListener('keyup', (e) => {
  if (e.key === 'Enter') document.getElementById('searchBtn').click();
});

function drawGraph(triples) {
  const container = document.getElementById('graph');
  if (!triples || triples.length === 0) {
    container.innerHTML = `
      <div class="d-flex flex-column align-items-center justify-content-center h-100 text-muted p-5">
        <i class="bi bi-diagram-3 fs-1 text-secondary opacity-50 mb-2"></i>
        <div class="fw-semibold">No connected triples</div>
        <div class="small">Try searching another term or broader keyword.</div>
      </div>`;
    return;
  }

  const nodes = [];
  const edges = [];
  const seenNodes = new Set();

  const palette = [
    { bg: '#e0e7ff', border: '#6366f1', text: '#312e81' },
    { bg: '#dcfce7', border: '#10b981', text: '#064e3b' },
    { bg: '#e0f2fe', border: '#0284c7', text: '#0c4a6e' },
    { bg: '#fef3c7', border: '#f59e0b', text: '#78350f' }
  ];

  function getColor(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) hash = (hash << 5) - hash + str.charCodeAt(i);
    return palette[Math.abs(hash) % palette.length];
  }

  triples.forEach(t => {
    const s = String(t.subject).trim();
    const o = String(t.object).trim();
    const r = String(t.relation).trim();
    if (s && !seenNodes.has(s)) {
      const c = getColor(s);
      nodes.push({ id: s, label: s, color: { background: c.bg, border: c.border }, font: { color: c.text, face: 'Plus Jakarta Sans', size: 12 } });
      seenNodes.add(s);
    }
    if (o && !seenNodes.has(o)) {
      const c = getColor(o);
      nodes.push({ id: o, label: o, color: { background: c.bg, border: c.border }, font: { color: c.text, face: 'Plus Jakarta Sans', size: 12 } });
      seenNodes.add(o);
    }
    if (s && o) {
      edges.push({
        from: s, to: o, label: r,
        arrows: { to: { enabled: true, scaleFactor: 0.8 } },
        color: { color: '#94a3b8', highlight: '#4f46e5' },
        font: { face: 'Plus Jakarta Sans', size: 10, align: 'middle', background: '#ffffff', strokeWidth: 2, strokeColor: '#ffffff' },
        smooth: { type: 'cubicBezier', roundness: 0.3 }
      });
    }
  });

  const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
  const options = {
    nodes: { shape: 'box', margin: 8, borderRadius: 6, shadow: { enabled: true, color: 'rgba(0,0,0,0.03)', size: 4 } },
    physics: { stabilization: { iterations: 120 }, barnesHut: { gravitationalConstant: -2800, springLength: 110 } }
  };
  network = new vis.Network(container, data, options);
}

function graphZoomIn() { if (network) network.moveTo({ scale: network.getScale() * 1.25 }); }
function graphZoomOut() { if (network) network.moveTo({ scale: network.getScale() * 0.8 }); }
function graphFit() { if (network) network.fit({ animation: { duration: 400 } }); }
function graphTogglePhysics() {
  if (network) {
    physicsEnabled = !physicsEnabled;
    network.setOptions({ physics: { enabled: physicsEnabled } });
    document.getElementById('physicsToggleBtn').classList.toggle('text-primary', physicsEnabled);
  }
}
function graphExportPNG() {
  const canvas = document.querySelector('#graph canvas');
  if (canvas) {
    const a = document.createElement('a'); a.href = canvas.toDataURL('image/png'); a.download = 'semantic_subgraph.png'; a.click();
  }
}

function updateTriplesList(triples) {
  const div = document.getElementById('triplesList');
  if (!triples || triples.length === 0) {
    div.innerHTML = '<div class="text-muted text-center p-3">No connected triples.</div>';
    return;
  }
  div.innerHTML = triples.map(t => `
    <div class="py-2 border-bottom d-flex align-items-center gap-2">
      <span class="badge bg-indigo-subtle text-primary border border-primary-subtle text-truncate" style="max-width: 120px;">${escapeHtml(t.subject)}</span>
      <span class="text-secondary small font-mono">--[${escapeHtml(t.relation)}]--></span>
      <span class="badge bg-success-subtle text-success border border-success-subtle text-truncate" style="max-width: 120px;">${escapeHtml(t.object)}</span>
    </div>
  `).join('');
}

document.getElementById('searchBtn').onclick = async () => {
  const q = document.getElementById('query').value.trim();
  const statusSpan = document.getElementById('searchStatus');
  if (!q) return showToast('Please enter a concept to search.', 'danger');

  statusSpan.innerHTML = '<span class="text-primary"><i class="bi bi-hourglass-split me-1"></i>Computing vector similarities...</span>';
  try {
    const res = await authFetch('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q })
    });
    const data = await res.json();
    if (res.ok) {
      statusSpan.innerHTML = `<span class="text-success fw-semibold">Matched ${data.top_nodes.length} concepts & ${data.triples.length} edges</span>`;
      drawGraph(data.triples);
      updateTriplesList(data.triples);

      const nodeList = document.getElementById('nodeList');
      nodeList.innerHTML = '';
      if (!data.top_nodes || data.top_nodes.length === 0) {
        nodeList.innerHTML = '<div class="text-muted p-2 text-center">No nodes matched.</div>';
      } else {
        data.top_nodes.forEach(n => {
          const item = document.createElement('a');
          item.href = '#';
          item.className = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center py-2 px-3 border-0 rounded-3 mb-1';
          item.style.backgroundColor = '#f8fafc';
          const pct = Math.round(n.score * 100);
          item.innerHTML = `
            <div>
              <div class="fw-semibold text-dark">${escapeHtml(n.name)}</div>
              <div class="progress mt-1" style="height: 4px; width: 100px;">
                <div class="progress-bar bg-primary" style="width: ${pct}%"></div>
              </div>
            </div>
            <span class="badge bg-white text-primary border rounded-pill">${pct}% Match</span>
          `;
          item.onclick = async (e) => {
            e.preventDefault();
            const r = await authFetch('/search_node', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ node: n.name })
            });
            const d = await r.json();
            drawGraph(d.triples);
            updateTriplesList(d.triples);
            showToast(`Focused on node: ${n.name}`, 'info');
          };
          nodeList.appendChild(item);
        });
      }
    } else {
      statusSpan.innerHTML = `<span class="text-danger">${escapeHtml(data.error || 'Search error')}</span>`;
    }
  } catch (err) {
    statusSpan.innerHTML = '<span class="text-danger">Search request failed.</span>';
  }
};
</script>
""" + BASE_FOOT

# -------------------- DASHBOARD PAGE --------------------
DASHBOARD_HTML = BASE_HEAD + NAVBAR + """
<div class="d-flex flex-column flex-md-row justify-content-between align-items-md-center gap-2 mb-4">
  <div>
    <h4 class="fw-bold mb-1">Executive Analytics & Quality Monitor</h4>
    <p class="text-muted small mb-0">Live pipeline ingestion statistics, weekly throughput, and community ratings.</p>
  </div>
  <div class="d-flex align-items-center gap-2">
    <button class="btn btn-sm btn-outline-secondary bg-white px-3" onclick="loadDashboard()">
      <i class="bi bi-arrow-clockwise me-1"></i> Refresh
    </button>
  </div>
</div>

<!-- 4 KPI Cards -->
<div class="row g-4 mb-4">
  <div class="col-6 col-lg-3">
    <div class="card card-custom p-4">
      <div class="d-flex justify-content-between align-items-center">
        <span class="text-muted small fw-semibold">Ingested Triples</span>
        <span class="badge bg-primary-subtle text-primary rounded-pill"><i class="bi bi-diagram-2"></i></span>
      </div>
      <h3 id="triplesTotal" class="fw-extrabold text-dark mt-2 mb-0">0</h3>
      <div class="small text-muted mt-1">Persistent SQLite edges</div>
    </div>
  </div>

  <div class="col-6 col-lg-3">
    <div class="card card-custom p-4">
      <div class="d-flex justify-content-between align-items-center">
        <span class="text-muted small fw-semibold">Unique Entities</span>
        <span class="badge bg-success-subtle text-success rounded-pill"><i class="bi bi-node-plus"></i></span>
      </div>
      <h3 id="uniqueNodes" class="fw-extrabold text-dark mt-2 mb-0">0</h3>
      <div class="small text-muted mt-1">Distinct graph nodes</div>
    </div>
  </div>

  <div class="col-6 col-lg-3">
    <div class="card card-custom p-4">
      <div class="d-flex justify-content-between align-items-center">
        <span class="text-muted small fw-semibold">CSV Datasets</span>
        <span class="badge bg-info-subtle text-info rounded-pill"><i class="bi bi-folder2-open"></i></span>
      </div>
      <h3 id="filesUploaded" class="fw-extrabold text-dark mt-2 mb-0">0</h3>
      <div class="small text-muted mt-1">Processed file batches</div>
    </div>
  </div>

  <div class="col-6 col-lg-3">
    <div class="card card-custom p-4">
      <div class="d-flex justify-content-between align-items-center">
        <span class="text-muted small fw-semibold">Average Rating</span>
        <span class="badge bg-warning-subtle text-warning rounded-pill"><i class="bi bi-star-fill"></i></span>
      </div>
      <h3 id="avgRating" class="fw-extrabold text-dark mt-2 mb-0">0.0</h3>
      <div class="small text-muted mt-1">User evaluation score</div>
    </div>
  </div>
</div>

<div class="row g-4">
  <!-- Left Column: Chart & Table -->
  <div class="col-lg-8">
    <div class="card card-custom p-4 mb-4">
      <div class="d-flex justify-content-between align-items-center mb-3">
        <h6 class="fw-bold mb-0">Weekly Ingestion Throughput</h6>
        <span class="badge bg-light text-secondary border font-mono small">Past 7 Days</span>
      </div>
      <div style="height: 220px;">
        <canvas id="usageChart"></canvas>
      </div>
    </div>

    <!-- Recent Triples Table -->
    <div class="card card-custom p-4">
      <div class="d-flex justify-content-between align-items-center mb-3">
        <h6 class="fw-bold mb-0">Recent Knowledge Triples</h6>
        <span class="badge bg-light text-muted border small font-mono">Live DB Feed</span>
      </div>
      <div class="table-responsive small" style="max-height: 240px; overflow-y: auto;">
        <table class="table table-hover align-middle mb-0">
          <thead class="table-light sticky-top">
            <tr><th>Subject</th><th>Relation</th><th>Object</th><th>Source</th></tr>
          </thead>
          <tbody id="triplesBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Right Column: Feedback Form & Community Feed -->
  <div class="col-lg-4">
    <!-- Feedback Form -->
    <div class="card card-custom p-4 mb-4">
      <h6 class="fw-bold mb-1">Submit System Evaluation</h6>
      <p class="text-muted small mb-3">Rate relevance and accuracy of graph retrieval.</p>
      
      <div id="stars" class="d-flex gap-2 justify-content-center mb-3">
        <span class="star-btn" data-v="1">&#9733;</span>
        <span class="star-btn" data-v="2">&#9733;</span>
        <span class="star-btn" data-v="3">&#9733;</span>
        <span class="star-btn" data-v="4">&#9733;</span>
        <span class="star-btn" data-v="5">&#9733;</span>
      </div>

      <textarea id="feedbackText" class="form-control mb-3 small" rows="3" placeholder="Share your evaluation or query feedback..."></textarea>
      
      <button id="submitFeedbackBtn" class="btn btn-primary-custom w-100 fw-semibold">
        <i class="bi bi-send-fill me-1"></i> Submit Feedback
      </button>
      <div id="feedbackStatus" class="small mt-2 text-center"></div>
    </div>

    <!-- Feedback Feed -->
    <div class="card card-custom p-4">
      <h6 class="fw-bold mb-1">Community Feedback</h6>
      <p class="text-muted small mb-3">Recent user ratings and reviews.</p>
      <div id="feedbackList" class="small" style="max-height: 230px; overflow-y: auto;"></div>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
let selectedRating = 0;
let chartInstance = null;

document.querySelectorAll('.star-btn').forEach(s => {
  s.onclick = () => {
    selectedRating = parseInt(s.dataset.v);
    document.querySelectorAll('.star-btn').forEach(x => {
      x.classList.toggle('active', parseInt(x.dataset.v) <= selectedRating);
    });
  };
});

document.getElementById('submitFeedbackBtn').onclick = async () => {
  const statusDiv = document.getElementById('feedbackStatus');
  if (!selectedRating) return showToast('Please select a star rating (1-5).', 'danger');
  const comments = document.getElementById('feedbackText').value.trim();

  try {
    const res = await authFetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rating: selectedRating, comments })
    });
    if (res.ok) {
      showToast('Thank you! Feedback recorded into SQLite.', 'success');
      document.getElementById('feedbackText').value = '';
      selectedRating = 0;
      document.querySelectorAll('.star-btn').forEach(x => x.classList.remove('active'));
      loadDashboard();
    } else {
      showToast('Failed to save feedback.', 'danger');
    }
  } catch (e) {
    showToast('Network error submitting feedback.', 'danger');
  }
};

async function loadDashboard() {
  try {
    const statsRes = await fetch('/stats');
    const s = await statsRes.json();
    document.getElementById('filesUploaded').textContent = s.files_uploaded;
    document.getElementById('triplesTotal').textContent = s.triples_total;
    document.getElementById('uniqueNodes').textContent = s.unique_nodes;
    document.getElementById('avgRating').textContent = (s.avg_rating || 0).toFixed(1) + ' ★';

    const feedbackRes = await fetch('/get_feedback');
    const fb = await feedbackRes.json();
    const fbList = document.getElementById('feedbackList');
    fbList.innerHTML = '';
    if (!fb.feedback || fb.feedback.length === 0) {
      fbList.innerHTML = '<div class="text-muted text-center p-3">No feedback recorded yet.</div>';
    } else {
      fb.feedback.slice(-10).reverse().forEach(item => {
        const stars = '★'.repeat(item.rating) + '☆'.repeat(5 - item.rating);
        fbList.innerHTML += `
          <div class="border rounded-3 p-3 mb-2 bg-light">
            <div class="d-flex justify-content-between align-items-center mb-1">
              <strong class="text-dark">${escapeHtml(item.username)}</strong>
              <span class="text-warning fw-bold">${stars}</span>
            </div>
            <div class="text-secondary">${escapeHtml(item.comments || 'No comment')}</div>
            <div class="text-muted mt-1 font-mono" style="font-size: 10px;">${escapeHtml(item.timestamp || '')}</div>
          </div>
        `;
      });
    }

    const triplesRes = await fetch('/recent_triples');
    const trData = await triplesRes.json();
    const tbody = document.getElementById('triplesBody');
    tbody.innerHTML = '';
    (trData.triples || []).slice(0, 25).forEach(t => {
      tbody.innerHTML += `
        <tr>
          <td><span class="badge bg-indigo-subtle text-primary border border-primary-subtle">${escapeHtml(t.subject)}</span></td>
          <td><code class="text-secondary">${escapeHtml(t.relation)}</code></td>
          <td><span class="badge bg-success-subtle text-success border border-success-subtle">${escapeHtml(t.object)}</span></td>
          <td><span class="badge bg-light text-muted border">${escapeHtml(t.source || 'db')}</span></td>
        </tr>
      `;
    });

    const metricsRes = await fetch('/metrics');
    const mData = await metricsRes.json();
    const ctx = document.getElementById('usageChart').getContext('2d');
    
    if (chartInstance) {
      chartInstance.data.labels = mData.labels;
      chartInstance.data.datasets[0].data = mData.counts;
      chartInstance.update();
    } else {
      const gradient = ctx.createLinearGradient(0, 0, 0, 200);
      gradient.addColorStop(0, 'rgba(79, 70, 229, 0.28)');
      gradient.addColorStop(1, 'rgba(79, 70, 229, 0.02)');

      chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: mData.labels,
          datasets: [{
            label: 'Triples Ingested',
            data: mData.counts,
            fill: true,
            backgroundColor: gradient,
            borderColor: '#4f46e5',
            borderWidth: 2.5,
            tension: 0.35,
            pointBackgroundColor: '#4f46e5',
            pointRadius: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.04)' } },
            x: { grid: { display: false } }
          }
        }
      });
    }
  } catch (err) {
    console.error('Failed to load dashboard:', err);
  }
}

loadDashboard();
setInterval(loadDashboard, 15000);
</script>
""" + BASE_FOOT


# -------------------- API ROUTES & CONTROLLERS --------------------
@app.route("/")
def index():
    return render_template_string(FRONT_HTML)

@app.route("/login", methods=["GET", "POST"])
def login_route():
    if request.method == "GET":
        return render_template_string(LOGIN_HTML)

    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()

    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Invalid username or password"}), 401

    token = create_token(username)
    return jsonify({"token": token, "username": username})

@app.route("/signup", methods=["POST"])
def signup_route():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    conn = get_db()
    cur = conn.cursor()
    now_str = datetime.now(timezone.utc).isoformat()
    try:
        pw_hash = generate_password_hash(password)
        cur.execute("""
        INSERT INTO users (username, email, password_hash, created_at)
        VALUES (?, ?, ?, ?)
        """, (username, email, pw_hash, now_str))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Username already exists. Please choose another."}), 400

    conn.close()
    return jsonify({"ok": True, "message": "User registered successfully"})

@app.route("/logout")
def logout_route():
    return redirect(url_for("login_route"))

@app.route("/upload", methods=["GET", "POST"])
def upload_route():
    if request.method == "GET":
        return render_template_string(UPLOAD_HTML)

    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    try:
        content = file.stream.read().decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(content))
        header = next(reader, None)

        added = []
        conn = get_db()
        cur = conn.cursor()
        now_str = datetime.now(timezone.utc).isoformat()

        # If header wasn't a standard header, process it
        first_row = header
        if first_row and len(first_row) >= 3:
            s, r, o = first_row[0].strip(), first_row[1].strip(), first_row[2].strip()
            if s.lower() != "subject" and s and r and o:
                cur.execute("""
                INSERT OR IGNORE INTO triples (subject, relation, object, source, created_at)
                VALUES (?, ?, ?, ?, ?)
                """, (s, r, o, file.filename, now_str))
                if cur.rowcount > 0:
                    added.append({"subject": s, "relation": r, "object": o})

        for row in reader:
            if len(row) < 3:
                continue
            s, r, o = row[0].strip(), row[1].strip(), row[2].strip()
            if not s or not r or not o or s.lower() == "subject":
                continue
            cur.execute("""
            INSERT OR IGNORE INTO triples (subject, relation, object, source, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, (s, r, o, file.filename, now_str))
            if cur.rowcount > 0:
                added.append({"subject": s, "relation": r, "object": o})

        conn.commit()
        conn.close()

        bump_processing(len(added))
        return jsonify({
            "triples_added": len(added),
            "triples": added
        })
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV file: {str(e)}"}), 500

@app.route("/semantic")
def semantic_route():
    return render_template_string(SEMANTIC_HTML)

@app.route("/dashboard")
def dashboard_route():
    return render_template_string(DASHBOARD_HTML)

@app.route("/analyze", methods=["POST"])
def analyze_route():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"triples": [], "mode": "error", "message": "No text provided"}), 400

    # Step 1: Try NLP dependency parsing (requires a proper sentence with a verb)
    triples = extract_triples_from_text(text)

    if triples:
        # Save newly extracted triples to the database
        conn = get_db()
        cur = conn.cursor()
        now_str = datetime.now(timezone.utc).isoformat()
        added_count = 0
        for t in triples:
            cur.execute("""
            INSERT OR IGNORE INTO triples (subject, relation, object, source, created_at)
            VALUES (?, ?, ?, 'nlp_extract', ?)
            """, (t["subject"], t["relation"], t["object"], now_str))
            if cur.rowcount > 0:
                added_count += 1
        conn.commit()
        conn.close()
        bump_processing(max(1, added_count))
        return jsonify({
            "triples": triples,
            "mode": "nlp_extract",
            "message": f"Extracted {len(triples)} triple(s) from sentence using NLP dependency parsing."
        })

    # Step 2: NLP returned nothing — input is likely a keyword/phrase (not a full sentence).
    # Fall back to searching the existing knowledge graph database.
    graph_triples, found = lookup_graph_by_keyword(text)
    if found:
        return jsonify({
            "triples": graph_triples,
            "mode": "graph_lookup",
            "message": f"No sentence structure detected. Showing {len(graph_triples)} related triple(s) from the knowledge graph matching your keywords."
        })

    # Step 3: Nothing found anywhere
    return jsonify({
        "triples": [],
        "mode": "no_results",
        "message": "No triples found. Try a full sentence (e.g. 'Albert Einstein developed the theory of relativity') or upload a CSV dataset first."
    })

@app.route("/search", methods=["POST"])
def search_route():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"triples": [], "top_nodes": []}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT subject FROM triples UNION SELECT DISTINCT object FROM triples")
    all_nodes = [r[0] for r in cur.fetchall() if r[0]]

    if not all_nodes:
        conn.close()
        return jsonify({"triples": [], "top_nodes": []})

    model = get_embed_model()
    top_nodes = []

    if model is not None:
        try:
            node_embs = model.encode(all_nodes, convert_to_numpy=True)
            query_emb = model.encode([query], convert_to_numpy=True)
            from sklearn.metrics.pairwise import cosine_similarity
            scores = cosine_similarity(query_emb, node_embs)[0]
            top_k = min(8, len(all_nodes))
            top_indices = np.argsort(scores)[::-1][:top_k]
            top_nodes = [{"name": all_nodes[i], "score": float(scores[i])} for i in top_indices if scores[i] > 0.05]
        except Exception as e:
            print(f"[WARN] Embedding computation error: {e}")

    # Fallback to lexical search if transformer is unavailable or produced empty results
    if not top_nodes:
        q_lower = query.lower()
        matched = [n for n in all_nodes if q_lower in n.lower() or any(w in n.lower() for w in q_lower.split())]
        top_nodes = [{"name": n, "score": 1.0} for n in matched[:8]]
        if not top_nodes and all_nodes:
            top_nodes = [{"name": all_nodes[0], "score": 0.5}]

    # Retrieve ACTUAL connected knowledge graph triples for these top nodes
    matched_names = [n["name"] for n in top_nodes]
    if matched_names:
        placeholders = ",".join(["?"] * len(matched_names))
        cur.execute(f"""
        SELECT subject, relation, object FROM triples
        WHERE subject IN ({placeholders}) OR object IN ({placeholders})
        LIMIT 50
        """, matched_names + matched_names)
        connected_triples = [{"subject": r[0], "relation": r[1], "object": r[2]} for r in cur.fetchall()]
    else:
        connected_triples = []

    conn.close()
    return jsonify({
        "top_nodes": top_nodes,
        "triples": connected_triples
    })

@app.route("/search_node", methods=["POST"])
def search_node_route():
    data = request.get_json() or {}
    node = data.get("node", "").strip()
    if not node:
        return jsonify({"triples": []}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    SELECT subject, relation, object FROM triples
    WHERE subject = ? OR object = ?
    LIMIT 100
    """, (node, node))
    triples = [{"subject": r[0], "relation": r[1], "object": r[2]} for r in cur.fetchall()]
    conn.close()
    return jsonify({"triples": triples})

@app.route("/recent_triples", methods=["GET"])
def recent_triples_route():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT subject, relation, object, source FROM triples ORDER BY id DESC LIMIT 50")
    triples = [{"subject": r[0], "relation": r[1], "object": r[2], "source": r[3]} for r in cur.fetchall()]
    conn.close()
    return jsonify({"triples": triples})

@app.route("/stats", methods=["GET"])
def stats_route():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM triples")
    triples_total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM (SELECT subject FROM triples UNION SELECT object FROM triples)")
    unique_nodes = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT source) FROM triples WHERE source != 'sample_seed' AND source != 'nlp_extract'")
    files_uploaded = cur.fetchone()[0]

    cur.execute("SELECT AVG(rating) FROM feedback")
    avg_rating_val = cur.fetchone()[0]
    avg_rating = round(float(avg_rating_val), 2) if avg_rating_val is not None else 0.0

    conn.close()
    return jsonify({
        "files_uploaded": files_uploaded,
        "triples_total": triples_total,
        "unique_nodes": unique_nodes,
        "avg_rating": avg_rating
    })

@app.route("/metrics", methods=["GET"])
def metrics_route():
    conn = get_db()
    cur = conn.cursor()
    counts = [0] * 7
    cur.execute("""
    SELECT day_index, SUM(count) FROM pipeline_metrics
    GROUP BY day_index
    """)
    for row in cur.fetchall():
        idx = row[0]
        if 0 <= idx < 7:
            counts[idx] = row[1] or 0
    conn.close()
    return jsonify({
        "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "counts": counts
    })

@app.route("/feedback", methods=["POST"])
def feedback_route():
    data = request.get_json() or {}
    try:
        rating = int(data.get("rating", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Rating must be an integer between 1 and 5"}), 400

    if rating < 1 or rating > 5:
        return jsonify({"error": "Rating must be between 1 and 5"}), 400

    comments = data.get("comments", "").strip()
    user = get_optional_user()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO feedback (username, rating, comments, timestamp)
    VALUES (?, ?, ?, ?)
    """, (user, rating, comments, now_str))
    conn.commit()
    conn.close()

    return jsonify({"ok": True, "message": "Feedback recorded"})

@app.route("/get_feedback", methods=["GET"])
def get_feedback_route():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT username, rating, comments, timestamp FROM feedback ORDER BY id DESC LIMIT 20")
    records = [{"username": r[0], "rating": r[1], "comments": r[2], "timestamp": r[3]} for r in cur.fetchall()]
    conn.close()
    return jsonify({"feedback": records})

@app.route("/download_sample_csv")
def download_sample_csv():
    csv_path = os.path.join(BASE_DIR, "knowledge_graph_master_dataset.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, "semanticapp_sample_triples.csv")
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            content = f.read()
        return Response(
            content,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=knowledge_graph_sample.csv"}
        )
    except Exception as e:
        return jsonify({"error": f"Failed to download sample: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM triples")
        cnt = cur.fetchone()[0]
        conn.close()
        return jsonify({"status": "healthy", "database": "connected", "triples_count": cnt}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


# -------------------- MAIN RUNNER --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀 KnowMap with AI server running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

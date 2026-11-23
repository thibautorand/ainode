#!/usr/bin/env python3
"""AI de placement de graphe de flux de donnees (nodes + links) en positions x/y.

Entree : un JSON {"nodes": [...], "links": [...]} (schema de ton exemple).
Sortie : positions (x, y) pour chaque noeud + le trace (route) de chaque lien.

Principe
--------
1. Un initialiseur en couches (Sugiyama simplifie : profondeur topologique en x,
   tri par barycentres en y) donne un point de depart propre.
2. Un "surrogate" smooth du bareme de cout est minimise par descente de
   gradient (Adam + gradient numerique) : chaque penalite du tableau est
   remplacee par une version differentiable (softplus / sigmoide) qui vaut
   ~0 quand la situation est bonne et ~la penalite du bareme quand elle est
   violee. Les discontinuites sont ainsi lissees, donc le gradient est decent.
3. A chaque fin de run, le cout EXACT du bareme est evalue sur le meilleur
   layout (rapport CSV). Les poids multiplicateurs de chaque terme sont
   ensuitre mis a jour (les termes encore violes montent, les termes satisfaits
   baissent legerement) et sauvegardes dans weights.csv, une ligne par run.
   Le run suivant repart avec ces poids : l'IA apprend d'un run a l'autre.
4. Plusieurs runs et restarts, on garde le meilleur layout global.

Utilisation
-----------
    python flow_layout_ai.py graph.json
    python flow_layout_ai.py graph.json --runs 5 --iters 400 --restarts 4
    python flow_layout_ai.py --demo                      # graphe de test
    python flow_layout_ai.py graph.json --png            # apercu PNG (matplotlib)

Sorties dans --outdir (defaut ./layout_out) :
    weights.csv       poids des termes, une ligne par run (reprise possible)
    history.csv       cout exact par run + detail (brut) de chaque terme
    best_layout.json  meilleur layout (x, y par noeud + route de chaque lien)
    layout_runN.json  meilleur layout de chaque run
    preview_runN.png  apercu si --png

Dependances : numpy obligatoire, pandas optionnel (CSV), matplotlib optionnel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np

try:
    import pandas as pd  # optionnel, pour les CSV
except Exception:
    pd = None


# ---------------------------------------------------------------------------
# Bareme de cout (termes exacts). Les "raw" ci-dessous sont deja en unites de
# cout du bareme ; les poids w_k appris sont des multiplicateurs.
# ---------------------------------------------------------------------------

TERM_KEYS = [
    "overlap",     # deux etapes qui se chevauchent          (+100000)
    "traversed",   # lien derriere une etape                 (+50000 / etape)
    "crossing",    # deux liens qui se croisent              (+5000)
    "backward",    # lien de droite a gauche                 (+15/px puis (px-50)^2)
    "near",        # etapes reliees a moins de 70 px         (+800)
    "touching",    # etapes presque collees sans se toucher  (+600)
    "far",         # etapes reliees a plus de 650 px         (+400 / tranche 300px)
    "linknear",    # deux liens a moins de 30 px             (+300)
    "bends",       # coude                                   (+200)
    "steep",       # diagonale abrupte (plus haut que large) (+50 / px au-dela)
    "length",      # longueur d'un lien au-dela de 60 px     (+1/px)
]

CAPS = {  # valeurs de normalisation pour l'apprentissage des poids
    "overlap": 100000.0, "traversed": 50000.0, "crossing": 5000.0,
    "backward": 400.0, "near": 800.0, "touching": 600.0, "far": 400.0,
    "linknear": 300.0, "bends": 400.0, "steep": 300.0, "length": 200.0,
}

WEIGHT_COLS = ["w_" + k for k in TERM_KEYS]


# ---------------------------------------------------------------------------
# Petites maths smooth
# ---------------------------------------------------------------------------

def softplus(z):
    return np.logaddexp(0.0, z)


def smax0(z, s):
    """max(0, z) lisse, rampe de largeur ~s."""
    return s * softplus(z / s)


def sig(z):
    """Sigmoide numeriquement stable (sature a 0/1 loin du point d'inflexion)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def smooth_min(a, b, k=5.0):
    """min(a, b) lisse (log-sum-exp negatif)."""
    return -k * np.logaddexp(-a / k, -b / k)


# ---------------------------------------------------------------------------
# Graphe : chargement + precalcs (ancrages de ports, paires reliees...)
# ---------------------------------------------------------------------------

class FlowGraph:
    def __init__(self, spec):
        if "nodes" not in spec:
            raise ValueError("JSON invalide : cle 'nodes' absente")
        self.spec = spec
        nodes = spec["nodes"]

        # liens : on dedoublonne les liens strictement identiques
        uniq = {}
        for lk in spec.get("links", []):
            key = (lk.get("from"), lk.get("to"), lk.get("output_port"),
                   lk.get("input_port"), lk.get("kind"))
            uniq[key] = lk
        self.links = list(uniq.values())

        self.node_ids = [n["id"] for n in nodes]
        self.N = len(nodes)
        self.idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.wid = np.array([float(n.get("width", 240)) for n in nodes])
        self.hei = np.array([float(n.get("height", 117)) for n in nodes])

        self.L = len(self.links)
        self.src = np.array([self.idx[lk["from"]] for lk in self.links], int)
        self.dst = np.array([self.idx[lk["to"]] for lk in self.links], int)

        # ancres de ports : fraction de la largeur / hauteur du noeud
        self.o_wx = np.zeros(self.L)
        self.o_hy = np.zeros(self.L)
        self.i_wx = np.zeros(self.L)
        self.i_hy = np.zeros(self.L)
        cnt = {}
        for li, lk in enumerate(self.links):
            s, d = self.src[li], self.dst[li]
            op = lk.get("output_port") or "output"
            ip = lk.get("input_port") or "input"
            if op == "linked_value":
                key = (s, "lvo")
                k = cnt.get(key, 0)
                total = int(nodes[s].get("linked_values_out", 1) or 1)
                self.o_wx[li] = (k + 1) / (total + 1)
                self.o_hy[li] = 1.0
                cnt[key] = k + 1
            else:
                outs = nodes[s].get("outputs") or ["output"]
                j = outs.index(op) if op in outs else 0
                self.o_wx[li] = 1.0
                self.o_hy[li] = (j + 1) / (len(outs) + 1)
            if ip == "linked_value":
                key = (d, "lvi")
                k = cnt.get(key, 0)
                total = int(nodes[d].get("linked_values_in", 1) or 1)
                self.i_wx[li] = (k + 1) / (total + 1)
                self.i_hy[li] = 1.0
                cnt[key] = k + 1
            else:
                ins = nodes[d].get("inputs") or ["input"]
                j = ins.index(ip) if ip in ins else 0
                self.i_wx[li] = 0.0
                self.i_hy[li] = (j + 1) / (len(ins) + 1)

        # paires reliees (non ordonnees, uniques) pour near/touching/far
        pairs = {(min(int(a), int(b)), max(int(a), int(b)))
                 for a, b in zip(self.src, self.dst)}
        self.pair_i = np.array([p[0] for p in sorted(pairs)], int)
        self.pair_j = np.array([p[1] for p in sorted(pairs)], int)

    # ---- trace des liens (routes orthogonales) -----------------------------
    # vers l'avant  : (sx,sy) (mx,sy) (mx,ty) (tx,ty) + points rembourrage
    # vers l'arriere: (sx,sy) (sx+pad,sy) (sx+pad,ym) (tx-pad,ym) (tx-pad,ty) (tx,ty)
    # Tout est fonction (lisse par morceaux) de xs/ys : compatible gradient.
    def routes(self, xs, ys, pad=25.0):
        sx = xs[self.src] + self.o_wx * self.wid[self.src]
        sy = ys[self.src] + self.o_hy * self.hei[self.src]
        tx = xs[self.dst] + self.i_wx * self.wid[self.dst]
        ty = ys[self.dst] + self.i_hy * self.hei[self.dst]
        fwd = tx >= (sx - 1e-6)
        mx = 0.5 * (sx + tx)
        ym = 0.5 * (sy + ty)
        P = np.empty((self.L, 6, 2), dtype=float)
        P[:, 0, 0] = sx
        P[:, 0, 1] = sy
        x1 = np.where(fwd, mx, sx + pad)
        P[:, 1, 0] = x1
        P[:, 1, 1] = sy
        P[:, 2, 0] = x1
        P[:, 2, 1] = np.where(fwd, ty, ym)
        x3 = np.where(fwd, tx, tx - pad)
        P[:, 3, 0] = x3
        P[:, 3, 1] = np.where(fwd, ty, ym)
        P[:, 4, 0] = x3
        P[:, 4, 1] = ty
        P[:, 5, 0] = tx
        P[:, 5, 1] = ty
        return P, sy, ty


# Geometrie segment/rectangle (version float pur, utilisee dans le surrogate
# pour les paresseux candidats et dans le cout exact).

def point_rect_dist(px, py, x1, y1, x2, y2):
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def pt_seg_dist(p, a, b):
    abx, aby = b[0] - a[0], b[1] - a[1]
    apx, apy = p[0] - a[0], p[1] - a[1]
    ab2 = abx * abx + aby * aby
    tt = 0.0 if ab2 <= 1e-12 else max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    return math.hypot(apx - tt * abx, apy - tt * aby)


def seg_seg_dist(p1, p2, q1, q2):
    """Distance minimale entre deux segments (formule clamp + endpoints)."""
    best = min(pt_seg_dist(p1, q1, q2), pt_seg_dist(p2, q1, q2),
               pt_seg_dist(q1, p1, p2), pt_seg_dist(q2, p1, p2))
    rx, ry = p2[0] - p1[0], p2[1] - p1[1]
    sx_, sy_ = q2[0] - q1[0], q2[1] - q1[1]
    wx, wy = q1[0] - p1[0], q1[1] - p1[1]
    a = rx * rx + ry * ry
    bb = rx * sx_ + ry * sy_
    c = sx_ * sx_ + sy_ * sy_
    if a > 1e-12 and c > 1e-12:
        d = rx * wx + ry * wy
        e = sx_ * wx + sy_ * wy
        den = a * c - bb * bb
        if den > 1e-12:
            sc = min(1.0, max(0.0, (bb * e - c * d) / den))
            tc = min(1.0, max(0.0, (a * e - bb * d) / den))
            best = min(best, math.hypot(wx + sc * rx - tc * sx_,
                                        wy + sc * ry - tc * sy_))
    return best


def polyline_min_dist(A, B):
    """Distance minimale entre deux polylignes donnees par leurs sommets."""
    best = float("inf")
    for i in range(len(A) - 1):
        if (A[i + 1][0] - A[i][0]) ** 2 + (A[i + 1][1] - A[i][1]) ** 2 < 1e-12:
            continue
        for j in range(len(B) - 1):
            if (B[j + 1][0] - B[j][0]) ** 2 + (B[j + 1][1] - B[j][1]) ** 2 < 1e-12:
                continue
            best = min(best, seg_seg_dist(A[i], A[i + 1], B[j], B[j + 1]))
    return best


def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def segs_properly_intersect(p1, p2, q1, q2):
    o1 = _orient(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1])
    o2 = _orient(p1[0], p1[1], p2[0], p2[1], q2[0], q2[1])
    o3 = _orient(q1[0], q1[1], q2[0], q2[1], p1[0], p1[1])
    o4 = _orient(q1[0], q1[1], q2[0], q2[1], p2[0], p2[1])
    eps = 1e-9
    cross1 = (o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)
    cross2 = (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    return cross1 and cross2


def seg_intersects_rect(p1, p2, x1, y1, x2, y2):
    """Intersection inclusive segment/rectangle (Liang-Barsky)."""
    t0, t1 = 0.0, 1.0
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    for p_, q_ in ((-dx, p1[0] - x1), (dx, x2 - p1[0]),
                   (-dy, p1[1] - y1), (dy, y2 - p1[1])):
        if p_ == 0:
            if q_ < 0:
                return False
        else:
            r_ = q_ / p_
            if p_ < 0:
                if r_ > t1:
                    return False
                if r_ > t0:
                    t0 = r_
            else:
                if r_ < t0:
                    return False
                if r_ < t1:
                    t1 = r_
    return True


# ---------------------------------------------------------------------------
# Surrogate smooth du bareme (a minimiser)
# ---------------------------------------------------------------------------

def surrogate_cost(g: FlowGraph, params, w, sharp):
    """Retourne (cout total pondere, dictionnaire des termes bruts).

    params = concatenation [xs, ys] (coins superieurs gauches, en px).
    Chaque penalite du bareme est remplacee par une version differentiable
    qui tend vers la penalite exacte quand paremetre la situation est espanclee.
    `sharp` >= 1 aiguisiti progressivement les sigmoides (recuit du lissage).
    """
    N = g.N
    xs = params[:N]
    ys = params[N:]
    l = xs
    r = xs + g.wid
    t = ys
    b = ys + g.hei

    Lm = l[:, None]
    Rm = r[:, None]
    Tm = t[:, None]
    Bm = b[:, None]

    parts = {k: 0.0 for k in TERM_KEYS}

    # ---- chevauchement de noeuds (toutes paires) ---------------------------
    ovx = smax0(np.minimum(Rm, Rm.T) - np.maximum(Lm, Lm.T), 5.0)
    ovy = smax0(np.minimum(Bm, Bm.T) - np.maximum(Tm, Tm.T), 5.0)
    pmin = smooth_min(ovx, ovy, 5.0)
    iu = ~np.eye(N, dtype=bool)
    parts["overlap"] = 100000.0 * (sig((pmin - 0.5) * 0.5 * sharp) * iu).sum() * 0.5

    # ---- proximite / collage / trop loin entre etapes RELIEES --------------
    gx = smax0(np.maximum(Lm, Lm.T) - np.minimum(Rm, Rm.T), 6.0)
    gy = smax0(np.maximum(Tm, Tm.T) - np.minimum(Bm, Bm.T), 6.0)
    G = np.sqrt(gx * gx + gy * gy + 1e-9)
    gap = G[g.pair_i, g.pair_j]
    openg = sig((gap - 1.5) * 0.5 * sharp)  # annule a peu pres si chevauchement
    parts["near"] = (800.0 * sig((70.0 - gap) * (sharp / 8.0)) * openg).sum()
    parts["touching"] = (600.0 * sig((10.0 - gap) * (sharp / 3.0)) * openg).sum()

    cxi = l[g.pair_i] + 0.5 * g.wid[g.pair_i]
    cyi = t[g.pair_i] + 0.5 * g.hei[g.pair_i]
    cxj = l[g.pair_j] + 0.5 * g.wid[g.pair_j]
    cyj = t[g.pair_j] + 0.5 * g.hei[g.pair_j]
    dc = np.hypot(cxi - cxj, cyi - cyj)
    parts["far"] = (400.0 * smax0(dc - 650.0, 40.0) / 300.0).sum()

    # ---- liens : routes, longueur, retour arriere, coudes, pente ----------
    P, sy, ty = g.routes(xs, ys)
    dx = P[:, 1:, 0] - P[:, :-1, 0]
    dy = P[:, 1:, 1] - P[:, :-1, 1]
    Ltot = np.hypot(dx, dy).sum(axis=1)
    parts["length"] = smax0(Ltot - 60.0, 5.0).sum()

    hm = (dy == 0.0)  # segments horizontaux (dy exactement 0 par construction)
    B = (smax0(-dx, 3.0) * hm).sum(axis=1)
    parts["backward"] = (15.0 * B + smax0(B - 50.0, 2.0) ** 2).sum()

    dys = np.abs(ty - sy)
    dxs = np.abs(xs[g.dst] - xs[g.src])
    parts["bends"] = (200.0 * (2.0 * sig((dys - 0.5) * 0.5 * sharp)
                               + 2.0 * sig((B - 0.5) * 0.5 * sharp))).sum()
    parts["steep"] = (50.0 * smax0(dys - (dxs + 1e-6), 4.0)).sum()

    # ---- lien qui passe derriere une etape ---------------------------------
    bminx = P[:, :, 0].min(axis=1)
    bmaxx = P[:, :, 0].max(axis=1)
    bminy = P[:, :, 1].min(axis=1)
    bmaxy = P[:, :, 1].max(axis=1)
    gl = np.maximum(bminx[:, None], l[None, :]) - np.minimum(bmaxx[:, None], r[None, :])
    gt = np.maximum(bminy[:, None], t[None, :]) - np.minimum(bmaxy[:, None], b[None, :])
    cand = (gl < 14.0) & (gt < 14.0)
    for li in range(g.L):
        cand[li, g.src[li]] = False
        cand[li, g.dst[li]] = False
    pen = 0.0
    for li, ni in np.argwhere(cand):
        dmin = min(point_rect_dist(float(px), float(py),
                                   float(l[ni]), float(t[ni]),
                                   float(r[ni]), float(b[ni]))
                   for px, py in P[li])
        pen += 50000.0 * sig((10.0 - dmin) * (sharp / 2.5))
    parts["traversed"] = pen

    # ---- croisements et proximity entre liens ------------------------------
    shared = ((g.src[:, None] == g.src[None, :]) |
              (g.dst[:, None] == g.dst[None, :]) |
              (g.src[:, None] == g.dst[None, :]) |
              (g.dst[:, None] == g.src[None, :]))
    iuL = np.triu(np.ones((g.L, g.L), dtype=bool), 1)
    gl2 = np.maximum(bminx[:, None], bminx[None, :]) - np.minimum(bmaxx[:, None], bmaxx[None, :])
    gt2 = np.maximum(bminy[:, None], bminy[None, :]) - np.minimum(bmaxy[:, None], bmaxy[None, :])
    cand2 = (gl2 < 40.0) & (gt2 < 40.0) & iuL & ~shared & (g.L > 1)
    for li, lj in np.argwhere(cand2):
        d = polyline_min_dist(P[li], P[lj])
        parts["crossing"] += 5000.0 * sig((20.0 - d) * (sharp / 6.0))
        parts["linknear"] += 300.0 * sig((30.0 - d) * (sharp / 6.0))

    total = 0.0
    for k in TERM_KEYS:
        total += w[k] * parts[k]
    return total, parts


# ---------------------------------------------------------------------------
# Cout EXACT du bareme (evaluation, rapport, apprentissage des poids)
# ---------------------------------------------------------------------------

def exact_breakdown(g: FlowGraph, xs, ys):
    l = np.asarray(xs, float)
    t = np.asarray(ys, float)
    r = l + g.wid
    b = t + g.hei
    raw = {k: 0.0 for k in TERM_KEYS}

    # chevauchement
    for i in range(g.N):
        for j in range(i + 1, g.N):
            if (min(r[i], r[j]) - max(l[i], l[j]) > 1e-6 and
                    min(b[i], b[j]) - max(t[i], t[j]) > 1e-6):
                raw["overlap"] += 100000.0

    P, sy, ty = g.routes(l, t)

    # par lien : backward, longueur, coudes, pente
    for li in range(g.L):
        pts = [tuple(P[li, k]) for k in range(6)]
        segs = [(pts[k], pts[k + 1]) for k in range(5)
                if (pts[k + 1][0] - pts[k][0]) ** 2 + (pts[k + 1][1] - pts[k][1]) ** 2 > 1e-12]
        B = sum(max(0.0, -(p2[0] - p1[0])) for p1, p2 in segs if p1[1] == p2[1])
        raw["backward"] += 15.0 * B + (max(0.0, B - 50.0) ** 2)
        Ltot = sum(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in segs)
        raw["length"] += max(0.0, Ltot - 60.0)
        if segs:
            dirs = [(1 if p2[0] > p1[0] else -1 if p2[0] < p1[0] else 0,
                     1 if p2[1] > p1[1] else -1 if p2[1] < p1[1] else 0)
                    for p1, p2 in segs]
            corners = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
            raw["bends"] += 200.0 * corners
        dys = abs(float(ty[li]) - float(sy[li]))
        dxs = abs(float(P[li, -1, 0]) - float(P[li, 0, 0]))
        if dys > dxs:
            raw["steep"] += 50.0 * (dys - dxs)

    # paires reliees : near / touching / far
    for i, j in zip(g.pair_i, g.pair_j):
        gxv = max(0.0, max(l[i], l[j]) - min(r[i], r[j]))
        gyv = max(0.0, max(t[i], t[j]) - min(b[i], b[j]))
        gap = math.hypot(gxv, gyv)
        if gap < 70.0:
            raw["near"] += 800.0
        if 0.0 < gap < 10.0:
            raw["touching"] += 600.0
        d = math.hypot((l[i] + 0.5 * g.wid[i]) - (l[j] + 0.5 * g.wid[j]),
                       (t[i] + 0.5 * g.hei[i]) - (t[j] + 0.5 * g.hei[j]))
        if d > 650.0:
            raw["far"] += 400.0 * (1 + int((d - 650.0) // 300.0))

    # lien derriere une etape
    for li in range(g.L):
        s, dnode = g.src[li], g.dst[li]
        pts = [tuple(P[li, k]) for k in range(6)]
        segs = [(pts[k], pts[k + 1]) for k in range(5)
                if (pts[k + 1][0] - pts[k][0]) ** 2 + (pts[k + 1][1] - pts[k][1]) ** 2 > 1e-12]
        for ni in range(g.N):
            if ni == s or ni == dnode:
                continue
            x1_, y1_, x2_, y2_ = l[ni], t[ni], r[ni], b[ni]
            if any(seg_intersects_rect(p1, p2, x1_, y1_, x2_, y2_) for p1, p2 in segs):
                raw["traversed"] += 50000.0

    # croisements + liens trop proches
    for li in range(g.L):
        for lj in range(li + 1, g.L):
            if ({g.src[li], g.dst[li]} & {g.src[lj], g.dst[lj]}):
                continue
            A = [tuple(P[li, k]) for k in range(6)]
            B_ = [tuple(P[lj, k]) for k in range(6)]
            if polyline_min_dist(A, B_) < 30.0:
                raw["linknear"] += 300.0
            Asegs = [(A[k], A[k + 1]) for k in range(5)
                     if (A[k + 1][0] - A[k][0]) ** 2 + (A[k + 1][1] - A[k][1]) ** 2 > 1e-12]
            Bsegs = [(B_[k], B_[k + 1]) for k in range(5)
                     if (B_[k + 1][0] - B_[k][0]) ** 2 + (B_[k + 1][1] - B_[k][1]) ** 2 > 1e-12]
            cross = sum(1 for p1, p2 in Asegs for q1, q2 in Bsegs
                        if segs_properly_intersect(p1, p2, q1, q2))
            raw["crossing"] += 5000.0 * cross

    return raw


# ---------------------------------------------------------------------------
# Optimisation : Adam + gradient numerique (forward differences)
# ---------------------------------------------------------------------------

def optimize(g: FlowGraph, w, x0, y0, iters=250, lr=3.0,
             sharp_final=4.0, patience=80):
    dim = 2 * g.N
    p = np.concatenate([np.asarray(x0, float), np.asarray(y0, float)])
    m = np.zeros(dim)
    v = np.zeros(dim)
    b1, b2, eps = 0.9, 0.999, 1e-8
    f_best = float("inf")
    p_best = p.copy()
    last_improve = 0

    for it in range(iters):
        sharp = 1.0 + (sharp_final - 1.0) * (it / max(1, iters - 1))
        f0, _ = surrogate_cost(g, p, w, sharp)
        grad = np.empty(dim)
        for i in range(dim):
            pp = p.copy()
            pp[i] += 1.0
            grad[i] = (surrogate_cost(g, pp, w, sharp)[0] - f0) / 1.0
        step = it + 1
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        mh = m / (1 - b1 ** step)
        vh = v / (1 - b2 ** step)
        lr_t = lr * max(0.05, 1.0 - it / iters)

        if f0 < f_best - 1e-9:
            f_best = f0
            p_best = p.copy()
            last_improve = it
        elif it - last_improve > patience:
            break

        p = p - lr_t * mh / (np.sqrt(vh) + eps)

    return p_best, f_best


def update_weights(w, raw):
    """Monte le poids des termes encore violes, baisse doucement les satisfaits."""
    w2 = {}
    for k in TERM_KEYS:
        e = min(2.5, raw.get(k, 0.0) / CAPS[k])
        fac = 0.93 if e <= 0.02 else 1.0 + 0.7 * e
        w2[k] = min(25.0, max(0.25, w.get(k, 1.0) * fac))
    return w2


# ---------------------------------------------------------------------------
# Initialisation en couches (profondeur en x, barycentres en y)
# ---------------------------------------------------------------------------

def layered_init(g: FlowGraph, rng, jitter=0.0, col_w=400.0):
    N = g.N
    depth = np.zeros(N)
    for _ in range(N + 2):
        cand = depth[g.src] + 1.0
        upd = cand > depth[g.dst] + 1e-12
        if not upd.any():
            break
        depth[g.dst[upd]] = cand[upd]

    nbr_left = [[] for _ in range(N)]
    nbr_right = [[] for _ in range(N)]
    for li in range(g.L):
        s, d = g.src[li], g.dst[li]
        if s != d:
            nbr_right[s].append(int(d))
            nbr_left[d].append(int(s))

    col_nodes = {}
    for i, dp in enumerate(depth):
        col_nodes.setdefault(int(dp), []).append(i)
    order = {c: list(v) for c, v in col_nodes.items()}

    ypos = np.arange(N) * 150.0
    for p in range(8):
        cols = sorted(order) if p % 2 == 0 else sorted(order, reverse=True)
        for c in cols:
            lst = order[c]

            def bary(i):
                nbs = (nbr_left[i] + nbr_right[i]) if 0 < c < max(order) else \
                      (nbr_right[i] if c == 0 else nbr_left[i])
                vals = [ypos[k] for k in nbs] or [i * 137.0]
                return sum(vals) / len(vals)

            lst.sort(key=bary)
            cur = 60.0
            for i in lst:
                ypos[i] = cur
                cur += g.hei[i] + 60.0

    xs = 100.0 + depth * col_w
    ys = ypos.copy()
    if jitter > 0:
        xs = xs + rng.normal(0.0, jitter * 0.3, N)
        ys = ys + rng.normal(0.0, jitter, N)
    return xs, ys


# ---------------------------------------------------------------------------
# Sorties : CSV (poids, historique), JSON du layout, apercu PNG
# ---------------------------------------------------------------------------

def append_row_csv(path, row):
    exists = os.path.exists(path)
    if pd is not None:
        pd.DataFrame([row]).to_csv(path, mode="a", header=not exists, index=False)
    else:
        with open(path, "a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                wr.writeheader()
            wr.writerow(row)


def load_last_weights(path):
    if not os.path.exists(path):
        return None
    rows = []
    if pd is not None:
        df = pd.read_csv(path)
        if df.empty:
            return None
        rows = [df.iloc[-1].to_dict()]
    else:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    if not rows:
        return None
    row = rows[-1]
    out = {}
    for k, col in zip(TERM_KEYS, WEIGHT_COLS):
        try:
            out[k] = float(row[col])
        except (KeyError, TypeError, ValueError):
            pass
    return out or None


def _clean_polyline(P):
    pts, last = [], None
    for p in P:
        q = (round(float(p[0]), 2), round(float(p[1]), 2))
        if last is None or q != last:
            pts.append([q[0], q[1]])
        last = q
    return pts


def save_layout(path, g, xs, ys, meta):
    P, _, _ = g.routes(np.asarray(xs, float), np.asarray(ys, float))
    nodes_out = []
    for i, n in enumerate(g.spec["nodes"]):
        d = dict(n)
        d["x"] = round(float(xs[i]), 2)
        d["y"] = round(float(ys[i]), 2)
        nodes_out.append(d)
    links_out = []
    for li, lk in enumerate(g.links):
        d = dict(lk)
        d["route"] = _clean_polyline(P[li])
        links_out.append(d)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "nodes": nodes_out, "links": links_out},
                  f, ensure_ascii=False, indent=2)


def render_png(path, g, xs, ys, title=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except Exception:
        print("  (matplotlib absent : PNG ignore)")
        return
    P, _, _ = g.routes(np.asarray(xs, float), np.asarray(ys, float))
    fig, ax = plt.subplots(figsize=(15, 8.5))
    facec = {"input": "#3f6f4f", "output": "#6f4b3f", "combine": "#3f4f6f"}
    for li in range(g.L):
        pts = _clean_polyline(P[li])
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                lw=1.1, color="#6b78b8", zorder=1)
    for i, n in enumerate(g.spec["nodes"]):
        fc = facec.get(n.get("family", ""), "#2e3958")
        ax.add_patch(FancyBboxPatch(
            (float(xs[i]), float(ys[i])), float(g.wid[i]), float(g.hei[i]),
            boxstyle="round,pad=0,rounding_size=8",
            fc=fc, ec="#8891c0", lw=1.0, zorder=2))
        ax.text(float(xs[i]) + 12, float(ys[i]) + 22,
                n.get("title", n["id"]), color="white", fontsize=7, zorder=3)
    ax.set_title(title)
    ax.autoscale_view()
    ax.margins(0.08)
    ax.axis("off")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe de demo (meme schema que ton exemple)
# ---------------------------------------------------------------------------

TITLES = {
    "source": ("Fichier source", "input"), "published": ("Source publiee", "input"),
    "select": ("Select columns".replace("Select columns", "Selectionner les colonnes"), "transform"),
    "filter": ("Filtrer les lignes", "transform"), "sort": ("Trier les lignes", "transform"),
    "dedup": ("Supprimer les doublons", "transform"), "formula": ("Colonne calculee", "transform"),
    "group": ("Grouper les etapes", "transform"), "lookup": ("Rechercher des valeurs", "combine"),
    "clean": ("Nettoyer les donnees", "transform"), "dates": ("Dates", "transform"),
    "check": ("Controler", "transform"), "changes": ("Changements depuis la derniere fois", "transform"),
    "append": ("Ajouter les lignes", "combine"), "workbook": ("Combiner les feuilles", "combine"),
    "compare": ("Comparer", "combine"), "split": ("Fractionner une colonne", "transform"),
    "layout": ("Mise en page", "transform"),
    "pilot": ("Publier vers Pilotage", "output"), "mail": ("Courrier Outlook", "output"),
    "output": ("Publier une sortie", "output"),
}


def make_demo_graph(seed=7, n_middle=14, n_out=3):
    rng = random.Random(seed)
    kinds_middle = ["select", "filter", "sort", "dedup", "formula", "group",
                    "lookup", "clean", "dates", "check", "changes", "append",
                    "workbook", "compare", "split", "layout"]
    nodes = []
    for i in range(2):
        nodes.append({"id": f"node_{i + 1}", "kind": "source" if i == 0 else "published",
                      "family": "input", "title": TITLES["source" if i == 0 else "published"][0],
                      "width": 240, "height": 117, "inputs": [], "outputs": ["output"],
                      "linked_values_in": 0, "linked_values_out": 0})
    for k in range(n_middle):
        kind = kinds_middle[k % len(kinds_middle)]
        title, fam = TITLES[kind]
        ins = ["input"]
        outs = ["output"]
        if kind == "lookup":
            ins = ["input", "lookup"]
        elif kind == "check":
            ins = ["input", "reference"]
        elif kind == "compare":
            ins = ["input", "compare"]
            outs = ["only_first", "both", "only_second"]
        elif kind == "split":
            outs = ["output", "remainder"]
        d = max(rng.random(), 0.15)
        nodes.append({"id": f"node_{k + 3}", "kind": kind, "family": fam,
                      "title": title, "width": 240, "height": 117,
                      "inputs": ins, "outputs": outs,
                      "linked_values_in": 0, "linked_values_out": 0,
                      "_rnd": d})
    out_kinds = ["pilot", "mail", "output"]
    for k in range(n_out):
        kind = out_kinds[k % len(out_kinds)]
        title, fam = TITLES[kind]
        nodes.append({"id": f"node_{2 + n_middle + k + 1}", "kind": kind,
                      "family": fam, "title": title, "width": 240, "height": 117,
                      "inputs": ["input"], "outputs": [],
                      "linked_values_in": 0, "linked_values_out": 0})

    links = []
    n_all = len(nodes)
    for i in range(2, n_all):
        n_parents = 1 if rng.random() < 0.55 else 2
        seen = set()
        node = nodes[i]
        free_ports = list(node.get("inputs", ["input"]))
        for p_i in range(n_parents):
            j = rng.randrange(0, i)
            if j in seen:
                continue
            seen.add(j)
            if len(free_ports) > 1:
                ip = free_ports.pop(0) if p_i == 0 else free_ports[-1]
            else:
                ip = "input"
            op = node["outputs"][0] if node.get("outputs") else "output"
            links.append({"from": nodes[j]["id"], "to": node["id"],
                          "output_port": op, "input_port": ip, "kind": "data"})
    # deux liens "linked_value" pour tester les valeurs reliees
    for _ in range(3):
        a = rng.randrange(0, max(1, n_all - 2))
        b = rng.randrange(min(a + 1, n_all - 1), n_all)
        if a != b:
            links.append({"from": nodes[a]["id"], "to": nodes[b]["id"],
                          "output_port": "linked_value",
                          "input_port": "linked_value", "kind": "linked_value"})
    for lk in links:
        if lk["kind"] == "linked_value":
            for nd in nodes:
                if nd["id"] == lk["from"]:
                    nd["linked_values_out"] += 1
                elif nd["id"] == lk["to"]:
                    nd["linked_values_in"] += 1
    for nd in nodes:
        nd.pop("_rnd", None)
    return {"note": "graphe de demo genere aleatoirement", "nodes": nodes, "links": links}


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def progressive(spec, rng, iters, lr):
    """Placement causal : seuls les deux axes du nouveau noeud sont libres."""
    nodes = {n["id"]: n for n in spec["nodes"]}
    if len(nodes) != len(spec["nodes"]):
        raise ValueError("Identifiants de noeuds dupliqués")
    links = spec.get("links", [])
    parents = {key: set() for key in nodes}
    for link in links:
        parents[link["to"]].add(link["from"])
    placed = {}
    remaining = dict(nodes)
    while remaining:
        # Continuer la procédure active. Dans un cycle, ouvrir une boucle.
        key = min(remaining, key=lambda k: (len(parents[k] - placed.keys()),
                                            -len(parents[k] & placed.keys())))
        node = remaining.pop(key)
        predecessors = parents[key] & placed.keys()
        if predecessors:
            x = max(placed[k][0] + nodes[k].get("width", 240) for k in predecessors) + 100
            y = np.mean([placed[k][1] for k in placed if k in predecessors])
        else:
            x = 100.
            y = max((p[1] + nodes[k].get("height", 117) for k, p in placed.items()), default=-20.) + 80
        ids = [*placed, key]
        active = set(ids)
        graph = FlowGraph({"nodes": [nodes[k] for k in ids], "links": [
            link for link in links if link["from"] in active and link["to"] in active]})
        xs = np.array([placed[k][0] for k in placed] + [x], dtype=float)
        ys = np.array([placed[k][1] for k in placed] + [y], dtype=float)
        weights = dict.fromkeys(TERM_KEYS, 1.)
        best = sum(exact_breakdown(graph, xs, ys).values())
        winner = (x, y)
        if placed and predecessors:
            xs[-1], ys[-1] = np.array(winner) + rng.normal(0, [35, 70])
            for step in range(iters + 1):
                score = sum(exact_breakdown(graph, xs, ys).values())
                if score < best:
                    winner, best = (xs[-1], ys[-1]), score
                def objective():
                    return surrogate_cost(graph, np.r_[xs, ys], weights, 2.)[0]
                base = objective()
                gradient = []
                for axis in (xs, ys):
                    axis[-1] += 1
                    gradient.append(objective() - base)
                    axis[-1] -= 1
                norm = math.hypot(*gradient)
                if norm < 1e-6:
                    break
                scale = lr / (norm * math.sqrt(1 + step / 10))
                xs[-1] -= scale * gradient[0]
                ys[-1] -= scale * gradient[1]
        placed[key] = winner
    return tuple(np.array([placed[k][axis] for k in nodes]) for axis in (0, 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Placement AI d'un graphe de flux (nodes + links).")
    ap.add_argument("graph", nargs="?", help="fichier JSON du graphe")
    ap.add_argument("--runs", type=int, default=3, help="runs d'entrainement (les poids se propagent)")
    ap.add_argument("--restarts", type=int, default=3, help="departs aleatoires par run")
    ap.add_argument("--iters", type=int, default=250, help="iterations Adam par restart")
    ap.add_argument("--lr", type=float, default=3.0, help="taux d'apprentissage Adam")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="layout_out")
    ap.add_argument("--png", action="store_true", help="apercus PNG (matplotlib requis)")
    ap.add_argument("--demo", action="store_true", help="genere un graphe de demo et l'optimise")
    ap.add_argument("--progressive", action="store_true", help="placer puis figer chaque noeud")
    args = ap.parse_args(argv)

    if args.demo:
        spec = make_demo_graph(seed=args.seed or 7)
    elif args.graph:
        with open(args.graph, encoding="utf-8") as f:
            spec = json.load(f)
    else:
        ap.error("fournis un JSON de graphe ou --demo")

    g = FlowGraph(spec)
    os.makedirs(args.outdir, exist_ok=True)
    if args.progressive:
        xs, ys = progressive(spec, np.random.default_rng(args.seed), args.iters, args.lr)
        raw = exact_breakdown(g, xs, ys)
        save_layout(os.path.join(args.outdir, "best_layout.json"), g, xs, ys,
                    {"mode": "progressive", "cost_exact": sum(raw.values()), "raw": raw})
        if args.png:
            render_png(os.path.join(args.outdir, "preview.png"), g, xs, ys)
        return
    w_path = os.path.join(args.outdir, "weights.csv")
    hist_path = os.path.join(args.outdir, "history.csv")

    w = load_last_weights(w_path) or {k: 1.0 for k in TERM_KEYS}
    if load_last_weights(w_path):
        print(f"Poids repris de {w_path} (entrainement continue).")

    global_best = None
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")

    for run in range(1, args.runs + 1):
        run_best = None
        for rs in range(args.restarts):
            seed = args.seed * 100000 + run * 1000 + rs
            rng = np.random.default_rng(seed)
            if rs == 0:
                j = 0.0 if run == 1 else 20.0
            elif rs == args.restarts - 1:
                j = 120.0
            else:
                j = 50.0 + 25.0 * rs
            x0, y0 = layered_init(g, rng, jitter=j)
            pb, fb = optimize(g, w, x0, y0, iters=args.iters, lr=args.lr)
            xs, ys = pb[:g.N], pb[g.N:]
            raw = exact_breakdown(g, xs, ys)
            cost = sum(w[k] * raw[k] for k in TERM_KEYS)
            print(f"Run {run}/{args.runs} restart {rs + 1}/{args.restarts}: "
                  f"cout exact = {cost:,.0f}")
            if run_best is None or cost < run_best["cost"]:
                run_best = {"x": xs, "y": ys, "raw": raw, "cost": cost}

        w_next = update_weights(w, run_best["raw"])

        append_row_csv(w_path, {"run": run, "timestamp": stamp,
                                "best_cost": round(run_best["cost"], 1),
                                **{c: round(w_next[k], 4) for k, c in zip(TERM_KEYS, WEIGHT_COLS)}})
        append_row_csv(hist_path, {"run": run, "restart": "", "iters": args.iters,
                                   "best_cost": round(run_best["cost"], 1),
                                   **{"raw_" + k: round(run_best["raw"][k], 1) for k in TERM_KEYS}})

        lp = os.path.join(args.outdir, f"layout_run{run}.json")
        save_layout(lp, g, run_best["x"], run_best["y"],
                    {"run": run, "cost_exact": run_best["cost"], "weights": dict(w)})
        if args.png:
            render_png(os.path.join(args.outdir, f"preview_run{run}.png"), g,
                       run_best["x"], run_best["y"], f"Run {run}, cout {run_best['cost']:,.0f}")

        detail = " | ".join(f"{k}={run_best['raw'][k]:,.0f}" for k in TERM_KEYS
                            if run_best["raw"][k] > 0)
        print(f"  -> layout: {lp}")
        if detail:
            print(f"     {detail}")
        if global_best is None or run_best["cost"] < global_best["cost"]:
            global_best = dict(run_best, run=run)
        w = w_next

    if global_best:
        bp = os.path.join(args.outdir, "best_layout.json")
        save_layout(bp, g, global_best["x"], global_best["y"],
                    {"run": global_best["run"], "cost_exact": global_best["cost"],
                     "weights": dict(w)})
        print(f"\nMeilleur global : run {global_best['run']}, "
              f"cout exact = {global_best['cost']:,.0f}")
        print(f"  layout final : {bp}")
        print(f"  poids pour le prochain lancement : weights.csv")


if __name__ == "__main__":
    main()
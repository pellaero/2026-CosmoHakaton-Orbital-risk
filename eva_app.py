import gradio as gr
import pandas as pd
import numpy as np
import requests
import json
import html
import re
import os
import base64
import csv
import io
import time
from datetime import datetime, timezone, timedelta
from skyfield.api import load, EarthSatellite
import plotly.graph_objects as go

# ==========================================
# 1. ГЛОБАЛЬНЫЕ НАСТРОЙКИ И ЭФЕМЕРИДЫ
# ==========================================
ALGORITHM_VERSION = "4.3.0-ContinuousRisk"
print("Загрузка астрономических эфемерид (de421)...")
eph = load('de421.bsp')
ts = load.timescale()
print("Эфемериды успешно загружены.")

THRESHOLDS = {
    'kp_critical': 7.0, 'kp_warning': 5.0,
    'proton_critical': 10.0, 'mmod_warning_km': 5.0,
    'shadow_warning_ratio': 0.6, 'glare_critical_deg': 5.0
}
ISS_NORAD_ID = "25544"

# ==========================================
# 2. КЭШИРОВАНИЕ И SNAPSHOT (Критерий Т6)
# ==========================================
class LocalDataSnapshot:
    """Локальный снапшот для Graceful Degradation при бане IP или недоступности API."""
    KP_HIST = {
        "2024-05-11T00:00:00Z": 4.0,  "2024-05-11T03:00:00Z": 5.3,
        "2024-05-11T06:00:00Z": 6.7,  "2024-05-11T09:00:00Z": 8.0,
        "2024-05-11T12:00:00Z": 8.7,  "2024-05-11T15:00:00Z": 7.3,
        "2024-05-11T18:00:00Z": 6.0,  "2024-05-11T21:00:00Z": 5.0
    }
    PROTONS_HIST = {
        "2024-05-11T09:00:00Z": 1.2,  "2024-05-11T10:00:00Z": 5.5,
        "2024-05-11T11:00:00Z": 15.4, "2024-05-11T12:00:00Z": 45.2,
        "2024-05-11T13:00:00Z": 22.1, "2024-05-11T14:00:00Z": 8.0
    }
    TLE_LINE1 = "1 25544U 98067A   24131.50000000  .00016660  00000+0  29600-3 0  9993"
    TLE_LINE2 = "2 25544  51.6416 120.0000 0005000 100.0000 260.0000 15.50000000000000"

CACHE = LocalDataSnapshot()

class DataCache:
    def __init__(self):
        self._store = {}

    def get(self, key, max_age_seconds, force_refresh=False):
        if force_refresh or key not in self._store:
            return None, 0
        fetch_time, data = self._store[key]
        age_seconds = time.time() - fetch_time
        if age_seconds > max_age_seconds:
            return None, 0
        return data, age_seconds

    def set(self, key, data):
        self._store[key] = (time.time(), data)

    def clear(self):
        self._store = {}
        print("Кэш полностью очищен.")

    def get_status_md(self):
        if not self._store:
            return "Кэш пуст."
        lines = []
        for key, (ft, _) in self._store.items():
            lines.append(f"- {key}: {int((time.time()-ft)/60)} мин. назад")
        return "Статус кэша:\n" + "\n".join(lines)

eva_cache = DataCache()

# ==========================================
# 3. ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ (4 механизма)
# ==========================================
def fetch_tle_robust(mode, target_date=None, force_refresh=False):
    cache_key = f"tle_{mode}"
    cached, age = eva_cache.get(cache_key, 43200, force_refresh)
    if cached: return cached

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(f"https://celestrak.org/NORAD/elements/gp.php?CATNR={ISS_NORAD_ID}&FORMAT=tle", headers=headers, timeout=15)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
        for i in range(len(lines) - 2):
            if "ISS (ZARYA)" in lines[i]:
                name = lines[i].strip()
                l1 = lines[i+1].strip()
                l2 = lines[i+2].strip()
                sat = EarthSatellite(l1, l2, name, ts)
                result = (sat, f"CelesTrak (эпоха {sat.epoch.utc_strftime('%Y-%m-%d')})")
                eva_cache.set(cache_key, result)
                return result
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code in [403, 429]:
            print(f"[КЭШ] CelesTrak: HTTP {e.response.status_code} (Rate Limit). Переход на снапшот.")
        else:
            print(f"[КЭШ] CelesTrak: {e}. Переход на снапшот.")
    except Exception as e:
        print(f"[КЭШ] CelesTrak недоступен ({type(e).__name__}). Переход на снапшот.")

    sat = EarthSatellite(CACHE.TLE_LINE1, CACHE.TLE_LINE2, "ISS [LOCAL CACHE]", ts)
    result = (sat, f"Local Snapshot (эпоха {sat.epoch.utc_strftime('%Y-%m-%d')})")
    eva_cache.set(cache_key, result)
    return result

def fetch_kp_data(mode, target_date, force_refresh=False):
    cache_key = f"kp_{mode}_{target_date.strftime('%Y%m%d')}"
    is_historical = mode.startswith("Исторический")
    ttl = 1800 if not is_historical else 86400
    cached, age = eva_cache.get(cache_key, ttl, force_refresh)
    if cached: return cached

    try:
        if is_historical:
            url = f"https://kp.gfz.de/app/json/?start={target_date.strftime('%Y-%m-%d')}T00:00:00Z&end={target_date.strftime('%Y-%m-%d')}T23:59:59Z&index=Kp"
        else:
            url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
        
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        if is_historical:
            df = pd.DataFrame({'time_tag': pd.to_datetime(data['datetime'], utc=True), 'kp': data['Kp']})
            result = (df, "GFZ Potsdam (Итоговый архив)", "Наблюдение (Факт)")
        else:
            df = pd.DataFrame(data[1:], columns=data[0])
            df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
            df["kp"] = pd.to_numeric(df["kp"], errors="coerce")
            result = (df[['time_tag', 'kp']], "NOAA SWPC (Прогноз)", "Внешний прогноз")
            
        eva_cache.set(cache_key, result)
        return result
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        print(f"[КЭШ] Kp API: HTTP {code}. Переход на снапшот.")
    except Exception as e:
        print(f"[КЭШ] Kp API недоступен ({type(e).__name__}). Переход на снапшот.")

    df = pd.DataFrame(list(CACHE.KP_HIST.items()), columns=['time_tag', 'kp'])
    df['time_tag'] = pd.to_datetime(df['time_tag'], utc=True)
    result = (df, "GFZ Potsdam (Локальный Кэш)", "Наблюдение (Факт) [Кэш]")
    eva_cache.set(cache_key, result)
    return result

def fetch_proton_data(target_date, force_refresh=False):
    cache_key = f"protons_{target_date.strftime('%Y%m%d')}"
    cached, age = eva_cache.get(cache_key, 900, force_refresh)
    if cached: return cached

    try:
        url = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        df = df[df["energy"] == ">=10 MeV"].copy()
        df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
        df["proton_flux"] = pd.to_numeric(df["flux"], errors="coerce")
        
        next_day = target_date + timedelta(days=1)
        df_f = df[(df["time_tag"] >= target_date) & (df["time_tag"] < next_day)]
        
        if not df_f.empty:
            result = (df_f[['time_tag', 'proton_flux']], "NOAA GOES", "Наблюдение/Прогноз")
            eva_cache.set(cache_key, result)
            return result
            
        print("[КЭШ] Protons API: нет данных за выбранную дату. Переход на снапшот.")
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        print(f"[КЭШ] Protons API: HTTP {code}. Переход на снапшот.")
    except Exception as e:
        print(f"[КЭШ] Protons API недоступен ({type(e).__name__}). Переход на снапшот.")

    df = pd.DataFrame(list(CACHE.PROTONS_HIST.items()), columns=['time_tag', 'proton_flux'])
    df['time_tag'] = pd.to_datetime(df['time_tag'], utc=True)
    result = (df, "NOAA GOES (Локальный Кэш)", "Наблюдение [Кэш]")
    eva_cache.set(cache_key, result)
    return result

def fetch_mmod_socrates(force_refresh=False):
    cache_key = "mmod_socrates"
    cached, age = eva_cache.get(cache_key, 3600, force_refresh)
    if cached: return cached

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get("https://celestrak.org/SOCRATES/sort-minRange.csv", headers=headers, timeout=60)
        resp.raise_for_status()
        mmod_list = []
        for row in csv.DictReader(io.StringIO(resp.text)):
            try:
                if ISS_NORAD_ID not in (row['NORAD_CAT_ID_1'], row['NORAD_CAT_ID_2']):
                    continue
                if float(row['TCA_RELATIVE_SPEED']) < 0.01:
                    continue
                tca = datetime.strptime(row['TCA'], '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
                mmod_list.append({'tca': tca, 'min_distance_km': float(row['TCA_RANGE'])})
            except (KeyError, ValueError):
                continue
        result = (mmod_list, "CelesTrak SOCRATES", True)
        eva_cache.set(cache_key, result)
        return result
    except Exception as e:
        print(f"[КЭШ] SOCRATES недоступен ({type(e).__name__}). Риск MMOD не оценен.")
        result = (None, "SOCRATES (Недоступно)", False)
        eva_cache.set(cache_key, result)
        return result

def get_kp_forecast():
    """Загрузка прогноза Kp на 3 дня."""
    url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df['time_tag'] = pd.to_datetime(df['time_tag'], utc=True)
        df['kp_predicted'] = pd.to_numeric(df['kp'], errors='coerce')
        return df[['time_tag', 'kp_predicted']]
    except Exception as e:
        print(f"Не удалось загрузить прогноз Kp: {e}")
        return pd.DataFrame()

# ==========================================
# 4. ДВИЖОК ОЦЕНКИ РИСКОВ (ГРАДИЕНТЫ + БАЗА)
# ==========================================
def calculate_solar_glare(sat, start_time, duration_hours):
    end_time = start_time + timedelta(hours=duration_hours)
    times = [start_time + timedelta(minutes=5 * i) for i in range(int((end_time-start_time).total_seconds()//300) + 1)]
    t = ts.utc([d.year for d in times], [d.month for d in times], [d.day for d in times],
               [d.hour for d in times], [d.minute for d in times], [d.second for d in times])
    
    geocentric = sat.at(t)
    sun_vec = eph['sun'].at(t).position.au - geocentric.position.au
    earth_vec = -geocentric.position.au
    
    sun_norm = sun_vec / np.linalg.norm(sun_vec, axis=0)
    earth_norm = earth_vec / np.linalg.norm(earth_vec, axis=0)
    
    dot_prod = np.clip(np.sum(sun_norm * earth_norm, axis=0), -1.0, 1.0)
    angles = np.degrees(np.arccos(dot_prod))
    
    min_angle = float(np.min(angles))
    critical_glare_mins = int(np.sum(angles < THRESHOLDS['glare_critical_deg'])) * 5
    return {'min_angle_deg': round(min_angle, 1), 'critical_glare_mins': critical_glare_mins}

def calculate_naive_baseline(kp_max):
    """Простой подход (Т5): только Kp >= 5 = отмена"""
    if kp_max is None: return 50
    return 100 if kp_max >= THRESHOLDS['kp_warning'] else 0

def calculate_advanced_risk(sat, start_time, duration_hours, kp_df, proton_df, mmod_data, mmod_available):
    """Полный расчет с непрерывными функциями и базовым фоном"""
    end_time = start_time + timedelta(hours=duration_hours)
    
    # --- РАСЧЕТ ГЕОМЕТРИИ (Тень и Блики) ---
    times = [start_time + timedelta(minutes=5 * i) for i in range(int((end_time-start_time).total_seconds()//300) + 1)]
    t = ts.utc([d.year for d in times], [d.month for d in times], [d.day for d in times],
               [d.hour for d in times], [d.minute for d in times], [d.second for d in times])
    sunlit = sat.at(t).is_sunlit(eph)
    shadow_ratio = 1.0 - (np.sum(sunlit) / len(sunlit))
    sunlit_minutes = int(np.sum(sunlit)) * 5
    shadow_minutes = len(sunlit) * 5 - sunlit_minutes
    
    glare = calculate_solar_glare(sat, start_time, duration_hours)
    
    # --- СБОР ВНЕШНИХ ДАННЫХ ---
    kp_max = kp_df[(kp_df['time_tag'] >= start_time) & (kp_df['time_tag'] < end_time)]['kp'].max() if not kp_df.empty else None
    prot_max = proton_df[(proton_df['time_tag'] >= start_time) & (proton_df['time_tag'] < end_time)]['proton_flux'].max() if not proton_df.empty else None
    
    mmod_hits = 0
    if mmod_available and mmod_data:
        for obj in mmod_data:
            if start_time <= obj['tca'] < end_time and obj['min_distance_km'] < THRESHOLDS['mmod_warning_km']:
                mmod_hits += 1

    # ==========================================
    # НОВЫЙ ДВИЖОК ОЦЕНКИ (ГРАДИЕНТЫ + БАЗА)
    # ==========================================
    risk_score = 12  # БАЗОВЫЙ РИСК ВКД (фон, штатные микрометеороиды, нагрузка на скафандр)
    critical_flags, warning_flags = [], []
    factors_proof = []

    # 1. КОСМИЧЕСКАЯ ПОГОДА (Непрерывный Kp)
    if kp_max is not None:
        if kp_max >= 7.0:
            risk_score += 35
            critical_flags.append(f"Kp {kp_max:.1f} (Экстрем. буря)")
            factors_proof.append({"factor": "Kp-индекс", "value": float(kp_max), "threshold": 7.0,
                                  "status": "CRITICAL", "proof": "Экстремальная буря. Риск сбоя навигации и связи.", "mechanism": "Космическая погода"})
        elif kp_max >= 5.0:
            risk_score += 20
            warning_flags.append(f"Kp {kp_max:.1f} (Буря)")
            factors_proof.append({"factor": "Kp-индекс", "value": float(kp_max), "threshold": 5.0,
                                  "status": "WARNING", "proof": "Геомагнитная буря. Требуется мониторинг.", "mechanism": "Космическая погода"})
        elif kp_max >= 3.0:
            risk_score += int((kp_max - 2.0) * 5) 
            factors_proof.append({"factor": "Kp-индекс", "value": float(kp_max), "threshold": 3.0,
                                  "status": "INFO", "proof": "Повышенная геомагнитная активность (фоновый риск).", "mechanism": "Космическая погода"})

    # 2. ПРОТОНЫ (Непрерывный рост)
    if prot_max and prot_max > 0:
        if prot_max > 10.0:
            risk_score += 40
            critical_flags.append(f"Протоны {prot_max:.1f} pfu")
            factors_proof.append({"factor": "Протоны >10 МэВ", "value": float(prot_max), "threshold": 10.0,
                                  "status": "CRITICAL", "proof": "Превышен порог радиационной безопасности.", "mechanism": "Космическая погода"})
        elif prot_max > 1.0:
            risk_score += int(min(prot_max * 2, 15))
            factors_proof.append({"factor": "Протоны >10 МэВ", "value": float(prot_max), "threshold": 1.0,
                                  "status": "INFO", "proof": "Повышенный радиационный фон.", "mechanism": "Космическая погода"})

    # 3. ОРБИТАЛЬНАЯ ОБСТАНОВКА (MMOD)
    if mmod_hits > 0:
        risk_score += 25
        critical_flags.append(f"MMOD: {mmod_hits} сближений <{THRESHOLDS['mmod_warning_km']}км")
        factors_proof.append({"factor": "MMOD", "value": mmod_hits, "threshold": THRESHOLDS['mmod_warning_km'],
                              "status": "CRITICAL", "proof": "Зафиксированы сближения с космическим мусором.", "mechanism": "Орбитальная обстановка"})
    elif not mmod_available:
        risk_score += 5  # Штраф за неопределенность
        warning_flags.append("MMOD: Данные недоступны")
        factors_proof.append({"factor": "MMOD", "value": "N/A", "threshold": "N/A",
                              "status": "UNKNOWN", "proof": "Сервис SOCRATES недоступен. Риск не оценен.", "mechanism": "Орбитальная обстановка"})
    else:
        factors_proof.append({"factor": "MMOD", "value": 0, "threshold": THRESHOLDS['mmod_warning_km'],
                              "status": "INFO", "proof": "Угроз сближения не выявлено (учтен штатный фон).", "mechanism": "Орбитальная обстановка"})

    # 4. ОРБИТАЛЬНАЯ ГЕОМЕТРИЯ (Тень и Термическое циклирование)
    if shadow_ratio > 0.6:
        risk_score += 12
        warning_flags.append(f"Тень {shadow_ratio*100:.0f}% (Риск переохлаждения)")
        factors_proof.append({"factor": "Доля тени", "value": f"{shadow_ratio*100:.0f}%", "threshold": "60%",
                              "status": "WARNING", "proof": "Длительное нахождение в тени. Риск переохлаждения скафандра.", "mechanism": "Орбитальная геометрия"})
    elif shadow_ratio > 0.35:
        risk_score += 6
        factors_proof.append({"factor": "Термоциклирование", "value": f"{shadow_ratio*100:.0f}% тени", "threshold": "35%",
                              "status": "INFO", "proof": "Частые перепады температур (тень/свет). Нагрузка на термосистему скафандра.", "mechanism": "Орбитальная геометрия"})

    # 5. СОЛНЕЧНЫЕ БЛИКИ (Смягчение порога)
    if glare['critical_glare_mins'] > 0:
        risk_score += 10
        warning_flags.append(f"Блики <{THRESHOLDS['glare_critical_deg']}°: {glare['critical_glare_mins']} мин")
        factors_proof.append({"factor": "Солнечные блики", "value": f"<{glare['min_angle_deg']}°", "threshold": f"<{THRESHOLDS['glare_critical_deg']}°",
                              "status": "WARNING", "proof": "Ослепление визора. Невозможность работы с оптикой.", "mechanism": "Орбитальная геометрия"})
    elif glare['min_angle_deg'] < 15.0:
        risk_score += 4
        factors_proof.append({"factor": "Солнечные блики", "value": f"{glare['min_angle_deg']}°", "threshold": "15°",
                              "status": "INFO", "proof": "Повышенная нагрузка на визор и терморегуляцию скафандра.", "mechanism": "Орбитальная геометрия"})

    risk_score = min(risk_score, 100)
    baseline_risk = calculate_naive_baseline(kp_max)

    return {
        'risk_score': risk_score,
        'baseline_risk': baseline_risk,
        'critical_flags': critical_flags,
        'warning_flags': warning_flags,
        'factors_proof': factors_proof,
        'metrics': {
            'shadow_ratio': round(shadow_ratio, 2),
            'shadow_minutes': shadow_minutes,
            'sunlit_minutes': sunlit_minutes,
            'kp_max': round(kp_max, 2) if kp_max else None,
            'proton_max': round(prot_max, 2) if prot_max else None,
            'mmod_hits': mmod_hits,
            'glare_min_angle': glare['min_angle_deg'],
            'glare_critical_mins': glare['critical_glare_mins']
        }
    }

def compare_windows_dynamic(sat, base_date, search_window_hours, eva_duration_hours, kp_df, proton_df, mmod_data, mmod_available):
    """Сравнение окон с поиском лучшего (критерий О3)"""
    results = []
    for h in range(0, search_window_hours + 1):
        start_time = base_date + timedelta(hours=h)
        risk_data = calculate_advanced_risk(sat, start_time, eva_duration_hours, kp_df, proton_df, mmod_data, mmod_available)
        risk_data['window_start'] = start_time
        results.append(risk_data)
        
    safe_windows = [r for r in results if r['risk_score'] < 40]
    if safe_windows:
        best_window = min(safe_windows, key=lambda x: x['risk_score'])
    else:
        best_window = min(results, key=lambda x: x['risk_score'])
        
    return results, best_window

# ==========================================
# 5. ФУНКЦИИ ПРОГНОЗА НА 3 ДНЯ (С БАЗОВЫМ РИСКОМ)
# ==========================================
def evaluate_forecast_window(sat, start_time, duration_hours, kp_forecast_df):
    end_time = start_time + timedelta(hours=duration_hours)
    window_kp = kp_forecast_df[
        (kp_forecast_df['time_tag'] >= start_time) & 
        (kp_forecast_df['time_tag'] < end_time)
    ]
    
    if window_kp.empty:
        return {
            'start_time': start_time, 'duration_hours': duration_hours,
            'kp_max': None, 'kp_mean': None,
            'risk_level': 'UNKNOWN', 'risk_score': 12, # Базовый риск даже без данных
            'recommendation': 'Нет данных прогноза, учтен базовый фон',
            'critical_factors': [], 'warning_factors': [],
            'shadow_ratio': None, 'sunlit_minutes': None
        }

    kp_max = window_kp['kp_predicted'].max()
    kp_mean = window_kp['kp_predicted'].mean()
    
    times = [start_time + timedelta(minutes=5*i) for i in range(int(duration_hours * 60 / 5) + 1)]
    t = ts.utc([d.year for d in times], [d.month for d in times], [d.day for d in times],
               [d.hour for d in times], [d.minute for d in times], [d.second for d in times])
    geocentric = sat.at(t)
    sunlit = geocentric.is_sunlit(eph)
    shadow_ratio = (len(sunlit) - int(np.sum(sunlit))) / len(sunlit) if len(sunlit) > 0 else 0
    sunlit_minutes = int(np.sum(sunlit)) * 5

    risk_score = 12  # БАЗОВЫЙ РИСК ВКД
    critical_factors = []
    warning_factors = []

    # Kp (Непрерывный)
    if kp_max >= 7:
        risk_score += 35
        critical_factors.append(f"Kp={kp_max:.1f} (экстремальная геомагнитная буря)")
    elif kp_max >= 5:
        risk_score += 20
        warning_factors.append(f"Kp={kp_max:.1f} (геомагнитная буря)")
    elif kp_max >= 3:
        risk_score += int((kp_max - 2.0) * 5)
        warning_factors.append(f"Kp={kp_max:.1f} (повышенная активность)")

    # Тень (Термоциклирование)
    if shadow_ratio > 0.6:
        risk_score += 12
        warning_factors.append(f"Длительная тень ({shadow_ratio*100:.0f}% времени)")
    elif shadow_ratio > 0.35:
        risk_score += 6
        warning_factors.append(f"Термоциклирование ({shadow_ratio*100:.0f}% времени в тени)")

    if len(critical_factors) > 0 or risk_score >= 70:
        risk_level = "CRITICAL"
        recommendation = "НЕ РЕКОМЕНДУЕТСЯ - критический риск"
    elif risk_score >= 40:
        risk_level = "WARNING"
        recommendation = "Выходить с осторожностью, требуется обоснование"
    else:
        risk_level = "SAFE"
        recommendation = "Безопасно (базовый фон учтен)"

    return {
        'start_time': start_time, 'duration_hours': duration_hours,
        'kp_max': kp_max, 'kp_mean': kp_mean, 'risk_score': risk_score,
        'risk_level': risk_level, 'recommendation': recommendation,
        'critical_factors': critical_factors, 'warning_factors': warning_factors,
        'shadow_ratio': shadow_ratio, 'sunlit_minutes': sunlit_minutes
    }

def compare_forecast_windows(sat, base_date, duration_hours, kp_forecast_df):
    results = []
    for day_offset in range(3):
        target_date = base_date + timedelta(days=day_offset)
        window_start = target_date.replace(hour=12, minute=0, second=0, tzinfo=timezone.utc)
        risk = evaluate_forecast_window(sat, window_start, duration_hours, kp_forecast_df)
        results.append(risk)
        
    safe_days = [r for r in results if r['risk_level'] == 'SAFE']
    warning_days = [r for r in results if r['risk_level'] == 'WARNING']
    
    if safe_days:
        best = min(safe_days, key=lambda x: x['risk_score'])
        recommendation_text = f"Лучший день - {best['start_time'].strftime('%Y-%m-%d')} (риск {best['risk_score']})"
    elif warning_days:
        best = min(warning_days, key=lambda x: x['risk_score'])
        recommendation_text = f"Наименее рискованный день - {best['start_time'].strftime('%Y-%m-%d')} (риск {best['risk_score']})"
    else:
        best = results[0]
        recommendation_text = "Все 3 дня критичны. ВКД не рекомендуется."
        
    return results, best, recommendation_text

# ==========================================
# 6. ВИЗУАЛИЗАЦИЯ С ПОЯСНЕНИЯМИ
# ==========================================
def plot_two_windows(kp_df, window1_dt, window2_dt, duration, r1, r2):
    fig = go.Figure()
    if not kp_df.empty:
        fig.add_trace(go.Scatter(
            x=kp_df['time_tag'], y=kp_df['kp'],
            mode='lines+markers', name='Kp-индекс',
            line=dict(color='red', width=2)
        ))
    fig.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Kp=5", annotation_position="top right")
    fig.add_hline(y=7, line_dash="dash", line_color="red", annotation_text="Kp=7", annotation_position="top right")

    color1 = "green" if r1['risk_score'] < 40 else ("orange" if r1['risk_score'] < 70 else "red")
    color2 = "green" if r2['risk_score'] < 40 else ("orange" if r2['risk_score'] < 70 else "red")

    fig.add_shape(type="rect", x0=window1_dt, x1=window1_dt + timedelta(hours=duration),
                  xref="x", yref="paper", y0=0, y1=1, fillcolor=color1, opacity=0.2, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=window2_dt, x1=window2_dt + timedelta(hours=duration),
                  xref="x", yref="paper", y0=0, y1=1, fillcolor=color2, opacity=0.2, line_width=0, layer="below")

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=f"Окно 1: Риск {r1['risk_score']}", marker=dict(color=color1, size=12, symbol='square'), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=f"Окно 2: Риск {r2['risk_score']}", marker=dict(color=color2, size=12, symbol='square'), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Низкий риск (< 40)', marker=dict(color='green', size=8, symbol='square', opacity=0.6), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Средний риск (40 - 69)', marker=dict(color='orange', size=8, symbol='square', opacity=0.6), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Высокий риск (>= 70)', marker=dict(color='red', size=8, symbol='square', opacity=0.6), showlegend=True))

    fig.update_layout(
        title="Сравнение окон ВКД с оценкой 4 механизмов",
        xaxis_title="Время (UTC)", yaxis_title="Kp-индекс",
        height=450, showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.95)", bordercolor="black", borderwidth=1, font=dict(size=11))
    )
    return fig

def plot_full_analysis(kp_df, best_window, base_date, duration):
    fig = go.Figure()
    if not kp_df.empty:
        fig.add_trace(go.Scatter(
            x=kp_df['time_tag'], y=kp_df['kp'],
            mode='lines+markers', name='Kp-индекс',
            line=dict(color='red', width=2)
        ))
    fig.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Kp=5", annotation_position="top right")
    fig.add_hline(y=7, line_dash="dash", line_color="red", annotation_text="Kp=7", annotation_position="top right")

    risk_color = "green" if best_window['risk_score'] < 40 else ("orange" if best_window['risk_score'] < 70 else "red")

    fig.add_shape(type="rect", x0=best_window['window_start'], x1=best_window['window_start'] + timedelta(hours=duration),
                  xref="x", yref="paper", y0=0, y1=1, fillcolor=risk_color, opacity=0.3, line_width=0, layer="below")

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=f"Рекомендуемое окно: Риск {best_window['risk_score']}", marker=dict(color=risk_color, size=12, symbol='square'), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Низкий риск (< 40)', marker=dict(color='green', size=8, symbol='square', opacity=0.6), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Средний риск (40 - 69)', marker=dict(color='orange', size=8, symbol='square', opacity=0.6), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Высокий риск (>= 70)', marker=dict(color='red', size=8, symbol='square', opacity=0.6), showlegend=True))

    fig.update_layout(
        title=f"Прогноз на 3 дня. Рекомендация: {best_window['window_start'].strftime('%Y-%m-%d %H:%M')} UTC",
        xaxis_title="Время (UTC)", yaxis_title="Kp-индекс",
        height=450, showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.95)", bordercolor="black", borderwidth=1, font=dict(size=11))
    )
    return fig

# ==========================================
# 7. ГЛАВНАЯ ФУНКЦИЯ ВЕБ-ИНТЕРФЕЙСА
# ==========================================
def run_analysis(mode, date_str, w1_hour, w1_min, w2_hour, w2_min, duration, forecast_mode, force_refresh):
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        base_date = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)

        sat, tle_src = fetch_tle_robust(mode, base_date, force_refresh)
        kp_df, kp_src, kp_type = fetch_kp_data(mode, base_date, force_refresh)
        prot_df, prot_src, prot_type = fetch_proton_data(base_date, force_refresh)
        mmod_data, mmod_src, mmod_avail = fetch_mmod_socrates(force_refresh)

        report = []
        chart = None
        json_data = None

        if forecast_mode:
            base_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            kp_forecast = get_kp_forecast()
            if kp_forecast.empty:
                return "Ошибка: не удалось загрузить прогноз Kp", None, None, "Ошибка"
            
            kp_df = kp_forecast.rename(columns={'kp_predicted': 'kp'})
            prot_df, prot_src, prot_type = fetch_proton_data(base_date, force_refresh)
            
            window_results, best_window = compare_windows_dynamic(sat, base_date, 48, duration, kp_df, prot_df, mmod_data, mmod_avail)
            
            report.append("=" * 70)
            report.append("ПРОГНОЗ НА 3 ДНЯ С ПОЛНОЙ ОЦЕНКОЙ РИСКОВ")
            report.append("=" * 70)
            
            for i, risk in enumerate(window_results[:3]):
                report.append(f"\nОКНО {i+1}: {risk['window_start'].strftime('%Y-%m-%d %H:%M')} UTC")
                report.append(f"   Интегральный риск (Команда): {risk['risk_score']}/100")
                report.append(f"   Базовый подход (Только Kp): {risk['baseline_risk']}/100")
                report.append(f"   Освещенность: {risk['metrics']['sunlit_minutes']} мин на свету, {risk['metrics']['shadow_minutes']} мин в тени ({risk['metrics']['shadow_ratio']*100:.0f}%)")
                
                if risk['metrics']['glare_critical_mins'] > 0:
                    report.append(f"   Блики: {risk['metrics']['glare_critical_mins']} мин с углом <{THRESHOLDS['glare_critical_deg']}°")
                else:
                    report.append(f"   Блики: отсутствуют (мин. угол {risk['metrics']['glare_min_angle']}°)")
                    
                if risk['metrics']['mmod_hits'] > 0:
                    report.append(f"   MMOD: {risk['metrics']['mmod_hits']} сближений <{THRESHOLDS['mmod_warning_km']}км")
                elif not mmod_avail:
                    report.append(f"   MMOD: данные недоступны")
                    
                if risk['critical_flags']:
                    report.append(f"   КРИТИЧНО: {', '.join(risk['critical_flags'])}")
                if risk['warning_flags']:
                    report.append(f"   ВНИМАНИЕ: {', '.join(risk['warning_flags'])}")

            report.append("\n" + "=" * 70)
            report.append(f"РЕКОМЕНДАЦИЯ: Лучшее окно - {best_window['window_start'].strftime('%Y-%m-%d %H:%M')} UTC")
            report.append(f"Риск: {best_window['risk_score']}/100 (базовый: {best_window['baseline_risk']}/100)")
            
            if best_window['risk_score'] < best_window['baseline_risk']:
                report.append("ВЫВОД: Алгоритм команды нашел безопасное окно за счет учета всех 4 механизмов.")
            elif best_window['risk_score'] < 40:
                report.append("ВЫВОД: Обстановка спокойная, учтен базовый фон рисков ВКД.")
            else:
                report.append("ВЫВОД: Ситуация критическая, базовый и продвинутый подходы солидарны.")

            report.append("\nДОКАЗАТЕЛЬНАЯ БАЗА:")
            for p in best_window['factors_proof']:
                report.append(f"  [{p['status']}] {p['factor']}: {p['value']} | {p['proof']}")

            chart = plot_full_analysis(kp_df, best_window, base_date, duration)
            json_data = {
                "mode": "3-day forecast",
                "algorithm_version": ALGORITHM_VERSION,
                "base_date": date_str,
                "duration_hours": duration,
                "best_window": {
                    "start_time": best_window['window_start'].isoformat(),
                    "risk_score": best_window['risk_score'],
                    "baseline_risk": best_window['baseline_risk'],
                    "factors_proof": best_window['factors_proof']
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            window1_dt = base_date.replace(hour=int(w1_hour), minute=int(w1_min))
            window2_dt = base_date.replace(hour=int(w2_hour), minute=int(w2_min))
            
            r1 = calculate_advanced_risk(sat, window1_dt, duration, kp_df, prot_df, mmod_data, mmod_avail)
            r2 = calculate_advanced_risk(sat, window2_dt, duration, kp_df, prot_df, mmod_data, mmod_avail)
            
            report.append("=" * 70)
            report.append("СРАВНЕНИЕ ДВУХ ОКОН ВКД (4 МЕХАНИЗМА)")
            report.append("=" * 70)
            
            for w_dt, r in [(window1_dt, r1), (window2_dt, r2)]:
                w_name = "ОКНО 1" if w_dt == window1_dt else "ОКНО 2"
                report.append(f"\n{w_name}: {w_dt.strftime('%Y-%m-%d %H:%M')} UTC")
                report.append(f"   Интегральный риск (Команда): {r['risk_score']}/100")
                report.append(f"   Базовый подход (Только Kp): {r['baseline_risk']}/100")
                report.append(f"   Освещенность: {r['metrics']['sunlit_minutes']} мин на свету, {r['metrics']['shadow_minutes']} мин в тени ({r['metrics']['shadow_ratio']*100:.0f}%)")
                
                if r['metrics']['glare_critical_mins'] > 0:
                    report.append(f"   Блики: {r['metrics']['glare_critical_mins']} мин с углом <{THRESHOLDS['glare_critical_deg']}°")
                else:
                    report.append(f"   Блики: отсутствуют (мин. угол {r['metrics']['glare_min_angle']}°)")
                    
                if r['metrics']['mmod_hits'] > 0:
                    report.append(f"   MMOD: {r['metrics']['mmod_hits']} сближений <{THRESHOLDS['mmod_warning_km']}км")
                elif not mmod_avail:
                    report.append(f"   MMOD: данные недоступны")
                    
                if r['critical_flags']:
                    report.append(f"   КРИТИЧНО: {', '.join(r['critical_flags'])}")
                if r['warning_flags']:
                    report.append(f"   ВНИМАНИЕ: {', '.join(r['warning_flags'])}")

            report.append("\n" + "=" * 70)
            if r1['risk_score'] < r2['risk_score']:
                report.append(f"РЕКОМЕНДАЦИЯ: Окно 1 имеет меньший интегральный риск.")
            elif r2['risk_score'] < r1['risk_score']:
                report.append(f"РЕКОМЕНДАЦИЯ: Окно 2 имеет меньший интегральный риск.")
            else:
                report.append("РЕКОМЕНДАЦИЯ: Окна равнозначны по уровню риска.")

            report.append("\nСРАВНЕНИЕ С ПРОСТЫМ ПОДХОДОМ (Т5):")
            report.append(f"  Базовый подход учитывает только Kp >= 5")
            report.append(f"  Алгоритм команды учитывает 4 механизма + базовый фон ВКД")
            if r1['risk_score'] != r1['baseline_risk'] or r2['risk_score'] != r2['baseline_risk']:
                report.append(f"  ВЫВОД: Многофакторный анализ дал отличающуюся оценку")

            chart = plot_two_windows(kp_df, window1_dt, window2_dt, duration, r1, r2)
            json_data = {
                "mode": "two windows comparison",
                "algorithm_version": ALGORITHM_VERSION,
                "date": date_str,
                "window1": {
                    "start_time": window1_dt.isoformat(),
                    "risk_score": r1['risk_score'],
                    "baseline_risk": r1['baseline_risk'],
                    "factors_proof": r1['factors_proof']
                },
                "window2": {
                    "start_time": window2_dt.isoformat(),
                    "risk_score": r2['risk_score'],
                    "baseline_risk": r2['baseline_risk'],
                    "factors_proof": r2['factors_proof']
                },
                "recommendation": "Window 1" if r1['risk_score'] < r2['risk_score'] else ("Window 2" if r2['risk_score'] < r1['risk_score'] else "Equal"),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        text_report = "\n".join(report)
        
        if json_data:
            filename = f"eva_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
                
        cache_status = eva_cache.get_status_md()
        return text_report, chart, filename, cache_status

    except Exception as e:
        import traceback
        error_msg = f"Ошибка при расчёте:\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None, None, "Ошибка инициализации."

# ==========================================
# 8. ИНТЕРФЕЙС GRADIO
# ==========================================
UI_ACC, UI_LOW, UI_HIGH = "#FF5A13", "#8BEE5A", "#FF334A"
UI_FG, UI_MUT, UI_LINE2 = "#FFFFFF", "rgba(255,255,255,.58)", "rgba(255,255,255,.07)"

def _esc(value):
    return html.escape(str(value)).replace(">=", "≥")

def _risk_tone(score):
    if score is None: return "unk"
    return "high" if score >= 70 else ("mid" if score >= 40 else "low")

TONE_ICON = {"high": "▲", "mid": "◆", "low": "✓", "unk": "○"}

def _chip(text, tone):
    return f'<span class="eva-chip eva-{tone}">{TONE_ICON[tone]} {_esc(text)}</span>'

def _kv(rows):
    cells = " ".join(f'<span class="eva-k">{_esc(k)}</span><span class="{cls}">{_esc(v)}</span>' for k, v, cls in rows)
    return f'<div class="eva-kv">{cells}</div>'

def _parse_report(text):
    """Разбирает текстовый отчёт run_analysis на окна и итоговые блоки."""
    lines = text.splitlines()
    windows, tail, cur = [], [], None
    
    for raw in lines[3:]:
        s = raw.strip()
        if s.startswith("="):
            cur = None
        elif s.startswith("ОКНО "):
            name, _, when = s.partition(": ")
            cur = {"name": name, "when": when, "rows": [], "crit": [], "warn": [], "score": None}
            windows.append(cur)
        elif cur is not None:
            if not s: continue
            key, _, val = s.partition(": ")
            if key == "КРИТИЧНО":
                cur["crit"] = val.split(", ")
            elif key == "ВНИМАНИЕ":
                cur["warn"] = val.split(", ")
            else:
                if key.startswith("Интегральный риск"):
                    cur["score"] = int(val.split("/")[0])
                cur["rows"].append((key, val))
        else:
            tail.append(s)

    rec, rec_extra, compare, evidence, section = None, [], [], [], None
    for s in tail:
        if not s: continue
        if s.startswith("РЕКОМЕНДАЦИЯ:"):
            rec, section = s.partition(": ")[2], None
        elif s.startswith("СРАВНЕНИЕ С ПРОСТЫМ"):
            section = "compare"
        elif s.startswith("ДОКАЗАТЕЛЬНАЯ БАЗА"):
            section = "evidence"
        elif section == "compare":
            if s.startswith("ВЫВОД:"):
                compare.append(("Вывод", s.partition(": ")[2], "eva-acc-text"))
            else:
                k, _, v = s.partition(" учитывает ")
                compare.append((k, "учитывает " + v, ""))
        elif section == "evidence":
            m = re.match(r"\[(\w+)\]\s*(.*?):\s*(.*?)\s*\|\s*(.*)", s)
            if m:
                evidence.append(m.groups())
        else:
            rec_extra.append(s)

    return lines[1].strip(), windows, rec, rec_extra, compare, evidence

def report_to_html(text):
    if not text: return REPORT_EMPTY_HTML
    if not text.startswith("="):
        return (f'<div class="eva-panel eva-report"><div class="eva-report-head">'
                f'<span class="eva-badge eva-badge-high">Ошибка</span></div>'
                f'<pre class="eva-error">{_esc(text)}</pre></div>')

    title, windows, rec, rec_extra, compare, evidence = _parse_report(text)
    parts = []
    
    for w in windows:
        tone = _risk_tone(w["score"])
        rows = []
        for k, v in w["rows"]:
            cls = f"eva-{tone}-text eva-strong" if k.startswith("Интегральный риск") else ""
            rows.append((k, v.replace(" мин на свету, ", " мин на свету · "), cls))
            
        chips = [_chip(c, "high") for c in w["crit"]]
        chips += [_chip(c, "unk" if "недоступн" in c else "mid") for c in w["warn"]]
        
        parts.append(
            f'<div class="eva-win"><div class="eva-win-head">'
            f'<span class="eva-win-name">{_esc(w["name"])}</span>'
            f'<span class="eva-mono eva-mut">{_esc(w["when"])}</span>'
            f'<span class="eva-score eva-{tone}-text">{TONE_ICON[tone]} {w["score"]}/100</span></div>'
            f'{_kv(rows)}'
            + (f'<div class="eva-chips">{"".join(chips)}</div>' if chips else "")
            + '</div>'
        )

    body = '<div class="eva-divider"></div>'.join(parts)
    
    if rec:
        best = next((int(s.split(":")[1].split("/")[0]) for s in rec_extra if s.startswith("Риск:")), None)
        if best is None and windows:
            best = min(w["score"] for w in windows)
        tone = _risk_tone(best)
        extra = "".join(f'<span class="eva-rec-extra">{_esc(s)}</span>' for s in rec_extra)
        body += (f'<div class="eva-rec eva-rec-{tone}"><span class="eva-h">РЕКОМЕНДАЦИЯ</span>'
                 f'<span class="eva-rec-text eva-{tone}-text">{_esc(rec)}</span>{extra}</div>')

    if compare:
        body += (f'<div class="eva-sub"><span class="eva-h">СРАВНЕНИЕ С ПРОСТЫМ ПОДХОДОМ (T5)</span>'
                 f'{_kv(compare)}</div>')

    if evidence:
        status_tone = {"CRITICAL": "high", "WARNING": "mid", "UNKNOWN": "unk", "INFO": "low"}
        items = "".join(
            f'<div class="eva-ev"><div class="eva-ev-head">{_chip(status, status_tone.get(status, "unk"))}'
            f'<span class="eva-strong">{_esc(factor)}</span><span class="eva-mono eva-mut">{_esc(value)}</span></div>'
            f'<span class="eva-mut">{_esc(proof)}</span></div>'
            for status, factor, value, proof in evidence
        )
        body += f'<div class="eva-sub"><span class="eva-h">ДОКАЗАТЕЛЬНАЯ БАЗА</span>{items}</div>'

    title = title.replace(" (4 МЕХАНИЗМА)", " · 4 МЕХАНИЗМА")
    return (f'<div class="eva-panel eva-report"><div class="eva-report-head">'
            f'<span class="eva-badge">Отчёт</span><span class="eva-mono eva-mut">{_esc(title)}</span></div>'
            f'<div class="eva-report-body">{body}</div></div>')

def cache_to_html(status):
    rows = []
    for line in (status or " ").splitlines():
        if line.startswith("- "):
            key, _, age = line[2:].rpartition(": ")
            rows.append(f'<div class="eva-cache-row"><span class="eva-mono"><i class="eva-dot"></i>{_esc(key)}</span>'
                        f'<span class="eva-mono eva-mut">{_esc(age.replace("мин. ", "мин "))}</span></div>')
    if not rows:
        rows.append(f'<div class="eva-cache-row"><span class="eva-mut">{_esc(status or "Кэш пуст.")}</span></div>')
        
    return (f'<div class="eva-panel eva-cache"><div class="eva-cache-head"><span class="eva-h">СТАТУС КЭША</span>'
            f'<span class="eva-mut eva-small">обновлено</span></div>{"".join(rows)}</div>')

def style_chart(fig):
    """Перекрашивает график Plotly в палитру дизайна, не меняя данных."""
    if fig is None: return None
    palette = {"green": UI_LOW, "orange": UI_ACC, "red": UI_HIGH}
    
    for tr in fig.data:
        if tr.name == "Kp-индекс":
            tr.line.color, tr.line.width = UI_FG, 2.5
            tr.marker.color, tr.marker.size = UI_FG, 7
        elif tr.marker.color in palette:
            tr.marker.color = palette[tr.marker.color]
            
    for sh in fig.layout.shapes:
        if sh.fillcolor in palette: sh.fillcolor = palette[sh.fillcolor]
        if sh.line.color in palette: sh.line.color = palette[sh.line.color]
        
    for ann in fig.layout.annotations: 
        ann.font.color = palette.get("red") if "7" in (ann.text or "") else UI_ACC
        ann.font.family = "Azeret Mono, monospace"
        
    axis = dict(gridcolor=UI_LINE2, zerolinecolor=UI_LINE2, linecolor="rgba(255,255,255,.12)",
                tickfont=dict(family="Azeret Mono, monospace", size=11, color=UI_MUT),
                title_font=dict(family="Manrope, sans-serif", size=12, color=UI_MUT))
                
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.03)",
        font=dict(family="Manrope, sans-serif", color=UI_FG),
        title=dict(font=dict(size=13, color=UI_MUT), x=0, xanchor="left"),
        height=500, margin=dict(l=56, r=24, t=48, b=24),
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.18, yanchor="top",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(family="Azeret Mono, monospace", size=11, color=UI_FG)),
        xaxis=axis, yaxis=axis,
    )
    return fig

def run_analysis_ui(*args):
    text_report, chart, filename, cache_status = run_analysis(*args)
    return report_to_html(text_report), style_chart(chart), filename, cache_to_html(cache_status)

REPORT_EMPTY_HTML = (
    '<div class="eva-panel eva-report"><div class="eva-report-head"><span class="eva-badge">Отчёт</span>'
    '<span class="eva-mono eva-mut">ОЖИДАНИЕ ЗАПУСКА</span></div>'
    '<div class="eva-report-body"><span class="eva-mut">Задайте параметры слева и нажмите «Запустить анализ». '
    'Демо-кейс: 2024-05-11 — экстремальная геомагнитная буря. </span></div></div>'
)

HEADER_HTML = f"""
<header class="eva-panel eva-header">
 <div class="eva-header-top">
 <div class="eva-logo" aria-hidden="true">
 <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
 <ellipse cx="12" cy="12" rx="10" ry="4.2" transform="rotate(-24 12 12)"/>
 <circle cx="12" cy="12" r="3.2" fill="currentColor" stroke="none"/>
 </svg>
 </div>
 <div class="eva-title">
 <h1>Анализ рисков ВКД на МКС</h1>
 <span>Исследовательский прототип поддержки решений · КосмоХакатон 2026</span>
 </div>
 <span class="eva-version">v{ALGORITHM_VERSION.split('-')[0]}</span>
 </div>
 <div class="eva-mechs">
 <span class="eva-h">4 механизма воздействия</span>
 <span class="eva-pill"><i></i>Космическая погода (Kp, протоны)</span>
 <span class="eva-pill"><i></i>Орбитальная обстановка (MMOD)</span>
 <span class="eva-pill"><i></i>Освещённость</span>
 <span class="eva-pill"><i></i>Солнечные блики</span>
 </div>
</header>
"""

def _card_title(badge, subtitle=""):
    sub = f'<span class="eva-mut eva-small">{subtitle}</span>' if subtitle else ""
    return f'<div class="eva-card-title"><span class="eva-badge">{badge}</span>{sub}</div>'

def _background_css():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "bg.jpg")
    if not os.path.exists(path): return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"body::before{{background-image:url('data:image/jpeg;base64,{data}')}}"

UI_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Azeret+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""

UI_CSS = """
:root{--eva-bg:#040A0A;--eva-panel:rgba(18,24,24,.62);--eva-panel2:rgba(255,255,255,.05);--eva-line:rgba(255,255,255,.12);
--eva-line2:rgba(255,255,255,.07);--eva-fg:#FFFFFF;--eva-mut:rgba(255,255,255,.58);--eva-acc:#FF5A13;--eva-low:#8BEE5A;
--eva-high:#FF334A;--eva-onacc:#150500;--eva-veil:rgba(4,10,10,.62);--eva-mono:'Azeret Mono',ui-monospace,monospace}
html,body{background:var(--eva-bg)!important;color-scheme:dark}
body::before{content:"";position:fixed;inset:0;pointer-events:none;background-size:cover;background-position:62% 40%;opacity:.85;z-index:0}
body::after{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
background:linear-gradient(105deg,var(--eva-veil) 0%,var(--eva-veil) 48%,transparent 100%)}
gradio-app{position:relative;z-index:1;background:transparent!important}
footer{display:none!important}
@keyframes eva-pulse{0%{box-shadow:0 0 0 0 rgba(255,90,19,.45)}70%{box-shadow:0 0 0 10px rgba(255,90,19,0)}100%{box-shadow:0 0 0 0 rgba(255,90,19,0)}}
.gradio-container{
--body-background-fill:transparent;--background-fill-primary:transparent;--background-fill-secondary:var(--eva-panel2);
--block-background-fill:transparent;--block-border-width:0px;--block-border-color:transparent;--block-padding:0px;
--block-radius:0px;--block-shadow:none;--block-label-background-fill:transparent;--block-label-border-width:0px;
--block-label-text-color:var(--eva-mut);--block-title-text-color:var(--eva-mut);--block-info-text-color:var(--eva-mut);
--block-label-padding:0;--block-title-padding:0;--block-label-margin:0;
--body-text-color:var(--eva-fg);--body-text-color-subdued:var(--eva-mut);
--input-background-fill:var(--eva-panel2);--input-background-fill-focus:var(--eva-panel2);
--input-border-color:var(--eva-line2);--input-border-color-focus:var(--eva-acc);--input-border-width:1px;
--input-radius:16px;--input-padding:13px 15px;--input-shadow:none;--input-shadow-focus:none;--input-text-size:14px;
--border-color-primary:var(--eva-line2);--border-color-accent:var(--eva-acc);
--color-accent:var(--eva-acc);--color-accent-soft:rgba(255,90,19,.15);--slider-color:var(--eva-acc);
--checkbox-background-color:var(--eva-panel2);--checkbox-background-color-focus:var(--eva-panel2);
--checkbox-background-color-hover:var(--eva-panel2);--checkbox-background-color-selected:var(--eva-acc);
--checkbox-border-color:var(--eva-line);--checkbox-border-color-focus:var(--eva-acc);
--checkbox-border-color-hover:var(--eva-acc);--checkbox-border-color-selected:var(--eva-acc);
--checkbox-border-radius:6px;--checkbox-shadow:none;--checkbox-border-width:1px;
--checkbox-label-background-fill:transparent;--checkbox-label-background-fill-hover:transparent;
--checkbox-label-background-fill-selected:transparent;--checkbox-label-border-width:0px;--checkbox-label-padding:0px;
--checkbox-label-shadow:none;--checkbox-label-text-color:var(--eva-fg);--checkbox-label-text-color-selected:var(--eva-fg);
--checkbox-label-text-size:13.5px;--checkbox-label-gap:11px;
--button-primary-background-fill:var(--eva-acc);--button-primary-background-fill-hover:#FF6E30;
--button-primary-text-color:var(--eva-onacc);--button-primary-text-color-hover:var(--eva-onacc);
--button-primary-border-color:transparent;--button-primary-border-color-hover:transparent;
--button-secondary-background-fill:var(--eva-panel);--button-secondary-background-fill-hover:var(--eva-panel);
--button-secondary-text-color:var(--eva-fg);--button-secondary-text-color-hover:var(--eva-fg);
--button-secondary-border-color:var(--eva-line);--button-secondary-border-color-hover:var(--eva-acc);
--button-border-width:1px;--button-shadow:none;--button-shadow-hover:none;--button-shadow-active:none;
--button-large-radius:999px;--button-medium-radius:999px;--button-small-radius:999px;
--button-large-padding:14px;--button-medium-padding:13px;--button-large-text-size:14px;--button-medium-text-size:13.5px;
--button-large-text-weight:700;--button-medium-text-weight:600;
--font:'Manrope',system-ui,sans-serif;--font-mono:var(--eva-mono);
--panel-background-fill:transparent;--panel-border-color:transparent;--panel-border-width:0px;
--shadow-drop:none;--shadow-drop-lg:none;--shadow-inset:none;--loader-color:var(--eva-acc);
--layout-gap:12px;--form-gap-width:0px;--spacing-lg:9px;
--table-even-background-fill:transparent;--table-odd-background-fill:transparent;
max-width:1440px!important;margin:0 auto!important;padding:28px 22px 64px!important;
background:transparent!important;font-family:'Manrope',system-ui,sans-serif!important;color:var(--eva-fg);
font-size:14px;line-height:1.5}
.gradio-container .form{border:0!important;background:transparent!important;box-shadow:none!important;gap:18px!important}
.gradio-container >.main,.gradio-container .main.fillable{padding-left:0!important;padding-right:0!important}
.gradio-container .block{background:transparent!important;border:0!important;box-shadow:none!important}
/* заголовок */
.eva-header{display:flex;flex-direction:column;margin-bottom:24px;overflow:hidden}
.eva-header-top{display:flex;align-items:center;gap:16px;padding:20px 22px}
.eva-logo{width:44px;height:44px;border-radius:14px;background:var(--eva-acc);color:var(--eva-onacc);flex:none;
display:flex;align-items:center;justify-content:center;box-shadow:0 6px 20px rgba(255,90,19,.28)}
.eva-title{display:flex;flex-direction:column;gap:4px;min-width:0;flex:1}
.eva-title h1{margin:0!important;font-size:26px!important;font-weight:700!important;letter-spacing:-.02em;color:var(--eva-fg)!important;line-height:1.15}
.eva-title span{font-size:13.5px;color:var(--eva-mut)}
.eva-version{flex:none;align-self:flex-start;padding:5px 12px;border:1px solid var(--eva-line);border-radius:999px;
font-family:var(--eva-mono);font-size:11.5px;color:var(--eva-mut)}
.eva-mechs{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:14px 22px;border-top:1px solid var(--eva-line2);
background:rgba(255,255,255,.02)}
.eva-mechs .eva-h{margin-right:6px!important}
.eva-pill{display:inline-flex;align-items:center;gap:8px;padding:6px 13px;border:1px solid var(--eva-line2);border-radius:999px;
background:var(--eva-panel2);font-size:12.5px;color:var(--eva-fg);white-space:nowrap}
.eva-pill i{width:6px;height:6px;border-radius:50%;background:var(--eva-acc);flex:none}
@media (max-width:600px){.eva-title h1{font-size:21px!important}.eva-version{display:none}
.eva-header-top{padding:16px;align-items:flex-start}.eva-logo{width:38px;height:38px;border-radius:12px}
.eva-mechs{padding:12px 16px}.eva-header{margin-bottom:16px}}
/* общие элементы */
.eva-h,.eva-section-title h2{font-size:10.5px!important;font-weight:600!important;letter-spacing:.09em;color:var(--eva-mut)!important;
margin:0!important;text-transform:uppercase;line-height:1.5}
.eva-mono{font-family:var(--eva-mono)}
.eva-mut{color:var(--eva-mut)}
.eva-small{font-size:12.5px}
.eva-strong{font-weight:600}
.eva-low-text{color:var(--eva-low)!important}.eva-mid-text,.eva-acc-text{color:var(--eva-acc)!important}
.eva-high-text{color:var(--eva-high)!important}.eva-unk-text{color:var(--eva-mut)!important}
.eva-acc-text{font-weight:600}
.eva-badge{display:inline-block;padding:5px 13px;border-radius:999px;background:var(--eva-acc);color:var(--eva-onacc);font-size:12px;font-weight:700;white-space:nowrap}
.eva-badge-high{background:var(--eva-high)}
.eva-main{gap:20px!important;align-items:flex-start!important;margin-top:28px!important}
@media (max-width:600px){.eva-main{margin-top:16px!important}}
.eva-col{gap:12px!important}
/* карточки */
.eva-card,.eva-panel{border:1px solid var(--eva-line)!important;border-radius:26px!important;background:var(--eva-panel)!important;
backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px)}
.eva-card{padding:20px!important;gap:18px!important}
.eva-row-card{gap:16px!important;flex-wrap:nowrap!important}
.eva-row-card>*{min-width:0!important}
.eva-card-title{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
/* подписи полей */
.gradio-container [data-testid="block-info"]{font-size:10.5px!important;font-weight:600!important;letter-spacing:.09em;
color:var(--eva-mut)!important;text-transform:uppercase;margin-bottom:9px!important;display:inline-block}
.gradio-container input[type=text],.gradio-container input[type=number],.gradio-container textarea{
font-family:var(--eva-mono)!important;font-size:14px!important;color:var(--eva-fg)!important}
.gradio-container input::placeholder{color:var(--eva-mut)}
/* кнопки */
.gradio-container button.eva-btn{font-family:inherit!important;min-height:0!important}
.gradio-container button.eva-btn.primary{font-weight:700!important;border:none!important}
.gradio-container button.eva-btn.primary:hover{filter:brightness(1.08)}
.gradio-container button.eva-btn.secondary{backdrop-filter:blur(22px);font-weight:600!important;padding:13px!important}
.gradio-container button.eva-run{padding:16px!important;font-size:15px!important}
.eva-pair{gap:10px!important}
/* сегментированный переключатель */
.eva-seg .wrap{display:grid!important;grid-template-columns:1fr 1fr;gap:4px!important;padding:4px!important;
border:1px solid var(--eva-line2);border-radius:999px;background:var(--eva-panel2)}
.eva-seg .wrap label{justify-content:center!important;border-radius:999px!important;padding:10px!important;margin:0!important;
background:transparent!important;border:0!important;color:var(--eva-mut)!important;font-size:12.5px!important;font-weight:600;cursor:pointer}
.eva-seg .wrap label.selected{background:var(--eva-fg)!important;color:var(--eva-bg)!important}
.eva-seg .wrap label span{color:inherit!important;margin:0!important}
.eva-seg .wrap label input{position:absolute;opacity:0;pointer-events:none;width:0;height:0}
/* чекбоксы */
.gradio-container .eva-check label{gap:11px!important;align-items:flex-start!important}
.gradio-container .eva-check input[type=checkbox]{width:18px;height:18px;border-radius:6px;margin-top:1px}
.gradio-container .eva-check [data-testid="block-info"]{display:none}
.gradio-container .eva-check .label-text{font-size:13.5px}
.gradio-container .eva-check .info-text{padding-left:29px;margin-top:2px;font-size:12px;color:var(--eva-mut)}
/* слайдер */
.eva-slider .head{align-items:center!important}
.eva-slider input[type=number]{width:auto!important;min-width:56px;text-align:center;padding:6px 16px!important;border-radius:999px!important;
font-weight:600;font-size:13px!important}
.eva-slider input[type=range]{accent-color:var(--eva-acc)}
.eva-slider .reset-button,.eva-slider button[aria-label*="eset"]{display:none!important}
.eva-slider .min_value,.eva-slider .max_value{font-family:var(--eva-mono);font-size:11px;color:var(--eva-mut)}
/* отчёт */
.eva-report{overflow:hidden}
.eva-report-head{padding:16px 22px;border-bottom:1px solid var(--eva-line2);display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.eva-report-head .eva-mono{font-size:12px}
.eva-report-body{padding:20px 22px;display:flex;flex-direction:column;gap:18px;font-size:13.5px}
.eva-win{display:flex;flex-direction:column;gap:10px}
.eva-win-head{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.eva-win-name{font-size:13.5px;font-weight:700}
.eva-win-head .eva-mono{font-size:13px}
.eva-score{margin-left:auto;font-family:var(--eva-mono);font-size:13px;font-weight:700}
.eva-kv{display:grid;grid-template-columns:auto 1fr;gap:6px 16px;font-family:var(--eva-mono);font-size:12.5px}
.eva-kv .eva-k{color:var(--eva-mut)}
.eva-sub .eva-kv{font-family:inherit;font-size:13px}
.eva-chips{display:flex;gap:8px;flex-wrap:wrap}
.eva-chip{padding:7px 13px;border:1px solid;border-radius:999px;font-family:var(--eva-mono);font-size:12px;font-weight:600;white-space:nowrap}
.eva-chip.eva-high{border-color:var(--eva-high);background:rgba(255,51,74,.12);color:var(--eva-high)}
.eva-chip.eva-mid{border-color:var(--eva-acc);background:rgba(255,90,19,.12);color:var(--eva-acc)}
.eva-chip.eva-low{border-color:var(--eva-low);background:rgba(139,238,90,.11);color:var(--eva-low)}
.eva-chip.eva-unk{border-color:var(--eva-line);background:var(--eva-panel2);color:var(--eva-mut)}
.eva-divider{height:1px;background:var(--eva-line2)}
.eva-rec{padding:16px 18px;border:1px solid;border-radius:20px;display:flex;flex-direction:column;gap:5px}
.eva-rec-low{border-color:var(--eva-low);background:rgba(139,238,90,.11)}
.eva-rec-mid{border-color:var(--eva-acc);background:rgba(255,90,19,.11)}
.eva-rec-high{border-color:var(--eva-high);background:rgba(255,51,74,.11)}
.eva-rec-unk{border-color:var(--eva-line);background:var(--eva-panel2)}
.eva-rec-text{font-size:15px;font-weight:700}
.eva-rec-extra{font-size:13px;color:var(--eva-fg)}
.eva-sub{padding:16px 18px;border:1px solid var(--eva-line2);border-radius:20px;background:var(--eva-panel2);display:flex;flex-direction:column;gap:8px}
.eva-ev{display:flex;flex-direction:column;gap:6px;padding:10px 0;border-top:1px solid var(--eva-line2);font-size:13px}
.eva-ev-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.eva-error{margin:0;padding:20px 22px;white-space:pre-wrap;font-family:var(--eva-mono);font-size:12px;color:var(--eva-high)}
/* график и файл */
.eva-plot-card{padding:20px 22px!important;gap:4px!important}
.eva-plot-card .js-plotly-plot,.eva-plot-card .plot-container{background:transparent!important}
.eva-plot-card .block,.eva-file-card .block{padding:0!important}
.eva-file-card{padding:20px 22px!important;gap:12px!important}
.eva-file-card .file-preview-holder,.eva-file-card table{border:1px solid var(--eva-line2)!important;border-radius:18px!important;
background:var(--eva-panel2)!important;overflow:hidden}
.eva-file-card td,.eva-file-card a{font-family:var(--eva-mono)!important;font-size:13px!important;color:var(--eva-fg)!important;border:0!important}
.eva-file-card .download a{color:var(--eva-acc)!important;font-weight:700}
.eva-file-card .empty,.eva-file-card [data-testid="upload"]{min-height:0!important}
/* кэш */
.eva-cache{padding:20px 22px}
.eva-cache-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.eva-cache-row{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:9px 0;border-top:1px solid var(--eva-line2);font-size:12.5px}
.eva-cache-row .eva-mut{font-size:11.5px}
.eva-dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--eva-low);margin-right:9px;vertical-align:middle}
@media (max-width:900px){
.eva-main{flex-direction:column!important}
.eva-main>*{width:100%!important}
.gradio-container{padding:20px 16px 48px!important}
.eva-kv{grid-template-columns:1fr;gap:2px}
.eva-kv .eva-k{margin-top:6px}
}
"""

with gr.Blocks(title="Анализ рисков ВКД на МКС", fill_width=True) as app:
    gr.HTML(HEADER_HTML)
    
    with gr.Row(equal_height=False, elem_classes="eva-main"):
        with gr.Column(elem_classes="eva-col"):
            gr.HTML('<h2 class="eva-h">ПАРАМЕТРЫ АНАЛИЗА</h2>')
            forecast_btn = gr.Button("Прогноз на 3 дня", variant="primary", size="lg", elem_classes="eva-btn")
            with gr.Row(elem_classes="eva-pair"):
                today_btn = gr.Button("Сегодня", elem_classes="eva-btn")
                tomorrow_btn = gr.Button("Завтра", elem_classes="eva-btn")
                
            with gr.Column(elem_classes="eva-card"):
                mode = gr.Radio(
                    choices=[("Исторический (Replay)", "Исторический режим (Replay)"),
                             ("Текущая обстановка", "Текущая обстановка")],
                    value="Исторический режим (Replay)",
                    label="Режим работы",
                    elem_classes="eva-seg"
                )
                date_input = gr.Textbox(
                    label="Дата · YYYY-MM-DD",
                    value="2024-05-11",
                    placeholder="2024-05-11"
                )
                forecast_mode = gr.Checkbox(
                    label="Режим прогноза на 3 дня",
                    value=False,
                    info="Отметьте для анализа следующих 3 дней",
                    elem_classes="eva-check"
                )
                
            with gr.Row(elem_classes="eva-card eva-row-card"):
                w1_hour = gr.Number(label="Окно 1 · час (UTC)", value=12, minimum=0, maximum=23, step=1)
                w1_min = gr.Number(label="Окно 1 · минута", value=0, minimum=0, maximum=59, step=1)
                
            with gr.Row(elem_classes="eva-card eva-row-card"):
                w2_hour = gr.Number(label="Окно 2 · час (UTC)", value=18, minimum=0, maximum=23, step=1)
                w2_min = gr.Number(label="Окно 2 · минута", value=0, minimum=0, maximum=59, step=1)
                
            with gr.Column(elem_classes="eva-card"):
                duration = gr.Slider(
                    minimum=1, maximum=8, value=6, step=1,
                    label="Длительность ВКД · часы",
                    elem_classes="eva-slider"
                )
                force_refresh = gr.Checkbox(
                    label="Принудительно обновить кэш",
                    value=False,
                    elem_classes="eva-check"
                )
                
            analyze_btn = gr.Button("Запустить анализ", variant="primary", size="lg", elem_classes="eva-btn eva-run")
            
        with gr.Column(elem_classes="eva-col"):
            gr.HTML('<h2 class="eva-h">РЕЗУЛЬТАТЫ</h2>')
            text_output = gr.HTML(REPORT_EMPTY_HTML)
            
            with gr.Column(elem_classes="eva-card eva-plot-card"):
                gr.HTML(_card_title("Временная картина Kp-индекса"))
                chart_output = gr.Plot(show_label=False)
                
            with gr.Column(elem_classes="eva-card eva-file-card"):
                gr.HTML(_card_title("Сохранённый отчёт (JSON)"))
                file_output = gr.File(show_label=False)
                
            cache_md = gr.HTML(cache_to_html(eva_cache.get_status_md()))

    def set_forecast_mode():
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return True, today

    def set_today():
        today = datetime.now().strftime("%Y-%m-%d")
        return today, False

    def set_tomorrow():
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        return tomorrow, False

    forecast_btn.click(fn=set_forecast_mode, inputs=[], outputs=[forecast_mode, date_input])
    today_btn.click(fn=set_today, inputs=[], outputs=[date_input, forecast_mode])
    tomorrow_btn.click(fn=set_tomorrow, inputs=[], outputs=[date_input, forecast_mode])
    
    analyze_btn.click(
        fn=run_analysis_ui,
        inputs=[mode, date_input, w1_hour, w1_min, w2_hour, w2_min, duration, forecast_mode, force_refresh],
        outputs=[text_output, chart_output, file_output, cache_md]
    )

# ==========================================
# 9. ЗАПУСК
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ЗАПУСК ВЕБ-ПРИЛОЖЕНИЯ АНАЛИЗА ВКД (4 МЕХАНИЗМА + БАЗОВЫЙ ФОН)")
    print("=" * 70)
    print("Откройте браузер и перейдите по ссылке:")
    print("http://127.0.0.1:7860")
    print("=" * 70 + "\n")
    app.launch(server_name="127.0.0.1", server_port=7860, share=False,
               theme=gr.themes.Base(), css=UI_CSS + _background_css(), head=UI_HEAD)
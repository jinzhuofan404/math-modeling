import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '题目', 'B题：附件 数据集')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'data')

os.makedirs(OUT_DIR, exist_ok=True)


def load_pickdata1(chunksize=200000):
    """Load pickdata1 with chunked reading, aggregate per user."""
    path = os.path.join(DATA_DIR, 'pickdata1.csv')
    print(f'Loading {path}...')

    user_agg = {}
    total_rows = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        total_rows += len(chunk)
        if total_rows % 500000 == 0:
            print(f'  pickdata1: processed {total_rows} rows...')

        chunk['dt'] = pd.to_datetime(chunk['dt'])
        # Filter only the useful resources
        resource_map = {
            'food': 'food', 'wood': 'wood', 'stone': 'stone',
            'coins': 'coins', 'diamond': 'diamond', 'fund': 'fund'
        }

        for uid, grp in chunk.groupby('account_id'):
            if uid not in user_agg:
                user_agg[uid] = {
                    'n_records_p1': 0,
                    'first_date': None, 'last_date': None,
                    'days_active_p1': set(),
                    'total_get': 0.0, 'total_reduce': 0.0,
                    'res_get': {}, 'res_reduce': {},
                    'levels_p1': [],
                    'current_level_max': 0,
                    'league_name': None,
                    'country': None, 'platform': None, 'channel_id': None,
                }
            u = user_agg[uid]
            u['n_records_p1'] += len(grp)
            u['first_date'] = min(u['first_date'] or grp['dt'].min(), grp['dt'].min())
            u['last_date'] = max(u['last_date'] or grp['dt'].max(), grp['dt'].max())
            u['days_active_p1'].update(grp['dt'].dt.date.unique())

            # Resource aggregation
            for ct, label in [('get', 'res_get'), ('reduce', 'res_reduce')]:
                sub = grp[grp['change_type'] == ct]
                for rname, rsum in sub.groupby('resource_name')['change_num'].sum().items():
                    u[label][rname] = u[label].get(rname, 0) + rsum

            u['total_get'] += grp[grp['change_type'] == 'get']['change_num'].sum()
            u['total_reduce'] += grp[grp['change_type'] == 'reduce']['change_num'].sum()
            u['levels_p1'].extend(grp['current_level'].dropna().tolist())
            u['current_level_max'] = max(u['current_level_max'], grp['current_level'].max())

            last_row = grp.iloc[-1]
            if pd.notna(last_row.get('league_name')) and u['league_name'] is None:
                u['league_name'] = last_row['league_name']
            if u['country'] is None:
                u['country'] = last_row['country']
            if u['platform'] is None:
                u['platform'] = last_row['platform']
            if u['channel_id'] is None:
                u['channel_id'] = last_row['channel_id']

    print(f'  pickdata1 done: {total_rows} rows, {len(user_agg)} users')
    return user_agg


def load_pickdata2(user_agg, chunksize=100000):
    """Load pickdata2 with chunked reading, update user aggregation."""
    path = os.path.join(DATA_DIR, 'pickdata2.csv')
    print(f'Loading {path}...')

    total_rows = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        total_rows += len(chunk)
        if total_rows % 500000 == 0:
            print(f'  pickdata2: processed {total_rows} rows...')

        chunk['dt'] = pd.to_datetime(chunk['dt'])

        for uid, grp in chunk.groupby('account_id'):
            if uid not in user_agg:
                continue  # Skip users not in pickdata1 (shouldn't happen for training set)

            u = user_agg[uid]
            u.setdefault('n_records_p2', 0)
            u.setdefault('days_active_p2', set())
            u.setdefault('levels_p2', [])
            u.setdefault('total_pay', 0)
            u.setdefault('first_pay_time', None)
            u.setdefault('first_pay_level', None)
            u.setdefault('first_login_time', None)
            u.setdefault('register_time', None)
            u.setdefault('last_login_time', None)
            u.setdefault('vip_level_max', 0)
            u.setdefault('is_in_league', False)
            u.setdefault('event_types', set())
            u.setdefault('media_source', None)
            u.setdefault('diamond_snapshots', [])
            u.setdefault('gold_snapshots', [])
            u.setdefault('drug_snapshots', [])
            u.setdefault('duration_times', 0)

            u['n_records_p2'] += len(grp)
            u['days_active_p2'].update(grp['dt'].dt.date.unique())
            u['levels_p2'].extend(grp['current_level'].dropna().tolist())

            # Payment
            pay_vals = grp['total_pay'].dropna()
            if len(pay_vals) > 0:
                u['total_pay'] = max(u['total_pay'], pay_vals.max())

            # First pay
            fp = grp.dropna(subset=['first_pay_time'])
            if len(fp) > 0 and u['first_pay_time'] is None:
                u['first_pay_time'] = fp.iloc[0]['first_pay_time']
                u['first_pay_level'] = fp.iloc[0]['first_pay_level']

            # Login times
            fl = grp.dropna(subset=['first_login_time'])
            if len(fl) > 0 and u['first_login_time'] is None:
                u['first_login_time'] = fl.iloc[0]['first_login_time']

            rt = grp.dropna(subset=['register_time'])
            if len(rt) > 0 and u['register_time'] is None:
                u['register_time'] = rt.iloc[0]['register_time']

            ll = grp.dropna(subset=['last_login_time'])
            if len(ll) > 0:
                u['last_login_time'] = max(u['last_login_time'] or '', str(ll.iloc[-1]['last_login_time']))

            # VIP
            vip = grp['current_vip_level'].dropna()
            if len(vip) > 0:
                u['vip_level_max'] = max(u['vip_level_max'], vip.max())

            # League (from pickdata1 already set, supplement from event_data)
            # Check if any league-related info exists
            if not u['is_in_league']:
                u['is_in_league'] = (u.get('league_name') is not None and str(u['league_name']) != 'nan')

            # Event types
            u['event_types'].update(grp['event'].dropna().unique())

            # Resource snapshots
            for col, key in [('current_diamond', 'diamond_snapshots'),
                           ('current_gold', 'gold_snapshots'),
                           ('current_drug', 'drug_snapshots')]:
                vals = grp[col].dropna()
                if len(vals) > 0:
                    u[key].append(vals.median())

            # Duration
            dur = grp['duration_times'].dropna()
            if len(dur) > 0:
                u['duration_times'] = max(u['duration_times'], dur.max())

            # Media source
            ms = grp.dropna(subset=['media_source'])
            if len(ms) > 0 and u['media_source'] is None:
                u['media_source'] = ms.iloc[0]['media_source']

    print(f'  pickdata2 done: {total_rows} rows')
    return user_agg


def build_feature_table(user_agg):
    """Convert user aggregation dict to a clean feature DataFrame."""
    rows = []
    for uid, u in user_agg.items():
        # Active days
        all_days = u.get('days_active_p1', set()) | u.get('days_active_p2', set())
        n_active_days = len(all_days)

        # Date span
        first_date = u.get('first_date')
        last_date = u.get('last_date')
        if first_date and last_date:
            lifecycle_days = (last_date - first_date).days + 1
        else:
            lifecycle_days = n_active_days

        # Level stats
        all_levels = u.get('levels_p1', []) + u.get('levels_p2', [])
        level_start = min(all_levels) if all_levels else 0
        level_end = max(all_levels) if all_levels else 0
        level_growth = level_end - level_start
        level_growth_rate = level_growth / max(n_active_days, 1)

        # Resource totals
        res_get = u.get('res_get', {})
        res_reduce = u.get('res_reduce', {})

        # Diamond stats
        dia_snaps = u.get('diamond_snapshots', [])
        dia_median = np.median(dia_snaps) if dia_snaps else 0

        rows.append({
            'account_id': uid,
            'n_records_p1': u.get('n_records_p1', 0),
            'n_records_p2': u.get('n_records_p2', 0),
            'total_records': u.get('n_records_p1', 0) + u.get('n_records_p2', 0),
            'days_active': n_active_days,
            'lifecycle_days': lifecycle_days,
            'first_date': first_date,
            'last_date': last_date,
            'level_start': level_start,
            'level_end': level_end,
            'level_growth': level_growth,
            'level_growth_rate': level_growth_rate,
            'current_level_max': u.get('current_level_max', 0),
            'total_pay': u.get('total_pay', 0),
            'is_paying': 1 if u.get('total_pay', 0) > 0 else 0,
            'first_pay_time': u.get('first_pay_time'),
            'first_pay_level': u.get('first_pay_level'),
            'vip_level_max': u.get('vip_level_max', 0),
            'is_in_league': 1 if u.get('is_in_league', False) else 0,
            'league_name': u.get('league_name'),
            'country': u.get('country'),
            'platform': u.get('platform'),
            'channel_id': u.get('channel_id'),
            'media_source': u.get('media_source'),
            'food_get': res_get.get('food', 0),
            'food_reduce': res_reduce.get('food', 0),
            'wood_get': res_get.get('wood', 0),
            'wood_reduce': res_reduce.get('wood', 0),
            'stone_get': res_get.get('stone', 0),
            'stone_reduce': res_reduce.get('stone', 0),
            'coins_get': res_get.get('coins', 0),
            'coins_reduce': res_reduce.get('coins', 0),
            'diamond_get': res_get.get('diamond', 0),
            'diamond_reduce': res_reduce.get('diamond', 0),
            'diamond_median': dia_median,
            'total_get': u.get('total_get', 0),
            'total_reduce': u.get('total_reduce', 0),
            'duration_times': u.get('duration_times', 0),
            'n_event_types': len(u.get('event_types', set())),
        })

    df = pd.DataFrame(rows)
    return df


def main():
    print('=' * 60)
    print('Feature Engineering for B Problem')
    print('=' * 60)

    # Step 1: Load pickdata1
    user_agg = load_pickdata1()

    # Step 2: Load pickdata2 and merge
    user_agg = load_pickdata2(user_agg)

    # Step 3: Build feature table
    df = build_feature_table(user_agg)
    print(f'\nFeature table: {df.shape[0]} users, {df.shape[1]} features')

    # Summary stats
    print(f'\n=== Summary ===')
    print(f'Total users: {len(df)}')
    print(f'Paying users: {df["is_paying"].sum()} ({df["is_paying"].mean()*100:.1f}%)')
    print(f'Mean total_pay: {df[df["is_paying"]==1]["total_pay"].mean():.2f}')
    print(f'Max total_pay: {df["total_pay"].max():.2f}')
    print(f'Mean days_active: {df["days_active"].mean():.1f}')
    print(f'Mean lifecycle_days: {df["lifecycle_days"].mean():.1f}')
    print(f'League participation: {df["is_in_league"].mean()*100:.1f}%')
    print(f'Mean level_end: {df["level_end"].mean():.1f}')
    print(f'Top countries: {df["country"].value_counts().head(5).to_dict()}')

    # Save
    out_path = os.path.join(OUT_DIR, 'player_features.csv')
    df.to_csv(out_path, index=False)
    print(f'\nSaved to {out_path}')

    # Also save a smaller version for quick loading
    df_slim = df.drop(columns=['first_date', 'last_date', 'first_pay_time', 'league_name'], errors='ignore')
    slim_path = os.path.join(OUT_DIR, 'player_features_slim.csv')
    df_slim.to_csv(slim_path, index=False)
    print(f'Saved slim version to {slim_path}')

    return df


def build_day3_features():
    """Step 1: Build features using ONLY Day1-3 data. No future information leakage."""
    print('=' * 60)
    print('Building Day3-only Feature Table')
    print('=' * 60)

    path1 = os.path.join(DATA_DIR, 'pickdata1.csv')
    path2 = os.path.join(DATA_DIR, 'pickdata2.csv')

    # ── Pass 1: Determine Day1 and collect active dates from pickdata2 ──
    print('Pass 1: Reading pickdata2 for Day1 + active dates...')
    user_day1 = {}       # uid -> Day1 date
    user_active_dates = {}  # uid -> set of date objects

    total = 0
    for chunk in pd.read_csv(path2, chunksize=200000):
        total += len(chunk)
        chunk['dt'] = pd.to_datetime(chunk['dt'])
        chunk['date'] = chunk['dt'].dt.date
        chunk['register_time'] = pd.to_datetime(chunk['register_time'], errors='coerce')

        for uid, grp in chunk.groupby('account_id'):
            if uid not in user_day1:
                # Day1 = register_time if available, else first pickdata2 record
                rt = grp['register_time'].dropna()
                if len(rt) > 0:
                    user_day1[uid] = rt.iloc[0].date()
                else:
                    user_day1[uid] = grp['date'].min()
                user_active_dates[uid] = set()
            user_active_dates[uid].update(grp['date'].unique())

        if total % 500000 == 0:
            print(f'  processed {total} rows...')
    print(f'  Pass 1 done: {len(user_day1)} users')

    # ── Compute churn labels ──
    print('Computing churn labels...')
    churn_info = {}
    for uid, dates in user_active_dates.items():
        d1 = user_day1[uid]
        # Check for first "2 consecutive days without records" starting from d1
        all_days = sorted(dates)
        churn_day = None
        # Build a set of days relative to d1
        max_day = 30
        for offset in range(max_day):
            day = d1 + pd.Timedelta(days=offset)
            next_day = d1 + pd.Timedelta(days=offset + 1)
            if day not in dates and next_day not in dates:
                churn_day = offset  # 0-indexed: churned starting at this offset
                break
        if churn_day is None:
            churn_info[uid] = {'duration': 30, 'event': 0}
        else:
            churn_info[uid] = {'duration': max(1, churn_day), 'event': 1}

    # ── Pass 2: Extract Day1-3 features from pickdata2 ──
    print('Pass 2: Extracting Day1-3 features from pickdata2...')
    user_feat_p2 = {}
    for uid in user_day1:
        user_feat_p2[uid] = {
            'days_logged_p2': 0,
            'levels_day3': [],
            'dur_times': [],
            'diamond_snaps': [],
            'gold_snaps': [],
            'event_types': set(),
            'is_pay_day3': False,
            'is_league_day3': False,
        }

    total = 0
    for chunk in pd.read_csv(path2, chunksize=200000):
        total += len(chunk)
        chunk['dt'] = pd.to_datetime(chunk['dt'])
        chunk['date'] = chunk['dt'].dt.date
        chunk['first_pay_time'] = pd.to_datetime(chunk['first_pay_time'], errors='coerce')

        for uid, grp in chunk.groupby('account_id'):
            if uid not in user_day1:
                continue
            d1 = user_day1[uid]
            d3_end = d1 + pd.Timedelta(days=2)  # Day3 = d1+2
            # Filter to Day1-3
            mask_d3 = grp['date'] <= d3_end
            d3 = grp[mask_d3]
            if len(d3) == 0:
                continue

            u = user_feat_p2[uid]
            u['days_logged_p2'] = d3['date'].nunique()
            u['levels_day3'].extend(d3['current_level'].dropna().tolist())
            u['dur_times'].extend(d3['duration_times'].dropna().tolist())
            u['diamond_snaps'].extend(d3['current_diamond'].dropna().tolist())
            u['gold_snaps'].extend(d3['current_gold'].dropna().tolist())
            u['event_types'].update(d3['event'].dropna().unique())

            # First pay within Day3
            fpt = d3['first_pay_time'].dropna()
            if len(fpt) > 0:
                u['is_pay_day3'] = True

            # No league check from p2 - p1 has league_name

        if total % 500000 == 0:
            print(f'  processed {total} rows...')
    print(f'  Pass 2 done.')

    # ── Pass 3: Extract Day1-3 resource features from pickdata1 ──
    print('Pass 3: Extracting Day1-3 resource features from pickdata1...')
    user_feat_p1 = {}
    for uid in user_day1:
        user_feat_p1[uid] = {
            'food_reduce': 0.0, 'wood_reduce': 0.0, 'stone_reduce': 0.0,
            'diamond_reduce': 0.0, 'coins_reduce': 0.0,
            'levels_p1': [], 'league_day3': False, 'country': None,
            'platform': None, 'channel_id': None,
        }

    total = 0
    for chunk in pd.read_csv(path1, chunksize=200000):
        total += len(chunk)
        chunk['dt'] = pd.to_datetime(chunk['dt'])
        chunk['date'] = chunk['dt'].dt.date

        for uid, grp in chunk.groupby('account_id'):
            if uid not in user_day1:
                continue
            d1 = user_day1[uid]
            d3_end = d1 + pd.Timedelta(days=2)
            mask_d3 = grp['date'] <= d3_end
            d3 = grp[mask_d3]
            if len(d3) == 0:
                continue

            u = user_feat_p1[uid]
            # Resource consumption (reduce only)
            for res in ['food', 'wood', 'stone', 'diamond', 'coins']:
                sub = d3[(d3['resource_name'] == res) & (d3['change_type'] == 'reduce')]
                if len(sub) > 0:
                    u[f'{res}_reduce'] += sub['change_num'].sum()

            u['levels_p1'].extend(d3['current_level'].dropna().tolist())

            # League
            league = d3['league_name'].dropna()
            if len(league) > 0:
                u['league_day3'] = True

            # Demographics
            last = d3.iloc[-1]
            if u['country'] is None and pd.notna(last.get('country')):
                u['country'] = last['country']
            if u['platform'] is None:
                u['platform'] = last.get('platform', 1)
            if u['channel_id'] is None:
                u['channel_id'] = last.get('channel_id', '')

        if total % 500000 == 0:
            print(f'  processed {total} rows...')
    print(f'  Pass 3 done.')

    # ── Build final feature table ──
    print('Building feature table...')
    rows = []
    for uid in user_day1:
        p2 = user_feat_p2[uid]
        p1 = user_feat_p1[uid]
        ci = churn_info[uid]

        # Level: Day3 max level - Day1 min level
        all_lvls = p2['levels_day3'] + p1['levels_p1']
        level_d1 = min(all_lvls) if all_lvls else 0
        level_d3 = max(all_lvls) if all_lvls else 0
        level_change = level_d3 - level_d1

        # Online time
        avg_duration = np.mean(p2['dur_times']) if p2['dur_times'] else 0

        # Diamond and gold
        dia_d3 = np.median(p2['diamond_snaps']) if p2['diamond_snaps'] else 0
        gold_d3 = np.median(p2['gold_snaps']) if p2['gold_snaps'] else 0

        rows.append({
            'account_id': uid,
            'days_logged_d3': p2['days_logged_p2'],
            'level_d3': level_d3,
            'level_change_d3': level_change,
            'avg_duration_d3': avg_duration,
            'food_reduce_d3': p1['food_reduce'],
            'wood_reduce_d3': p1['wood_reduce'],
            'stone_reduce_d3': p1['stone_reduce'],
            'diamond_reduce_d3': p1['diamond_reduce'],
            'coins_reduce_d3': p1['coins_reduce'],
            'diamond_d3': dia_d3,
            'gold_d3': gold_d3,
            'is_pay_d3': 1 if p2['is_pay_day3'] else 0,
            'is_league_d3': 1 if p1['league_day3'] else 0,
            'n_event_types_d3': len(p2['event_types']),
            'country': p1['country'],
            'platform': p1['platform'],
            'channel_id': p1['channel_id'],
            # Churn targets
            'duration': ci['duration'],
            'event_churned': ci['event'],
        })

    df = pd.DataFrame(rows)

    # Summary
    print(f'\n=== Day3 Feature Table Summary ===')
    print(f'Users: {len(df)}')
    print(f'Features: {len(df.columns)}')
    print(f'Paying Day3: {df["is_pay_d3"].sum()} ({df["is_pay_d3"].mean()*100:.1f}%)')
    print(f'League Day3: {df["is_league_d3"].sum()} ({df["is_league_d3"].mean()*100:.1f}%)')
    print(f'Mean days logged (Day1-3): {df["days_logged_d3"].mean():.1f}')
    print(f'Churn rate: {df["event_churned"].mean()*100:.1f}%')
    print(f'Mean duration: {df["duration"].mean():.1f} days')
    print(f'Censored: {(df["event_churned"]==0).sum()} users')

    # Verify no leakage variables
    forbidden = ['lifecycle_days', 'level_end', 'total_pay', 'vip_level_max', 'total_records']
    leaked = [c for c in forbidden if c in df.columns]
    if leaked:
        print(f'[WARNING] Leakage variables found: {leaked}')
    else:
        print(f'[OK] No leakage variables detected.')

    # Save
    out_path = os.path.join(OUT_DIR, 'player_features_day3.csv')
    df.to_csv(out_path, index=False)
    print(f'\nSaved to {out_path}')
    return df


if __name__ == '__main__':
    # Run original full-period feature engineering
    # df_full = main()

    # Run Day3-only feature engineering (for Problem 1)
    df_day3 = build_day3_features()

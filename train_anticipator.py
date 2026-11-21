"""Train on sessions, select on validation, and report event-level anticipation.

python train_anticipator.py --data data --output models/anticipator.joblib
Family sessions use leave-one-performer-out and training-only selection.
Legacy side-only sessions retain pass-1 reports plus leave-one-session-out checks.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance
from pose_features import (FeatureStream, CueGate, reactive_onset, feature_names, MotionOnset,
                           CandidateConfirm, FAMILY_CUES, NEGATIVES)


def load_session(path):
    with np.load(path, allow_pickle=False) as z:
        session = {k: z[k] for k in z.files}
    t = session['timestamps']
    if not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(f'{path}: timestamps must be finite and strictly increasing')
    if session['keypoints'].shape != (len(t), 17, 2) or session['confidences'].shape != (len(t), 17):
        raise ValueError(f'{path}: invalid pose shape')
    frames = session['onset_frames'].astype(int)
    if np.any(frames < 0) or np.any(frames >= len(t)):
        raise ValueError(f'{path}: onset out of bounds')
    if len(frames) != len(session['onset_sides']) or not set(session['onset_sides']).issubset({'LEFT', 'RIGHT', 'CENTER'}):
        raise ValueError(f'{path}: invalid onset sides')
    order = np.argsort(frames)
    session['onset_frames'] = frames[order]
    session['onset_sides'] = session['onset_sides'][order]
    for name in ('onset_families', 'onset_sources', 'detected_times'):
        if name in session:
            if len(session[name]) != len(frames):
                raise ValueError(f'{path}: invalid {name} length')
            session[name] = session[name][order]
    if 'glove_blobs' in session and session['glove_blobs'].shape != (len(t), 2, 3):
        raise ValueError(f'{path}: invalid glove blob shape')
    if len(session.get('commands', [])) != len(session.get('command_times', [])):
        raise ValueError(f'{path}: command/time count differs')
    if np.any(np.diff(session.get('command_times', [])) <= 0):
        raise ValueError(f'{path}: command times must increase')
    human_path = Path(path).with_suffix('.labels.json')
    if human_path.exists():
        human = json.loads(human_path.read_text())
        session['human_labels'] = human.get('labelers', {})
    return session


def onsets(session):
    human = session.get('human_labels', {})
    if any(human.values()):
        # Pair independent labelers' marks by TIME (within 150 ms), not by list order:
        # if one labeler skips a punch, order-based pairing shifts every later pair.
        # Keep the full range; use its midpoint for training and score range bounds separately.
        ts = session['timestamps']
        flat = []
        for name, group in human.items():
            for m in group:
                if m.get('departure') is None:
                    continue
                f = int(m['departure'])
                if f < 0 or f >= len(ts):
                    raise ValueError('Human departure outside session')
                flat.append((float(ts[f]), name, m))
        flat.sort(key=lambda x: x[0])
        clusters = []
        for t0, name, m in flat:
            c = clusters[-1] if clusters else None
            if c is not None and t0-c['t'] <= .15 and name not in c['names']:
                c['marks'].append(m)
                c['names'].add(name)
            else:
                clusters.append(dict(t=t0, names={name}, marks=[m]))
        events = []
        for c in clusters:
            marks = c['marks']
            families = {m['family'] for m in marks}
            frames = [int(m['departure']) for m in marks]
            family = next(iter(families)) if len(families) == 1 else 'UNKNOWN'
            if family in NEGATIVES:
                continue
            lo, hi = [float(session['timestamps'][j]) for j in (min(frames), max(frames))]
            events.append(dict(time=(lo+hi)/2, side=family, family=family,
                               departure_range=(lo, hi), source='human'))
        return events
    families = session.get('onset_families')
    events = []
    for j, (i, side) in enumerate(zip(session['onset_frames'], session['onset_sides'])):
        family = str(families[j]) if families is not None else str(side)
        if family in NEGATIVES:
            continue
        if family == 'UNKNOWN':
            # Legacy unscripted files retain their side labels for compatibility.
            family = str(side)
        events.append(dict(time=float(session['timestamps'][i]), side=family, family=family))
    return events


def build_labels(timestamps, events, horizon=.3, recovery=.4):
    t = np.asarray(timestamps)
    labels, keep = np.full(len(t), 'NONE', dtype='<U16'), np.ones(len(t), dtype=bool)
    # Reverse order means the nearest future onset wins overlapping horizons.
    for e in sorted(events, key=lambda e: e['time'], reverse=True):
        delta = e['time']-t
        if e['side'] == 'UNKNOWN':
            keep &= ~((delta >= -.4) & (delta <= horizon))
            continue
        labels[(delta > 0) & (delta <= horizon+1e-10)] = e['side']
        keep &= ~((t >= e['time']) & (t <= e['time']+recovery+1e-10))
    return labels, keep


def session_features(session, window=8):
    stream, rows, valid = FeatureStream(window), [], []
    for i, (t, xy, conf) in enumerate(zip(session['timestamps'], session['keypoints'], session['confidences'])):
        blobs = session.get('glove_blobs')
        rows.append(stream.update(t, xy, conf, blobs[i] if blobs is not None else None))
        f = stream.history[-1]
        valid.append(bool(f.mask[5] and f.mask[6] and (f.mask[9] or f.mask[10])))
    return np.asarray(rows), np.asarray(valid)


def warnings_from_probabilities(timestamps, probabilities, classes, threshold=.5, valid=None):
    gate, events = CueGate(threshold), []
    if valid is None:
        valid = np.ones(len(timestamps), dtype=bool)
    for t, p, ok in zip(timestamps, probabilities, valid):
        event = gate.update(t, dict(zip(classes, p)) if ok else {})
        if event:
            events.append(event)
    return events


def _non_punch_seconds(start, end, events, recovery):
    intervals = sorted((max(start, e['time']), min(end, e['time']+recovery)) for e in events)
    occupied, right = 0., start
    for a, b in intervals:
        if b > max(a, right):
            occupied += b-max(a, right)
        right = max(right, b)
    return max(0., end-start-occupied)


def event_metrics(warnings, events, start, end, horizon=.3, recovery=.4, reactive=False):
    """One cue matches at most one event, regardless of side. Side scored separately.

    Unmatched cues during punch/recovery are counted separately; the false alarm
    rate uses only unmatched cues outside those intervals and their duration.
    Reactive matches allow [onset,onset+recovery], but never count as early.
    """
    events = sorted(events, key=lambda e: e['time'])
    matched, leads, correct, false, during = set(), [], 0, 0, 0
    for warning in sorted(warnings, key=lambda e: e['time']):
        matches = [i for i, e in enumerate(events) if i not in matched and
                   ((not reactive and 0 < e['time']-warning['time'] <= horizon+1e-10) or
                    (reactive and 0 <= warning['time']-e['time'] <= recovery))]
        if matches:
            i = matches[0]
            matched.add(i)
            leads.append((events[i]['time']-warning['time'])*1000)
            correct += warning['side'] == events[i]['side']
        elif any(e['time'] <= warning['time'] <= e['time']+recovery for e in events):
            during += 1
        else:
            false += 1
    early = np.asarray([v for v in leads if v > 0])
    total = len(events)
    duration = _non_punch_seconds(start, end, events, recovery)
    return dict(punches=total, matched=len(matched), warned=int(len(early)),
                warned_pct=100*len(early)/total if total else 0.,
                useful_pct=100*sum(v >= 100-1e-8 for v in leads)/total if total else 0.,
                median_ms=float(np.median(leads)) if leads else float('nan'),
                p25_ms=float(np.percentile(leads, 25)) if leads else float('nan'),
                p75_ms=float(np.percentile(leads, 75)) if leads else float('nan'),
                side_accuracy=100*correct/len(matched) if matched else float('nan'),
                false_alarms=false, false_per_min=false*60/duration if duration > 0 else float('nan'),
                non_punch_seconds=duration, recovery_cues=during)


def print_metrics(label, m):
    print(f"{label:18s} before {m['warned']}/{m['punches']} ({m['warned_pct']:.1f}%) | "
          f">=100ms {m['useful_pct']:.1f}% | lead median/p25/p75 "
          f"{m['median_ms']:.1f}/{m['p25_ms']:.1f}/{m['p75_ms']:.1f} ms | "
          f"false/min {m['false_per_min']:.2f} | side {m['side_accuracy']:.1f}% | "
          f"recovery cues {m['recovery_cues']}")


def fit_candidates(x, y):
    weights = compute_sample_weight('balanced', y)
    hist = HistGradientBoostingClassifier(class_weight='balanced', max_iter=120,
                                         max_leaf_nodes=15, l2_regularization=1., random_state=42)
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16),
                             max_iter=350, random_state=42, early_stopping=False))
    hist.fit(x, y)
    mlp.fit(x, y, mlpclassifier__sample_weight=weights)
    return {'hist': hist, 'mlp': mlp}


def evaluate(model, session, x, valid, threshold, horizon):
    warnings = warnings_from_probabilities(session['timestamps'], model.predict_proba(x),
                                         model.classes_, threshold, valid)
    t = session['timestamps']
    return event_metrics(warnings, onsets(session), t[0], t[-1], horizon)


def main():
    global FAMILY_CUES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='data')
    ap.add_argument('--output', default='models/anticipator.joblib')
    ap.add_argument('--window', type=int, default=8)
    ap.add_argument('--horizon', type=float, default=.3)
    ap.add_argument('--cue-map', help='JSON family-to-game-command overrides, also saved in model')
    args = ap.parse_args()
    if args.cue_map:
        overrides = json.loads(Path(args.cue_map).read_text())
        if not set(overrides).issubset(FAMILY_CUES) or not set(overrides.values()).issubset({'DUCK', 'BACK', 'DODGE LEFT', 'DODGE RIGHT'}):
            ap.error('Invalid family or game command in --cue-map')
        FAMILY_CUES = dict(FAMILY_CUES, **overrides)
    paths = sorted(Path(args.data).glob('*.npz'))
    # Re-posing the same recording must never place copies on both sides of a split.
    unique = {}
    for p in paths:
        session = load_session(p)
        identity = str(session.get('session_id', p.stem))
        if identity in unique:
            print(f'Re-posed session {identity}: using {p.name} only.')
        unique[identity] = (p, session)
    pairs = list(unique.values())
    paths, sessions = [p for p, s in pairs], [s for p, s in pairs]
    count = sum(len(onsets(s)) for s in sessions)
    if count < 30:
        print(f'Too little data: {count} onsets; need at least 30 (aim for 100+ across sessions). No model trained.')
        return
    if len(sessions) < 2:
        print('Need at least two independent sessions for a session holdout. No model trained.')
        return
    if args.horizon <= 0:
        ap.error('--horizon must be positive')
    prepared = [session_features(s, args.window) for s in sessions]
    labels = [build_labels(s['timestamps'], onsets(s), args.horizon) for s in sessions]
    for session, (y, keep) in zip(sessions, labels):
        if len(session.get('commands', [])):
            y[negative_mask(session)] = 'NONE'
    performers = {str(s.get('performer', 'unknown')) for s in sessions}
    if len(performers - {'unknown', ''}) >= 2:
        train_performer_folds(sessions, paths, prepared, labels, args)
        return
    print('WARNING: fewer than two named performers; falling back to leave-one-session-out.')
    if any(len(s.get('commands', [])) or s.get('human_labels') or
           any(f in FAMILY_CUES for f in s.get('onset_families', [])) for s in sessions):
        train_performer_folds(sessions, paths, prepared, labels, args, by_performer=False)
        return
    ntrain = len(sessions)-2 if len(sessions) >= 3 else 1
    validation = ntrain
    print('Train:', ', '.join(p.name for p in paths[:ntrain]))
    print('Validation:', paths[validation].name, '| Test:', paths[-1].name)
    if len(sessions) == 2:
        print('CAUTION: only two sessions; model selection and reported evaluation share the holdout. Results are optimistic.')
    x = np.concatenate([prepared[i][0][labels[i][1] & prepared[i][1]] for i in range(ntrain)])
    y = np.concatenate([labels[i][0][labels[i][1] & prepared[i][1]] for i in range(ntrain)])
    if len(np.unique(y)) < 2 or not np.any(y == 'NONE'):
        print('Training sessions need both NONE and pre-punch examples. No model trained.')
        return
    candidates = fit_candidates(x, y)
    scores = {}
    for name, model in candidates.items():
        m = evaluate(model, sessions[validation], *prepared[validation], .5, args.horizon)
        print_metrics('validation '+name, m)
        scores[name] = m['useful_pct'] + .25*m['warned_pct'] - 5*np.nan_to_num(m['false_per_min'])
    chosen = max(scores, key=scores.get)
    model = candidates[chosen]
    print(f'Selected {chosen}: useful% + 0.25*warned% - 5*false/min at threshold 0.5.')
    print('Offline timestamps exclude inference/audio latency. Auto onsets are heuristic; '
          'review recordings and correct labels before interpreting these as real punch results.')
    s, (xt, valid) = sessions[-1], prepared[-1]
    for threshold in (.3, .5, .7):
        print_metrics(f'test threshold {threshold}', evaluate(model, s, xt, valid, threshold, args.horizon))
    reactive = reactive_onset(s['timestamps'], s['keypoints'], s['confidences'])
    baseline = [dict(time=e['detected_time'], side=e['side']) for e in reactive]
    print_metrics('reactive baseline', event_metrics(baseline, onsets(s), s['timestamps'][0],
                  s['timestamps'][-1], args.horizon, reactive=True))
    keep = labels[-1][1] & valid
    if keep.any():
        importance = permutation_importance(model, xt[keep], labels[-1][0][keep],
                         scoring='balanced_accuracy', n_repeats=3, random_state=42)
        print('Top 15 held-out permutation importances (balanced frame accuracy drop):')
        for i in np.argsort(importance.importances_mean)[-15:][::-1]:
            print(f'  {feature_names()[i]:30s} {importance.importances_mean[i]:.5f}')
    bundle = dict(model=model, scaler=model.named_steps['standardscaler'] if chosen == 'mlp' else None,
                  window=args.window, horizon=args.horizon, recovery=.4, feature_names=feature_names(),
                  pose_model='yolo11n-pose.pt', selected=chosen, train_sessions=[p.name for p in paths[:ntrain]],
                  validation_session=paths[validation].name, test_session=paths[-1].name)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)
    print('Saved', args.output, '(not refit on held-out sessions).')
    # Legacy side-only sessions retain their pass-1 artifact but get every session
    # scored independently as well; model and threshold selection stay inside train.
    for train, test in performer_splits(sessions, by_performer=False):
        fit_ids, val = train[:-1], train[-1:]
        if not len(fit_ids):
            fit_ids, val = train, []
        xx = np.concatenate([prepared[i][0][labels[i][1] & prepared[i][1]] for i in fit_ids])
        yy = np.concatenate([labels[i][0][labels[i][1] & prepared[i][1]] for i in fit_ids])
        if len(np.unique(yy)) < 2:
            continue
        models = fit_candidates(xx, yy)
        chosen_fold, th_fold, best = 'hist', .5, -np.inf
        for name, candidate in models.items():
            for th in (.3, .5, .7):
                scores = [evaluate(candidate, sessions[i], *prepared[i], th, args.horizon) for i in val]
                if scores:
                    score = np.mean([m['useful_pct']+.25*m['warned_pct']-5*np.nan_to_num(m['false_per_min']) for m in scores])
                    if score > best:
                        chosen_fold, th_fold, best = name, th, score
        for i in test:
            print_metrics('LOSO '+paths[i].stem, evaluate(models[chosen_fold], sessions[i], *prepared[i], th_fold, args.horizon))
    print('Legacy side labels cannot establish gesture-family accuracy; collect scripted or human family labels.')


def performer_splits(sessions, by_performer=True):
    groups = [str(s.get('performer', 'unknown')) if by_performer else str(i)
              for i, s in enumerate(sessions)]
    return [(np.array([i for i, g in enumerate(groups) if g != held], dtype=int),
             np.array([i for i, g in enumerate(groups) if g == held], dtype=int))
            for held in sorted(set(groups))]


def negative_mask(session):
    times = session['timestamps']
    commands = session.get('commands', [])
    command_times = session.get('command_times', [])
    if not len(commands):
        events = onsets(session)
        return np.array([not any(e['time']-.3 <= t <= e['time']+.4 for e in events) for t in times])
    indices = np.searchsorted(command_times, times, side='right')-1
    return np.array([i < 0 or str(commands[i]) in NEGATIVES for i in indices])


def family_metrics(warnings, session, horizon=.3, recovery=.4):
    """Score signed cue-minus-departure delays; negative intervals remain separate.

    Human disagreement is preserved in delay_ranges_ms and conservative prediction
    uses the earliest marked departure. Scripted negatives always take precedence.
    """
    times = session['timestamps']
    negative = negative_mask(session)
    steps = np.r_[np.diff(times), np.median(np.diff(times)) if len(times) > 1 else 0.]
    seconds = float(np.sum(steps[negative]))
    events, used = onsets(session), set()
    stats = {f: dict(attempts=0, on_time_correct=0, wrong_direction=0, unknowns=0,
                     predicted_delays_ms=[], detection_delays_ms=[], delay_ranges_ms=[]) for f in FAMILY_CUES}
    false = 0
    for e in events:
        if e['side'] in stats:
            stats[e['side']]['attempts'] += 1
    for w in sorted(warnings, key=lambda x: x['time']):
        index = int(np.clip(np.searchsorted(times, w['time'], side='right')-1, 0, len(times)-1))
        if negative[index]:
            false += 1
            continue
        candidates = [i for i, e in enumerate(events) if i not in used
                      and -horizon <= w['time']-e['time'] <= recovery]
        if not candidates:
            continue
        i = min(candidates, key=lambda i: abs(events[i]['time']-w['time']))
        used.add(i)
        e = events[i]
        if e['side'] not in stats:
            continue
        row = stats[e['side']]
        expected = FAMILY_CUES[e['side']]
        actual = FAMILY_CUES.get(w.get('family', w.get('side')))
        if actual != expected:
            row['wrong_direction'] += 1
        elif w['time'] <= times[min(len(times)-1, np.searchsorted(times, e.get('departure_range', (e['time'], e['time']))[0])+2)]:
            row['on_time_correct'] += 1
        lo, hi = e.get('departure_range', (e['time'], e['time']))
        delay = (w['time']-e['time'])*1000
        row['delay_ranges_ms'].append(((w['time']-hi)*1000, (w['time']-lo)*1000))
        if w['time'] < lo:
            row['predicted_delays_ms'].append(delay)
        elif w['time'] >= hi:
            row['detection_delays_ms'].append(delay)
        # Cues within labeler disagreement cannot establish prediction vs detection.
    validity = None
    for i, e in enumerate(events):
        if i not in used and e['side'] in stats:
            # Unknowns mean invalid tracking within the scoring interval, not every miss.
            if validity is None:
                _, validity = session_features(session)
            interval = (times >= e['time']-horizon) & (times <= e['time']+recovery)
            stats[e['side']]['unknowns'] += int(not validity[interval].all())
    return dict(families=stats, false_alarms=false, negative_seconds=seconds,
                false_per_min=false*60/seconds if seconds else float('nan'))


def session_features_slice(session, start, end):
    # Preserve preceding history while measuring validity in an event's interval.
    x, valid = session_features(session)
    mask = (session['timestamps'] >= start) & (session['timestamps'] <= end)
    return x[mask], valid[mask]


def offline_cues(model, session, x, threshold=.5, mode='predict', glove=True, predictor=True):
    stream, detector = FeatureStream(), MotionOnset(glove_thresh=3. if glove else np.inf)
    machine = CandidateConfirm(threshold, min(.3, threshold), mode)
    events = []
    probabilities = model.predict_proba(x) if predictor else None
    for i, (t, xy, conf) in enumerate(zip(session['timestamps'], session['keypoints'], session['confidences'])):
        blobs = session.get('glove_blobs')
        stream.update(t, xy, conf, blobs[i] if blobs is not None and glove else None)
        f = stream.history[-1]
        signal = detector.update(f, i)
        probs = dict(zip(model.classes_, probabilities[i])) if predictor and not f.unknown else {}
        if predictor:
            event = machine.update(t, probs, signal, f.unknown)
        else:
            event = dict(time=float(t), side=signal['side'], classification_time=None,
                         detection_time=signal['detected_time']) if signal else None
            # Wrist/growth rules cannot classify a forward vs rising gesture.
            # Score the original positional game command, making that limitation visible.
            if event:
                event['family'] = {'LEFT': 'HOOK_L', 'RIGHT': 'HOOK_R', 'CENTER': 'JAB'}[event['side']]
        if event:
            events.append(event)
    return events


def print_family_metrics(label, result):
    print(label, f"false buzzes {result['false_alarms']} / {result['negative_seconds']:.2f}s negative footage "
          f"= {result['false_per_min']:.2f}/min")
    for family, row in result['families'].items():
        if not row['attempts']:
            continue
        print(f"  {family}: on-time correct {row['on_time_correct']}/{row['attempts']} | "
              f"wrong direction {row['wrong_direction']} | unknown {row['unknowns']}")
        for name in ('predicted_delays_ms', 'detection_delays_ms'):
            delays = row[name]
            worst = max(delays) if delays else float('nan')
            p99 = f'{np.percentile(delays, 99):.1f} ms' if len(delays) >= 50 else 'too few trials'
            print(f"    {name}: n={len(delays)}, median {np.median(delays) if delays else float('nan'):.1f} ms, "
                  f"worst {worst:.1f} ms, p99 {p99}")
    count = sum(len(r['predicted_delays_ms']) for r in result['families'].values())
    print('  Unseen pre-departure prediction evidence' if count and result['false_alarms'] == 0 and result['negative_seconds'] > 0
          else '  Prediction not established: requires unseen pre-departure cues and silence on negatives.')


def train_performer_folds(sessions, paths, prepared, labels, args, by_performer=True):
    """Outer held-out group is never used for model or probability threshold choice."""
    last_bundle = None
    folds = performer_splits(sessions, by_performer)
    for train, test in folds:
        # An inner held-out training session chooses model and threshold. With only
        # one training session, fixed defaults avoid using the outer test at all.
        training_groups = {str(sessions[i].get('performer', 'unknown')) for i in train}
        if by_performer and len(training_groups) > 1:
            held = sorted(training_groups)[-1]
            fit_ids = np.array([i for i in train if str(sessions[i].get('performer', 'unknown')) != held])
            validation = np.array([i for i in train if str(sessions[i].get('performer', 'unknown')) == held])
        else:
            fit_ids, validation = train[:-1], train[-1:]
        if not len(fit_ids):
            fit_ids, validation = train, []
            print('Only one training session: fixed hist/0.5; no held-out threshold calibration possible.')
        x = np.concatenate([prepared[i][0][labels[i][1] & prepared[i][1]] for i in fit_ids])
        y = np.concatenate([labels[i][0][labels[i][1] & prepared[i][1]] for i in fit_ids])
        if len(np.unique(y)) < 2 or not np.any(y == 'NONE'):
            print('Skipping fold: training requires NONE and gesture examples. Held out:', [paths[i].name for i in test])
            continue
        candidates = fit_candidates(x, y)
        chosen, threshold, best = 'hist', .5, -np.inf
        for name, model in candidates.items():
            for th in (.3, .5, .7):
                if not len(validation):
                    continue
                rows = []
                for i in validation:
                    cues = offline_cues(model, sessions[i], prepared[i][0], th)
                    result = family_metrics(cues, sessions[i], args.horizon)
                    hits = sum(r['on_time_correct'] for r in result['families'].values())
                    attempts = sum(r['attempts'] for r in result['families'].values())
                    rows.append(100*hits/max(attempts, 1)-5*np.nan_to_num(result['false_per_min']))
                score = np.mean(rows)
                if score > best:
                    chosen, threshold, best = name, th, score
        model = candidates[chosen]
        print('Held out:', ', '.join(paths[i].name for i in test), '| training-only selection:', chosen, threshold)
        for i in test:
            print('Ablation table on identical held-out clip:', paths[i].name)
            baseline_lead, baseline_false = None, None
            for label, glove, predictor, mode in (
                ('(a) wrist only', False, False, 'confirm'),
                ('(b) + relative glove area', True, False, 'confirm'),
                ('(c) + pose candidate predict', True, True, 'predict'),
                ('(c) + pose candidate confirm', True, True, 'confirm')):
                cues = offline_cues(model, sessions[i], prepared[i][0], threshold, mode, glove, predictor)
                result = family_metrics(cues, sessions[i], args.horizon)
                print_family_metrics(label, result)
                delays = [d for r in result['families'].values()
                          for k in ('predicted_delays_ms', 'detection_delays_ms') for d in r[k]]
                lead = -float(np.median(delays)) if delays else float('nan')
                if baseline_lead is None:
                    baseline_lead, baseline_false = lead, result['false_alarms']
                print(f"  median lead {lead:.1f} ms | lead bought {lead-baseline_lead:.1f} ms | "
                      f"false alarms added {result['false_alarms']-baseline_false}")
                # Detection and classification have independent timestamps in each cue.
                detections = [1000*(e['classification_time']-e['detection_time']) for e in cues
                              if e.get('classification_time') is not None and e.get('detection_time') is not None]
                print('  classification minus motion detection ms:', detections)
        for i in test:
            keep = labels[i][1] & prepared[i][1]
            if keep.any():
                importance = permutation_importance(model, prepared[i][0][keep], labels[i][0][keep],
                                                   scoring='balanced_accuracy', n_repeats=3, random_state=42)
                print('Top 15 held-out permutation importances (never used for selection):')
                for j in np.argsort(importance.importances_mean)[-15:][::-1]:
                    print(f'  {feature_names()[j]:30s} {importance.importances_mean[j]:.5f}')
        last_bundle = dict(model=model, window=args.window, horizon=args.horizon, recovery=.4,
                           feature_names=feature_names(), selected=chosen, threshold=threshold,
                           family_cues=FAMILY_CUES, pose_model='yolo11n-pose.pt',
                           train_sessions=[paths[i].name for i in fit_ids],
                           validation_sessions=[paths[i].name for i in validation],
                           test_sessions=[paths[i].name for i in test])
    if last_bundle:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(last_bundle, args.output)
        print('Saved', args.output, '(last fold, not refit on held-out performers).')


if __name__ == '__main__':
    main()

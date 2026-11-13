#!/usr/bin/env python3
"""Build-only, bounded-memory analysis of the eleven authorized raw sort demos.

Run from icl_ws with openpi/.venv/bin/python -B <this file>. No robot I/O.
Events are command threshold crossings, not proof of successful grasps.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
WS = HERE.parents[3]
sys.path.insert(0, str(WS))
from yor_fk import fk_states

DATA = WS / 'assets/icl-demo-dataset'
EPISODES = range(89, 100)
FPS = 30


def quantiles(x):
    return dict(zip(('p10', 'median', 'p90', 'max'), np.quantile(x, [.1, .5, .9, 1]).tolist()))


def main():
    metadata = {}
    for line in (DATA / 'meta/episodes.jsonl').open():
        row = json.loads(line)
        if row['episode_index'] in EPISODES:
            metadata[row['episode_index']] = row
    index, stats = {}, []
    keydir = HERE / 'keyframes'
    keydir.mkdir(exist_ok=True)
    for ep in EPISODES:
        m = metadata[ep]
        assert m['keep'] and m['valid'] and m['tasks'] == ['sort the items into their containers']
        table = pq.read_table(DATA / f'data/chunk-000/episode_{ep:06d}.parquet')
        assert len(table) == m['length']
        assert table['frame_index'].to_pylist() == list(range(len(table)))
        assert set(table['episode_index'].to_pylist()) == {ep}
        assert all(table['keep'].to_pylist()) and all(table['valid'].to_pylist())
        index[str(ep)] = len(table)
        states = np.array(table['observation.state'].to_pylist())
        grips = np.array(table['action.gripper'].to_pylist())
        measured = fk_states(states)
        row = dict(episode=ep, length=len(table), duration_s=len(table)/FPS, arms={})
        for j, arm in enumerate(('left', 'right')):
            cmd = np.array(table[f'action.{arm}_ee'].to_pylist())
            assert np.isfinite(cmd).all() and np.isfinite(states).all() and np.isfinite(grips).all()
            speed = np.linalg.norm(np.diff(cmd[:, 4:], axis=0), axis=1)*FPS
            # Visual audit of 98:219/250/290/330 and 98:446/500/587/620:
            # LOW intervals carry objects. This conflicts with PIPELINE's dataset
            # convention; record raw transitions and never turn these into commands.
            closed = grips[:, j] < .5
            closes = (np.flatnonzero(~closed[:-1] & closed[1:])+1).tolist()
            opens = (np.flatnonzero(closed[:-1] & ~closed[1:])+1).tolist()
            cycles = []
            for close in closes:
                later = [f for f in opens if f > close]
                if not later:
                    continue
                release = later[0]
                prior = [f for f in opens if f < close]
                approach = prior[-1] if prior else 0
                peak = close + int(np.argmax(cmd[close:release+1, 6]))
                verified_height = np.flatnonzero(cmd[close:release+1, 6] >= cmd[close, 6]+.03)
                cycles.append(dict(close=close, release=release, approach_start=approach, peak=peak,
                    approach_s=(close-approach)/FPS, closed_s=(release-close)/FPS,
                    lift_30mm_s=float(verified_height[0]/FPS) if len(verified_height) else None,
                    lift_m=float(cmd[peak, 6]-cmd[close, 6]),
                    release_dz_m=float(cmd[release, 6]-cmd[close, 6]),
                    transfer_dxy_m=(cmd[release, 4:6]-cmd[close, 4:6]).tolist(),
                    horizontal_distance_m=float(np.linalg.norm(cmd[release, 4:6]-cmd[close, 4:6])),
                    measured_lift_m=float(np.max(measured[j][close:release+1, 6])-measured[j][close, 6]),
                    measured_release_dz_m=float(measured[j][release, 6]-measured[j][close, 6]),
                    closed_speed_m_s=quantiles(speed[close:release]),
                    close_grip=float(grips[close, j]), release_grip=float(grips[release, j])))
            row['arms'][arm] = dict(path_m=float(speed.sum()/FPS), speed_m_s=quantiles(speed),
                joint_range_rad=np.ptp(states[:, j*7:j*7+7], axis=0).tolist(),
                measured_path_m=float(np.linalg.norm(np.diff(measured[j][:, 4:], axis=0), axis=1).sum()),
                command_measured_position_error_m=quantiles(np.linalg.norm(cmd[:, 4:]-measured[j][:, 4:], axis=1)),
                low_crossings=closes, high_crossings=opens, cycles=cycles)
        stats.append(row)
        # Six head keyframes: start/end plus four evenly spread times. Never upscale.
        frames = np.linspace(0, len(table)-1, 6).round().astype(int).tolist()
        cap = cv2.VideoCapture(str(DATA / f'videos/chunk-000/observation.images.zed/episode_{ep:06d}.mp4'))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(table)
        sheet = Image.new('RGB', (960, 520), 'white')
        draw = ImageDraw.Draw(sheet)
        for k, frame in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = cap.read()
            assert ok, (ep, frame)
            im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            im.thumbnail((320, 240))
            im.save(keydir / f'{ep}_{frame}_zed.jpg', quality=85)
            x, y = (k % 3)*320, (k // 3)*260
            sheet.paste(im, (x, y+20));draw.text((x+4, y+3), f'{ep}:{frame}', fill='black')
        cap.release()
        sheet.save(keydir / f'{ep}_overview.jpg', quality=90)
        row['keyframes'] = frames
    # Focused visual audit: every episode has six overview views; episode 98
    # additionally supplies cycle endpoints and eight right-wrist views.
    focused = {
        '98_cycles': ('zed', [0,219,290,446,587,746,827,926,980,1095,1175,1268,1488,1558,1723,1921,2091,2296]),
        '98_wrist': ('fish1', [219,250,290,330,446,500,587,620]),
    }
    for name, (camera, frames) in focused.items():
        cap = cv2.VideoCapture(str(DATA / f'videos/chunk-000/observation.images.{camera}/episode_000098.mp4'))
        sheet = Image.new('RGB', (1280, ((len(frames)+3)//4)*260), 'white')
        draw = ImageDraw.Draw(sheet)
        for k, frame in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = cap.read()
            assert ok
            im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            im.thumbnail((320,240))
            x,y = (k%4)*320,(k//4)*260
            sheet.paste(im,(x,y+20));draw.text((x+4,y+3), f'98:{frame} {camera}',fill='black')
        cap.release();sheet.save(keydir / f'{name}.jpg',quality=90)
    (HERE / 'demo_index.json').write_text(json.dumps(index, indent=2)+'\n')
    (HERE / 'demo_statistics.json').write_text(json.dumps(stats, indent=2)+'\n')
    print(json.dumps([dict(episode=r['episode'], length=r['length'],
        cycles={a:[(c['close'], c['release']) for c in r['arms'][a]['cycles']] for a in ('left','right')}) for r in stats], indent=2))


if __name__ == '__main__':
    main()

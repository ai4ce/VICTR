"""One-time statistics from the approved raw-pool extraction, without reloading 6GB."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT.parents[2]/'agent_runs/astra_2026-10-09/square_task'


def analyze():
    episodes=[];segments=[];index={}
    for ep in range(241,252):
        file=PRIOR/f'ep{ep}.npz';d=np.load(file);a=d['actions'];n=len(a)
        frame=np.concatenate([np.arange(f,f+5) for f in d['frames']])
        index[str(ep)]=d['frames'].astype(int).tolist()
        speeds=[]
        for arm,j in [('left',0),('right',1)]:
            xyz=a[:,j*7+4:j*7+7]
            v=np.linalg.norm(np.diff(xyz,axis=0),axis=1)*30
            moving=v[v>.002]
            speeds.append(dict(arm=arm,median_moving_m_s=float(np.median(moving)) if len(moving) else 0,
                               p90_moving_m_s=float(np.percentile(moving,90)) if len(moving) else 0))
            closed=a[:,14+j]<.2;edges=np.diff(np.r_[False,closed,False].astype(int))
            for lo,hi in zip(np.where(edges==1)[0],np.where(edges==-1)[0]):
                if hi-lo<15:continue
                before=np.where(a[:lo,14+j]>.8)[0];closing=int(before[-1]+1) if len(before) else lo
                peak=lo+int(np.argmax(xyz[lo:hi,2]));rel=min(hi,n-1)
                clear=np.where(xyz[lo:hi,2]>=xyz[lo,2]+.04)[0]
                clear=lo+int(clear[0]) if len(clear) else None
                quat=a[:,j*7:j*7+4]
                rot=Rotation.from_quat(quat[[lo,rel]][:,[1,2,3,0]])
                segments.append(dict(episode=ep,arm=arm,closing_frame=int(frame[closing]),grasp_frame=int(frame[lo]),
                    clearance_frame=int(frame[clear]) if clear is not None else None,
                    peak_frame=int(frame[peak]),release_frame=int(frame[rel]),
                    preclose_s=closing/30,closing_s=(lo-closing)/30,held_s=(hi-lo)/30,
                    lift_40mm_s=(clear-lo)/30 if clear is not None else None,
                    grasp_to_peak_s=(peak-lo)/30,lowering_s=(rel-peak)/30,postrelease_s=(n-rel)/30,
                    grasp_xyz=xyz[lo].tolist(),approach_z=float(np.median(xyz[max(0,closing-30):closing+1,2])),
                    peak_z=float(xyz[peak,2]),release_xyz=xyz[rel].tolist(),
                    lift_m=float(xyz[peak,2]-xyz[lo,2]),lowering_m=float(xyz[peak,2]-xyz[rel,2]),
                    transfer_xy_m=float(np.linalg.norm(xyz[rel,:2]-xyz[lo,:2])),
                    orientation_change_deg=float((rot[0].inv()*rot[1]).magnitude()*180/np.pi)))
        episodes.append(dict(episode=ep,sampled_frames=n,sampled_duration_s=n/30,
            arms_with_grasp=[s['arm'] for s in segments if s['episode']==ep],speeds=speeds,
            raw_extraction_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    clean=[s for s in segments if s['arm']=='right']
    fields=['closing_s','held_s','lift_40mm_s','grasp_to_peak_s','lowering_s','grasp_to_peak_s','approach_z','peak_z','lift_m','lowering_m','transfer_xy_m','orientation_change_deg']
    summary={k:float(np.median([s[k] for s in clean if s[k] is not None])) for k in fields}
    data=dict(source='Prior extract.py copied pool raw actions/proprio and images unchanged; no embeddings/values',
              fps=30,episodes=episodes,segments=segments,right_segment_medians=summary,
              frame_coverage_note='Each overlapping pool chunk contributed its first five frames; last five-frame tail is absent.')
    (ROOT/'notes/demo_statistics.json').write_text(json.dumps(data,indent=2)+'\n')
    (ROOT/'notes/demo_index.json').write_text(json.dumps(index,separators=(',',':'))+'\n')
    print(json.dumps(summary,indent=2))
    for e in episodes:
        s=next(s for s in clean if s['episode']==e['episode'])
        print(e['episode'],e['sampled_duration_s'],e['arms_with_grasp'],'frames',s['closing_frame'],s['grasp_frame'],s['peak_frame'],s['release_frame'],
              'phase_s',round(s['closing_s'],2),round(s['held_s'],2),round(s['lift_40mm_s'],2) if s['lift_40mm_s'] else None,round(s['lowering_s'],2),
              'z',*[round(v,3) for v in (s['approach_z'],s['grasp_xyz'][2],s['peak_z'],s['release_xyz'][2])],
              'v',round(e['speeds'][1]['median_moving_m_s'],3),round(e['speeds'][1]['p90_moving_m_s'],3))


if __name__=='__main__':analyze()
